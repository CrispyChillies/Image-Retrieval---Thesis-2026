import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor
from transformers import AutoModel, AutoProcessor, Trainer, TrainingArguments

from read_data import ChestXrayDataSet


COVIDX_LABEL_TO_NAME = {
    0: "normal",
    1: "pneumonia",
    2: "COVID-19",
}

COVIDX_LABEL_TO_TEXT = {
    0: "A chest X-ray showing no evidence of pneumonia or COVID-19 infection.",
    1: "A chest X-ray showing findings consistent with pneumonia.",
    2: "A chest X-ray showing findings consistent with COVID-19 pneumonia.",
}


def retrieval_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.cpu()
        target = target.cpu()
        pred = target[pred].t()
        correct = pred.eq(target[None])

        res = []
        batch_size = target.size(0)
        for k in topk:
            correct_k = correct[:k].any(dim=0).sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
    return res


class MedSigLIPCOVIDXDataset(Dataset):
    """Wrap ChestXrayDataSet to expose paired image-text examples for retrieval fine-tuning."""

    def __init__(self, base_dataset, processor, max_text_length=64):
        self.base_dataset = base_dataset
        self.processor = processor
        self.max_text_length = max_text_length

        size = processor.image_processor.size
        if isinstance(size, dict):
            image_size = size.get("height") or size.get("shortest_edge")
        else:
            image_size = size
        mean = processor.image_processor.image_mean
        std = processor.image_processor.image_std

        self.image_transform = Compose([
            Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image_path = self.base_dataset.image_names[index]
        class_label = int(self.base_dataset.labels[index])

        image = self.base_dataset[index][0]
        if not torch.is_tensor(image):
            image = self.image_transform(image.convert("RGB"))

        # The text paired with each image defines the positive image-text pair.
        # MedSigLIP then uses all other pairs in the batch as in-batch negatives,
        # which is the correct retrieval objective for this setup.
        text = COVIDX_LABEL_TO_TEXT[class_label]
        text_inputs = self.processor.tokenizer(
            text,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )

        return {
            "pixel_values": image,
            "input_ids": torch.tensor(text_inputs["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(text_inputs["attention_mask"], dtype=torch.long),
            "class_labels": torch.tensor(class_label, dtype=torch.long),
            "text": text,
            "image_path": image_path,
        }


def collate_fn(examples):
    return {
        "pixel_values": torch.stack([example["pixel_values"] for example in examples]),
        "input_ids": torch.stack([example["input_ids"] for example in examples]),
        "attention_mask": torch.stack([example["attention_mask"] for example in examples]),
        "class_labels": torch.stack([example["class_labels"] for example in examples]),
        "return_loss": True,
    }


class RetrievalTrainer(Trainer):
    """Trainer that keeps MedSigLIP contrastive training but evaluates with retrieval metrics."""

    @torch.no_grad()
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            return metrics

        eval_loader = DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=collate_fn,
        )

        model = self.model
        model.eval()
        device = self.args.device

        embeds = []
        labels = []
        for batch in eval_loader:
            pixel_values = batch["pixel_values"].to(device)
            if hasattr(model, "get_image_features"):
                image_features = model.get_image_features(pixel_values=pixel_values)
            else:
                outputs = model(pixel_values=pixel_values)
                image_features = outputs.image_embeds

            embeds.append(F.normalize(image_features, dim=-1).cpu())
            labels.append(batch["class_labels"].cpu())

        embeds = torch.cat(embeds, dim=0)
        labels = torch.cat(labels, dim=0)

        dists = embeds @ embeds.t()
        dists.fill_diagonal_(float("-inf"))

        r1, r5, r10 = retrieval_accuracy(dists, labels, topk=(1, 5, 10))
        metrics[f"{metric_key_prefix}_r1"] = r1.item()
        metrics[f"{metric_key_prefix}_r5"] = r5.item()
        metrics[f"{metric_key_prefix}_r10"] = r10.item()

        self.log(metrics)
        return metrics


def build_datasets(args, processor):
    train_base = ChestXrayDataSet(
        data_dir=os.path.join(args.dataset_dir, "train"),
        image_list_file=args.train_image_list,
        use_covid=not args.anomaly,
        mask_dir=os.path.join(args.mask_dir, "train") if args.mask_dir else None,
        transform=None,
    )
    eval_base = ChestXrayDataSet(
        data_dir=os.path.join(args.dataset_dir, "test"),
        image_list_file=args.test_image_list,
        mask_dir=os.path.join(args.mask_dir, "test") if args.mask_dir else None,
        transform=None,
    )

    train_dataset = MedSigLIPCOVIDXDataset(train_base, processor=processor, max_text_length=args.max_text_length)
    eval_dataset = MedSigLIPCOVIDXDataset(eval_base, processor=processor, max_text_length=args.max_text_length)
    return train_dataset, eval_dataset


def maybe_freeze_model(model, freeze_vision, freeze_text):
    if freeze_vision and hasattr(model, "vision_model"):
        for param in model.vision_model.parameters():
            param.requires_grad = False

    if freeze_text and hasattr(model, "text_model"):
        for param in model.text_model.parameters():
            param.requires_grad = False


def maybe_enable_gradient_checkpointing(model):
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()


def build_trainer(args, model, processor, train_dataset, eval_dataset):
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        remove_unused_columns=False,
        do_train=True,
        do_eval=(args.eval_strategy != "no"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_num_workers=args.workers,
        report_to=args.report_to,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="eval_r1",
        greater_is_better=True,
        seed=args.seed,
    )

    return RetrievalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        processing_class=processor,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune MedSigLIP on COVIDx CXR for retrieval")
    parser.add_argument("--model-name", default="google/medsiglip-448", help="Hugging Face model id")
    parser.add_argument("--dataset-dir", required=True, help="Root COVIDx dataset directory")
    parser.add_argument("--train-image-list", default="./train_split.txt", help="Training split file")
    parser.add_argument("--test-image-list", default="./test_COVIDx4.txt", help="Evaluation split file")
    parser.add_argument("--mask-dir", default=None, help="Optional segmentation mask root directory")
    parser.add_argument("--anomaly", action="store_true", help="Exclude COVID-19 from training")
    parser.add_argument("--output-dir", default="./medsiglip-covidx-ft", help="Where checkpoints will be saved")
    parser.add_argument("--epochs", default=2, type=int, help="Number of training epochs")
    parser.add_argument("--train-batch-size", default=4, type=int, help="Per-device train batch size")
    parser.add_argument("--eval-batch-size", default=4, type=int, help="Per-device eval batch size")
    parser.add_argument("--gradient-accumulation-steps", default=8, type=int, help="Gradient accumulation steps")
    parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate")
    parser.add_argument("--weight-decay", default=0.01, type=float, help="Weight decay")
    parser.add_argument("--warmup-steps", default=50, type=int, help="Warmup steps")
    parser.add_argument("--lr-scheduler-type", default="cosine", help="Scheduler type")
    parser.add_argument("--max-text-length", default=64, type=int, help="Tokenizer max text length")
    parser.add_argument("--workers", default=4, type=int, help="Dataloader workers")
    parser.add_argument("--logging-steps", default=20, type=int, help="Logging interval")
    parser.add_argument("--eval-steps", default=100, type=int, help="Evaluation interval when using step eval")
    parser.add_argument("--save-steps", default=100, type=int, help="Checkpoint interval when using step save")
    parser.add_argument("--eval-strategy", default="steps", choices=["no", "steps", "epoch"], help="Evaluation schedule")
    parser.add_argument("--save-strategy", default="steps", choices=["no", "steps", "epoch"], help="Checkpoint schedule")
    parser.add_argument("--save-total-limit", default=2, type=int, help="Maximum checkpoints to keep")
    parser.add_argument("--load-best-model-at-end", action="store_true", help="Reload best checkpoint at the end")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 mixed precision")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 mixed precision")
    parser.add_argument("--report-to", default="tensorboard", help="Trainer reporting backend")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--push-to-hub", action="store_true", help="Push final model to Hugging Face Hub")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--freeze-vision", action="store_true", help="Freeze the vision tower")
    parser.add_argument("--freeze-text", action="store_true", help="Freeze the text tower")
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)

    if args.gradient_checkpointing:
        maybe_enable_gradient_checkpointing(model)
    maybe_freeze_model(model, freeze_vision=args.freeze_vision, freeze_text=args.freeze_text)

    train_dataset, eval_dataset = build_datasets(args, processor)
    trainer = build_trainer(args, model, processor, train_dataset, eval_dataset)

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    if args.eval_strategy != "no":
        metrics = trainer.evaluate()
        print(
            f"Final retrieval metrics: "
            f"R@1={metrics.get('eval_r1', 0.0):.2f}, "
            f"R@5={metrics.get('eval_r5', 0.0):.2f}, "
            f"R@10={metrics.get('eval_r10', 0.0):.2f}"
        )

    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
