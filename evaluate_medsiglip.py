import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoProcessor

from read_data import ChestXrayDataSet
from train_medsiglip import COVIDX_LABEL_TO_TEXT


LABEL_NAMES = {
    0: "normal",
    1: "pneumonia",
    2: "COVID-19",
}


class MedSigLIPImageDataset(torch.utils.data.Dataset):
    """Image-only view of COVIDx that still exposes the paired text prompt and path."""

    def __init__(self, base_dataset, processor):
        self.base_dataset = base_dataset
        self.processor = processor

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image_path = self.base_dataset.image_names[index]
        label = int(self.base_dataset.labels[index])
        image = self.base_dataset[index][0]
        text = COVIDX_LABEL_TO_TEXT[label]
        return {
            "image": image,
            "label": label,
            "text": text,
            "image_path": image_path,
        }


def collate_examples(examples, processor):
    images = [example["image"] for example in examples]
    labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)
    texts = [example["text"] for example in examples]
    image_paths = [example["image_path"] for example in examples]

    image_inputs = processor.image_processor(images=images, return_tensors="pt")
    return {
        "pixel_values": image_inputs["pixel_values"],
        "labels": labels,
        "texts": texts,
        "image_paths": image_paths,
    }


def build_loader(args, processor):
    dataset = ChestXrayDataSet(
        data_dir=os.path.join(args.dataset_dir, "test"),
        image_list_file=args.test_image_list,
        mask_dir=os.path.join(args.mask_dir, "test") if args.mask_dir else None,
        transform=None,
    )
    eval_dataset = MedSigLIPImageDataset(dataset, processor)
    return DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=lambda batch: collate_examples(batch, processor),
    )


def get_text_features(model, processor, device, prompts, max_text_length):
    inputs = processor.tokenizer(
        prompts,
        max_length=max_text_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if hasattr(model, "get_text_features"):
        text_features = model.get_text_features(**inputs)
    else:
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        text_features = outputs.text_embeds
    return F.normalize(text_features, dim=-1)


@torch.no_grad()
def collect_image_features(model, loader, device):
    all_features = []
    all_labels = []
    all_paths = []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        if hasattr(model, "get_image_features"):
            image_features = model.get_image_features(pixel_values=pixel_values)
        else:
            outputs = model(pixel_values=pixel_values)
            image_features = outputs.image_embeds

        all_features.append(F.normalize(image_features, dim=-1).cpu())
        all_labels.append(batch["labels"].cpu())
        all_paths.extend(batch["image_paths"])

    return (
        torch.cat(all_features, dim=0),
        torch.cat(all_labels, dim=0),
        all_paths,
    )


def evaluate_zeroshot(image_features, labels, text_features):
    logits = image_features @ text_features.T
    preds = logits.argmax(dim=1).numpy()
    targets = labels.numpy()

    return {
        "accuracy": accuracy_score(targets, preds) * 100.0,
        "macro_f1": f1_score(targets, preds, average="macro") * 100.0,
        "weighted_f1": f1_score(targets, preds, average="weighted") * 100.0,
        "classification_report": classification_report(
            targets,
            preds,
            target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
            digits=4,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(targets, preds).tolist(),
    }


def evaluate_retrieval(image_features, labels, topk_values):
    sim = image_features @ image_features.T
    sim.fill_diagonal_(-1.0)
    ranks = sim.argsort(dim=1, descending=True)

    labels_np = labels.numpy()
    results = {}
    for k in topk_values:
        topk = ranks[:, :k].numpy()
        retrieved = labels_np[topk]
        query = labels_np[:, None]
        hit = (retrieved == query).any(axis=1).mean() * 100.0
        majority = []
        for row in retrieved:
            values, counts = np.unique(row, return_counts=True)
            majority.append(int(values[np.argmax(counts)]))
        majority = np.array(majority)
        results[f"r_at_{k}"] = hit
        results[f"majority_accuracy_at_{k}"] = accuracy_score(labels_np, majority) * 100.0
        results[f"majority_macro_f1_at_{k}"] = f1_score(labels_np, majority, average="macro") * 100.0

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MedSigLIP on COVIDx CXR")
    parser.add_argument("--model-path", required=True, help="Fine-tuned checkpoint directory or HF model id")
    parser.add_argument("--dataset-dir", required=True, help="Root COVIDx dataset directory")
    parser.add_argument("--test-image-list", default="./test_COVIDx4.txt", help="Evaluation split file")
    parser.add_argument("--mask-dir", default=None, help="Optional segmentation mask root directory")
    parser.add_argument("--batch-size", default=8, type=int, help="Evaluation batch size")
    parser.add_argument("--workers", default=4, type=int, help="Dataloader workers")
    parser.add_argument("--max-text-length", default=64, type=int, help="Tokenizer max text length")
    parser.add_argument("--topk", default="1,5,10", help="Comma-separated retrieval k values")
    parser.add_argument("--output-json", default=None, help="Optional path to save metrics as json")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).to(device)
    model.eval()

    loader = build_loader(args, processor)
    image_features, labels, image_paths = collect_image_features(model, loader, device)

    class_prompts = [COVIDX_LABEL_TO_TEXT[i] for i in sorted(COVIDX_LABEL_TO_TEXT)]
    text_features = get_text_features(model, processor, device, class_prompts, args.max_text_length).cpu()

    zeroshot_metrics = evaluate_zeroshot(image_features, labels, text_features)
    topk_values = [int(x) for x in args.topk.split(",") if x.strip()]
    retrieval_metrics = evaluate_retrieval(image_features, labels, topk_values)

    metrics = {
        "model_path": args.model_path,
        "dataset_dir": args.dataset_dir,
        "num_samples": len(image_paths),
        "class_prompts": class_prompts,
        "zero_shot": zeroshot_metrics,
        "retrieval": retrieval_metrics,
    }

    print("\n=== MedSigLIP Zero-Shot Classification ===")
    print(f"Accuracy: {zeroshot_metrics['accuracy']:.2f}%")
    print(f"Macro F1: {zeroshot_metrics['macro_f1']:.2f}%")
    print(f"Weighted F1: {zeroshot_metrics['weighted_f1']:.2f}%")
    print("\nClassification report:")
    print(zeroshot_metrics["classification_report"])

    print("Confusion matrix:")
    for row in zeroshot_metrics["confusion_matrix"]:
        print(row)

    print("\n=== MedSigLIP Image Retrieval ===")
    for key, value in retrieval_metrics.items():
        print(f"{key}: {value:.2f}%")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved metrics to {args.output_json}")


if __name__ == "__main__":
    main()
