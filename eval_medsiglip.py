import argparse
import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoProcessor

from read_data import ChestXrayDataSet
from train_medsiglip import COVIDX_LABEL_TO_TEXT


def retrieval_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.cpu()
        target = target.cpu()
        pred = target[pred].t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].any(dim=0).sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
    return res


def compute_ap(ranks, nres):
    ap = 0
    recall_step = 1.0 / nres
    for j in np.arange(len(ranks)):
        rank = ranks[j]
        precision_0 = 1.0 if rank == 0 else float(j) / rank
        precision_1 = float(j + 1) / (rank + 1)
        ap += (precision_0 + precision_1) * recall_step / 2.0
    return ap


def compute_map(ranks, gnd, kappas=None):
    if kappas is None:
        kappas = []

    m_ap = 0.0
    nq = len(gnd)
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.where(gnd == gnd[i])[0]
        if qgnd.shape[0] == 0:
            aps[i] = float("nan")
            prs[i, :] = float("nan")
            nempty += 1
            continue

        pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgnd)]
        ap = compute_ap(pos, len(qgnd))
        m_ap += ap
        aps[i] = ap

        pos += 1
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j])
            prs[i, j] = (pos <= kq).sum() / kq
        pr += prs[i, :]

    m_ap = m_ap / (nq - nempty)
    pr = pr / (nq - nempty)
    return m_ap, aps, pr, prs


def majority_vote(retrieved_labels):
    if len(retrieved_labels) == 0:
        return None
    counter = Counter(retrieved_labels)
    return counter.most_common(1)[0][0]


def compute_classification_metrics(labels, dists, k_values=(1, 5, 10, 15, 20)):
    labels_np = labels.cpu().numpy()
    n_samples = labels.size(0)
    ranks = torch.argsort(dists, dim=0, descending=True).cpu().numpy()

    results = {}
    for k in k_values:
        predicted_labels = []
        true_labels = []
        for i in range(n_samples):
            top_k_indices = ranks[:k, i]
            retrieved_labels = labels_np[top_k_indices]
            pred_label = majority_vote(retrieved_labels)
            predicted_labels.append(pred_label)
            true_labels.append(labels_np[i])

        results[k] = {
            "precision_macro": precision_score(true_labels, predicted_labels, average="macro", zero_division=0) * 100.0,
            "recall_macro": recall_score(true_labels, predicted_labels, average="macro", zero_division=0) * 100.0,
            "f1_macro": f1_score(true_labels, predicted_labels, average="macro", zero_division=0) * 100.0,
            "precision_weighted": precision_score(true_labels, predicted_labels, average="weighted", zero_division=0) * 100.0,
            "recall_weighted": recall_score(true_labels, predicted_labels, average="weighted", zero_division=0) * 100.0,
            "f1_weighted": f1_score(true_labels, predicted_labels, average="weighted", zero_division=0) * 100.0,
            "accuracy": accuracy_score(true_labels, predicted_labels) * 100.0,
        }
    return results


class MedSigLIPEvalDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, label = self.base_dataset[index]
        return {
            "image": image,
            "label": label,
            "text": COVIDX_LABEL_TO_TEXT[int(label)],
            "path": self.base_dataset.image_names[index],
        }


def collate_fn(batch, processor):
    images = [item["image"] for item in batch]
    labels = torch.stack([item["label"] if isinstance(item["label"], torch.Tensor) else torch.tensor(item["label"]) for item in batch])
    paths = [item["path"] for item in batch]
    texts = [item["text"] for item in batch]
    pixel_values = processor.image_processor(images=images, return_tensors="pt")["pixel_values"]
    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "paths": paths,
        "texts": texts,
    }


def build_loader(args, processor):
    base_dataset = ChestXrayDataSet(
        data_dir=os.path.join(args.dataset_dir, "test"),
        image_list_file=args.test_image_list,
        mask_dir=os.path.join(args.mask_dir, "test") if args.mask_dir else None,
        transform=None,
    )
    dataset = MedSigLIPEvalDataset(base_dataset)
    return DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=lambda batch: collate_fn(batch, processor),
    )


@torch.no_grad()
def get_text_features(model, processor, device, prompts, max_text_length):
    text_inputs = processor.tokenizer(
        prompts,
        max_length=max_text_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    ).to(device)

    if hasattr(model, "get_text_features"):
        text_features = model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        )
    else:
        outputs = model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        )
        text_features = outputs.text_embeds

    return F.normalize(text_features, dim=-1)


@torch.no_grad()
def evaluate(model, processor, loader, device, args):
    model.eval()

    class_prompts = [COVIDX_LABEL_TO_TEXT[i] for i in sorted(COVIDX_LABEL_TO_TEXT)]
    text_features = get_text_features(model, processor, device, class_prompts, args.max_text_length)

    all_predictions = []
    all_labels = []
    embeds = []
    logit_scale = model.logit_scale.exp() if hasattr(model, "logit_scale") else torch.tensor(100.0, device=device)

    for batch_idx, batch in enumerate(loader):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        if hasattr(model, "get_image_features"):
            image_features = model.get_image_features(pixel_values=pixel_values)
        else:
            outputs = model(pixel_values=pixel_values)
            image_features = outputs.image_embeds

        image_features = F.normalize(image_features, dim=-1)
        logits = logit_scale * image_features @ text_features.t()
        predictions = torch.argmax(logits, dim=-1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        embeds.append(image_features.cpu())

        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {(batch_idx + 1) * args.eval_batch_size} images...")

    labels = torch.tensor(np.array(all_labels), dtype=torch.long)
    embeds = torch.cat(embeds, dim=0)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    accuracy_cls = accuracy_score(all_labels, all_predictions) * 100.0
    precision_macro = precision_score(all_labels, all_predictions, average="macro", zero_division=0) * 100.0
    recall_macro = recall_score(all_labels, all_predictions, average="macro", zero_division=0) * 100.0
    f1_macro = f1_score(all_labels, all_predictions, average="macro", zero_division=0) * 100.0

    print("\n>> Zero-shot Classification Metrics:")
    print(f"   Accuracy: {accuracy_cls:.2f}%")
    print(f"   Precision (macro): {precision_macro:.2f}%")
    print(f"   Recall (macro): {recall_macro:.2f}%")
    print(f"   F1 (macro): {f1_macro:.2f}%")

    dists = embeds @ embeds.t()
    dists.fill_diagonal_(float("-inf"))

    kappas = [1, 5, 10]
    accuracy = retrieval_accuracy(dists, labels, topk=kappas)
    accuracy = torch.stack(accuracy).cpu().numpy()
    print(f">> R@K{kappas}: {np.around(accuracy, 2)}%")

    ranks = torch.argsort(dists, dim=0, descending=True)
    m_ap, _, pr, _ = compute_map(ranks.cpu().numpy(), labels.cpu().numpy(), kappas)
    print(f">> mAP: {m_ap * 100.0:.2f}%")
    print(f">> mP@K{kappas}: {np.around(pr * 100.0, 2)}%")

    print("\n>> Retrieval Classification Metrics (Majority Voting):")
    k_values = [1, 5, 10, 15, 20]
    classification_results = compute_classification_metrics(labels, dists, k_values)
    for k in k_values:
        metrics = classification_results[k]
        print(f"\n>> Top-{k} Retrieved Images:")
        print(f"   Accuracy: {metrics['accuracy']:.2f}%")
        print(f"   Precision (macro): {metrics['precision_macro']:.2f}%")
        print(f"   Recall (macro): {metrics['recall_macro']:.2f}%")
        print(f"   F1 (macro): {metrics['f1_macro']:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MedSigLIP on COVIDx retrieval and zero-shot classification")
    parser.add_argument("--model-path", required=True, help="Fine-tuned checkpoint directory or Hugging Face model id")
    parser.add_argument("--dataset-dir", required=True, help="Root COVIDx dataset directory")
    parser.add_argument("--test-image-list", default="./test_COVIDx4.txt", help="Evaluation split file")
    parser.add_argument("--mask-dir", default=None, help="Optional segmentation mask root directory")
    parser.add_argument("--eval-batch-size", default=8, type=int, help="Evaluation batch size")
    parser.add_argument("--workers", default=4, type=int, help="Dataloader workers")
    parser.add_argument("--max-text-length", default=64, type=int, help="Tokenizer max text length")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).to(device)

    loader = build_loader(args, processor)
    evaluate(model, processor, loader, device, args)


if __name__ == "__main__":
    main()
