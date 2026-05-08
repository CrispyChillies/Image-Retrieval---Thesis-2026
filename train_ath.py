import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ath_model import ATHNet, TripletHashLoss
from read_data import ChestXrayDataSet, ISICDataSet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class OnlineTripletDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.labels = [int(label) for label in base_dataset.labels]
        self.indices_by_label = defaultdict(list)
        for index, label in enumerate(self.labels):
            self.indices_by_label[label].append(index)
        self.label_values = list(self.indices_by_label.keys())
        self.negative_label_choices = {
            label: [other for other in self.label_values if other != label]
            for label in self.label_values
        }

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        anchor_image, anchor_label = self.base_dataset[index]
        anchor_label = int(anchor_label)

        positive_candidates = self.indices_by_label[anchor_label]
        if len(positive_candidates) > 1:
            positive_index = index
            while positive_index == index:
                positive_index = random.choice(positive_candidates)
        else:
            positive_index = index

        negative_label = random.choice(self.negative_label_choices[anchor_label])
        negative_index = random.choice(self.indices_by_label[negative_label])

        positive_image, positive_label = self.base_dataset[positive_index]
        negative_image, negative_label = self.base_dataset[negative_index]

        return (
            anchor_image,
            positive_image,
            negative_image,
            torch.tensor(anchor_label, dtype=torch.long),
            torch.tensor(int(positive_label), dtype=torch.long),
            torch.tensor(int(negative_label), dtype=torch.long),
        )


def build_transforms(image_size):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    return train_transform, eval_transform


def build_datasets(args, train_transform, eval_transform):
    if args.dataset == "covid":
        train_dataset = ChestXrayDataSet(
            data_dir=os.path.join(args.dataset_dir, "train"),
            image_list_file=args.train_image_list,
            use_covid=True,
            transform=train_transform,
        )
        train_eval_dataset = ChestXrayDataSet(
            data_dir=os.path.join(args.dataset_dir, "train"),
            image_list_file=args.train_image_list,
            use_covid=True,
            transform=eval_transform,
        )
        query_dataset = ChestXrayDataSet(
            data_dir=os.path.join(args.dataset_dir, "test"),
            image_list_file=args.val_image_list,
            use_covid=True,
            transform=eval_transform,
        )
    elif args.dataset == "isic":
        train_dataset = ISICDataSet(
            data_dir=os.path.join(args.dataset_dir, "ISIC-2017_Training_Data"),
            image_list_file=args.train_image_list,
            use_melanoma=True,
            transform=train_transform,
        )
        train_eval_dataset = ISICDataSet(
            data_dir=os.path.join(args.dataset_dir, "ISIC-2017_Training_Data"),
            image_list_file=args.train_image_list,
            use_melanoma=True,
            transform=eval_transform,
        )
        query_dataset = ISICDataSet(
            data_dir=os.path.join(args.dataset_dir, "ISIC-2017_Test_v2_Data"),
            image_list_file=args.val_image_list,
            use_melanoma=True,
            transform=eval_transform,
        )
    else:
        raise NotImplementedError(f"Unsupported dataset: {args.dataset}")

    return train_dataset, train_eval_dataset, query_dataset


@torch.no_grad()
def extract_codes_and_logits(model, data_loader, device, binarize=False):
    model.eval()
    all_codes = []
    all_logits = []
    all_labels = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        codes, logits = model(images)
        if binarize:
            codes = (codes >= 0).float()
        all_codes.append(codes.cpu())
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    return (
        torch.cat(all_codes, dim=0),
        torch.cat(all_logits, dim=0),
        torch.cat(all_labels, dim=0),
    )


def pairwise_distance(query_codes, gallery_codes, binary_codes):
    if binary_codes:
        query_binary = query_codes.to(torch.int16)
        gallery_binary = gallery_codes.to(torch.int16)
        return (
            (query_binary[:, None, :] != gallery_binary[None, :, :]).sum(dim=2).float()
        )
    return torch.cdist(query_codes.float(), gallery_codes.float(), p=2)


def compute_retrieval_metrics(
    query_codes, query_labels, gallery_codes, gallery_labels, topk_values, binary_codes
):
    distances = pairwise_distance(query_codes, gallery_codes, binary_codes)
    sorted_indices = torch.argsort(distances, dim=1, descending=False)

    results = {}
    for topk in topk_values:
        hits = []
        aps = []
        rrs = []
        majority_accuracy = []

        for i in range(query_labels.shape[0]):
            label = int(query_labels[i].item())
            ranked = sorted_indices[i, :topk]
            ranked_labels = gallery_labels[ranked]
            matches = (ranked_labels == label).cpu().numpy().astype(np.int32)

            hits.append(float(matches.any()))

            if matches.sum() == 0:
                aps.append(0.0)
                rrs.append(0.0)
            else:
                precision_sum = 0.0
                positives = 0
                first_rank = None
                for rank, match in enumerate(matches, start=1):
                    if match:
                        positives += 1
                        precision_sum += positives / rank
                        if first_rank is None:
                            first_rank = rank
                aps.append(precision_sum / positives)
                rrs.append(1.0 / first_rank)

            majority_label = torch.mode(ranked_labels).values.item()
            majority_accuracy.append(float(majority_label == label))

        results[topk] = {
            "mhr": float(np.mean(hits)),
            "map": float(np.mean(aps)),
            "mrr": float(np.mean(rrs)),
            "majority_acc": float(np.mean(majority_accuracy)),
        }

    return results


def train_one_epoch(model, data_loader, optimizer, triplet_loss, ce_loss, device, args):
    model.train()
    running_loss = 0.0

    for batch in data_loader:
        anchor, positive, negative, anchor_y, positive_y, negative_y = batch
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        anchor_y = anchor_y.to(device)
        positive_y = positive_y.to(device)
        negative_y = negative_y.to(device)

        optimizer.zero_grad()

        anchor_hash, anchor_logits = model(anchor)
        positive_hash, positive_logits = model(positive)
        negative_hash, negative_logits = model(negative)

        hash_loss = triplet_loss(anchor_hash, positive_hash, negative_hash)
        type_loss = (
            ce_loss(anchor_logits, anchor_y)
            + ce_loss(positive_logits, positive_y)
            + ce_loss(negative_logits, negative_y)
        ) / 3.0

        loss = args.triplet_weight * hash_loss + args.ce_weight * type_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(len(data_loader), 1)


@torch.no_grad()
def evaluate(model, gallery_loader, query_loader, device, args):
    gallery_codes, _, gallery_labels = extract_codes_and_logits(
        model, gallery_loader, device, binarize=args.binary_eval
    )
    query_codes, query_logits, query_labels = extract_codes_and_logits(
        model, query_loader, device, binarize=args.binary_eval
    )

    retrieval = compute_retrieval_metrics(
        query_codes=query_codes,
        query_labels=query_labels,
        gallery_codes=gallery_codes,
        gallery_labels=gallery_labels,
        topk_values=args.eval_topk,
        binary_codes=args.binary_eval,
    )
    classification_acc = (
        query_logits.argmax(dim=1).eq(query_labels).float().mean().item()
    )

    return {
        "classification_acc": classification_acc,
        "retrieval": retrieval,
    }


def save_checkpoint(model, optimizer, args, epoch, metrics, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "metrics": metrics,
        },
        output_path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train ATH on COVIDx CXR or ISIC.")
    parser.add_argument("--dataset", choices=["covid", "isic"], required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--train-image-list", default=None)
    parser.add_argument("--val-image-list", default=None)
    parser.add_argument("--output-dir", default="./checkpoints/ath")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--hash-size", type=int, default=36)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--triplet-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--binary-eval", action="store_true")
    parser.add_argument("--eval-topk", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--resume", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.train_image_list is None:
        args.train_image_list = (
            "./train_split.txt"
            if args.dataset == "covid"
            else "./ISIC-2017_Training_Part3_GroundTruth.csv"
        )
    if args.val_image_list is None:
        args.val_image_list = (
            "./test_COVIDx4.txt"
            if args.dataset == "covid"
            else "./ISIC-2017_Test_v2_Part3_GroundTruth_balanced.csv"
        )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform, eval_transform = build_transforms(args.image_size)
    train_base, gallery_base, query_base = build_datasets(
        args, train_transform, eval_transform
    )

    num_classes = len(set(int(label) for label in train_base.labels))
    args.num_classes = num_classes
    train_dataset = OnlineTripletDataset(train_base)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    gallery_loader = DataLoader(
        gallery_base,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    query_loader = DataLoader(
        query_base,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = ATHNet(
        hash_size=args.hash_size,
        num_classes=num_classes,
        input_size=args.image_size,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    triplet_loss = TripletHashLoss(margin=args.margin).to(device)
    ce_loss = nn.CrossEntropyLoss().to(device)

    start_epoch = 0
    best_score = float("-inf")
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0))
        saved_metrics = checkpoint.get("metrics", {})
        best_score = (
            saved_metrics.get("retrieval", {})
            .get(max(args.eval_topk), {})
            .get("map", float("-inf"))
        )
        print(f"Loaded checkpoint from {args.resume} at epoch {start_epoch}.")

    best_checkpoint_path = os.path.join(args.output_dir, f"ath_{args.dataset}_best.pth")
    last_checkpoint_path = os.path.join(args.output_dir, f"ath_{args.dataset}_last.pth")
    metrics_path = os.path.join(
        args.output_dir, f"ath_{args.dataset}_best_metrics.json"
    )

    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            triplet_loss=triplet_loss,
            ce_loss=ce_loss,
            device=device,
            args=args,
        )
        metrics = evaluate(model, gallery_loader, query_loader, device, args)
        selected_topk = max(args.eval_topk)
        score = metrics["retrieval"][selected_topk]["map"]

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "
            f"loss={train_loss:.4f} | "
            f"cls_acc={metrics['classification_acc'] * 100.0:.2f}% | "
            f"mAP@{selected_topk}={score * 100.0:.2f}%"
        )

        save_checkpoint(
            model, optimizer, args, epoch + 1, metrics, last_checkpoint_path
        )

        if score > best_score:
            best_score = score
            save_checkpoint(
                model, optimizer, args, epoch + 1, metrics, best_checkpoint_path
            )
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

    print(f"Best checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
