import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from ath_model import ATHNet
from read_data import ChestXrayDataSet, ISICDataSet


def build_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def build_datasets(args, transform):
    if args.dataset == "covid":
        gallery_dataset = ChestXrayDataSet(
            data_dir=os.path.join(args.dataset_dir, "train"),
            image_list_file=args.gallery_image_list,
            use_covid=True,
            transform=transform,
        )
        query_dataset = ChestXrayDataSet(
            data_dir=os.path.join(args.dataset_dir, "test"),
            image_list_file=args.query_image_list,
            use_covid=True,
            transform=transform,
        )
    elif args.dataset == "isic":
        gallery_dataset = ISICDataSet(
            data_dir=os.path.join(args.dataset_dir, "ISIC-2017_Training_Data"),
            image_list_file=args.gallery_image_list,
            use_melanoma=True,
            transform=transform,
        )
        query_dataset = ISICDataSet(
            data_dir=os.path.join(args.dataset_dir, "ISIC-2017_Test_v2_Data"),
            image_list_file=args.query_image_list,
            use_melanoma=True,
            transform=transform,
        )
    else:
        raise NotImplementedError(f"Unsupported dataset: {args.dataset}")

    return gallery_dataset, query_dataset


@torch.no_grad()
def extract_codes_logits_labels(model, data_loader, device, binary_codes):
    model.eval()
    all_codes = []
    all_logits = []
    all_labels = []

    for images, labels in data_loader:
        images = images.to(device)
        codes, logits = model(images)
        if binary_codes:
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


def compute_metrics(
    query_codes,
    query_labels,
    gallery_codes,
    gallery_labels,
    query_logits,
    topk_values,
    binary_codes,
):
    distances = pairwise_distance(query_codes, gallery_codes, binary_codes)
    sorted_indices = torch.argsort(distances, dim=1)

    # Precompute total relevant items for each query (for recall calculation)
    total_relevant_per_query = {}
    for row, label_tensor in enumerate(query_labels):
        label = int(label_tensor.item())
        total_relevant = (gallery_labels == label).sum().item()
        total_relevant_per_query[row] = total_relevant

    retrieval = {}
    for topk in topk_values:
        hit_scores = []
        ap_scores = []
        rr_scores = []
        vote_scores = []
        precision_at_k_scores = []
        recall_at_k_scores = []

        for row, label_tensor in enumerate(query_labels):
            label = int(label_tensor.item())
            ranked_indices = sorted_indices[row, :topk]
            ranked_labels = gallery_labels[ranked_indices]
            matches = (ranked_labels == label).cpu().numpy().astype(np.int32)

            hit_scores.append(float(matches.any()))

            # Precision@K: number of relevant items in top-K / K
            num_relevant_at_k = matches.sum()
            precision_at_k = num_relevant_at_k / topk
            precision_at_k_scores.append(precision_at_k)

            # Recall@K: number of relevant items in top-K / total relevant items
            total_relevant = total_relevant_per_query[row]
            recall_at_k = (
                num_relevant_at_k / total_relevant if total_relevant > 0 else 0.0
            )
            recall_at_k_scores.append(recall_at_k)

            if matches.sum() == 0:
                ap_scores.append(0.0)
                rr_scores.append(0.0)
            else:
                first_rank = None
                precision_sum = 0.0
                positives = 0
                for rank, match in enumerate(matches, start=1):
                    if match:
                        positives += 1
                        precision_sum += positives / rank
                        if first_rank is None:
                            first_rank = rank
                ap_scores.append(precision_sum / positives)
                rr_scores.append(1.0 / first_rank)

            vote = Counter(ranked_labels.tolist()).most_common(1)[0][0]
            vote_scores.append(float(vote == label))

        retrieval[topk] = {
            "mhr": float(np.mean(hit_scores)),
            "map": float(np.mean(ap_scores)),
            "mrr": float(np.mean(rr_scores)),
            "mp@k": float(np.mean(precision_at_k_scores)),
            "r@k": float(np.mean(recall_at_k_scores)),
            "majority_acc": float(np.mean(vote_scores)),
        }

    classification_acc = (
        query_logits.argmax(dim=1).eq(query_labels).float().mean().item()
    )
    return {
        "classification_acc": classification_acc,
        "retrieval": retrieval,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test ATH retrieval on COVIDx CXR or ISIC."
    )
    parser.add_argument("--dataset", choices=["covid", "isic"], required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--gallery-image-list", default=None)
    parser.add_argument("--query-image-list", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--hash-size", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--binary-codes", action="store_true")
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--save-json", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gallery_image_list is None:
        args.gallery_image_list = (
            "./train_split.txt"
            if args.dataset == "covid"
            else "./ISIC-2017_Training_Part3_GroundTruth.csv"
        )
    if args.query_image_list is None:
        args.query_image_list = (
            "./test_COVIDx4.txt"
            if args.dataset == "covid"
            else "./ISIC-2017_Test_v2_Part3_GroundTruth_balanced.csv"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.resume, map_location=device)
    saved_args = checkpoint.get("args", {})

    image_size = args.image_size or saved_args.get("image_size", 256)
    hash_size = args.hash_size or saved_args.get("hash_size", 36)
    num_classes = args.num_classes or saved_args.get("num_classes")
    if num_classes is None:
        num_classes = 3 if args.dataset in {"covid", "isic"} else None
    if num_classes is None:
        raise ValueError("Unable to infer num_classes. Pass --num-classes explicitly.")

    transform = build_transform(image_size)
    gallery_dataset, query_dataset = build_datasets(args, transform)

    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = ATHNet(hash_size=hash_size, num_classes=num_classes, input_size=image_size)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    gallery_codes, _, gallery_labels = extract_codes_logits_labels(
        model, gallery_loader, device, args.binary_codes
    )
    query_codes, query_logits, query_labels = extract_codes_logits_labels(
        model, query_loader, device, args.binary_codes
    )

    metrics = compute_metrics(
        query_codes=query_codes,
        query_labels=query_labels,
        gallery_codes=gallery_codes,
        gallery_labels=gallery_labels,
        query_logits=query_logits,
        topk_values=args.topk,
        binary_codes=args.binary_codes,
    )

    print(f"Classification accuracy: {metrics['classification_acc'] * 100.0:.2f}%")
    for topk in args.topk:
        result = metrics["retrieval"][topk]
        print(
            f"Top-{topk} | "
            f"mHR={result['mhr'] * 100.0:.2f}% | "
            f"mP@K={result['mp@k'] * 100.0:.2f}% | "
            f"R@K={result['r@k'] * 100.0:.2f}% | "
            f"mAP={result['map'] * 100.0:.2f}% | "
            f"mRR={result['mrr'] * 100.0:.2f}% | "
            f"vote_acc={result['majority_acc'] * 100.0:.2f}%"
        )

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {args.save_json}")


if __name__ == "__main__":
    main()
