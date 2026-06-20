import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model import ConvNeXtV2, ConvNeXtV2_SRA
from read_data import (
    ChestXrayDataSet,
    ISICDataSet,
    TBX11kDataSet,
    VINDRConceptCLIPDataSet,
)


class VinDRRetrievalDataSet(VINDRConceptCLIPDataSet):
    LEGACY_FINDING6_COLUMNS = [
        "Aortic enlargement",
        "Cardiomegaly",
        "Pleural effusion",
        "Pleural thickening",
        "Lung Opacity",
        "No finding",
    ]

    def __init__(self, *args, label_mode="all", **kwargs):
        super().__init__(*args, return_pil=False, **kwargs)
        self.label_mode = label_mode

    def legacy_finding6_label(self, index):
        row = self.data.iloc[index]
        for label_idx, column in enumerate(self.LEGACY_FINDING6_COLUMNS[:-1]):
            if int(row[column]) == 1:
                return torch.tensor(label_idx, dtype=torch.long)
        return torch.tensor(len(self.LEGACY_FINDING6_COLUMNS) - 1, dtype=torch.long)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if self.label_mode == "concept":
            label = sample["concept_labels"]
        elif self.label_mode == "disease":
            label = sample["disease_labels"]
        elif self.label_mode == "legacy_finding6":
            label = self.legacy_finding6_label(index)
        elif self.label_mode == "all":
            label = sample["all_labels"]
        else:
            raise ValueError(f"Unsupported VinDR label mode: {self.label_mode}")
        return sample["image"], label


def build_model(args):
    if args.model == "convnextv2":
        return ConvNeXtV2(embedding_dim=args.embedding_dim)
    if args.model == "convnextv2_sra":
        return ConvNeXtV2_SRA(
            embedding_dim=args.embedding_dim,
            num_heads=args.sra_num_heads,
            lam=args.sra_lam,
        )
    raise ValueError(f"Unsupported model: {args.model}")


def load_checkpoint(model, resume, device):
    if not resume:
        print("=> no checkpoint path provided")
        return
    if not os.path.isfile(resume):
        raise FileNotFoundError(f"Checkpoint not found: {resume}")

    print(f"=> loading checkpoint: {resume}")
    checkpoint = torch.load(resume, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("state-dict", "state_dict", "model_state_dict"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break

    if isinstance(checkpoint, dict):
        checkpoint = {
            key.replace("module.", "", 1): value
            for key, value in checkpoint.items()
        }

    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    print(f"=> loaded checkpoint; missing={len(missing)}, unexpected={len(unexpected)}")


def build_transform():
    normalize = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def build_dataset(args, transform):
    if args.dataset == "covid":
        return ChestXrayDataSet(
            data_dir=args.test_dataset_dir,
            image_list_file=args.test_image_list,
            mask_dir=args.mask_dir,
            transform=transform,
        )
    if args.dataset == "isic":
        return ISICDataSet(
            data_dir=args.test_dataset_dir,
            image_list_file=args.test_image_list,
            mask_dir=args.mask_dir,
            transform=transform,
        )
    if args.dataset == "tbx11k":
        return TBX11kDataSet(
            data_dir=args.test_dataset_dir,
            csv_file=args.test_image_list,
            transform=transform,
        )
    if args.dataset == "vindr":
        return VinDRRetrievalDataSet(
            data_dir=args.test_dataset_dir,
            csv_file=args.test_image_list,
            transform=transform,
            label_mode=args.vindr_label_mode,
        )
    raise ValueError(f"Unsupported dataset: {args.dataset}")


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    embeds = []
    labels = []

    for batch_idx, (samples, batch_labels) in enumerate(loader, 1):
        samples = samples.to(device)
        output = model(samples)
        if isinstance(output, dict):
            output = output["embedding"]
        embeds.append(output.detach().cpu())
        labels.append(batch_labels.detach().cpu())

        if batch_idx % 10 == 0:
            print(f"Processed {batch_idx * loader.batch_size} images...")

    return torch.cat(embeds, dim=0), torch.cat(labels, dim=0)


def relevance_matrix(labels):
    if labels.ndim == 1:
        rel = labels[:, None].eq(labels[None, :])
    else:
        labels = labels.float()
        rel = (labels @ labels.t()) > 2
    rel.fill_diagonal_(False)
    return rel


def average_precision(relevance):
    relevant_positions = np.flatnonzero(relevance)
    if relevant_positions.size == 0:
        return np.nan

    hits = 0
    precision_sum = 0.0
    for rank_idx, is_relevant in enumerate(relevance, 1):
        if is_relevant:
            hits += 1
            precision_sum += hits / rank_idx
    return precision_sum / relevant_positions.size


def reciprocal_rank(relevance):
    relevant_positions = np.flatnonzero(relevance)
    if relevant_positions.size == 0:
        return np.nan
    return 1.0 / (relevant_positions[0] + 1)


def reciprocal_rank_at_k(relevance, k):
    relevant_positions = np.flatnonzero(relevance[:k])
    if relevant_positions.size == 0:
        return 0.0
    return 1.0 / (relevant_positions[0] + 1)


def average_precision_at_k(relevance, k):
    topk_relevance = relevance[:k]
    denominator = min(int(relevance.sum()), k)
    if denominator == 0:
        return np.nan

    hits = 0
    precision_sum = 0.0
    for rank_idx, is_relevant in enumerate(topk_relevance, 1):
        if is_relevant:
            hits += 1
            precision_sum += hits / rank_idx
    return precision_sum / denominator


def ndcg_at_k(relevance, k):
    gains = relevance[:k].astype(np.float32)
    if gains.size == 0:
        return np.nan
    if gains.sum() == 0:
        return 0.0

    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float(np.sum(gains * discounts))

    ideal_len = min(int(relevance.sum()), k)
    ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_len + 2))
    idcg = float(np.sum(ideal_discounts))
    return dcg / idcg if idcg > 0 else np.nan


def mean_without_nan(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")


def compute_retrieval_metrics(embeds, labels, k_values):
    dists = -torch.cdist(embeds, embeds)
    dists.fill_diagonal_(float("-inf"))
    rankings = torch.argsort(dists, dim=1, descending=True).cpu().numpy()
    relevant = relevance_matrix(labels).cpu().numpy()

    per_query_full_ap = []
    per_query_full_rr = []
    metrics_at_k = {
        k: {"ndcg": [], "mrr": [], "map": []}
        for k in k_values
    }

    for query_idx in range(rankings.shape[0]):
        ranked_relevance = relevant[query_idx, rankings[query_idx]]
        if not ranked_relevance.any():
            per_query_full_ap.append(np.nan)
            per_query_full_rr.append(np.nan)
            for k in k_values:
                metrics_at_k[k]["ndcg"].append(np.nan)
                metrics_at_k[k]["mrr"].append(np.nan)
                metrics_at_k[k]["map"].append(np.nan)
            continue

        per_query_full_ap.append(average_precision(ranked_relevance))
        per_query_full_rr.append(reciprocal_rank(ranked_relevance))

        for k in k_values:
            metrics_at_k[k]["ndcg"].append(ndcg_at_k(ranked_relevance, k))
            metrics_at_k[k]["mrr"].append(reciprocal_rank_at_k(ranked_relevance, k))
            metrics_at_k[k]["map"].append(average_precision_at_k(ranked_relevance, k))

    results = {
        "num_queries": int(rankings.shape[0]),
        "num_evaluated_queries": int(np.sum(~np.isnan(per_query_full_ap))),
        "MAP": mean_without_nan(per_query_full_ap),
        "MRR": mean_without_nan(per_query_full_rr),
        "at_k": {},
    }

    for k in k_values:
        results["at_k"][str(k)] = {
            "nDCG": mean_without_nan(metrics_at_k[k]["ndcg"]),
            "MRR": mean_without_nan(metrics_at_k[k]["mrr"]),
            "MAP": mean_without_nan(metrics_at_k[k]["map"]),
        }

    return results, dists


def print_metrics(results):
    print("\n=== Retrieval Ranking Metrics ===")
    print(f"Queries: {results['num_queries']}")
    print(f"Evaluated queries: {results['num_evaluated_queries']}")
    print(f"MAP: {results['MAP'] * 100.0:.2f}%")
    print(f"MRR: {results['MRR'] * 100.0:.2f}%")

    print("\nK     nDCG@K     MRR@K      MAP@K")
    print("------------------------------------")
    for k, values in results["at_k"].items():
        print(
            f"{int(k):<5} "
            f"{values['nDCG'] * 100.0:>7.2f}% "
            f"{values['MRR'] * 100.0:>9.2f}% "
            f"{values['MAP'] * 100.0:>9.2f}%"
        )


def save_results(args, results, embeds, labels, dists):
    if not args.save_dir:
        return

    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_name = os.path.splitext(os.path.basename(args.resume or args.model))[0]
    base_path = os.path.join(args.save_dir, f"{checkpoint_name}_ranking_metrics")

    with open(f"{base_path}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    np.savez(
        base_path,
        embeds=embeds.numpy(),
        labels=labels.numpy(),
        dists=dists.numpy(),
        metrics_json=json.dumps(results),
    )
    print(f"\n>> Results saved to {base_path}.json and {base_path}.npz")


def parse_k_values(raw_value):
    values = [int(item.strip()) for item in raw_value.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one K value is required.")
    if any(k <= 0 for k in values):
        raise ValueError("All K values must be positive.")
    return sorted(set(values))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ConvNeXtV2 retrieval ranking metrics."
    )
    parser.add_argument(
        "--dataset",
        default="covid",
        choices=["covid", "isic", "tbx11k", "vindr"],
        help="Dataset to use.",
    )
    parser.add_argument(
        "--test-dataset-dir",
        default="/data/brian.hu/COVID/data/test",
        help="Test dataset directory path.",
    )
    parser.add_argument(
        "--test-image-list",
        default="./test_COVIDx4.txt",
        help="Test image list or CSV.",
    )
    parser.add_argument("--mask-dir", default=None, help="Segmentation masks path.")
    parser.add_argument(
        "--vindr-label-mode",
        default="all",
        choices=["all", "concept", "disease", "legacy_finding6"],
        help=(
            "VinDR labels used to define retrieval relevance. "
            "'all' uses 28 concept+disease labels, 'concept' uses 22 finding "
            "labels, 'disease' uses 6 disease labels, and 'legacy_finding6' "
            "uses the older 6-class primary finding setup."
        ),
    )
    parser.add_argument(
        "--model",
        default="convnextv2",
        choices=["convnextv2", "convnextv2_sra"],
        help="Model to evaluate.",
    )
    parser.add_argument("--embedding-dim", default=None, type=int)
    parser.add_argument(
        "--sra-num-heads",
        default=8,
        type=int,
        help="Number of attention heads for ConvNeXtV2_SRA.",
    )
    parser.add_argument(
        "--sra-lam",
        default=0.1,
        type=float,
        help="Lambda for residual attention in ConvNeXtV2_SRA.",
    )
    parser.add_argument("--resume", default="", help="Checkpoint path.")
    parser.add_argument("--eval-batch-size", default=64, type=int)
    parser.add_argument("-j", "--workers", default=4, type=int)
    parser.add_argument(
        "--k-values",
        default="1,5,10",
        help="Comma-separated K values for nDCG@K, MRR@K, and MAP@K.",
    )
    parser.add_argument("--save-dir", default="./results", help="Result save directory.")
    return parser.parse_args()


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    k_values = parse_k_values(args.k_values)

    model = build_model(args)
    load_checkpoint(model, args.resume, device)
    model.to(device)

    transform = build_transform()
    dataset = build_dataset(args, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    print(f"Evaluating {args.model} on {args.dataset}...")
    embeds, labels = extract_embeddings(model, loader, device)
    results, dists = compute_retrieval_metrics(embeds, labels, k_values)
    print_metrics(results)
    save_results(args, results, embeds, labels, dists)


if __name__ == "__main__":
    main(parse_args())
