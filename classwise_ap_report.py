import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model import ConvNeXtV2, ConvNeXtV2_SRA
from read_data import ChestXrayDataSet, TBX11kDataSet, VINDRConceptCLIPDataSet


COVID_LABELS = ["normal", "pneumonia", "COVID-19"]
TBX_LABELS = ["tb", "healthy", "sick_but_no_tb"]
VINDR_CONCEPT_COLUMNS = VINDRConceptCLIPDataSet.CONCEPT_COLUMNS
VINDR_DISEASE_COLUMNS = VINDRConceptCLIPDataSet.DISEASE_COLUMNS
VINDR_ALL_COLUMNS = VINDRConceptCLIPDataSet.ALL_COLUMNS


@dataclass
class ClassAPRow:
    dataset: str
    run_name: str
    class_name: str
    class_index: int
    positives_in_split: int
    num_queries: int
    mean_ap: float
    freq_group: str


class VinDRRetrievalDataSet(VINDRConceptCLIPDataSet):
    """Return tensor image + selected label mode for retrieval evaluation."""

    LEGACY_FINDING6_COLUMNS = [
        "Aortic enlargement",
        "Cardiomegaly",
        "Pleural effusion",
        "Pleural thickening",
        "Lung Opacity",
        "No finding",
    ]

    def __init__(self, *args, label_mode="disease", **kwargs):
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


def build_transform(model_name: str):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img_size = 384 if model_name in ["convnextv2", "convnextv2_sra"] else 224
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


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


def load_checkpoint(model, checkpoint_path: str, device: torch.device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "state-dict"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break

    if isinstance(checkpoint, dict):
        checkpoint = {
            key.replace("module.", "", 1): value for key, value in checkpoint.items()
        }

    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    print(
        f"Loaded checkpoint: {checkpoint_path} | missing={len(missing)} unexpected={len(unexpected)}"
    )


def build_dataset(args, transform):
    if args.dataset == "covid":
        ds = ChestXrayDataSet(
            data_dir=args.dataset_dir,
            image_list_file=args.image_list,
            transform=transform,
        )
        class_names = COVID_LABELS
    elif args.dataset == "tbx11k":
        ds = TBX11kDataSet(
            data_dir=args.dataset_dir,
            csv_file=args.image_list,
            transform=transform,
        )
        class_names = TBX_LABELS
    elif args.dataset == "vindr":
        ds = VinDRRetrievalDataSet(
            data_dir=args.dataset_dir,
            csv_file=args.image_list,
            transform=transform,
            label_mode=args.vindr_label_mode,
        )
        if args.vindr_label_mode == "concept":
            class_names = VINDR_CONCEPT_COLUMNS
        elif args.vindr_label_mode == "disease":
            class_names = VINDR_DISEASE_COLUMNS
        elif args.vindr_label_mode == "all":
            class_names = VINDR_ALL_COLUMNS
        else:
            class_names = VinDRRetrievalDataSet.LEGACY_FINDING6_COLUMNS
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return ds, class_names


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    embeds = []
    labels = []
    for samples, batch_labels in loader:
        samples = samples.to(device)
        output = model(samples)
        if isinstance(output, dict):
            embedding = output["embedding"]
        elif isinstance(output, tuple):
            embedding = output[0]
        else:
            embedding = output

        embeds.append(embedding.detach().cpu())
        labels.append(batch_labels.detach().cpu())

    return torch.cat(embeds, dim=0), torch.cat(labels, dim=0)


def parse_runs(run_args: List[str]) -> List[Tuple[str, str]]:
    runs = []
    for item in run_args:
        if "=" not in item:
            raise ValueError(
                f"Invalid --run format: '{item}'. Use --run name=/path/to/checkpoint.pth"
            )
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid --run format: '{item}'")
        runs.append((name, path))
    return runs


def _average_precision_from_binary_relevance(relevance: np.ndarray) -> float:
    num_pos = int(relevance.sum())
    if num_pos == 0:
        return float("nan")

    hits = 0
    precision_sum = 0.0
    for rank_idx, is_rel in enumerate(relevance, start=1):
        if is_rel:
            hits += 1
            precision_sum += hits / rank_idx
    return precision_sum / num_pos


def _compute_class_counts(labels: torch.Tensor, num_classes: int) -> np.ndarray:
    if labels.dim() == 1:
        counts = np.bincount(labels.numpy().astype(np.int64), minlength=num_classes)
    else:
        counts = labels.numpy().astype(np.int64).sum(axis=0)
    return counts.astype(np.int64)


def _assign_freq_group(counts: np.ndarray) -> List[str]:
    n_classes = len(counts)
    groups = ["mid"] * n_classes

    if n_classes == 1:
        groups[0] = "common"
        return groups

    min_idx = int(np.argmin(counts))
    max_idx = int(np.argmax(counts))

    if min_idx == max_idx:
        groups[min_idx] = "common"
        return groups

    groups[min_idx] = "rare"
    groups[max_idx] = "common"

    if n_classes > 3:
        q1 = float(np.quantile(counts, 0.33))
        q2 = float(np.quantile(counts, 0.67))
        for i, c in enumerate(counts):
            if c <= q1:
                groups[i] = "rare"
            elif c >= q2:
                groups[i] = "common"
            else:
                groups[i] = "mid"

    return groups


def compute_classwise_ap(
    embeds: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    dataset: str,
    run_name: str,
) -> List[ClassAPRow]:
    embeds = F.normalize(embeds, p=2, dim=1)
    sim = embeds @ embeds.t()
    sim.fill_diagonal_(float("-inf"))
    rankings = torch.argsort(sim, dim=1, descending=True).numpy()

    num_samples = labels.size(0)
    num_classes = len(class_names)

    if labels.dim() == 1:
        labels_np = labels.numpy().astype(np.int64)
        class_membership = np.zeros((num_samples, num_classes), dtype=np.int64)
        class_membership[np.arange(num_samples), labels_np] = 1
    else:
        class_membership = labels.numpy().astype(np.int64)
        if class_membership.shape[1] != num_classes:
            raise ValueError(
                f"Label dimension mismatch: labels have {class_membership.shape[1]} columns, "
                f"but class_names has {num_classes} classes."
            )

    class_counts = _compute_class_counts(labels, num_classes)
    freq_groups = _assign_freq_group(class_counts)

    rows: List[ClassAPRow] = []
    for class_idx, class_name in enumerate(class_names):
        query_indices = np.where(class_membership[:, class_idx] == 1)[0]
        ap_values = []

        for q in query_indices:
            ranked_idx = rankings[q]
            relevance = class_membership[ranked_idx, class_idx].astype(bool)
            ap_q = _average_precision_from_binary_relevance(relevance)
            if not np.isnan(ap_q):
                ap_values.append(ap_q)

        mean_ap = float(np.mean(ap_values)) if ap_values else float("nan")
        rows.append(
            ClassAPRow(
                dataset=dataset,
                run_name=run_name,
                class_name=class_name,
                class_index=class_idx,
                positives_in_split=int(class_counts[class_idx]),
                num_queries=int(len(query_indices)),
                mean_ap=mean_ap,
                freq_group=freq_groups[class_idx],
            )
        )

    return rows


def save_reports(rows: List[ClassAPRow], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    per_class_df = pd.DataFrame([r.__dict__ for r in rows])
    per_class_df["mean_ap_percent"] = per_class_df["mean_ap"] * 100.0

    per_class_csv = os.path.join(out_dir, "per_class_ap.csv")
    per_class_df.to_csv(per_class_csv, index=False)

    group_df = (
        per_class_df.groupby(["dataset", "run_name", "freq_group"], as_index=False)
        .agg(
            mean_ap=("mean_ap", "mean"),
            mean_ap_percent=("mean_ap_percent", "mean"),
            total_queries=("num_queries", "sum"),
            num_classes=("class_name", "count"),
        )
        .sort_values(["dataset", "run_name", "freq_group"])
    )

    grouped_csv = os.path.join(out_dir, "grouped_common_rare_ap.csv")
    group_df.to_csv(grouped_csv, index=False)

    print("\nSaved:")
    print(f"- {per_class_csv}")
    print(f"- {grouped_csv}")

    print("\n=== Per-class AP (head) ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(per_class_df.head(20).to_string(index=False))

    print("\n=== Grouped AP: common/rare ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(group_df.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute class-wise retrieval AP and common/rare grouped AP for "
            "COVIDx, TBX11K, or VinDr using ConvNeXtV2/ConvNeXtV2_SRA checkpoints."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["covid", "tbx11k", "vindr"],
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Image directory for evaluation split.",
    )
    parser.add_argument(
        "--image-list",
        required=True,
        help="Manifest file for evaluation split.",
    )
    parser.add_argument(
        "--vindr-label-mode",
        default="disease",
        choices=["all", "concept", "disease", "legacy_finding6"],
        help="VinDr label mode.",
    )
    parser.add_argument(
        "--model",
        default="convnextv2_sra",
        choices=["convnextv2", "convnextv2_sra"],
    )
    parser.add_argument("--embedding-dim", default=None, type=int)
    parser.add_argument("--sra-num-heads", default=8, type=int)
    parser.add_argument("--sra-lam", default=0.1, type=float)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run definition: name=/path/to/checkpoint.pth (repeatable).",
    )
    parser.add_argument("--eval-batch-size", default=64, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--out-dir", default="./summary_classwise_ap")
    return parser.parse_args()


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = build_transform(args.model)
    dataset, class_names = build_dataset(args, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    runs = parse_runs(args.run)
    all_rows: List[ClassAPRow] = []

    for run_name, ckpt_path in runs:
        print(f"\n=== Evaluating run: {run_name} ===")
        model = build_model(args)
        load_checkpoint(model, ckpt_path, device)
        model.to(device)

        embeds, labels = extract_embeddings(model, loader, device)
        run_rows = compute_classwise_ap(
            embeds=embeds,
            labels=labels,
            class_names=class_names,
            dataset=args.dataset,
            run_name=run_name,
        )
        all_rows.extend(run_rows)

    save_reports(all_rows, args.out_dir)


if __name__ == "__main__":
    main(parse_args())
