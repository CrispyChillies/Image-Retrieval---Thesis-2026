import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


COVIDX_LABEL_MAP = {
    "normal": "normal",
    "pneumonia": "pneumonia",
    "covid-19": "COVID-19",
}

TBX_LABEL_MAP = {
    "tb": "tb",
    "healthy": "healthy",
    "sick_but_no_tb": "sick_but_no_tb",
}

VINDR_CONCEPT_COLUMNS = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Clavicle fracture",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Enlarged PA",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Lung cavity",
    "Lung cyst",
    "Mediastinal shift",
    "Nodule/Mass",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
    "Rib fracture",
    "Other lesion",
]

VINDR_DISEASE_COLUMNS = [
    "COPD",
    "Lung tumor",
    "Pneumonia",
    "Tuberculosis",
    "Other diseases",
    "No finding",
]


@dataclass
class SplitStats:
    dataset: str
    split: str
    n_samples: int
    n_classes: int
    max_class: str
    max_count: int
    min_class: str
    min_count: int
    imbalance_ratio: float


def _safe_ratio(max_count: int, min_count: int) -> float:
    if min_count <= 0:
        return float("inf")
    return float(max_count) / float(min_count)


def _counts_to_stats(dataset: str, split: str, counts: Dict[str, int]) -> SplitStats:
    if not counts:
        raise ValueError(f"No class counts found for {dataset}-{split}")

    sorted_items = sorted(counts.items(), key=lambda kv: kv[1])
    min_class, min_count = sorted_items[0]
    max_class, max_count = sorted_items[-1]

    return SplitStats(
        dataset=dataset,
        split=split,
        n_samples=int(sum(counts.values())),
        n_classes=len(counts),
        max_class=max_class,
        max_count=int(max_count),
        min_class=min_class,
        min_count=int(min_count),
        imbalance_ratio=_safe_ratio(int(max_count), int(min_count)),
    )


def _read_covidx_counts(list_file: str) -> Dict[str, int]:
    counts = {"normal": 0, "pneumonia": 0, "COVID-19": 0}
    with open(list_file, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            label_key = parts[2].strip().lower()
            mapped = COVIDX_LABEL_MAP.get(label_key)
            if mapped is None:
                continue
            counts[mapped] += 1
    return counts


def _read_tbx_counts(csv_file: str) -> Dict[str, int]:
    counts = {"tb": 0, "healthy": 0, "sick_but_no_tb": 0}
    with open(csv_file, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return counts

        normalized = {
            col.strip().lstrip("\ufeff").lower(): col
            for col in reader.fieldnames
            if col is not None
        }
        image_type_col = normalized.get("image_type")
        if image_type_col is None:
            return counts

        for row in reader:
            value = str(row.get(image_type_col, "")).strip().lower()
            mapped = TBX_LABEL_MAP.get(value)
            if mapped is None:
                continue
            counts[mapped] += 1
    return counts


def _normalize_vindr_df(df: pd.DataFrame) -> pd.DataFrame:
    if "Other disease" in df.columns and "Other diseases" not in df.columns:
        df = df.rename(columns={"Other disease": "Other diseases"})
    return df


def _aggregate_vindr(df: pd.DataFrame, label_cols: List[str]) -> pd.DataFrame:
    missing = [c for c in label_cols if c not in df.columns]
    if missing:
        raise ValueError(f"VinDr CSV missing label columns: {missing}")

    if "rad_id" in df.columns:
        df = df.groupby("image_id", as_index=False)[label_cols].max()
    else:
        keep_cols = ["image_id"] + label_cols if "image_id" in df.columns else label_cols
        df = df[keep_cols].copy()
    return df


def _read_vindr_counts(csv_file: str, mode: str) -> Tuple[Dict[str, int], int]:
    df = pd.read_csv(csv_file)
    df = _normalize_vindr_df(df)

    if mode == "disease":
        label_cols = VINDR_DISEASE_COLUMNS
    elif mode == "concept":
        label_cols = VINDR_CONCEPT_COLUMNS
    else:
        label_cols = VINDR_CONCEPT_COLUMNS + VINDR_DISEASE_COLUMNS

    agg = _aggregate_vindr(df, label_cols)
    counts = {c: int(agg[c].fillna(0).astype(int).sum()) for c in label_cols}
    n_images = int(len(agg))
    return counts, n_images


def _merge_counts(counts_a: Dict[str, int], counts_b: Dict[str, int]) -> Dict[str, int]:
    merged = dict(counts_a)
    for key, value in counts_b.items():
        merged[key] = merged.get(key, 0) + int(value)
    return merged


def _plot_counts(dataset: str, split: str, counts: Dict[str, int], out_dir: str) -> None:
    labels = list(counts.keys())
    values = [counts[k] for k in labels]

    fig_width = max(8, int(len(labels) * 0.5))
    plt.figure(figsize=(fig_width, 5))
    plt.bar(labels, values)
    plt.title(f"{dataset} - {split}: class counts")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{dataset.lower()}_{split.lower()}_class_counts.png")
    plt.savefig(save_path, dpi=200)
    plt.close()


def _append_result(
    rows: List[SplitStats],
    dataset: str,
    split: str,
    counts: Dict[str, int],
    n_samples_override: Optional[int] = None,
) -> None:
    stats = _counts_to_stats(dataset, split, counts)
    if n_samples_override is not None:
        stats.n_samples = n_samples_override
    rows.append(stats)


def _print_table(df: pd.DataFrame) -> None:
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("\n=== Class Imbalance Summary (max_count / min_count) ===")
        print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize class imbalance for COVIDx, TBX11K, and VinDr from label files. "
            "Outputs summary CSV + per-split class-count plots."
        )
    )

    parser.add_argument("--covid-train-list", type=str, default=None)
    parser.add_argument("--covid-test-list", type=str, default=None)
    parser.add_argument("--tbx-train-csv", type=str, default=None)
    parser.add_argument("--tbx-test-csv", type=str, default=None)
    parser.add_argument("--vindr-train-csv", type=str, default=None)
    parser.add_argument("--vindr-test-csv", type=str, default=None)
    parser.add_argument(
        "--vindr-mode",
        choices=["all", "disease", "concept"],
        default="all",
        help="Which VinDr labels to summarize.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./imbalance_reports",
        help="Directory to save summary CSV and plots.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving class-count bar charts.",
    )

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows: List[SplitStats] = []

    # COVIDx
    covid_train = None
    covid_test = None
    if args.covid_train_list and os.path.isfile(args.covid_train_list):
        covid_train = _read_covidx_counts(args.covid_train_list)
        _append_result(rows, "COVIDx", "train", covid_train)
        if not args.no_plot:
            _plot_counts("COVIDx", "train", covid_train, args.out_dir)

    if args.covid_test_list and os.path.isfile(args.covid_test_list):
        covid_test = _read_covidx_counts(args.covid_test_list)
        _append_result(rows, "COVIDx", "test", covid_test)
        if not args.no_plot:
            _plot_counts("COVIDx", "test", covid_test, args.out_dir)

    if covid_train is not None and covid_test is not None:
        covid_all = _merge_counts(covid_train, covid_test)
        _append_result(rows, "COVIDx", "all", covid_all)
        if not args.no_plot:
            _plot_counts("COVIDx", "all", covid_all, args.out_dir)

    # TBX11K
    tbx_train = None
    tbx_test = None
    if args.tbx_train_csv and os.path.isfile(args.tbx_train_csv):
        tbx_train = _read_tbx_counts(args.tbx_train_csv)
        _append_result(rows, "TBX11K", "train", tbx_train)
        if not args.no_plot:
            _plot_counts("TBX11K", "train", tbx_train, args.out_dir)

    if args.tbx_test_csv and os.path.isfile(args.tbx_test_csv):
        tbx_test = _read_tbx_counts(args.tbx_test_csv)
        _append_result(rows, "TBX11K", "test", tbx_test)
        if not args.no_plot:
            _plot_counts("TBX11K", "test", tbx_test, args.out_dir)

    if tbx_train is not None and tbx_test is not None:
        tbx_all = _merge_counts(tbx_train, tbx_test)
        _append_result(rows, "TBX11K", "all", tbx_all)
        if not args.no_plot:
            _plot_counts("TBX11K", "all", tbx_all, args.out_dir)

    # VinDr
    vindr_train = None
    vindr_test = None
    vindr_train_n = None
    vindr_test_n = None
    if args.vindr_train_csv and os.path.isfile(args.vindr_train_csv):
        vindr_train, vindr_train_n = _read_vindr_counts(args.vindr_train_csv, args.vindr_mode)
        _append_result(rows, "VinDr", "train", vindr_train, n_samples_override=vindr_train_n)
        if not args.no_plot:
            _plot_counts("VinDr", "train", vindr_train, args.out_dir)

    if args.vindr_test_csv and os.path.isfile(args.vindr_test_csv):
        vindr_test, vindr_test_n = _read_vindr_counts(args.vindr_test_csv, args.vindr_mode)
        _append_result(rows, "VinDr", "test", vindr_test, n_samples_override=vindr_test_n)
        if not args.no_plot:
            _plot_counts("VinDr", "test", vindr_test, args.out_dir)

    if vindr_train is not None and vindr_test is not None:
        vindr_all = _merge_counts(vindr_train, vindr_test)
        total_images = (vindr_train_n or 0) + (vindr_test_n or 0)
        _append_result(rows, "VinDr", "all", vindr_all, n_samples_override=total_images)
        if not args.no_plot:
            _plot_counts("VinDr", "all", vindr_all, args.out_dir)

    if not rows:
        raise ValueError("No valid input files were provided. Please pass at least one dataset label file.")

    summary_df = pd.DataFrame([r.__dict__ for r in rows])
    summary_df = summary_df[
        [
            "dataset",
            "split",
            "n_samples",
            "n_classes",
            "max_class",
            "max_count",
            "min_class",
            "min_count",
            "imbalance_ratio",
        ]
    ]
    summary_csv = os.path.join(args.out_dir, "imbalance_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    _print_table(summary_df)
    print(f"\nSaved summary CSV: {summary_csv}")
    if not args.no_plot:
        print(f"Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
