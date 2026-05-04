from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import average_precision_score


def jaccard_score(query_label: List[float], gallery_label: List[float]) -> float:
    query = np.asarray(query_label, dtype=np.float32)
    gallery = np.asarray(gallery_label, dtype=np.float32)
    intersection = float((query * gallery).sum())
    union = float(np.clip(query + gallery, 0.0, 1.0).sum())
    return intersection / (union + 1e-8)


def precision_at_k(binary_relevance: List[float], k: int) -> float:
    if not binary_relevance:
        return 0.0
    k = min(k, len(binary_relevance))
    return float(np.mean(binary_relevance[:k]))


def recall_at_k(binary_relevance: List[float], total_positives: int, k: int) -> float:
    if total_positives <= 0:
        return 0.0
    k = min(k, len(binary_relevance))
    return float(np.sum(binary_relevance[:k]) / total_positives)


def evaluate_results(items: List[Dict[str, object]], jaccard_threshold: float, ks: List[int]) -> Dict[str, float]:
    aps = []
    precision_scores = {k: [] for k in ks}
    recall_scores = {k: [] for k in ks}

    for item in items:
        query_label = item["query_label_vector"]
        hits = item["results"]
        scores = [hit["score"] for hit in hits]
        relevances = [
            1.0 if jaccard_score(query_label, hit["label_vector"]) > jaccard_threshold else 0.0
            for hit in hits
        ]
        total_positives = int(sum(relevances))

        if total_positives > 0:
            aps.append(average_precision_score(relevances, scores))

        for k in ks:
            precision_scores[k].append(precision_at_k(relevances, k))
            recall_scores[k].append(recall_at_k(relevances, total_positives, k))

    metrics: Dict[str, float] = {
        "mAP": float(np.mean(aps) * 100.0) if aps else 0.0,
        "num_queries": float(len(items)),
        "num_valid_ap_queries": float(len(aps)),
    }
    for k in ks:
        metrics[f"P@{k}"] = float(np.mean(precision_scores[k]) * 100.0) if precision_scores[k] else 0.0
        metrics[f"R@{k}"] = float(np.mean(recall_scores[k]) * 100.0) if recall_scores[k] else 0.0
    return metrics


def main(args: argparse.Namespace) -> None:
    with open(args.results_json, "r", encoding="utf-8") as handle:
        items = json.load(handle)

    ks = [int(token) for token in args.ks.split(",") if token.strip()]
    metrics = evaluate_results(items, args.jaccard_threshold, ks)

    print(f"results_json={args.results_json}")
    for key, value in metrics.items():
        if key.startswith("num_"):
            print(f"{key}={int(value)}")
        else:
            print(f"{key}={value:.3f}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(f"saved_metrics={output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NIH Zilliz retrieval results")
    parser.add_argument("--results-json", required=True, help="Output from query_nih_zilliz.py")
    parser.add_argument("--jaccard-threshold", type=float, default=0.4)
    parser.add_argument("--ks", default="1,5,10,20,50")
    parser.add_argument("--output-json", default=None)
    main(parser.parse_args())
