"""Embedding-level and score-level retrieval metrics for late-fusion evaluation."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np

from fusion_eval.fuse import l2_normalize


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity for normalized embeddings."""
    normalized = l2_normalize(embeddings.astype(np.float32))
    return normalized @ normalized.T


def rank_indices(similarity: np.ndarray) -> np.ndarray:
    """Rank gallery items for every query after removing self-match."""
    similarity = similarity.copy()
    np.fill_diagonal(similarity, -np.inf)
    return np.argsort(-similarity, axis=1)


def evaluate_retrieval_metrics(
    embeddings: np.ndarray,
    labels: Sequence[str],
    image_paths: Sequence[str],
    k_values: Iterable[int] = (1, 5, 10),
) -> Dict[str, float]:
    """Evaluate retrieval metrics on one embedding set."""
    similarity = compute_similarity_matrix(embeddings)
    return evaluate_retrieval_metrics_from_similarity(
        similarity=similarity,
        labels=labels,
        image_paths=image_paths,
        k_values=k_values,
    )


def evaluate_retrieval_metrics_from_similarity(
    similarity: np.ndarray,
    labels: Sequence[str],
    image_paths: Sequence[str],
    k_values: Iterable[int] = (1, 5, 10),
) -> Dict[str, float]:
    """Evaluate retrieval metrics from a precomputed similarity matrix."""
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("Similarity matrix must be square")
    if len(labels) != len(image_paths) or len(labels) != similarity.shape[0]:
        raise ValueError("Labels, image_paths, and similarity matrix must have matching sizes")

    k_values = sorted(set(int(k) for k in k_values))
    ranks = rank_indices(similarity)
    labels_np = np.asarray(labels)
    image_paths_np = np.asarray(image_paths)

    metrics: Dict[str, float] = {"num_samples": float(len(labels_np))}
    average_precisions = []
    precision_at_k_values = {k: [] for k in k_values}
    recall_at_k_values = {k: [] for k in k_values}

    for query_index in range(len(labels_np)):
        ranked_indices = ranks[query_index]
        ranked_indices = ranked_indices[image_paths_np[ranked_indices] != image_paths_np[query_index]]
        relevant = labels_np[ranked_indices] == labels_np[query_index]
        relevant_count = int(np.sum(labels_np == labels_np[query_index]) - 1)

        if relevant_count <= 0:
            average_precisions.append(0.0)
            for k in k_values:
                precision_at_k_values[k].append(0.0)
                recall_at_k_values[k].append(0.0)
            continue

        cumulative_hits = np.cumsum(relevant.astype(np.int32))
        hit_positions = np.flatnonzero(relevant)
        if len(hit_positions) == 0:
            average_precisions.append(0.0)
        else:
            precisions = cumulative_hits[hit_positions] / (hit_positions + 1)
            average_precisions.append(float(np.sum(precisions) / relevant_count))

        for k in k_values:
            topk_relevant = relevant[:k]
            hits = int(np.sum(topk_relevant))
            precision_at_k_values[k].append(hits / k)
            recall_at_k_values[k].append(1.0 if hits > 0 else 0.0)

    metrics["mAP"] = float(np.mean(average_precisions) * 100.0)
    for k in k_values:
        metrics[f"mP@{k}"] = float(np.mean(precision_at_k_values[k]) * 100.0)
        metrics[f"R@{k}"] = float(np.mean(recall_at_k_values[k]) * 100.0)
    return metrics
