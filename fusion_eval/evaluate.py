"""Run late-fusion experiments on aligned embedding sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from fusion_eval.align import AlignedEmbeddings
from fusion_eval.fuse import concat_fusion, l2_normalize, weighted_sum_fusion
from fusion_eval.metrics import (
    compute_similarity_matrix,
    evaluate_retrieval_metrics,
    evaluate_retrieval_metrics_from_similarity,
)


@dataclass(frozen=True)
class ExperimentResult:
    """One baseline or fusion experiment result."""

    experiment_name: str
    num_samples: int
    metrics: Dict[str, float]
    skipped: bool = False
    skipped_reason: Optional[str] = None


def run_late_fusion_experiments(
    aligned: AlignedEmbeddings,
    alpha_values: Sequence[float] = (0.2, 0.4, 0.5, 0.6, 0.8),
    k_values: Iterable[int] = (1, 5, 10),
    include_score_fusion: bool = True,
    score_normalization: str = "none",
) -> List[ExperimentResult]:
    """Evaluate baselines and late-fusion variants."""
    results: List[ExperimentResult] = []

    conv_baseline = l2_normalize(aligned.conv_embeddings)
    dino_baseline = l2_normalize(aligned.dino_embeddings)
    baselines = {
        "convnext_baseline": conv_baseline,
        "dino_baseline": dino_baseline,
        "concat_fusion": concat_fusion(aligned.conv_embeddings, aligned.dino_embeddings),
    }
    for name, embeddings in baselines.items():
        metrics = evaluate_retrieval_metrics(
            embeddings=embeddings,
            labels=aligned.labels,
            image_paths=aligned.image_paths,
            k_values=k_values,
        )
        results.append(
            ExperimentResult(
                experiment_name=name,
                num_samples=len(aligned.image_paths),
                metrics=metrics,
            )
        )

    if include_score_fusion:
        conv_similarity = normalize_similarity_matrix(
            compute_similarity_matrix(conv_baseline),
            mode=score_normalization,
        )
        dino_similarity = normalize_similarity_matrix(
            compute_similarity_matrix(dino_baseline),
            mode=score_normalization,
        )
        for alpha in alpha_values:
            fused_similarity = alpha * conv_similarity + (1.0 - alpha) * dino_similarity
            metrics = evaluate_retrieval_metrics_from_similarity(
                similarity=fused_similarity,
                labels=aligned.labels,
                image_paths=aligned.image_paths,
                k_values=k_values,
            )
            results.append(
                ExperimentResult(
                    experiment_name=f"score_fusion_alpha_{alpha:.1f}",
                    num_samples=len(aligned.image_paths),
                    metrics=metrics,
                )
            )

    for alpha in alpha_values:
        fusion = weighted_sum_fusion(
            conv_embeddings=aligned.conv_embeddings,
            dino_embeddings=aligned.dino_embeddings,
            alpha=alpha,
        )
        if fusion.embeddings is None:
            results.append(
                ExperimentResult(
                    experiment_name=f"weighted_sum_alpha_{alpha:.1f}",
                    num_samples=len(aligned.image_paths),
                    metrics={},
                    skipped=True,
                    skipped_reason=fusion.skipped_reason,
                )
            )
            continue

        metrics = evaluate_retrieval_metrics(
            embeddings=fusion.embeddings,
            labels=aligned.labels,
            image_paths=aligned.image_paths,
            k_values=k_values,
        )
        results.append(
            ExperimentResult(
                experiment_name=f"weighted_sum_alpha_{alpha:.1f}",
                num_samples=len(aligned.image_paths),
                metrics=metrics,
            )
        )

    return results


def normalize_similarity_matrix(similarity: np.ndarray, mode: str = "none") -> np.ndarray:
    """Optionally normalize similarity scores before score-level fusion."""
    if mode == "none":
        return similarity.astype(np.float32, copy=True)

    similarity = similarity.astype(np.float32, copy=True)
    diag = np.diag(similarity).copy()

    if mode == "zscore":
        means = np.mean(similarity, axis=1, keepdims=True)
        stds = np.std(similarity, axis=1, keepdims=True)
        stds = np.maximum(stds, 1e-12)
        normalized = (similarity - means) / stds
    elif mode == "minmax":
        mins = np.min(similarity, axis=1, keepdims=True)
        maxs = np.max(similarity, axis=1, keepdims=True)
        scales = np.maximum(maxs - mins, 1e-12)
        normalized = (similarity - mins) / scales
    else:
        raise ValueError(
            f"Unsupported score normalization mode: {mode}. "
            "Use one of: none, zscore, minmax"
        )

    np.fill_diagonal(normalized, diag)
    return normalized


def maybe_save_fused_embeddings(
    path: str,
    image_paths: Sequence[str],
    labels: Sequence[str],
    embeddings: np.ndarray,
) -> None:
    """Optionally save fused embeddings for later reuse."""
    np.savez_compressed(
        path,
        image_paths=np.asarray(image_paths),
        labels=np.asarray(labels),
        embeddings=np.asarray(embeddings, dtype=np.float32),
    )
