"""Run late-fusion experiments on aligned embedding sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from fusion_eval.align import AlignedEmbeddings
from fusion_eval.fuse import concat_fusion, l2_normalize, weighted_sum_fusion
from fusion_eval.metrics import evaluate_retrieval_metrics


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
) -> List[ExperimentResult]:
    """Evaluate baselines and late-fusion variants."""
    results: List[ExperimentResult] = []

    baselines = {
        "convnext_baseline": l2_normalize(aligned.conv_embeddings),
        "dino_baseline": l2_normalize(aligned.dino_embeddings),
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
