"""Late-fusion strategies for aligned embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def l2_normalize(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize embedding rows."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return embeddings / norms


def concat_fusion(conv_embeddings: np.ndarray, dino_embeddings: np.ndarray) -> np.ndarray:
    """Normalize, concatenate, and normalize again."""
    conv = l2_normalize(conv_embeddings)
    dino = l2_normalize(dino_embeddings)
    fused = np.concatenate([conv, dino], axis=1)
    return l2_normalize(fused)


@dataclass(frozen=True)
class WeightedSumResult:
    """Result of weighted-sum fusion attempt."""

    embeddings: Optional[np.ndarray]
    skipped_reason: Optional[str] = None


def weighted_sum_fusion(
    conv_embeddings: np.ndarray,
    dino_embeddings: np.ndarray,
    alpha: float,
) -> WeightedSumResult:
    """Fuse two embedding sets with a weighted sum if dimensions match."""
    if conv_embeddings.shape[1] != dino_embeddings.shape[1]:
        return WeightedSumResult(
            embeddings=None,
            skipped_reason=(
                "weighted_sum_skipped_dimension_mismatch:"
                f" conv_dim={conv_embeddings.shape[1]}, dino_dim={dino_embeddings.shape[1]}"
            ),
        )

    conv = l2_normalize(conv_embeddings)
    dino = l2_normalize(dino_embeddings)
    fused = alpha * conv + (1.0 - alpha) * dino
    return WeightedSumResult(embeddings=l2_normalize(fused))
