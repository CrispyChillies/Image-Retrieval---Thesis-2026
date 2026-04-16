"""Correctness evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from retrieval_analysis.milvus_adapter import RetrievedItem


@dataclass(frozen=True)
class CorrectnessConfig:
    """Correctness policy for retrieval evaluation."""

    top_k: int = 1


def is_retrieval_correct(
    query_label: str | None,
    results: Sequence[RetrievedItem],
    config: CorrectnessConfig,
) -> bool:
    """Return whether any of the first top_k labels matches the query label."""
    if not query_label or not results:
        return False
    return any(item.label == query_label for item in results[: config.top_k])
