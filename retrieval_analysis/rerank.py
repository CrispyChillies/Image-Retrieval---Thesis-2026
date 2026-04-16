"""Simple reranking hooks for retrieval analysis."""

from __future__ import annotations

from typing import Iterable, Protocol

from retrieval_analysis.milvus_adapter import QueryRecord, RetrievedItem


class Reranker(Protocol):
    """Protocol for optional rerankers."""

    def rerank(
        self, query: QueryRecord, results: Iterable[RetrievedItem]
    ) -> Iterable[RetrievedItem]:
        """Return a reranked iterable of retrieved items."""


class IdentityReranker:
    """Default no-op reranker."""

    def rerank(
        self, query: QueryRecord, results: Iterable[RetrievedItem]
    ) -> Iterable[RetrievedItem]:
        return list(results)
