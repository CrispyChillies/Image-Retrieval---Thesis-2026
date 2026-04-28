"""End-to-end comparison pipeline for dual-cluster retrieval analysis."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from retrieval_analysis.evaluator import CorrectnessConfig, is_retrieval_correct
from retrieval_analysis.milvus_adapter import (
    MilvusCollectionAdapter,
    QueryRecord,
    SearchResult,
    filter_present_queries,
)
from retrieval_analysis.rerank import IdentityReranker, Reranker


GROUP_BOTH_CORRECT = "both_correct"
GROUP_BOTH_WRONG = "both_wrong"
GROUP_DINO_CORRECT_CONV_WRONG = "dino_correct_conv_wrong"
GROUP_CONV_CORRECT_DINO_WRONG = "conv_correct_dino_wrong"


@dataclass
class ComparisonConfig:
    """Configuration for retrieval comparison."""

    top_k: int = 5
    conv_search_params: Optional[Dict] = None
    dino_search_params: Optional[Dict] = None
    correctness: CorrectnessConfig = CorrectnessConfig()
    skip_missing_queries: bool = True
    preload_batch_size: int = 100
    search_batch_size: int = 10


def load_query_set(path: str | Path) -> List[QueryRecord]:
    """Load an ordered query set from JSON, CSV, or whitespace text."""
    query_path = Path(path)
    suffix = query_path.suffix.lower()
    if suffix == ".json":
        with query_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            data = data.get("queries", data.get("results", []))
        return [
            QueryRecord(
                image_path=item.get("image_path", item.get("query_image_path")),
                label=item.get("label"),
            )
            for item in data
            if item.get("image_path", item.get("query_image_path"))
        ]

    if suffix == ".csv":
        with query_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [
                QueryRecord(
                    image_path=row.get("image_path", row.get("query_image_path", "")),
                    label=row.get("label", row.get("query_label")),
                )
                for row in reader
                if row.get("image_path") or row.get("query_image_path")
            ]

    queries: List[QueryRecord] = []
    with query_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) == 1:
                queries.append(QueryRecord(image_path=parts[0]))
            elif len(parts) >= 2:
                queries.append(QueryRecord(image_path=parts[0], label=parts[1]))
    return queries


def compare_models(
    conv_adapter: MilvusCollectionAdapter,
    dino_adapter: MilvusCollectionAdapter,
    queries: Sequence[QueryRecord],
    config: ComparisonConfig,
    reranker: Optional[Reranker] = None,
) -> Dict[str, object]:
    """Run aligned retrieval analysis for ConvNeXt and DINO collections."""
    reranker = reranker or IdentityReranker()
    requested_paths = [query.image_path for query in queries if query.image_path]
    conv_records = conv_adapter.fetch_records_by_image_paths(
        requested_paths,
        include_embedding=True,
        batch_size=config.preload_batch_size,
    )
    dino_records = dino_adapter.fetch_records_by_image_paths(
        requested_paths,
        include_embedding=True,
        batch_size=config.preload_batch_size,
    )

    conv_paths = set(conv_records)
    dino_paths = set(dino_records)
    coverage = {
        "conv_only": sorted(conv_paths - dino_paths),
        "dino_only": sorted(dino_paths - conv_paths),
        "present_in_both": sorted(conv_paths & dino_paths),
    }
    query_partition = filter_present_queries(queries, coverage)
    valid_queries = query_partition["valid"]
    missing_queries = query_partition["missing"]

    if missing_queries and not config.skip_missing_queries:
        missing_paths = ", ".join(query.image_path for query in missing_queries[:5])
        raise ValueError(
            "Some query image_paths are not present in both collections: "
            f"{missing_paths}"
        )

    results: List[Dict[str, object]] = []
    summary = Counter()
    per_query_errors: List[Dict[str, str]] = []

    for start in range(0, len(valid_queries), config.search_batch_size):
        query_batch = valid_queries[start : start + config.search_batch_size]
        try:
            aligned_queries: List[QueryRecord] = []
            conv_embeddings = []
            dino_embeddings = []

            for query in query_batch:
                conv_record = conv_records.get(query.image_path)
                dino_record = dino_records.get(query.image_path)
                if conv_record is None or dino_record is None:
                    per_query_errors.append(
                        {
                            "query_image_path": query.image_path,
                            "error": "missing_query_embedding_on_one_side",
                        }
                    )
                    continue

                query_label = (
                    query.label
                    or conv_record.get(conv_adapter.config.label_field)
                    or dino_record.get(dino_adapter.config.label_field)
                )
                aligned_query = QueryRecord(image_path=query.image_path, label=query_label)
                aligned_queries.append(aligned_query)
                conv_embeddings.append(conv_record[conv_adapter.config.vector_field])
                dino_embeddings.append(dino_record[dino_adapter.config.vector_field])

            if not aligned_queries:
                continue

            conv_results = conv_adapter.search_by_embeddings(
                queries=aligned_queries,
                query_embeddings=conv_embeddings,
                top_k=config.top_k,
                search_params=config.conv_search_params,
                reranker=reranker,
                exclude_self=True,
                batch_size=config.search_batch_size,
            )
            dino_results = dino_adapter.search_by_embeddings(
                queries=aligned_queries,
                query_embeddings=dino_embeddings,
                top_k=config.top_k,
                search_params=config.dino_search_params,
                reranker=reranker,
                exclude_self=True,
                batch_size=config.search_batch_size,
            )

            for aligned_query, conv_result, dino_result in zip(
                aligned_queries, conv_results, dino_results
            ):
                conv_correct = is_retrieval_correct(
                    aligned_query.label, conv_result.retrieved, config.correctness
                )
                dino_correct = is_retrieval_correct(
                    aligned_query.label, dino_result.retrieved, config.correctness
                )
                assigned_group = assign_group(
                    conv_correct=conv_correct,
                    dino_correct=dino_correct,
                )
                summary[assigned_group] += 1

                results.append(
                    build_query_analysis_row(
                        query=aligned_query,
                        conv_result=conv_result,
                        dino_result=dino_result,
                        conv_correct=conv_correct,
                        dino_correct=dino_correct,
                        assigned_group=assigned_group,
                    )
                )
        except Exception as exc:
            for query in query_batch:
                per_query_errors.append(
                    {"query_image_path": query.image_path, "error": str(exc)}
                )

    return {
        "coverage": {
            "present_in_conv_only": coverage["conv_only"],
            "present_in_dino_only": coverage["dino_only"],
            "present_in_both": coverage["present_in_both"],
        },
        "missing_queries": [
            {"image_path": query.image_path, "label": query.label}
            for query in missing_queries
        ],
        "errors": per_query_errors,
        "summary": {
            GROUP_BOTH_CORRECT: summary[GROUP_BOTH_CORRECT],
            GROUP_BOTH_WRONG: summary[GROUP_BOTH_WRONG],
            GROUP_DINO_CORRECT_CONV_WRONG: summary[
                GROUP_DINO_CORRECT_CONV_WRONG
            ],
            GROUP_CONV_CORRECT_DINO_WRONG: summary[
                GROUP_CONV_CORRECT_DINO_WRONG
            ],
            "evaluated_queries": len(results),
        },
        "results": results,
    }


def assign_group(conv_correct: bool, dino_correct: bool) -> str:
    """Assign one of the four required comparison groups."""
    if conv_correct and dino_correct:
        return GROUP_BOTH_CORRECT
    if not conv_correct and not dino_correct:
        return GROUP_BOTH_WRONG
    if dino_correct and not conv_correct:
        return GROUP_DINO_CORRECT_CONV_WRONG
    return GROUP_CONV_CORRECT_DINO_WRONG


def build_query_analysis_row(
    query: QueryRecord,
    conv_result: SearchResult,
    dino_result: SearchResult,
    conv_correct: bool,
    dino_correct: bool,
    assigned_group: str,
) -> Dict[str, object]:
    """Build one per-query analysis row."""
    return {
        "query_image_path": query.image_path,
        "query_label": query.label,
        "conv": serialize_search_result(conv_result),
        "dino": serialize_search_result(dino_result),
        "conv_correct": conv_correct,
        "dino_correct": dino_correct,
        "assigned_group": assigned_group,
    }


def serialize_search_result(result: SearchResult) -> Dict[str, object]:
    """Serialize search results for JSON export."""
    return {
        "image_paths": [item.image_path for item in result.retrieved],
        "labels": [item.label for item in result.retrieved],
        "scores": [item.score for item in result.retrieved],
        "distances": [item.distance for item in result.retrieved],
        "hits": [
            {
                "id": item.id,
                "image_path": item.image_path,
                "label": item.label,
                "score": item.score,
                "distance": item.distance,
            }
            for item in result.retrieved
        ],
    }
