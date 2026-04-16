"""Dual-cluster Milvus access helpers for retrieval analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pymilvus import MilvusClient


@dataclass(frozen=True)
class MilvusCollectionConfig:
    """Connection and collection settings for one Milvus-backed model."""

    name: str
    collection_name: str
    uri: Optional[str] = None
    token: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    db_name: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    vector_field: str = "embedding"
    id_field: str = "id"
    image_path_field: str = "image_path"
    label_field: str = "label"
    output_fields: Sequence[str] = field(
        default_factory=lambda: ("id", "image_path", "label")
    )


@dataclass
class QueryRecord:
    """Query sample definition shared across both models."""

    image_path: str
    label: Optional[str] = None


@dataclass
class RetrievedItem:
    """Normalized retrieval hit."""

    id: Optional[Any]
    image_path: Optional[str]
    label: Optional[str]
    score: Optional[float]
    distance: Optional[float]
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Search output for one query."""

    query: QueryRecord
    query_source: str
    retrieved: List[RetrievedItem]
    query_embedding: Sequence[float]


class MilvusCollectionAdapter:
    """Thin wrapper around one Milvus client and one collection."""

    def __init__(self, config: MilvusCollectionConfig):
        self.config = config
        self.client = MilvusClient(**self._build_client_kwargs(config))

    @staticmethod
    def _build_client_kwargs(config: MilvusCollectionConfig) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if config.uri:
            kwargs["uri"] = config.uri
        elif config.host:
            kwargs["uri"] = f"http://{config.host}:{config.port or 19530}"
        else:
            raise ValueError(
                f"{config.name}: provide either uri or host/port for Milvus access"
            )

        if config.token:
            kwargs["token"] = config.token
        elif config.user is not None or config.password is not None:
            kwargs["user"] = config.user or ""
            kwargs["password"] = config.password or ""

        if config.db_name:
            kwargs["db_name"] = config.db_name

        return kwargs

    def list_image_paths(self, batch_size: int = 1000) -> List[str]:
        """Load all stored image paths from this collection."""
        rows = self.client.query(
            collection_name=self.config.collection_name,
            filter=f'{self.config.image_path_field} != ""',
            output_fields=[self.config.image_path_field],
            limit=batch_size,
        )
        image_paths = [row.get(self.config.image_path_field) for row in rows]
        offset = len(rows)

        while rows:
            rows = self.client.query(
                collection_name=self.config.collection_name,
                filter=f'{self.config.image_path_field} != ""',
                output_fields=[self.config.image_path_field],
                limit=batch_size,
                offset=offset,
            )
            image_paths.extend(row.get(self.config.image_path_field) for row in rows)
            offset += len(rows)

        return [path for path in image_paths if path]

    def fetch_record_by_image_path(
        self, image_path: str, include_embedding: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Fetch one stored row by stable image path."""
        output_fields = list(self.config.output_fields)
        if include_embedding and self.config.vector_field not in output_fields:
            output_fields.append(self.config.vector_field)

        rows = self.client.query(
            collection_name=self.config.collection_name,
            filter=self._eq_expr(self.config.image_path_field, image_path),
            output_fields=output_fields,
            limit=2,
        )
        if not rows:
            return None
        if len(rows) > 1:
            raise ValueError(
                f"{self.config.name}: multiple rows found for image_path={image_path}"
            )
        return rows[0]

    def search_by_embedding(
        self,
        query: QueryRecord,
        query_embedding: Sequence[float],
        top_k: int,
        search_params: Optional[Dict[str, Any]] = None,
        reranker: Optional[Any] = None,
        exclude_self: bool = True,
        metadata_fields: Optional[Sequence[str]] = None,
    ) -> SearchResult:
        """Search this collection using a supplied query embedding."""
        fields = list(metadata_fields or self.config.output_fields)
        if self.config.image_path_field not in fields:
            fields.append(self.config.image_path_field)
        if self.config.label_field not in fields:
            fields.append(self.config.label_field)

        raw_hits = self.client.search(
            collection_name=self.config.collection_name,
            data=[list(query_embedding)],
            anns_field=self.config.vector_field,
            search_params=search_params or {},
            limit=top_k + 1 if exclude_self else top_k,
            output_fields=fields,
        )

        hits = raw_hits[0] if raw_hits else []
        retrieved = [self._normalize_hit(hit) for hit in hits]
        if exclude_self:
            retrieved = [
                item for item in retrieved if item.image_path != query.image_path
            ]
        if reranker is not None:
            retrieved = list(reranker.rerank(query=query, results=retrieved))

        return SearchResult(
            query=query,
            query_source=self.config.name,
            retrieved=retrieved[:top_k],
            query_embedding=query_embedding,
        )

    def _normalize_hit(self, hit: Dict[str, Any]) -> RetrievedItem:
        entity = hit.get("entity", {})
        score = hit.get("score")
        distance = hit.get("distance", score)
        item_id = entity.get(self.config.id_field, hit.get("id"))
        return RetrievedItem(
            id=item_id,
            image_path=entity.get(self.config.image_path_field),
            label=entity.get(self.config.label_field),
            score=score,
            distance=distance,
            raw=hit,
        )

    @staticmethod
    def _eq_expr(field_name: str, value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'{field_name} == "{escaped}"'


def compare_collection_coverage(
    conv_adapter: MilvusCollectionAdapter,
    dino_adapter: MilvusCollectionAdapter,
) -> Dict[str, List[str]]:
    """Compare image_path coverage across both collections."""
    conv_paths = set(conv_adapter.list_image_paths())
    dino_paths = set(dino_adapter.list_image_paths())
    return {
        "conv_only": sorted(conv_paths - dino_paths),
        "dino_only": sorted(dino_paths - conv_paths),
        "present_in_both": sorted(conv_paths & dino_paths),
    }


def filter_present_queries(
    queries: Iterable[QueryRecord],
    coverage: Dict[str, List[str]],
) -> Dict[str, List[QueryRecord]]:
    """Split queries by whether they exist in both collections."""
    present = set(coverage["present_in_both"])
    valid: List[QueryRecord] = []
    missing: List[QueryRecord] = []
    for query in queries:
        if query.image_path in present:
            valid.append(query)
        else:
            missing.append(query)
    return {"valid": valid, "missing": missing}
