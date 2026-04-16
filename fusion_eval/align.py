"""Embedding alignment utilities for late-fusion experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import json
import numpy as np

from retrieval_analysis.comparison import load_query_set
from retrieval_analysis.milvus_adapter import MilvusCollectionAdapter, MilvusCollectionConfig


@dataclass(frozen=True)
class EmbeddingRecord:
    """One stored embedding record."""

    image_path: str
    label: Optional[str]
    embedding: np.ndarray
    source_name: str
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class AlignedEmbeddings:
    """Intersection-aligned embedding payload for fusion experiments."""

    image_paths: List[str]
    labels: List[str]
    conv_embeddings: np.ndarray
    dino_embeddings: np.ndarray
    coverage: Dict[str, List[str]]


class EmbeddingSource:
    """Abstract embedding source."""

    def fetch_all(self) -> List[EmbeddingRecord]:
        raise NotImplementedError


class MilvusEmbeddingSource(EmbeddingSource):
    """Load embeddings from one Milvus collection."""

    def __init__(self, config: MilvusCollectionConfig, batch_size: int = 1000):
        self.config = config
        self.batch_size = batch_size
        self.adapter = MilvusCollectionAdapter(config)

    def fetch_all(self) -> List[EmbeddingRecord]:
        fields = list(self.config.output_fields)
        for required_field in (
            self.config.image_path_field,
            self.config.label_field,
            self.config.vector_field,
        ):
            if required_field not in fields:
                fields.append(required_field)

        rows = self.adapter.client.query(
            collection_name=self.config.collection_name,
            filter=f'{self.config.image_path_field} != ""',
            output_fields=fields,
            limit=self.batch_size,
        )
        all_rows = list(rows)
        offset = len(rows)

        while rows:
            rows = self.adapter.client.query(
                collection_name=self.config.collection_name,
                filter=f'{self.config.image_path_field} != ""',
                output_fields=fields,
                limit=self.batch_size,
                offset=offset,
            )
            all_rows.extend(rows)
            offset += len(rows)

        return [
            EmbeddingRecord(
                image_path=row[self.config.image_path_field],
                label=row.get(self.config.label_field),
                embedding=np.asarray(row[self.config.vector_field], dtype=np.float32),
                source_name=self.config.name,
                raw=row,
            )
            for row in all_rows
            if row.get(self.config.image_path_field) is not None
        ]


class FileEmbeddingSource(EmbeddingSource):
    """Load embeddings from local JSON or NPZ files."""

    def __init__(self, path: str | Path, source_name: str):
        self.path = Path(path)
        self.source_name = source_name

    def fetch_all(self) -> List[EmbeddingRecord]:
        suffix = self.path.suffix.lower()
        if suffix == ".json":
            return self._load_json()
        if suffix == ".npz":
            return self._load_npz()
        raise ValueError(f"Unsupported embedding file format: {self.path}")

    def _load_json(self) -> List[EmbeddingRecord]:
        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        rows = data.get("records", data)
        return [
            EmbeddingRecord(
                image_path=row["image_path"],
                label=row.get("label"),
                embedding=np.asarray(row["embedding"], dtype=np.float32),
                source_name=self.source_name,
                raw=row,
            )
            for row in rows
        ]

    def _load_npz(self) -> List[EmbeddingRecord]:
        payload = np.load(self.path, allow_pickle=True)
        image_paths = payload["image_paths"].tolist()
        labels = payload["labels"].tolist() if "labels" in payload else [None] * len(image_paths)
        embeddings = payload["embeddings"]
        return [
            EmbeddingRecord(
                image_path=image_path,
                label=label,
                embedding=np.asarray(embedding, dtype=np.float32),
                source_name=self.source_name,
                raw={},
            )
            for image_path, label, embedding in zip(image_paths, labels, embeddings)
        ]


def build_embedding_source(config: Mapping[str, Any]) -> EmbeddingSource:
    """Build an embedding source from config."""
    source_type = config.get("type", "milvus")
    if source_type == "milvus":
        return MilvusEmbeddingSource(
            MilvusCollectionConfig(**config["milvus"]),
            batch_size=config.get("batch_size", 1000),
        )
    if source_type == "file":
        return FileEmbeddingSource(
            path=config["path"],
            source_name=config["name"],
        )
    raise ValueError(f"Unsupported source type: {source_type}")


def align_embedding_sources(
    conv_source: EmbeddingSource,
    dino_source: EmbeddingSource,
    query_set_path: Optional[str | Path] = None,
    strict_label_check: bool = True,
) -> AlignedEmbeddings:
    """Align ConvNeXt and DINO embeddings by image_path."""
    conv_records = _index_records(conv_source.fetch_all(), "ConvNeXt")
    dino_records = _index_records(dino_source.fetch_all(), "DINO")

    conv_paths = set(conv_records)
    dino_paths = set(dino_records)
    coverage = {
        "present_in_conv_only": sorted(conv_paths - dino_paths),
        "present_in_dino_only": sorted(dino_paths - conv_paths),
        "present_in_both": sorted(conv_paths & dino_paths),
    }

    if query_set_path:
        query_paths = [query.image_path for query in load_query_set(query_set_path)]
        target_paths = [path for path in query_paths if path in conv_paths and path in dino_paths]
    else:
        target_paths = coverage["present_in_both"]

    labels: List[str] = []
    conv_embeddings: List[np.ndarray] = []
    dino_embeddings: List[np.ndarray] = []
    final_paths: List[str] = []

    for image_path in target_paths:
        conv_record = conv_records[image_path]
        dino_record = dino_records[image_path]
        conv_label = conv_record.label
        dino_label = dino_record.label

        if strict_label_check and conv_label != dino_label:
            raise ValueError(
                f"Label mismatch for image_path={image_path}: "
                f"conv={conv_label!r}, dino={dino_label!r}"
            )

        final_paths.append(image_path)
        labels.append(conv_label or dino_label or "unknown")
        conv_embeddings.append(conv_record.embedding)
        dino_embeddings.append(dino_record.embedding)

    if not final_paths:
        raise ValueError("No aligned samples found across the requested sources")

    return AlignedEmbeddings(
        image_paths=final_paths,
        labels=labels,
        conv_embeddings=np.stack(conv_embeddings).astype(np.float32),
        dino_embeddings=np.stack(dino_embeddings).astype(np.float32),
        coverage=coverage,
    )


def _index_records(records: Iterable[EmbeddingRecord], source_name: str) -> Dict[str, EmbeddingRecord]:
    indexed: Dict[str, EmbeddingRecord] = {}
    for record in records:
        if record.image_path in indexed:
            raise ValueError(
                f"Duplicate image_path found in {source_name}: {record.image_path}"
            )
        indexed[record.image_path] = record
    return indexed
