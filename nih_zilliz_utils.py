from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import unquote

import numpy as np
import torch
from PIL import Image
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from tqdm import tqdm

from nih_multilabel_retrieval import (
    NIH_RETRIEVAL_PATHOLOGIES,
    build_nih_val_transform,
    get_backbone_image_config,
)
from nih_multilabel_training import BACKBONE_SPECS


EMBEDDING_DIM = 256


def normalize_nih_label(label_name: str) -> str:
    return (
        label_name.strip()
        .replace("%20", " ")
        .replace("_", " ")
        .replace("-", " ")
        .lower()
    )


def parse_nih_labels_from_path(
    image_path: str,
    pathology_names: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[float]]:
    pathology_names = list(pathology_names or NIH_RETRIEVAL_PATHOLOGIES)
    pathology_to_index = {name: idx for idx, name in enumerate(pathology_names)}
    aliases = {
        "pleural_thickening": "Pleural Thickening",
        "pleural thickening": "Pleural Thickening",
        "pleuralthickening": "Pleural Thickening",
    }
    for name in pathology_names:
        aliases[normalize_nih_label(name)] = name

    stem = Path(image_path).stem
    prefix = "Chest_X-ray_"
    prefix_index = stem.find(prefix)
    if prefix_index < 0:
        raise ValueError(
            f"Unsupported NIH file name '{Path(image_path).name}'. Expected token '{prefix}'."
        )

    stem_without_prefix = stem[prefix_index + len(prefix):]
    encoded_labels, _ = stem_without_prefix.rsplit("_", 1)
    raw_label_names = [label.strip() for label in unquote(encoded_labels).split("|")]

    label_names: List[str] = []
    multi_hot = [0.0] * len(pathology_names)
    unknown_labels = []
    for raw_label in raw_label_names:
        normalized = normalize_nih_label(raw_label)
        canonical = aliases.get(normalized)
        if canonical is None or canonical not in pathology_to_index:
            unknown_labels.append(raw_label)
            continue
        multi_hot[pathology_to_index[canonical]] = 1.0
        label_names.append(canonical)

    if unknown_labels:
        raise ValueError(
            f"Unknown pathologies in '{Path(image_path).name}': {unknown_labels}."
        )

    return label_names, multi_hot


def load_npy_as_pil(image_path: str) -> Image.Image:
    image_array = np.load(image_path)
    image_array = np.asarray(image_array)

    if image_array.ndim == 3 and image_array.shape[0] in (1, 3):
        image_array = np.transpose(image_array, (1, 2, 0))
    if image_array.ndim == 3 and image_array.shape[-1] == 1:
        image_array = image_array[..., 0]

    if image_array.dtype != np.uint8:
        image_array = image_array.astype(np.float32)
        min_value = float(image_array.min())
        max_value = float(image_array.max())
        if max_value <= min_value:
            image_array = np.zeros_like(image_array, dtype=np.uint8)
        else:
            image_array = (image_array - min_value) / (max_value - min_value)
            image_array = np.clip(image_array * 255.0, 0.0, 255.0).astype(np.uint8)

    return Image.fromarray(image_array).convert("L")


def resolve_npy_paths(data_dir: str, image_list_file: Optional[str] = None) -> List[str]:
    paths: List[str] = []
    if image_list_file:
        manifest_path = Path(image_list_file)
        with open(manifest_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                candidate = Path(line.split(",")[0].strip())
                if not candidate.is_absolute():
                    candidate = Path(data_dir) / candidate
                paths.append(str(candidate))
    else:
        paths = sorted(str(path) for path in Path(data_dir).rglob("*.npy"))

    if not paths:
        raise ValueError("No .npy files found for NIH ingestion/query.")
    return paths


def connect_zilliz(uri: str, token: str) -> None:
    connections.connect(alias="default", uri=uri, token=token)


def disconnect_zilliz() -> None:
    connections.disconnect("default")


def build_collection_name(model_name: str, suffix: str) -> str:
    return f"nih_{model_name}_{suffix}"


def create_nih_collection(
    collection_name: str,
    drop_old: bool = False,
    metric_type: str = "COSINE",
    index_type: str = "IVF_FLAT",
    nlist: int = 1024,
) -> Collection:
    if drop_old and utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="image_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="label_text", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="label_vector_json", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        ]
        schema = CollectionSchema(fields=fields, description="NIH multi-label gallery embeddings")
        collection = Collection(name=collection_name, schema=schema, using="default")

    if not collection.indexes:
        index_params = {
            "metric_type": metric_type,
            "index_type": index_type,
            "params": {"nlist": nlist} if "IVF" in index_type else {},
        }
        collection.create_index(field_name="embedding", index_params=index_params)

    collection.load()
    return collection


def build_model_and_transform(
    model_name: str,
    checkpoint_path: str,
    backbone_name: Optional[str],
    device: torch.device,
    num_labels: int = 14,
) -> Tuple[torch.nn.Module, object]:
    spec = BACKBONE_SPECS[model_name]
    image_config = get_backbone_image_config(model_name)
    transform = build_nih_val_transform(
        image_size=image_config["image_size"],
        resize_size=image_config["resize_size"],
    )

    model = spec.model_builder(
        num_labels,
        backbone_name or spec.default_backbone_name,
        False,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif "state-dict" in checkpoint:
        checkpoint = checkpoint["state-dict"]
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model, transform


def encode_npy_paths(
    model: torch.nn.Module,
    transform,
    image_paths: Sequence[str],
    device: torch.device,
    batch_size: int,
    progress_desc: str = "Encoding",
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for start in tqdm(
        range(0, len(image_paths), batch_size),
        desc=progress_desc,
        unit="batch",
    ):
        batch_paths = image_paths[start : start + batch_size]
        tensors = []
        batch_rows = []
        for image_path in batch_paths:
            label_names, multi_hot = parse_nih_labels_from_path(image_path)
            image = load_npy_as_pil(image_path)
            tensors.append(transform(image))
            batch_rows.append(
                {
                    "image_path": image_path,
                    "image_name": Path(image_path).name,
                    "label_names": label_names,
                    "multi_hot": multi_hot,
                }
            )

        batch_tensor = torch.stack(tensors).to(device, non_blocking=True)
        with torch.no_grad():
            outputs = model(batch_tensor)
            embeddings = outputs["embedding"].detach().cpu().numpy()

        for row, embedding in zip(batch_rows, embeddings):
            row["embedding"] = embedding.astype(np.float32)
            rows.append(row)

    return rows


def insert_rows(collection: Collection, rows: Sequence[Dict[str, object]]) -> None:
    image_paths = [row["image_path"] for row in rows]
    image_names = [row["image_name"] for row in rows]
    label_texts = ["|".join(row["label_names"]) for row in rows]
    label_vectors = [json.dumps(row["multi_hot"]) for row in rows]
    embeddings = [row["embedding"].tolist() for row in rows]
    collection.insert([image_paths, image_names, label_texts, label_vectors, embeddings])
    collection.flush()


def search_collection(
    collection: Collection,
    query_vector: List[float],
    top_k: int,
    nprobe: int = 10,
) -> List[Dict[str, object]]:
    search_results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": nprobe}},
        limit=top_k,
        output_fields=["image_path", "image_name", "label_text", "label_vector_json"],
    )
    results: List[Dict[str, object]] = []
    for hits in search_results:
        for hit in hits:
            results.append(
                {
                    "id": hit.id,
                    "score": float(hit.distance),
                    "image_path": hit.entity.get("image_path"),
                    "image_name": hit.entity.get("image_name"),
                    "label_text": hit.entity.get("label_text"),
                    "label_vector": json.loads(hit.entity.get("label_vector_json")),
                }
            )
    return results
