from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

try:
    import numpy as np
except Exception:  # pragma: no cover - defensive fallback
    np = None


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_COLLECTION_NAME = "covid_test_mir"
DEFAULT_SPLIT_FILE = HERE / "datasets" / "covid" / "test_split.txt"
DEFAULT_DATA_DIR = HERE / "datasets" / "covid" / "data" / "test"
DEFAULT_VINDR_LABELS_FILE = REPO_ROOT / "vindr" / "image_labels_test.csv"
DEFAULT_VINDR_GLOBAL_DIR = HERE / "datasets" / "vindr-384" / "test_png_384"
DEFAULT_VINDR_REGION_DIR = HERE / "datasets" / "vindr-640" / "test_png_640"
VARCHAR_LONG = 65535


def _connect_milvus(host: str, port: str) -> None:
    print(f"[milvus] Connecting to {host}:{port} ...")
    connections.connect(alias="default", host=host, port=port)
    print("[milvus] Connected")


def _supports_kw(func: Callable[..., Any], kw: str) -> bool:
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return False

    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return kw in sig.parameters


def _invoke_with_patterns(func: Callable[..., Any], image_path: str, device: str | None) -> Any:
    attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
        ((image_path,), {}),
        ((), {"image_path": image_path}),
        ((), {"path": image_path}),
        ((), {"image": image_path}),
    ]

    if device:
        for idx, (args, kwargs) in enumerate(attempts):
            if _supports_kw(func, "device"):
                merged = dict(kwargs)
                merged["device"] = device
                attempts[idx] = (args, merged)

    last_error: Exception | None = None
    for args, kwargs in attempts:
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: PERF203
            last_error = exc

    if last_error is None:
        raise RuntimeError("No invocation patterns attempted")
    raise last_error


def _pick_callable(module: Any, names: list[str]) -> Callable[..., Any] | None:
    for name in names:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    return None


def _build_embedder(module_path: str, kind: str, device: str | None) -> Callable[[str], Any]:
    module = importlib.import_module(module_path)

    if kind == "global":
        fn_names = [
            "compute_global_embedding",
            "extract_global_embedding",
            "get_global_embedding",
            "global_embedding",
            "embed_global",
            "embed_from_path",
            "embed_image",
            "encode_image",
        ]
        class_names = ["GlobalEmbedder", "GlobalEmbedding", "ImageEmbedder"]
        method_names = ["compute", "extract", "embed", "encode", "__call__"]
    else:
        fn_names = [
            "compute_region_embeddings_from_path",
            "compute_region_embeddings",
            "extract_region_embeddings",
            "get_region_embeddings",
            "region_embeddings",
            "embed_regions",
            "compute_regions",
        ]
        class_names = ["RegionEmbedder", "RegionEmbedding", "RegionEncoder"]
        method_names = ["compute", "extract", "embed", "encode", "__call__"]

    fn = _pick_callable(module, fn_names)
    if fn is not None:
        return lambda image_path: _invoke_with_patterns(fn, image_path, device)

    for cls_name in class_names:
        cls = getattr(module, cls_name, None)
        if cls is None or not callable(cls):
            continue

        instance = None
        for init_kwargs in ({"device": device} if device else {}, {}):
            try:
                instance = cls(**init_kwargs)
                break
            except TypeError:
                continue

        if instance is None:
            continue

        for method_name in method_names:
            method = getattr(instance, method_name, None)
            if callable(method):
                return lambda image_path, _m=method: _invoke_with_patterns(_m, image_path, device)

    for attr in dir(module):
        lower = attr.lower()
        if kind in lower and "embed" in lower:
            candidate = getattr(module, attr)
            if callable(candidate):
                return lambda image_path, _c=candidate: _invoke_with_patterns(_c, image_path, device)

    raise RuntimeError(
        f"Could not find callable {kind} embedder in module '{module_path}'. "
        "Expected a function/class for embedding extraction."
    )


def _to_builtin(obj: Any) -> Any:
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        try:
            return obj.tolist()
        except Exception:
            pass
    return obj


def _normalize_global_vector(output: Any) -> list[float]:
    output = _to_builtin(output)

    if isinstance(output, dict):
        for key in ("global_vector", "embedding", "vector", "features"):
            if key in output:
                output = output[key]
                break

    if isinstance(output, tuple):
        if not output:
            raise ValueError("Empty tuple returned for global embedding")
        output = output[0]

    output = _to_builtin(output)

    if isinstance(output, list) and output and isinstance(output[0], list):
        output = output[0]

    if not isinstance(output, list):
        raise ValueError(f"Unsupported global embedding output type: {type(output).__name__}")

    vec = [float(x) for x in output]
    if not vec:
        raise ValueError("Global embedding is empty")
    return vec


def _normalize_region_output(output: Any) -> tuple[list[Any], list[Any], list[Any]]:
    output = _to_builtin(output)

    labels: list[Any] = []
    boxes: list[Any] = []
    vectors: list[Any] = []

    if output is None:
        return labels, boxes, vectors

    if isinstance(output, dict):
        labels = _to_builtin(
            output.get("region_labels")
            or output.get("labels")
            or output.get("classes")
            or output.get("class_names")
            or []
        )
        boxes = _to_builtin(output.get("region_boxes") or output.get("boxes") or output.get("bboxes") or [])
        vectors = _to_builtin(
            output.get("region_vectors") or output.get("vectors") or output.get("embeddings") or []
        )
    elif isinstance(output, tuple):
        if len(output) == 3:
            a, b, c = (_to_builtin(x) for x in output)
            candidates = [a, b, c]

            for candidate in candidates:
                if isinstance(candidate, list) and candidate and isinstance(candidate[0], (list, tuple)) and len(candidate[0]) == 4:
                    boxes = [list(map(float, box)) for box in candidate]
                    break

            for candidate in candidates:
                if candidate is boxes:
                    continue
                if isinstance(candidate, list) and candidate and isinstance(candidate[0], (list, tuple)) and len(candidate[0]) > 4:
                    vectors = [list(map(float, vec)) for vec in candidate]
                    break

            for candidate in candidates:
                if candidate is boxes or candidate is vectors:
                    continue
                if isinstance(candidate, list):
                    labels = [str(x) for x in candidate]
                    break
        else:
            vectors = _to_builtin(list(output))
    elif isinstance(output, list):
        if output and isinstance(output[0], dict):
            for item in output:
                label_value = item.get(
                    "label",
                    item.get(
                        "class_name_en",
                        item.get(
                            "class_name",
                            item.get("class", item.get("class_id", "unknown")),
                        ),
                    ),
                )
                labels.append(str(label_value))
                box = item.get("box", item.get("bbox", item.get("region_box", [])))
                boxes.append(_to_builtin(box))
                vec = item.get("vector", item.get("embedding", item.get("region_vector", [])))
                vectors.append(_to_builtin(vec))
        else:
            vectors = output
    else:
        raise ValueError(f"Unsupported region embedding output type: {type(output).__name__}")

    labels = [] if labels is None else [str(x) for x in _to_builtin(labels)]
    boxes = [] if boxes is None else [_to_builtin(b) for b in _to_builtin(boxes)]
    vectors = [] if vectors is None else [_to_builtin(v) for v in _to_builtin(vectors)]

    clean_boxes: list[list[float]] = []
    for box in boxes:
        if isinstance(box, (list, tuple)):
            clean_boxes.append([float(x) for x in box])

    clean_vectors: list[list[float]] = []
    for vec in vectors:
        if isinstance(vec, (list, tuple)):
            clean_vectors.append([float(x) for x in vec])

    return labels, clean_boxes, clean_vectors


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _round_vectors(vectors: list[list[float]], decimals: int) -> list[list[float]]:
    rounded: list[list[float]] = []
    for vec in vectors:
        rounded.append([round(float(v), decimals) for v in vec])
    return rounded


def _fit_region_payload(
    region_labels: list[Any],
    region_boxes: list[list[float]],
    region_vectors: list[list[float]],
    *,
    max_len: int = VARCHAR_LONG,
    max_regions: int | None = None,
    decimals_trials: tuple[int, ...] = (6, 5, 4, 3, 2, 1, 0),
) -> tuple[str, str, str, int, bool]:
    """
    Make region JSON payload fit VARCHAR limits by progressively:
    1) reducing float precision
    2) reducing number of stored regions
    """
    labels = list(region_labels)
    boxes = list(region_boxes)
    vectors = list(region_vectors)

    if max_regions is not None and max_regions >= 0:
        labels = labels[:max_regions]
        boxes = boxes[:max_regions]
        vectors = vectors[:max_regions]

    original_count = max(len(labels), len(boxes), len(vectors))
    for decimals in decimals_trials:
        vectors_rounded = _round_vectors(vectors, decimals)
        keep = max(len(labels), len(boxes), len(vectors_rounded))
        while keep >= 0:
            cur_labels = labels[:keep]
            cur_boxes = boxes[:keep]
            cur_vectors = vectors_rounded[:keep]
            labels_json = _safe_json(cur_labels)
            boxes_json = _safe_json(cur_boxes)
            vectors_json = _safe_json(cur_vectors)
            if (
                len(labels_json) <= max_len
                and len(boxes_json) <= max_len
                and len(vectors_json) <= max_len
            ):
                truncated = keep < original_count
                return labels_json, boxes_json, vectors_json, keep, truncated
            keep -= 1

    # Last-resort fallback: store empty region payload instead of failing image.
    return "[]", "[]", "[]", 0, original_count > 0


def parse_test_split(split_path: Path, data_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")

    with split_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 3:
                print(f"[split] Skipping malformed line {line_no}: {line}")
                continue

            image_id = parts[0]
            if len(parts) >= 4:
                label = parts[-2]
                filename = " ".join(parts[1:-2]).strip()
            else:
                filename = parts[1]
                label = parts[2] if len(parts) > 2 else "unknown"

            if not filename:
                print(f"[split] Skipping line {line_no} with empty filename: {line}")
                continue

            image_path = data_dir / filename
            if not image_path.exists():
                continue

            rows.append(
                {
                    "image_id": image_id,
                    "image_name": filename,
                    "image_path": str(image_path.resolve()),
                    "region_image_path": str(image_path.resolve()),
                    "label": label,
                }
            )

    return rows


def _is_positive_label(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in {"1", "1.0", "true", "yes", "y"}


def _compact_label_text(labels: list[str], max_len: int = 120) -> str:
    if not labels:
        return "No finding"

    chosen: list[str] = []
    for label in labels:
        candidate = ", ".join(chosen + [label]) if chosen else label
        if len(candidate) > max_len:
            break
        chosen.append(label)

    if not chosen:
        return labels[0][:max_len]

    remaining = len(labels) - len(chosen)
    if remaining <= 0:
        return ", ".join(chosen)

    suffix = f" (+{remaining} more)"
    base = ", ".join(chosen)
    if len(base) + len(suffix) <= max_len:
        return f"{base}{suffix}"

    trimmed_base = base[: max(0, max_len - len(suffix))].rstrip(", ")
    return f"{trimmed_base}{suffix}"


def parse_vindr_labels(
    labels_csv_path: Path,
    global_dir: Path,
    region_dir: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    if not labels_csv_path.exists():
        raise FileNotFoundError(f"Missing ViNDR labels file: {labels_csv_path}")
    if not global_dir.exists():
        raise FileNotFoundError(f"Missing ViNDR global image directory: {global_dir}")
    if not region_dir.exists():
        raise FileNotFoundError(f"Missing ViNDR region image directory: {region_dir}")

    with labels_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "image_id" not in reader.fieldnames:
            raise ValueError(f"Invalid ViNDR labels CSV (missing image_id): {labels_csv_path}")

        label_columns = [name for name in reader.fieldnames if name != "image_id"]
        for line_no, record in enumerate(reader, start=2):
            image_id = (record.get("image_id") or "").strip()
            if not image_id:
                print(f"[vindr] Skipping row {line_no}: missing image_id")
                continue

            image_name = f"{image_id}.png"
            global_path = global_dir / image_name
            region_path = region_dir / image_name

            if not global_path.exists():
                continue

            if not region_path.exists():
                print(
                    f"[vindr] Missing region image for {image_name}; "
                    "falling back to global image path"
                )
                region_path = global_path

            positive_labels = [
                col for col in label_columns if _is_positive_label(str(record.get(col, "")))
            ]
            label = _compact_label_text(positive_labels, max_len=120)

            rows.append(
                {
                    "image_id": image_id,
                    "image_name": image_name,
                    "image_path": str(global_path.resolve()),
                    "region_image_path": str(region_path.resolve()),
                    "label": label,
                }
            )

    return rows


def load_dataset_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    dataset = args.dataset.lower()
    if dataset == "covid":
        return parse_test_split(Path(args.split_file), Path(args.data_dir))
    if dataset == "vindr":
        return parse_vindr_labels(
            Path(args.vindr_labels_file),
            Path(args.vindr_global_dir),
            Path(args.vindr_region_dir),
        )
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def _build_collection(collection_name: str, global_dim: int, drop_old: bool) -> Collection:
    if drop_old and utility.has_collection(collection_name):
        print(f"[milvus] Dropping existing collection: {collection_name}")
        utility.drop_collection(collection_name)

    if utility.has_collection(collection_name):
        collection = Collection(name=collection_name)

        dim = None
        for field in collection.schema.fields:
            if field.name == "global_vector":
                dim = field.params.get("dim")
                break

        if dim is not None and int(dim) != int(global_dim):
            raise ValueError(
                f"Collection '{collection_name}' has global_vector dim={dim}, "
                f"but current embeddings have dim={global_dim}. Use --drop-old."
            )

        return collection

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="image_name", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="global_vector", dtype=DataType.FLOAT_VECTOR, dim=global_dim),
        FieldSchema(name="region_count", dtype=DataType.INT64),
        FieldSchema(name="region_labels_json", dtype=DataType.VARCHAR, max_length=VARCHAR_LONG),
        FieldSchema(name="region_boxes_json", dtype=DataType.VARCHAR, max_length=VARCHAR_LONG),
        FieldSchema(name="region_vectors_json", dtype=DataType.VARCHAR, max_length=VARCHAR_LONG),
    ]

    schema = CollectionSchema(fields=fields, description="Image embeddings with global + region vectors")
    print(f"[milvus] Creating collection: {collection_name} (global dim={global_dim})")
    return Collection(name=collection_name, schema=schema, using="default")


def _get_global_dim_from_collection(collection: Collection) -> int | None:
    for field in collection.schema.fields:
        if field.name == "global_vector":
            dim = field.params.get("dim")
            return int(dim) if dim is not None else None
    return None


def _ensure_index(
    collection: Collection,
    *,
    metric_type: str = "L2",
    index_type: str = "FLAT",
) -> None:
    has_global_index = False
    for idx in collection.indexes:
        field_name = getattr(idx, "field_name", None)
        if field_name == "global_vector":
            has_global_index = True
            break

        params = getattr(idx, "params", {}) or {}
        if params.get("field_name") == "global_vector":
            has_global_index = True
            break

    if has_global_index:
        print("[milvus] Index on global_vector already exists")
        return

    metric_type = metric_type.upper()
    index_type = index_type.upper()
    if metric_type not in {"L2", "IP", "COSINE"}:
        raise ValueError(f"Unsupported metric type: {metric_type}")
    if index_type not in {"FLAT", "IVF_FLAT", "HNSW"}:
        raise ValueError(f"Unsupported index type: {index_type}")

    params: dict[str, Any] = {}
    if index_type == "IVF_FLAT":
        params = {"nlist": 1024}
    elif index_type == "HNSW":
        params = {"M": 16, "efConstruction": 200}

    index_params = {
        "index_type": index_type,
        "metric_type": metric_type,
        "params": params,
    }
    print(f"[milvus] Creating {index_type} index on global_vector ({metric_type})")
    collection.create_index(field_name="global_vector", index_params=index_params)


def _insert_batch(collection: Collection, batch: dict[str, list[Any]]) -> int:
    if not batch["image_name"]:
        return 0

    entities = [
        batch["image_name"],
        batch["image_path"],
        batch["label"],
        batch["global_vector"],
        batch["region_count"],
        batch["region_labels_json"],
        batch["region_boxes_json"],
        batch["region_vectors_json"],
    ]

    collection.insert(entities)
    inserted = len(batch["image_name"])
    for k in batch:
        batch[k].clear()
    return inserted


def run(args: argparse.Namespace) -> None:
    rows = load_dataset_rows(args)
    if args.limit is not None:
        rows = rows[: args.limit]

    effective_collection_name = args.collection_name
    if args.dataset.lower() == "vindr" and args.collection_name == DEFAULT_COLLECTION_NAME:
        effective_collection_name = "vindr_test_mir"
        print(f"[config] Using default ViNDR collection name: {effective_collection_name}")

    print(f"[data] Dataset={args.dataset} entries with existing files: {len(rows)}")
    if not rows:
        print("[data] Nothing to ingest")
        return

    _connect_milvus(args.host, str(args.port))

    global_embed = _build_embedder("utils.global_embedding", kind="global", device=args.device)
    region_embed = _build_embedder("utils.region_embed", kind="region", device=args.device)

    collection: Collection | None = None
    inserted_total = 0
    failed_total = 0
    processed = 0

    batch: dict[str, list[Any]] = {
        "image_name": [],
        "image_path": [],
        "label": [],
        "global_vector": [],
        "region_count": [],
        "region_labels_json": [],
        "region_boxes_json": [],
        "region_vectors_json": [],
    }

    total = len(rows)
    for row in rows:
        processed += 1
        image_path = row["image_path"]
        region_image_path = row.get("region_image_path", image_path)
        image_name = row["image_name"]
        label = row["label"]

        try:
            g_raw = global_embed(image_path)
            global_vector = _normalize_global_vector(g_raw)

            if collection is None:
                collection = _build_collection(effective_collection_name, len(global_vector), args.drop_old)
                _ensure_index(
                    collection,
                    metric_type=args.metric_type,
                    index_type=args.index_type,
                )

            expected_dim = _get_global_dim_from_collection(collection)
            if expected_dim is not None and len(global_vector) != expected_dim:
                raise ValueError(
                    f"Embedding dim mismatch for {image_name}: got {len(global_vector)}"
                )

            r_raw = region_embed(region_image_path)
            region_labels, region_boxes, region_vectors = _normalize_region_output(r_raw)
            detected_count = max(len(region_labels), len(region_boxes), len(region_vectors))
            (
                labels_json,
                boxes_json,
                vectors_json,
                region_count,
                truncated,
            ) = _fit_region_payload(
                region_labels,
                region_boxes,
                region_vectors,
                max_len=VARCHAR_LONG,
                max_regions=args.max_region_store,
            )
            if truncated:
                print(
                    f"[warn] Region payload trimmed for {image_name}: "
                    f"stored={region_count}, detected={detected_count}"
                )

            batch["image_name"].append(image_name)
            batch["image_path"].append(image_path)
            batch["label"].append(label)
            batch["global_vector"].append(global_vector)
            batch["region_count"].append(int(region_count))
            batch["region_labels_json"].append(labels_json)
            batch["region_boxes_json"].append(boxes_json)
            batch["region_vectors_json"].append(vectors_json)

            if len(batch["image_name"]) >= args.batch_size:
                if collection is None:
                    raise RuntimeError("Collection is not initialized")
                inserted_total += _insert_batch(collection, batch)
                print(
                    f"[ingest] Inserted {inserted_total}/{total} (processed={processed}, failed={failed_total})"
                )

        except Exception as exc:
            failed_total += 1
            print(f"[warn] Failed {image_name}: {exc}")
            print(traceback.format_exc(limit=1).strip())
            continue

    if collection is None:
        print("[done] No valid embeddings were produced; nothing inserted")
        return

    inserted_total += _insert_batch(collection, batch)
    collection.flush()
    collection.load()

    print("=" * 72)
    print("[done] Ingestion finished")
    print(f"[done] Collection: {effective_collection_name}")
    print(f"[done] Total candidates: {total}")
    print(f"[done] Inserted: {inserted_total}")
    print(f"[done] Failed: {failed_total}")
    print(f"[done] Entities in collection: {collection.num_entities}")
    print("=" * 72)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest dataset embeddings into local Milvus")
    parser.add_argument("--dataset", type=str, default="covid", choices=["covid", "vindr"])
    parser.add_argument("--collection-name", type=str, default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--drop-old", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19530)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--split-file",
        type=str,
        default=str(DEFAULT_SPLIT_FILE),
        help="COVID split file path (used when --dataset=covid)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="COVID image directory path (used when --dataset=covid)",
    )
    parser.add_argument(
        "--vindr-labels-file",
        type=str,
        default=str(DEFAULT_VINDR_LABELS_FILE),
        help="ViNDR labels CSV (image_labels_test.csv, used when --dataset=vindr)",
    )
    parser.add_argument(
        "--vindr-global-dir",
        type=str,
        default=str(DEFAULT_VINDR_GLOBAL_DIR),
        help="ViNDR 384 PNG directory for global embeddings (used when --dataset=vindr)",
    )
    parser.add_argument(
        "--vindr-region-dir",
        type=str,
        default=str(DEFAULT_VINDR_REGION_DIR),
        help="ViNDR 640 PNG directory for region embeddings (used when --dataset=vindr)",
    )
    parser.add_argument(
        "--metric-type",
        type=str,
        default="L2",
        choices=["L2", "IP", "COSINE", "l2", "ip", "cosine"],
        help="Milvus distance metric for global_vector index",
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default="FLAT",
        choices=["FLAT", "IVF_FLAT", "HNSW", "flat", "ivf_flat", "hnsw"],
        help="Milvus index type for global_vector",
    )
    parser.add_argument(
        "--max-region-store",
        type=int,
        default=8,
        help="Maximum number of region vectors to store per image (prevents VARCHAR overflow)",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.batch_size <= 0:
        print("--batch-size must be > 0")
        return 2
    args.metric_type = args.metric_type.upper()
    args.index_type = args.index_type.upper()

    try:
        run(args)
    except KeyboardInterrupt:
        print("Interrupted")
        return 130
    except Exception as exc:
        print(f"Fatal error: {exc}")
        return 1
    finally:
        try:
            connections.disconnect("default")
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
