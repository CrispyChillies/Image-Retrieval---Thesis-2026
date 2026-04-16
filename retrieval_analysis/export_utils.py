"""Export helpers for retrieval comparison outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


def ensure_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_json(path: str | Path, payload: Mapping) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def write_csv(path: str | Path, rows: Iterable[Mapping]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def flatten_query_result(result: Mapping) -> Dict[str, object]:
    """Flatten nested retrieval fields for CSV export."""
    conv = result.get("conv", {})
    dino = result.get("dino", {})
    return {
        "query_image_path": result.get("query_image_path"),
        "query_label": result.get("query_label"),
        "group": result.get("assigned_group"),
        "conv_correct": result.get("conv_correct"),
        "dino_correct": result.get("dino_correct"),
        "conv_topk_image_paths": json.dumps(conv.get("image_paths", [])),
        "conv_topk_labels": json.dumps(conv.get("labels", [])),
        "conv_topk_scores": json.dumps(conv.get("scores", [])),
        "dino_topk_image_paths": json.dumps(dino.get("image_paths", [])),
        "dino_topk_labels": json.dumps(dino.get("labels", [])),
        "dino_topk_scores": json.dumps(dino.get("scores", [])),
    }
