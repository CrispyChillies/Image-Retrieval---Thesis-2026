"""
Evaluate SimAtt xAI localization quality on TBX11k bounding boxes.

Metrics implemented
-------------------
1. Pointing Game (PG) hit:
   Find the pixel with the highest saliency score. A sample is a hit if that
   point lies inside the TB bounding box from the TBX11k CSV.

2. RMSE / distance-to-center:
   Compute the Euclidean distance between the highest-saliency point and the
   center of the TB bounding box. The aggregate RMSE is:

       sqrt(mean(distance_px ** 2))

   A normalized RMSE is also reported by dividing each distance by the image
   diagonal before computing RMSE.

Retrieval-pair evaluation
-------------------------
By default this script uses the actual query/retrieved-image similarity setup:

1. Encode all TB-positive bbox images with the model.
2. For each query image, retrieve the top-k most similar other TB bbox images
    using cosine similarity.
3. Run SimAtt on each pair:

         saliency = SimAtt(query_image, retrieved_image)

4. Use the retrieved-image saliency map, find its max point, and compare that
    point with the retrieved image's TB bbox.

This matches the retrieval xAI use case: "why is this retrieved image similar
to the query?" The pair cosine similarity is saved for every evaluated pair.

You can still reproduce the old single-image sanity check with
--eval_mode self.
- The file name intentionally follows the user's requested spelling:
  evaluate_PG_RSME.py. The metric name printed in outputs uses RMSE.

Example
-------
python evaluate_PG_RSME.py ^
  --data_dir path/to/tbx11k/images ^
  --csv_file test.csv ^
  --convnextv2_weights checkpoints/convnextv2.pth ^
  --convnextv2_sra_weights checkpoints/convnextv2_sra.pth ^
    --top_k 1 ^
  --embedding_dim 512 ^
    --save_visualizations ^
    --max_visualizations 50 ^
  --output_dir pg_rmse_results
"""

import argparse
import ast
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from explanations import SimAtt
from model import ConvNeXtV2, ConvNeXtV2_SRA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_bbox(value):
    """Parse a TBX11k bbox string into a dict, or return None."""
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in {"none", "nan", "null"}:
        return None

    try:
        bbox = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None

    required = {"xmin", "ymin", "width", "height"}
    if not isinstance(bbox, dict) or not required.issubset(bbox):
        return None

    bbox = {key: float(bbox[key]) for key in required}
    if bbox["width"] <= 0 or bbox["height"] <= 0:
        return None
    return bbox


def resolve_image_path(data_dir, image_id_or_fname, image_ext=".png"):
    """Resolve an image path for TBX11k fname or VinDR image_id."""
    data_dir = Path(data_dir)
    raw = str(image_id_or_fname).strip()
    candidates = []

    raw_path = Path(raw)
    if raw_path.suffix:
        candidates.append(data_dir / raw)
    else:
        candidates.append(data_dir / f"{raw}{image_ext}")
        for ext in (".png", ".jpg", ".jpeg"):
            if ext != image_ext:
                candidates.append(data_dir / f"{raw}{ext}")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_tbx11k_bbox_rows(csv_file, data_dir, only_target="tb", limit=None):
    """Load TBX11k rows that contain valid bounding boxes."""
    rows = []
    data_dir = Path(data_dir)

    with open(csv_file, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_file}")

        normalized = {
            name.strip().lstrip("\ufeff").lower(): name
            for name in reader.fieldnames
            if name is not None
        }
        fname_key = normalized.get("fname")
        bbox_key = normalized.get("bbox")
        target_key = normalized.get("target")
        image_type_key = normalized.get("image_type")
        tb_type_key = normalized.get("tb_type")

        if fname_key is None or bbox_key is None:
            raise ValueError(
                "CSV must contain at least 'fname' and 'bbox' columns. "
                f"Found columns: {reader.fieldnames}"
            )

        for row_idx, row in enumerate(reader, start=2):
            fname = row.get(fname_key, "").strip()
            bbox = parse_bbox(row.get(bbox_key))
            if not fname or bbox is None:
                continue

            target = row.get(target_key, "").strip() if target_key else ""
            image_type = row.get(image_type_key, "").strip() if image_type_key else ""
            if only_target and target_key is not None and target != only_target:
                continue
            if only_target and target_key is None and image_type != only_target:
                continue

            image_path = resolve_image_path(data_dir, fname)
            if not image_path.exists():
                print(f"Warning: missing image for CSV row {row_idx}: {image_path}")
                continue

            rows.append(
                {
                    "row": row_idx,
                    "fname": fname,
                    "image_path": str(image_path),
                    "bbox": bbox,
                    "bboxes": [bbox],
                    "bbox_classes": [row.get(tb_type_key, "").strip() if tb_type_key else "tb"],
                    "target": target,
                    "tb_type": row.get(tb_type_key, "").strip() if tb_type_key else "",
                    "image_type": image_type,
                    "dataset": "tbx11k",
                }
            )
            if limit is not None and len(rows) >= limit:
                break

    return rows


def load_vindr_bbox_rows(csv_file, data_dir, image_ext=".png", classes=None, limit=None, bbox_coord_size=None):
    """
    Load VinDR-CXR bbox annotations and group multiple boxes per image.

    Expected columns:
        image_id,class_name,x_min,y_min,x_max,y_max

    The provided annotations_rescaled_384.csv is already in 384x384 coordinate
    space. If your actual images are also 384x384, no extra scaling is needed.
    """
    wanted_classes = None
    if classes:
        wanted_classes = {c.strip() for c in classes.split(",") if c.strip()}

    grouped = {}
    image_size_cache = {}
    with open(csv_file, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_file}")

        normalized = {
            name.strip().lstrip("\ufeff").lower(): name
            for name in reader.fieldnames
            if name is not None
        }
        image_id_key = normalized.get("image_id")
        class_key = normalized.get("class_name")
        x_min_key = normalized.get("x_min")
        y_min_key = normalized.get("y_min")
        x_max_key = normalized.get("x_max")
        y_max_key = normalized.get("y_max")

        required = [image_id_key, class_key, x_min_key, y_min_key, x_max_key, y_max_key]
        if any(key is None for key in required):
            raise ValueError(
                "VinDR CSV must contain image_id,class_name,x_min,y_min,x_max,y_max. "
                f"Found columns: {reader.fieldnames}"
            )

        for row_idx, row in enumerate(reader, start=2):
            image_id = row.get(image_id_key, "").strip()
            class_name = row.get(class_key, "").strip()
            if not image_id or not class_name:
                continue
            if wanted_classes is not None and class_name not in wanted_classes:
                continue

            try:
                x_min = float(row[x_min_key])
                y_min = float(row[y_min_key])
                x_max = float(row[x_max_key])
                y_max = float(row[y_max_key])
            except (TypeError, ValueError):
                print(f"Warning: invalid VinDR bbox on CSV row {row_idx}: {row}")
                continue

            width = x_max - x_min
            height = y_max - y_min
            if width <= 0 or height <= 0:
                continue

            image_path = resolve_image_path(data_dir, image_id, image_ext=image_ext)
            if not image_path.exists():
                print(f"Warning: missing VinDR image for CSV row {row_idx}: {image_path}")
                continue

            if bbox_coord_size is not None:
                if image_id not in image_size_cache:
                    image_size_cache[image_id] = Image.open(image_path).size
                image_width, image_height = image_size_cache[image_id]
                scale_x = float(image_width) / float(bbox_coord_size)
                scale_y = float(image_height) / float(bbox_coord_size)
                x_min *= scale_x
                x_max *= scale_x
                y_min *= scale_y
                y_max *= scale_y
                width = x_max - x_min
                height = y_max - y_min

            bbox = {"xmin": x_min, "ymin": y_min, "width": width, "height": height}
            if image_id not in grouped:
                grouped[image_id] = {
                    "row": row_idx,
                    "fname": f"{image_id}{image_path.suffix}",
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "bbox": bbox,
                    "bboxes": [],
                    "bbox_classes": [],
                    "target": "abnormality",
                    "tb_type": "",
                    "image_type": "vindr_cxr",
                    "dataset": "vindr",
                }
            grouped[image_id]["bboxes"].append(bbox)
            grouped[image_id]["bbox_classes"].append(class_name)

    rows = list(grouped.values())
    for item in rows:
        item["bbox"] = item["bboxes"][0]

    if limit is not None:
        rows = rows[:limit]
    return rows


def item_image_id(item):
    """Return the image_id for an item, falling back to the fname stem."""
    image_id = item.get("image_id")
    if image_id:
        return str(image_id).strip()
    return Path(str(item.get("fname", ""))).stem


def load_excluded_image_ids(labels_csv, exclude_label="No finding"):
    """
    Return image_ids whose exclude_label column is set in a multi-label CSV.

    The CSV is expected to have an 'image_id' column plus one-hot label columns,
    e.g. vindr/image_labels_test.csv. A value is treated as "set" when it is a
    non-zero number (or a truthy string like true/yes).
    """
    excluded = set()
    with open(labels_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Labels CSV has no header row: {labels_csv}")

        normalized = {
            name.strip().lstrip("\ufeff").lower(): name
            for name in reader.fieldnames
            if name is not None
        }
        image_id_key = normalized.get("image_id")
        label_key = normalized.get(exclude_label.strip().lower())
        if image_id_key is None or label_key is None:
            raise ValueError(
                f"Labels CSV must contain 'image_id' and '{exclude_label}' columns. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            image_id = row.get(image_id_key, "").strip()
            if not image_id:
                continue
            value = (row.get(label_key, "") or "").strip()
            try:
                is_set = float(value) != 0.0
            except (TypeError, ValueError):
                is_set = value.lower() in {"true", "yes", "y"}
            if is_set:
                excluded.add(image_id)

    return excluded


def load_annotation_rows(dataset, csv_file, data_dir, limit=None, image_ext=".png", vindr_classes=None, bbox_coord_size=None):
    """Dispatch to the correct annotation parser."""
    if dataset == "tbx11k":
        return load_tbx11k_bbox_rows(csv_file, data_dir, only_target="tb", limit=limit)
    if dataset == "vindr":
        return load_vindr_bbox_rows(
            csv_file=csv_file,
            data_dir=data_dir,
            image_ext=image_ext,
            classes=vindr_classes,
            limit=limit,
            bbox_coord_size=bbox_coord_size,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_model(model_name, weights_path, device, embedding_dim=None, sra_num_heads=8, sra_lam=0.1):
    """Load ConvNeXtV2 or ConvNeXtV2_SRA with project checkpoint conventions."""
    if model_name == "convnextv2":
        model = ConvNeXtV2(embedding_dim=embedding_dim)
    elif model_name == "convnextv2_sra":
        model = ConvNeXtV2_SRA(
            embedding_dim=embedding_dim,
            num_heads=sra_num_heads,
            lam=sra_lam,
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing:
        print(f"[{model_name}] Warning: {len(missing)} missing checkpoint keys")
    if unexpected:
        print(f"[{model_name}] Warning: {len(unexpected)} unexpected checkpoint keys")

    model.to(device)
    model.eval()
    return model


def build_simatt(model):
    """Build SimAtt on the last spatial ConvNeXt stage."""
    target_layer = model.convnext.stages[-1]
    explainer = SimAtt(model, target_layer, target_layers=None)
    explainer.eval()
    return explainer


def normalize_saliency(saliency):
    saliency = np.asarray(saliency, dtype=np.float32)
    saliency = np.nan_to_num(saliency, nan=0.0, posinf=0.0, neginf=0.0)
    s_min = float(np.min(saliency))
    s_max = float(np.max(saliency))
    if s_max > s_min:
        saliency = (saliency - s_min) / (s_max - s_min)
    else:
        saliency = np.zeros_like(saliency, dtype=np.float32)
    return saliency


def select_saliency_map(saliency_tensor, saliency_index=-1):
    """
    Convert SimAtt output to a single 2D saliency map.

    SimAtt(image, image) usually returns shape (2, H, W). The last map is the
    positive/retrieved image map, which is the one evaluated by default.
    """
    saliency = saliency_tensor.detach().squeeze().cpu().numpy()

    if saliency.ndim == 3:
        idx = saliency_index
        if idx < 0:
            idx = saliency.shape[0] + idx
        if idx < 0 or idx >= saliency.shape[0]:
            raise IndexError(
                f"saliency_index {saliency_index} is invalid for saliency shape {saliency.shape}"
            )
        saliency = saliency[idx]

    if saliency.ndim != 2:
        raise ValueError(f"Expected a 2D saliency map, got shape {saliency.shape}")
    return normalize_saliency(saliency)


def argmax_point_original_coords(saliency, original_width, original_height):
    """
    Find max-saliency point and map it back to original image coordinates.

    The image transform resizes the original image directly to img_size x img_size,
    so this inverse mapping is a simple scale from saliency-map coordinates to
    original image coordinates.
    """
    map_h, map_w = saliency.shape
    flat_idx = int(np.argmax(saliency))
    y_map, x_map = np.unravel_index(flat_idx, saliency.shape)

    x_orig = (x_map + 0.5) * float(original_width) / float(map_w)
    y_orig = (y_map + 0.5) * float(original_height) / float(map_h)
    return x_orig, y_orig, x_map, y_map, float(saliency[y_map, x_map])


def bbox_center(bbox):
    return bbox["xmin"] + bbox["width"] / 2.0, bbox["ymin"] + bbox["height"] / 2.0


def get_item_bboxes(item):
    """Return all bboxes for an item; TBX11k has one, VinDR may have many."""
    bboxes = item.get("bboxes")
    if bboxes:
        return bboxes
    return [item["bbox"]]


def get_item_bbox_classes(item):
    classes = item.get("bbox_classes") or []
    bboxes = get_item_bboxes(item)
    if len(classes) < len(bboxes):
        classes = classes + [""] * (len(bboxes) - len(classes))
    return classes


def point_inside_bbox(x, y, bbox):
    return (
        bbox["xmin"] <= x <= bbox["xmin"] + bbox["width"]
        and bbox["ymin"] <= y <= bbox["ymin"] + bbox["height"]
    )


def find_best_bbox_for_peak(x, y, item):
    """
    Select the bbox used for distance/RMSE.

    If the peak falls inside one or more boxes, use the hit box whose center is
    nearest to the peak. Otherwise use the nearest bbox center among all boxes.
    """
    bboxes = get_item_bboxes(item)
    classes = get_item_bbox_classes(item)
    candidates = []
    for idx, bbox in enumerate(bboxes):
        cx, cy = bbox_center(bbox)
        dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        inside = point_inside_bbox(x, y, bbox)
        candidates.append((not inside, dist, idx, bbox, classes[idx], inside, cx, cy))
    candidates.sort(key=lambda value: (value[0], value[1]))
    _, dist, idx, bbox, cls, inside, cx, cy = candidates[0]
    return {
        "index": idx,
        "bbox": bbox,
        "class_name": cls,
        "inside": bool(inside),
        "center_x": cx,
        "center_y": cy,
        "distance_px": dist,
    }


def expand_bbox(bbox, image_width, image_height, padding_ratio=0.0):
    """Expand bbox by a ratio of its own size and clamp to image bounds."""
    pad_x = bbox["width"] * padding_ratio
    pad_y = bbox["height"] * padding_ratio
    xmin = max(0.0, bbox["xmin"] - pad_x)
    ymin = max(0.0, bbox["ymin"] - pad_y)
    xmax = min(float(image_width), bbox["xmin"] + bbox["width"] + pad_x)
    ymax = min(float(image_height), bbox["ymin"] + bbox["height"] + pad_y)
    return {
        "xmin": xmin,
        "ymin": ymin,
        "width": max(0.0, xmax - xmin),
        "height": max(0.0, ymax - ymin),
    }


def bbox_boolean_mask(shape, bbox):
    """Create a boolean mask for a bbox on an array with shape (height, width)."""
    height, width = shape
    x0 = max(0, int(math.floor(bbox["xmin"])))
    y0 = max(0, int(math.floor(bbox["ymin"])))
    x1 = min(width, int(math.ceil(bbox["xmin"] + bbox["width"])))
    y1 = min(height, int(math.ceil(bbox["ymin"] + bbox["height"])))
    mask = np.zeros((height, width), dtype=bool)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def compute_region_overlap_hit(
    saliency,
    bboxes,
    image_width,
    image_height,
    threshold=0.35,
    bbox_padding_ratio=0.20,
    min_pixels=1,
    bbox_classes=None,
):
    """
    Region-based hit rule for ConvNeXtV2_SRA.

    A hit occurs if any sufficiently salient region overlaps the ground-truth
    bbox after optional bbox expansion. This is more permissive than classic
    Pointing Game because it accepts a salient region touching the lesion area,
    not only the single max-saliency point.
    """
    if isinstance(bboxes, dict):
        bboxes = [bboxes]
    bbox_classes = bbox_classes or [""] * len(bboxes)

    saliency_on_image = resize_saliency_to_image(saliency, (image_width, image_height))
    salient_mask = saliency_on_image >= threshold
    salient_pixels = int(salient_mask.sum())

    best = None
    for idx, bbox in enumerate(bboxes):
        expanded = expand_bbox(bbox, image_width, image_height, padding_ratio=bbox_padding_ratio)
        bbox_mask = bbox_boolean_mask(saliency_on_image.shape, expanded)
        overlap_mask = salient_mask & bbox_mask
        overlap_pixels = int(overlap_mask.sum())
        bbox_pixels = int(bbox_mask.sum())
        candidate = {
            "hit": bool(overlap_pixels >= min_pixels),
            "bbox_index": int(idx),
            "bbox_class": bbox_classes[idx] if idx < len(bbox_classes) else "",
            "matched_bbox": bbox,
            "expanded_bbox": expanded,
            "overlap_pixels": overlap_pixels,
            "salient_pixels": salient_pixels,
            "bbox_pixels": bbox_pixels,
            "overlap_fraction_of_saliency": float(overlap_pixels / salient_pixels) if salient_pixels > 0 else 0.0,
            "overlap_fraction_of_bbox": float(overlap_pixels / bbox_pixels) if bbox_pixels > 0 else 0.0,
        }
        if best is None or candidate["overlap_pixels"] > best["overlap_pixels"]:
            best = candidate

    if best is None:
        best = {
            "hit": False,
            "bbox_index": -1,
            "bbox_class": "",
            "matched_bbox": None,
            "expanded_bbox": None,
            "overlap_pixels": 0,
            "salient_pixels": salient_pixels,
            "bbox_pixels": 0,
            "overlap_fraction_of_saliency": 0.0,
            "overlap_fraction_of_bbox": 0.0,
        }

    hit = bool(best["hit"])

    return {
        "hit": bool(hit),
        "bbox_index": best["bbox_index"],
        "bbox_class": best["bbox_class"],
        "matched_bbox": best["matched_bbox"],
        "expanded_bbox": best["expanded_bbox"],
        "overlap_pixels": best["overlap_pixels"],
        "salient_pixels": best["salient_pixels"],
        "bbox_pixels": best["bbox_pixels"],
        "overlap_fraction_of_saliency": best["overlap_fraction_of_saliency"],
        "overlap_fraction_of_bbox": best["overlap_fraction_of_bbox"],
        "threshold": float(threshold),
        "bbox_padding_ratio": float(bbox_padding_ratio),
        "min_pixels": int(min_pixels),
    }


def safe_stem(text):
    """Make a filesystem-safe short filename stem."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(text))


def resize_saliency_to_image(saliency, image_size):
    """Resize normalized 2D saliency to PIL image size=(width, height)."""
    saliency = normalize_saliency(saliency)
    saliency_img = Image.fromarray((saliency * 255).astype(np.uint8))
    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR
    return np.asarray(saliency_img.resize(image_size, resample=resample), dtype=np.float32) / 255.0


def add_bbox_and_peak(ax, bbox, peak_x=None, peak_y=None, title=None):
    """Draw TB bbox and optional max-saliency point on an axis."""
    rect = patches.Rectangle(
        (bbox["xmin"], bbox["ymin"]),
        bbox["width"],
        bbox["height"],
        linewidth=2.0,
        edgecolor="lime",
        facecolor="none",
    )
    ax.add_patch(rect)
    if peak_x is not None and peak_y is not None:
        ax.scatter([peak_x], [peak_y], s=70, c="red", marker="x", linewidths=2.5)
    if title:
        ax.set_title(title)
    ax.axis("off")


def add_item_bboxes_and_peak(ax, item, result, peak_x=None, peak_y=None, title=None):
    """Draw all bboxes for an item and highlight the matched/evaluated bbox."""
    bboxes = get_item_bboxes(item)
    classes = get_item_bbox_classes(item)
    matched_idx = int(result.get("matched_bbox_index", 0))
    for idx, bbox in enumerate(bboxes):
        is_matched = idx == matched_idx
        rect = patches.Rectangle(
            (bbox["xmin"], bbox["ymin"]),
            bbox["width"],
            bbox["height"],
            linewidth=2.4 if is_matched else 1.4,
            edgecolor="lime" if is_matched else "yellow",
            facecolor="none",
            linestyle="-" if is_matched else ":",
        )
        ax.add_patch(rect)
        if classes[idx]:
            ax.text(
                bbox["xmin"],
                max(0, bbox["ymin"] - 3),
                classes[idx],
                color="lime" if is_matched else "yellow",
                fontsize=7,
                bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1),
            )
    if peak_x is not None and peak_y is not None:
        ax.scatter([peak_x], [peak_y], s=70, c="red", marker="x", linewidths=2.5)
    if title:
        ax.set_title(title)
    ax.axis("off")


def add_expanded_bbox(ax, bbox):
    """Draw expanded SRA acceptance bbox as a cyan dashed rectangle."""
    rect = patches.Rectangle(
        (bbox["xmin"], bbox["ymin"]),
        bbox["width"],
        bbox["height"],
        linewidth=1.6,
        edgecolor="cyan",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)


def sra_masked_saliency_for_visualization(saliency, image_size, result, visual_threshold=0.25):
    """
    Return a saliency image masked to the expanded bbox acceptance region.

    Pixels outside the expanded bbox or below visual_threshold are hidden. This
    creates the requested fourth panel: only the saliency region that contributes
    to the bbox-region hit is displayed; other regions are non-displayed.
    """
    saliency_on_image = resize_saliency_to_image(saliency, image_size)
    expanded_bbox = result.get("region_expanded_bbox") or result.get("retrieved_bbox")
    bbox_mask = bbox_boolean_mask(saliency_on_image.shape, expanded_bbox)
    visible_mask = bbox_mask & (saliency_on_image >= visual_threshold)
    return np.ma.masked_where(~visible_mask, saliency_on_image), visible_mask


def save_retrieval_visualization(
    model_name,
    query_item,
    retrieved_item,
    result,
    saliency,
    output_path,
    dpi=150,
    sra_region_visual_threshold=0.25,
):
    """Save query/retrieved/overlay figure; SRA also gets a bbox-hit-only panel."""
    query_img = Image.open(query_item["image_path"]).convert("RGB")
    retrieved_img = Image.open(retrieved_item["image_path"]).convert("RGB")
    saliency_overlay = resize_saliency_to_image(saliency, retrieved_img.size)
    is_sra_region = result.get("hit_rule") == "region_overlap"

    ncols = 4 if is_sra_region else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    axes[0].imshow(query_img)
    axes[0].set_title(f"Query\n{query_item['fname']}")
    axes[0].axis("off")

    axes[1].imshow(retrieved_img)
    add_item_bboxes_and_peak(
        axes[1],
        retrieved_item,
        result,
        result["peak_x"],
        result["peak_y"],
        title=(
            f"Retrieved rank {result['retrieval_rank']}\n"
            f"{retrieved_item['fname']} | sim={result['query_retrieved_similarity']:.4f}"
        ),
    )

    axes[2].imshow(retrieved_img)
    axes[2].imshow(saliency_overlay, cmap="jet", alpha=0.45)
    add_item_bboxes_and_peak(
        axes[2],
        retrieved_item,
        result,
        result["peak_x"],
        result["peak_y"],
        title=(
            f"{model_name} SimAtt on retrieved image\n"
            f"PG={'hit' if result['pg_hit'] else 'miss'} | dist={result['distance_px']:.1f}px"
        ),
    )
    if is_sra_region and result.get("region_expanded_bbox") is not None:
        add_expanded_bbox(axes[2], result["region_expanded_bbox"])

    if is_sra_region:
        masked_saliency, visible_mask = sra_masked_saliency_for_visualization(
            saliency=saliency,
            image_size=retrieved_img.size,
            result=result,
            visual_threshold=sra_region_visual_threshold,
        )
        axes[3].imshow(retrieved_img, alpha=0.55)
        axes[3].imshow(masked_saliency, cmap="jet", alpha=0.90)
        add_item_bboxes_and_peak(
            axes[3],
            retrieved_item,
            result,
            result["peak_x"],
            result["peak_y"],
            title=(
                "SRA saliency inside bbox region only\n"
                f"thr={sra_region_visual_threshold:.2f} | overlap={result.get('region_overlap_pixels', 0)} px"
            ),
        )
        if result.get("region_expanded_bbox") is not None:
            add_expanded_bbox(axes[3], result["region_expanded_bbox"])

    legend_handles = [
        patches.Patch(edgecolor="lime", facecolor="none", label="Matched bbox"),
        patches.Patch(edgecolor="yellow", facecolor="none", linestyle=":", label="Other bbox"),
        patches.Patch(edgecolor="cyan", facecolor="none", linestyle="--", label="Expanded SRA hit region"),
        plt.Line2D([0], [0], color="red", marker="x", linestyle="None", markersize=8, label="Max saliency"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_self_visualization(model_name, item, result, saliency, output_path, dpi=150):
    """Save a 2-panel self-SimAtt sanity-check visualization."""
    image = Image.open(item["image_path"]).convert("RGB")
    saliency_overlay = resize_saliency_to_image(saliency, image.size)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    add_item_bboxes_and_peak(axes[0], item, result, result["peak_x"], result["peak_y"], title=f"Image\n{item['fname']}")
    axes[1].imshow(image)
    axes[1].imshow(saliency_overlay, cmap="jet", alpha=0.45)
    add_item_bboxes_and_peak(
        axes[1],
        item,
        result,
        result["peak_x"],
        result["peak_y"],
        title=(
            f"{model_name} self-SimAtt overlay\n"
            f"PG={'hit' if result['pg_hit'] else 'miss'} | dist={result['distance_px']:.1f}px"
        ),
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_summary_comparison_plot(summaries, output_path, dpi=150):
    """Save a compact bar plot comparing PG hit rate and RMSE across models."""
    if not summaries:
        return

    model_names = list(summaries.keys())
    pg_values = [summaries[name]["pg_hit_rate"] for name in model_names]
    rmse_values = [summaries[name]["rmse_distance_px"] for name in model_names]
    norm_rmse_values = [summaries[name]["rmse_normalized_distance"] for name in model_names]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"][: len(model_names)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(model_names, pg_values, color=colors)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Hit rate")
    axes[0].set_ylabel("Hit rate")

    axes[1].bar(model_names, rmse_values, color=colors)
    axes[1].set_title("RMSE distance")
    axes[1].set_ylabel("Pixels")

    axes[2].bar(model_names, norm_rmse_values, color=colors)
    axes[2].set_title("Normalized RMSE")
    axes[2].set_ylabel("Distance / image diagonal")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def compute_embeddings(model, rows, transform, device, batch_size=32):
    """Compute L2-normalized image embeddings for retrieval."""
    embeddings = []

    for start in tqdm(range(0, len(rows), batch_size), desc="Computing retrieval embeddings"):
        batch_rows = rows[start:start + batch_size]
        batch_tensors = []

        for item in batch_rows:
            image = Image.open(item["image_path"]).convert("RGB")
            batch_tensors.append(transform(image))

        batch = torch.stack(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            batch_embeddings = model(batch)
            if isinstance(batch_embeddings, dict):
                batch_embeddings = batch_embeddings["embedding"]
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
        embeddings.append(batch_embeddings.detach().cpu())

        del batch, batch_embeddings
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return torch.cat(embeddings, dim=0)


def build_retrieval_pairs(embeddings, top_k=1):
    """
    Build query->retrieved index pairs from cosine similarity matrix.

    Returns tuples: (query_idx, retrieved_idx, rank, similarity).
    Self matches are excluded.
    """
    if len(embeddings) < 2:
        raise ValueError("Retrieval-pair evaluation needs at least 2 bbox images.")

    sim_matrix = torch.matmul(embeddings, embeddings.T).numpy()
    pairs = []
    k = min(top_k, len(embeddings) - 1)

    for query_idx in range(sim_matrix.shape[0]):
        sims = sim_matrix[query_idx].copy()
        sims[query_idx] = -np.inf
        retrieved_indices = np.argsort(sims)[::-1][:k]
        for rank, retrieved_idx in enumerate(retrieved_indices, start=1):
            pairs.append((query_idx, int(retrieved_idx), rank, float(sims[retrieved_idx])))

    return pairs


def evaluate_saliency_against_item(
    saliency,
    item,
    query_item=None,
    similarity=None,
    rank=None,
    use_region_hit=False,
    region_threshold=0.35,
    region_bbox_padding_ratio=0.20,
    region_min_pixels=1,
):
    """Compute PG/RMSE for a saliency map over item['image_path']."""
    image = Image.open(item["image_path"]).convert("RGB")
    original_width, original_height = image.size

    x_peak, y_peak, x_map, y_map, peak_value = argmax_point_original_coords(
        saliency, original_width, original_height
    )
    matched = find_best_bbox_for_peak(x_peak, y_peak, item)
    x_center, y_center = matched["center_x"], matched["center_y"]
    distance_px = matched["distance_px"]
    diagonal = math.sqrt(original_width ** 2 + original_height ** 2)
    normalized_distance = distance_px / diagonal if diagonal > 0 else float("nan")
    classic_pg_hit = bool(matched["inside"])

    region_info = None
    if use_region_hit:
        region_info = compute_region_overlap_hit(
            saliency=saliency,
            bboxes=get_item_bboxes(item),
            image_width=original_width,
            image_height=original_height,
            threshold=region_threshold,
            bbox_padding_ratio=region_bbox_padding_ratio,
            min_pixels=region_min_pixels,
            bbox_classes=get_item_bbox_classes(item),
        )
        hit = region_info["hit"]
        hit_rule = "region_overlap"
    else:
        hit = classic_pg_hit
        hit_rule = "max_point_pg"

    result = {
        "retrieved_fname": item["fname"],
        "retrieved_image_path": item["image_path"],
        "retrieved_target": item.get("target", ""),
        "retrieved_tb_type": item.get("tb_type", ""),
        "retrieved_bbox": matched["bbox"],
        "retrieved_bboxes": get_item_bboxes(item),
        "retrieved_bbox_classes": get_item_bbox_classes(item),
        "matched_bbox_index": int(matched["index"]),
        "matched_bbox_class": matched["class_name"],
        "bbox_center_x": x_center,
        "bbox_center_y": y_center,
        "peak_x": x_peak,
        "peak_y": y_peak,
        "peak_map_x": int(x_map),
        "peak_map_y": int(y_map),
        "peak_saliency": peak_value,
        "pg_hit": bool(hit),
        "classic_pg_hit": bool(classic_pg_hit),
        "hit_rule": hit_rule,
        "distance_px": distance_px,
        "normalized_distance": normalized_distance,
        "image_width": original_width,
        "image_height": original_height,
        "saliency_height": int(saliency.shape[0]),
        "saliency_width": int(saliency.shape[1]),
    }

    if region_info is not None:
        result.update(
            {
                "region_expanded_bbox": region_info["expanded_bbox"],
                "region_matched_bbox": region_info["matched_bbox"],
                "region_matched_bbox_index": region_info["bbox_index"],
                "region_matched_bbox_class": region_info["bbox_class"],
                "region_overlap_pixels": region_info["overlap_pixels"],
                "region_salient_pixels": region_info["salient_pixels"],
                "region_bbox_pixels": region_info["bbox_pixels"],
                "region_overlap_fraction_of_saliency": region_info["overlap_fraction_of_saliency"],
                "region_overlap_fraction_of_bbox": region_info["overlap_fraction_of_bbox"],
                "region_threshold": region_info["threshold"],
                "region_bbox_padding_ratio": region_info["bbox_padding_ratio"],
                "region_min_pixels": region_info["min_pixels"],
            }
        )

    if query_item is None:
        result.update(
            {
                "fname": item["fname"],
                "image_path": item["image_path"],
                "target": item.get("target", ""),
                "tb_type": item.get("tb_type", ""),
                "bbox": item["bbox"],
                "bboxes": get_item_bboxes(item),
                "bbox_classes": get_item_bbox_classes(item),
            }
        )
    else:
        result.update(
            {
                "query_fname": query_item["fname"],
                "query_image_path": query_item["image_path"],
                "query_tb_type": query_item.get("tb_type", ""),
                "retrieval_rank": int(rank),
                "query_retrieved_similarity": float(similarity),
            }
        )

    return result


def evaluate_self_model(
    model_name,
    model,
    rows,
    transform,
    device,
    saliency_index=-1,
    visualization_dir=None,
    max_visualizations=0,
    visualization_dpi=150,
    use_region_hit=False,
    region_threshold=0.35,
    region_bbox_padding_ratio=0.20,
    region_min_pixels=1,
):
    explainer = build_simatt(model).to(device)
    results = []
    vis_count = 0

    for item in tqdm(rows, desc=f"Evaluating {model_name} self-SimAtt"):
        image = Image.open(item["image_path"]).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Self-similarity SimAtt: evaluate localization on the same image that has bbox.
        model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(True):
            saliency_tensor = explainer(image_tensor, image_tensor)
        saliency = select_saliency_map(saliency_tensor, saliency_index=saliency_index)
        result = evaluate_saliency_against_item(
            saliency,
            item,
            use_region_hit=use_region_hit,
            region_threshold=region_threshold,
            region_bbox_padding_ratio=region_bbox_padding_ratio,
            region_min_pixels=region_min_pixels,
        )
        results.append(result)

        if visualization_dir is not None and vis_count < max_visualizations:
            output_path = Path(visualization_dir) / model_name / f"self_{vis_count + 1:04d}_{safe_stem(item['fname'])}.png"
            save_self_visualization(model_name, item, result, saliency, output_path, dpi=visualization_dpi)
            vis_count += 1

        del image_tensor, saliency_tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


def evaluate_retrieval_model(
    model_name,
    model,
    rows,
    transform,
    device,
    top_k=1,
    embedding_batch_size=32,
    saliency_index=-1,
    visualization_dir=None,
    max_visualizations=0,
    visualization_dpi=150,
    use_region_hit=False,
    region_threshold=0.35,
    region_bbox_padding_ratio=0.20,
    region_min_pixels=1,
    sra_region_visual_threshold=0.25,
):
    """
    Evaluate SimAtt on query->retrieved pairs.

    The saliency map evaluated is the retrieved-image map from
    SimAtt(query_tensor, retrieved_tensor), selected by saliency_index=-1.
    """
    print(f"[{model_name}] Building retrieval set using cosine similarity...")
    embeddings = compute_embeddings(
        model=model,
        rows=rows,
        transform=transform,
        device=device,
        batch_size=embedding_batch_size,
    )
    pairs = build_retrieval_pairs(embeddings, top_k=top_k)
    print(f"[{model_name}] Evaluating {len(pairs)} query->retrieved SimAtt pairs")

    explainer = build_simatt(model).to(device)
    results = []
    vis_count = 0

    for query_idx, retrieved_idx, rank, similarity in tqdm(pairs, desc=f"Evaluating {model_name} retrieval-SimAtt"):
        query_item = rows[query_idx]
        retrieved_item = rows[retrieved_idx]

        query_image = Image.open(query_item["image_path"]).convert("RGB")
        retrieved_image = Image.open(retrieved_item["image_path"]).convert("RGB")
        query_tensor = transform(query_image).unsqueeze(0).to(device)
        retrieved_tensor = transform(retrieved_image).unsqueeze(0).to(device)

        model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(True):
            saliency_tensor = explainer(query_tensor, retrieved_tensor)
        saliency = select_saliency_map(saliency_tensor, saliency_index=saliency_index)

        result = evaluate_saliency_against_item(
            saliency=saliency,
            item=retrieved_item,
            query_item=query_item,
            similarity=similarity,
            rank=rank,
            use_region_hit=use_region_hit,
            region_threshold=region_threshold,
            region_bbox_padding_ratio=region_bbox_padding_ratio,
            region_min_pixels=region_min_pixels,
        )
        results.append(result)

        if visualization_dir is not None and vis_count < max_visualizations:
            fname = (
                f"pair_{vis_count + 1:04d}_"
                f"q_{safe_stem(query_item['fname'])}_"
                f"r{rank}_{safe_stem(retrieved_item['fname'])}.png"
            )
            output_path = Path(visualization_dir) / model_name / fname
            save_retrieval_visualization(
                model_name=model_name,
                query_item=query_item,
                retrieved_item=retrieved_item,
                result=result,
                saliency=saliency,
                output_path=output_path,
                dpi=visualization_dpi,
                sra_region_visual_threshold=sra_region_visual_threshold,
            )
            vis_count += 1

        del query_tensor, retrieved_tensor, saliency_tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


def summarize_results(results):
    if not results:
        return {
            "num_samples": 0,
            "pg_hits": 0,
            "pg_hit_rate": None,
            "mean_distance_px": None,
            "median_distance_px": None,
            "rmse_distance_px": None,
            "mean_normalized_distance": None,
            "rmse_normalized_distance": None,
        }

    hits = np.array([r["pg_hit"] for r in results], dtype=np.float32)
    distances = np.array([r["distance_px"] for r in results], dtype=np.float64)
    norm_distances = np.array([r["normalized_distance"] for r in results], dtype=np.float64)

    return {
        "num_samples": int(len(results)),
        "pg_hits": int(hits.sum()),
        "pg_hit_rate": float(hits.mean()),
        "mean_distance_px": float(distances.mean()),
        "median_distance_px": float(np.median(distances)),
        "rmse_distance_px": float(np.sqrt(np.mean(distances ** 2))),
        "mean_normalized_distance": float(norm_distances.mean()),
        "rmse_normalized_distance": float(np.sqrt(np.mean(norm_distances ** 2))),
    }


def save_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Pointing Game and RMSE localization for SimAtt on TBX11k."
    )
    parser.add_argument("--dataset", type=str, default="tbx11k", choices=["tbx11k", "vindr"], help="Annotation format/dataset to evaluate")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--csv_file", type=str, default="test.csv", help="Annotation CSV file")
    parser.add_argument("--image_ext", type=str, default=".png", help="Image extension for VinDR image_id lookup, e.g. .png, .jpg")
    parser.add_argument(
        "--vindr_classes",
        type=str,
        default=None,
        help="Optional comma-separated VinDR class filter, e.g. 'Nodule/Mass,ILD'. Default uses all annotated classes.",
    )
    parser.add_argument(
        "--bbox_coord_size",
        type=int,
        default=None,
        help="If annotation bboxes are in a fixed square coordinate size, e.g. 384 for annotations_rescaled_384.csv, scale them to actual image size.",
    )
    parser.add_argument(
        "--exclude_labels_csv",
        type=str,
        default=None,
        help="Optional multi-label CSV (image_id + one-hot labels). Images flagged with --exclude_label are removed before retrieval and evaluation.",
    )
    parser.add_argument(
        "--exclude_label",
        type=str,
        default="No finding",
        help="Label column in --exclude_labels_csv to exclude, e.g. 'No finding'.",
    )
    parser.add_argument("--convnextv2_weights", type=str, default=None, help="Checkpoint for ConvNeXtV2")
    parser.add_argument("--convnextv2_sra_weights", type=str, default=None, help="Checkpoint for ConvNeXtV2_SRA")
    parser.add_argument("--embedding_dim", type=int, default=None, help="Embedding dimension used during training")
    parser.add_argument("--sra_num_heads", type=int, default=8, help="Number of SRA heads")
    parser.add_argument("--sra_lam", type=float, default=0.1, help="SRA residual attention lambda")
    parser.add_argument("--img_size", type=int, default=384, help="ConvNeXtV2 input size")
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="retrieval",
        choices=["retrieval", "self"],
        help="retrieval = SimAtt(query, retrieved) from top-k cosine retrieval; self = SimAtt(image, image) sanity check",
    )
    parser.add_argument("--top_k", type=int, default=1, help="Number of retrieved bbox images per query for retrieval mode")
    parser.add_argument("--embedding_batch_size", type=int, default=32, help="Batch size for retrieval embedding computation")
    parser.add_argument("--saliency_index", type=int, default=-1, help="Map index if SimAtt returns multiple maps; -1 = retrieved/positive image map")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick debugging")
    parser.add_argument("--output_dir", type=str, default="pg_rmse_results", help="Output directory")
    parser.add_argument("--save_visualizations", action="store_true", help="Save saliency overlay visualizations with bbox and max-saliency marker")
    parser.add_argument("--max_visualizations", type=int, default=25, help="Maximum number of visualizations to save per model")
    parser.add_argument("--visualization_dpi", type=int, default=150, help="DPI for saved visualization PNG files")
    parser.add_argument(
        "--sra_region_hit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use region-overlap hit rule for convnextv2_sra. ConvNeXtV2 still uses classic max-point PG.",
    )
    parser.add_argument(
        "--sra_region_threshold",
        type=float,
        default=0.35,
        help="Normalized saliency threshold for SRA region hit. Lower values accept larger saliency regions.",
    )
    parser.add_argument(
        "--sra_region_visual_threshold",
        type=float,
        default=0.25,
        help="Lower threshold used only for the fourth SRA visualization panel to show a broader saliency region.",
    )
    parser.add_argument(
        "--sra_region_bbox_padding_ratio",
        type=float,
        default=0.20,
        help="Expand bbox by this fraction of bbox width/height for SRA region-hit acceptance.",
    )
    parser.add_argument(
        "--sra_region_min_pixels",
        type=int,
        default=1,
        help="Minimum thresholded saliency pixels inside expanded bbox required for SRA region hit.",
    )
    parser.add_argument(
        "--convnextv2_region_threshold",
        type=float,
        default=None,
        help="Normalized saliency threshold enabling region-overlap hit for convnextv2. "
        "If unset (default), convnextv2 uses classic max-point Pointing Game.",
    )
    parser.add_argument(
        "--convnextv2_region_bbox_padding_ratio",
        type=float,
        default=0.20,
        help="Expand bbox by this fraction of bbox width/height for convnextv2 region-hit acceptance.",
    )
    parser.add_argument(
        "--convnextv2_region_min_pixels",
        type=int,
        default=1,
        help="Minimum thresholded saliency pixels inside expanded bbox required for convnextv2 region hit.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.convnextv2_weights is None and args.convnextv2_sra_weights is None:
        raise ValueError("Provide at least one checkpoint: --convnextv2_weights or --convnextv2_sra_weights")

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rows = load_annotation_rows(
        dataset=args.dataset,
        csv_file=args.csv_file,
        data_dir=args.data_dir,
        limit=args.limit,
        image_ext=args.image_ext,
        vindr_classes=args.vindr_classes,
        bbox_coord_size=args.bbox_coord_size,
    )
    if not rows:
        raise RuntimeError(
            "No valid bbox rows found. Check --dataset, --csv_file, --data_dir, image extension, and optional class filters."
        )
    print(f"Loaded {len(rows)} {args.dataset} annotated image samples from {args.csv_file}")

    if args.exclude_labels_csv:
        excluded_ids = load_excluded_image_ids(args.exclude_labels_csv, args.exclude_label)
        before = len(rows)
        rows = [row for row in rows if item_image_id(row) not in excluded_ids]
        print(
            f"Excluded {before - len(rows)} images with label '{args.exclude_label}' "
            f"({len(excluded_ids)} flagged ids in {args.exclude_labels_csv}); {len(rows)} remain"
        )
        if not rows:
            raise RuntimeError(
                f"All images were excluded by label '{args.exclude_label}'. Nothing left to evaluate."
            )

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((args.img_size, args.img_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = output_dir / "visualizations" if args.save_visualizations else None

    all_summaries = {}
    started = datetime.now().isoformat(timespec="seconds")

    model_specs = []
    if args.convnextv2_weights:
        model_specs.append(("convnextv2", args.convnextv2_weights))
    if args.convnextv2_sra_weights:
        model_specs.append(("convnextv2_sra", args.convnextv2_sra_weights))

    for model_name, weights_path in model_specs:
        print("\n" + "=" * 80)
        print(f"Loading and evaluating {model_name}")
        print(f"Weights: {weights_path}")
        print("=" * 80)

        model = load_model(
            model_name=model_name,
            weights_path=weights_path,
            device=device,
            embedding_dim=args.embedding_dim,
            sra_num_heads=args.sra_num_heads,
            sra_lam=args.sra_lam,
        )
        use_region_hit = model_name == "convnextv2_sra" and args.sra_region_hit
        if model_name == "convnextv2_sra":
            region_threshold = args.sra_region_threshold
            region_bbox_padding_ratio = args.sra_region_bbox_padding_ratio
            region_min_pixels = args.sra_region_min_pixels
        else:
            # convnextv2 (and any non-SRA model) only uses region-overlap when a
            # threshold is explicitly provided; otherwise it stays classic PG.
            use_region_hit = args.convnextv2_region_threshold is not None
            region_threshold = args.convnextv2_region_threshold
            region_bbox_padding_ratio = args.convnextv2_region_bbox_padding_ratio
            region_min_pixels = args.convnextv2_region_min_pixels

        if use_region_hit:
            print(
                f"[{model_name}] Using region-overlap hit rule: "
                f"threshold={region_threshold}, "
                f"visual_threshold={args.sra_region_visual_threshold}, "
                f"bbox_padding_ratio={region_bbox_padding_ratio}, "
                f"min_pixels={region_min_pixels}"
            )
        else:
            print(f"[{model_name}] Using classic max-point Pointing Game hit rule")

        if args.eval_mode == "retrieval":
            per_image = evaluate_retrieval_model(
                model_name=model_name,
                model=model,
                rows=rows,
                transform=transform,
                device=device,
                top_k=args.top_k,
                embedding_batch_size=args.embedding_batch_size,
                saliency_index=args.saliency_index,
                visualization_dir=visualization_dir,
                max_visualizations=args.max_visualizations,
                visualization_dpi=args.visualization_dpi,
                use_region_hit=use_region_hit,
                region_threshold=region_threshold,
                region_bbox_padding_ratio=region_bbox_padding_ratio,
                region_min_pixels=region_min_pixels,
                sra_region_visual_threshold=args.sra_region_visual_threshold,
            )
        else:
            per_image = evaluate_self_model(
                model_name=model_name,
                model=model,
                rows=rows,
                transform=transform,
                device=device,
                saliency_index=args.saliency_index,
                visualization_dir=visualization_dir,
                max_visualizations=args.max_visualizations,
                visualization_dpi=args.visualization_dpi,
                use_region_hit=use_region_hit,
                region_threshold=region_threshold,
                region_bbox_padding_ratio=region_bbox_padding_ratio,
                region_min_pixels=region_min_pixels,
            )
        summary = summarize_results(per_image)
        all_summaries[model_name] = summary

        per_image_csv = output_dir / f"{model_name}_pg_rmse_per_image.csv"
        per_image_json = output_dir / f"{model_name}_pg_rmse_per_image.json"
        save_csv(per_image_csv, per_image)
        with open(per_image_json, "w", encoding="utf-8") as f:
            json.dump(per_image, f, indent=2)

        print(f"\n{model_name} summary:")
        print(f"  Samples:              {summary['num_samples']}")
        print(f"  PG hits:              {summary['pg_hits']}")
        print(f"  PG hit rate:          {summary['pg_hit_rate']:.4f}")
        print(f"  Mean distance (px):   {summary['mean_distance_px']:.2f}")
        print(f"  Median distance (px): {summary['median_distance_px']:.2f}")
        print(f"  RMSE distance (px):   {summary['rmse_distance_px']:.2f}")
        print(f"  RMSE normalized:      {summary['rmse_normalized_distance']:.4f}")
        print(f"  Saved per-image CSV:  {per_image_csv}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_payload = {
        "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "csv_file": os.path.abspath(args.csv_file),
        "data_dir": os.path.abspath(args.data_dir),
        "dataset": args.dataset,
        "image_ext": args.image_ext,
        "vindr_classes": args.vindr_classes,
        "bbox_coord_size": args.bbox_coord_size,
        "exclude_labels_csv": os.path.abspath(args.exclude_labels_csv) if args.exclude_labels_csv else None,
        "exclude_label": args.exclude_label if args.exclude_labels_csv else None,
        "img_size": args.img_size,
        "eval_mode": args.eval_mode,
        "top_k": args.top_k,
        "saliency_index": args.saliency_index,
        "save_visualizations": args.save_visualizations,
        "max_visualizations": args.max_visualizations,
        "sra_region_hit": args.sra_region_hit,
        "sra_region_threshold": args.sra_region_threshold,
        "sra_region_visual_threshold": args.sra_region_visual_threshold,
        "sra_region_bbox_padding_ratio": args.sra_region_bbox_padding_ratio,
        "sra_region_min_pixels": args.sra_region_min_pixels,
        "convnextv2_region_threshold": args.convnextv2_region_threshold,
        "convnextv2_region_bbox_padding_ratio": args.convnextv2_region_bbox_padding_ratio,
        "convnextv2_region_min_pixels": args.convnextv2_region_min_pixels,
        "num_bbox_samples": len(rows),
        "summaries": all_summaries,
    }
    summary_path = output_dir / "summary_pg_rmse.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    summary_plot_path = output_dir / "summary_comparison.png"
    save_summary_comparison_plot(all_summaries, summary_plot_path, dpi=args.visualization_dpi)

    print("\n" + "=" * 80)
    print("Final summary")
    print("=" * 80)
    for model_name, summary in all_summaries.items():
        print(
            f"{model_name}: PG={summary['pg_hit_rate']:.4f}, "
            f"RMSE_px={summary['rmse_distance_px']:.2f}, "
            f"RMSE_norm={summary['rmse_normalized_distance']:.4f}"
        )
    print(f"Saved summary JSON: {summary_path}")
    print(f"Saved summary plot: {summary_plot_path}")
    if visualization_dir is not None:
        print(f"Saved overlay visualizations under: {visualization_dir}")


if __name__ == "__main__":
    main()
