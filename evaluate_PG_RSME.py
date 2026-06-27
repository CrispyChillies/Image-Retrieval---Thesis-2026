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

Notes
-----
- This script evaluates only rows with a valid bbox, i.e. TB-positive rows.
- SimAtt is a pairwise/retrieval explainer. For localization against a single
  image-level bbox, this script uses self-similarity by default:

      saliency = SimAtt(image, image)

  You can switch to the first returned map with --saliency-index 0, but the
  default --saliency-index -1 uses the explained/retrieved image map.
- The file name intentionally follows the user's requested spelling:
  evaluate_PG_RSME.py. The metric name printed in outputs uses RMSE.

Example
-------
python evaluate_PG_RSME.py ^
  --data_dir path/to/tbx11k/images ^
  --csv_file test.csv ^
  --convnextv2_weights checkpoints/convnextv2.pth ^
  --convnextv2_sra_weights checkpoints/convnextv2_sra.pth ^
  --embedding_dim 512 ^
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
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from explanations import SimAtt
from model import ConvNeXtV2, ConvNeXtV2_SRA


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

            image_path = data_dir / fname
            if not image_path.exists():
                print(f"Warning: missing image for CSV row {row_idx}: {image_path}")
                continue

            rows.append(
                {
                    "row": row_idx,
                    "fname": fname,
                    "image_path": str(image_path),
                    "bbox": bbox,
                    "target": target,
                    "tb_type": row.get(tb_type_key, "").strip() if tb_type_key else "",
                    "image_type": image_type,
                }
            )
            if limit is not None and len(rows) >= limit:
                break

    return rows


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


def point_inside_bbox(x, y, bbox):
    return (
        bbox["xmin"] <= x <= bbox["xmin"] + bbox["width"]
        and bbox["ymin"] <= y <= bbox["ymin"] + bbox["height"]
    )


def evaluate_model(model_name, model, rows, transform, device, saliency_index=-1):
    explainer = build_simatt(model).to(device)
    results = []

    for item in tqdm(rows, desc=f"Evaluating {model_name}"):
        image = Image.open(item["image_path"]).convert("RGB")
        original_width, original_height = image.size
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Self-similarity SimAtt: evaluate localization on the same image that has bbox.
        model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(True):
            saliency_tensor = explainer(image_tensor, image_tensor)
        saliency = select_saliency_map(saliency_tensor, saliency_index=saliency_index)

        x_peak, y_peak, x_map, y_map, peak_value = argmax_point_original_coords(
            saliency, original_width, original_height
        )
        x_center, y_center = bbox_center(item["bbox"])
        distance_px = math.sqrt((x_peak - x_center) ** 2 + (y_peak - y_center) ** 2)
        diagonal = math.sqrt(original_width ** 2 + original_height ** 2)
        normalized_distance = distance_px / diagonal if diagonal > 0 else float("nan")
        hit = point_inside_bbox(x_peak, y_peak, item["bbox"])

        results.append(
            {
                "fname": item["fname"],
                "image_path": item["image_path"],
                "target": item.get("target", ""),
                "tb_type": item.get("tb_type", ""),
                "bbox": item["bbox"],
                "bbox_center_x": x_center,
                "bbox_center_y": y_center,
                "peak_x": x_peak,
                "peak_y": y_peak,
                "peak_map_x": int(x_map),
                "peak_map_y": int(y_map),
                "peak_saliency": peak_value,
                "pg_hit": bool(hit),
                "distance_px": distance_px,
                "normalized_distance": normalized_distance,
                "image_width": original_width,
                "image_height": original_height,
                "saliency_height": int(saliency.shape[0]),
                "saliency_width": int(saliency.shape[1]),
            }
        )

        del image_tensor, saliency_tensor
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
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing TBX11k images")
    parser.add_argument("--csv_file", type=str, default="test.csv", help="TBX11k test CSV with bbox column")
    parser.add_argument("--convnextv2_weights", type=str, default=None, help="Checkpoint for ConvNeXtV2")
    parser.add_argument("--convnextv2_sra_weights", type=str, default=None, help="Checkpoint for ConvNeXtV2_SRA")
    parser.add_argument("--embedding_dim", type=int, default=None, help="Embedding dimension used during training")
    parser.add_argument("--sra_num_heads", type=int, default=8, help="Number of SRA heads")
    parser.add_argument("--sra_lam", type=float, default=0.1, help="SRA residual attention lambda")
    parser.add_argument("--img_size", type=int, default=384, help="ConvNeXtV2 input size")
    parser.add_argument("--saliency_index", type=int, default=-1, help="Map index if SimAtt returns multiple maps; -1 = retrieved/self positive map")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick debugging")
    parser.add_argument("--output_dir", type=str, default="pg_rmse_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.convnextv2_weights is None and args.convnextv2_sra_weights is None:
        raise ValueError("Provide at least one checkpoint: --convnextv2_weights or --convnextv2_sra_weights")

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rows = load_tbx11k_bbox_rows(args.csv_file, args.data_dir, only_target="tb", limit=args.limit)
    if not rows:
        raise RuntimeError(
            "No valid TB bbox rows found. Check --csv_file, --data_dir, and the CSV bbox/target columns."
        )
    print(f"Loaded {len(rows)} TB bbox samples from {args.csv_file}")

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
        per_image = evaluate_model(
            model_name=model_name,
            model=model,
            rows=rows,
            transform=transform,
            device=device,
            saliency_index=args.saliency_index,
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
        "img_size": args.img_size,
        "saliency_index": args.saliency_index,
        "num_bbox_samples": len(rows),
        "summaries": all_summaries,
    }
    summary_path = output_dir / "summary_pg_rmse.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

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


if __name__ == "__main__":
    main()
