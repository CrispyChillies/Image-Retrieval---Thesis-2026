import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def read_json(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_image_path(raw_path: str, images_root: Path | None, basename_index: dict[str, Path] | None) -> Path | None:
    if not raw_path:
        return None

    candidate = Path(raw_path)

    # 1) Absolute path as-is.
    if candidate.is_absolute() and candidate.exists():
        return candidate

    # 2) Relative path under images_root.
    if images_root is not None:
        rel_candidate = images_root / candidate
        if rel_candidate.exists():
            return rel_candidate

        # 3) Basename fallback via prebuilt index.
        if basename_index is not None:
            hit = basename_index.get(candidate.name)
            if hit is not None and hit.exists():
                return hit

    return None


def build_basename_index(images_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for p in images_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            # Keep first hit for deterministic behavior.
            index.setdefault(p.name, p)
    return index


def extract_retrieved_image_path(item: dict) -> str | None:
    # Handle different schemas.
    for key in ("image_path", "path", "retrieved_image", "filename"):
        if key in item and item[key]:
            return str(item[key])
    return None


def overlay_heatmap(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float, colormap: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    hm = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    hm = hm.astype(np.float32)
    hm_min = float(hm.min())
    hm_max = float(hm.max())
    if hm_max - hm_min > 1e-8:
        hm = (hm - hm_min) / (hm_max - hm_min)
    else:
        hm = np.zeros_like(hm, dtype=np.float32)

    hm_u8 = np.uint8(np.clip(hm * 255.0, 0, 255))
    hm_color = cv2.applyColorMap(hm_u8, colormap)
    out = cv2.addWeighted(hm_color, alpha, image_bgr, 1.0 - alpha, 0)
    return out


def get_colormap(name: str) -> int:
    cmap = {
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
        "hot": cv2.COLORMAP_HOT,
        "viridis": cv2.COLORMAP_VIRIDIS,
    }
    return cmap.get(name.lower(), cv2.COLORMAP_JET)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw saliency overlays from results JSON + rank saliency .npy files")
    parser.add_argument("--results_json", required=True, type=str, help="Path to results_*.json")
    parser.add_argument("--saliency_root", required=True, type=str, help="Root folder of saliency maps: saliency_root/<query_image_id>/rank{n}_saliency.npy")
    parser.add_argument("--images_root", required=False, type=str, default=None, help="Root folder containing retrieved images (optional if JSON paths are absolute)")
    parser.add_argument("--output_dir", required=True, type=str, help="Where overlays will be written")
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap blending factor in [0,1]")
    parser.add_argument("--colormap", type=str, default="jet", choices=["jet", "turbo", "hot", "viridis"], help="OpenCV colormap")
    parser.add_argument("--rank_pattern", type=str, default="rank{rank}_saliency.npy", help="Filename pattern for rank saliency files")
    parser.add_argument("--index_by_basename", action="store_true", help="Build recursive basename index under images_root when paths in JSON are incomplete")
    args = parser.parse_args()

    results_json = Path(args.results_json)
    saliency_root = Path(args.saliency_root)
    output_dir = Path(args.output_dir)
    images_root = Path(args.images_root) if args.images_root else None

    output_dir.mkdir(parents=True, exist_ok=True)

    data = read_json(results_json)
    rows = data.get("results", [])
    if not rows:
        raise RuntimeError("No 'results' entries found in JSON.")

    basename_index = None
    if args.index_by_basename:
        if images_root is None:
            raise RuntimeError("--index_by_basename requires --images_root")
        print(f"Building basename index under: {images_root}")
        basename_index = build_basename_index(images_root)
        print(f"Indexed {len(basename_index)} unique image basenames")

    cmap = get_colormap(args.colormap)

    saved = 0
    skipped = 0

    for row in rows:
        qid = str(row.get("query_image_id") or Path(str(row.get("query_image", "unknown"))).stem)
        retrieved = row.get("retrieved", []) or []

        if not retrieved:
            skipped += 1
            continue

        query_out = output_dir / qid
        query_out.mkdir(parents=True, exist_ok=True)

        for idx, item in enumerate(retrieved, start=1):
            saliency_path = saliency_root / qid / args.rank_pattern.format(rank=idx)
            if not saliency_path.exists():
                # Optional compatibility with zero-padded rank names.
                alt = saliency_root / qid / f"rank{idx:02d}_saliency.npy"
                saliency_path = alt if alt.exists() else saliency_path

            if not saliency_path.exists():
                print(f"[WARN] Missing saliency: {saliency_path}")
                skipped += 1
                continue

            raw_img_path = extract_retrieved_image_path(item)
            img_path = find_image_path(raw_img_path, images_root, basename_index)

            if img_path is None:
                print(f"[WARN] Could not resolve image path for query={qid}, rank={idx}, raw={raw_img_path}")
                skipped += 1
                continue

            image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                print(f"[WARN] Failed to read image: {img_path}")
                skipped += 1
                continue

            heatmap = np.load(str(saliency_path))
            if heatmap.ndim == 3:
                # Handle shape like (1,H,W) or (H,W,1)
                heatmap = np.squeeze(heatmap)

            overlay = overlay_heatmap(image_bgr, heatmap, alpha=args.alpha, colormap=cmap)

            out_name = f"rank{idx}_overlay.jpg"
            out_path = query_out / out_name
            cv2.imwrite(str(out_path), overlay)
            saved += 1

    print("=" * 70)
    print(f"Done. Saved overlays: {saved}")
    print(f"Skipped entries: {skipped}")
    print(f"Output dir: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
