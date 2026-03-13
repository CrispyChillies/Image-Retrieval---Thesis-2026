import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def read_json(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_image_path(
    raw_path: str | None,
    roots: list[Path] | None,
    basename_index: dict[str, Path] | None,
) -> Path | None:
    if not raw_path:
        return None

    candidate = Path(raw_path)

    # 1) Absolute path as-is.
    if candidate.is_absolute() and candidate.exists():
        return candidate

    # 2) Relative path under provided roots.
    if roots:
        for root in roots:
            rel_candidate = root / candidate
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


def build_query_keys(row: dict) -> list[str]:
    keys: list[str] = []

    qid = row.get("query_image_id")
    if qid:
        keys.append(str(qid))

    qimg = str(row.get("query_image", "")).strip()
    if qimg:
        qname = Path(qimg).name
        qstem = Path(qimg).stem
        keys.append(qstem)
        keys.append(qname)

    # Deduplicate while preserving order.
    dedup: list[str] = []
    seen = set()
    for k in keys:
        if k not in seen:
            seen.add(k)
            dedup.append(k)
    return dedup


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw saliency overlays from results JSON + rank saliency .npy files")
    parser.add_argument("--results_json", required=True, type=str, help="Path to results_*.json")
    parser.add_argument("--saliency_root", required=True, type=str, help="Root folder of saliency maps: saliency_root/<query_image_id>/rank{n}_saliency.npy")
    parser.add_argument("--query_images_root", required=False, type=str, default=None, help="Root folder containing query images")
    parser.add_argument("--retrieved_images_root", required=False, type=str, default=None, help="Root folder containing retrieved images")
    parser.add_argument("--images_root", required=False, type=str, default=None, help="Backward-compatible alias: used for both query and retrieved roots")
    parser.add_argument("--output_dir", required=True, type=str, help="Where overlays will be written")
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap blending factor in [0,1]")
    parser.add_argument("--colormap", type=str, default="jet", choices=["jet", "turbo", "hot", "viridis"], help="OpenCV colormap")
    parser.add_argument("--rank_pattern", type=str, default="rank{rank}_saliency.npy", help="Filename pattern for rank saliency files")
    parser.add_argument("--index_by_basename", action="store_true", help="Build recursive basename index under query/retrieved roots when paths in JSON are incomplete")
    parser.add_argument("--save_query_preview", action="store_true", help="Save query image preview into each query output folder")
    args = parser.parse_args()

    results_json = Path(args.results_json)
    saliency_root = Path(args.saliency_root)
    output_dir = Path(args.output_dir)

    query_images_root = Path(args.query_images_root) if args.query_images_root else None
    retrieved_images_root = Path(args.retrieved_images_root) if args.retrieved_images_root else None

    # Backward compatibility: if only --images_root is provided, use it for both.
    if args.images_root:
        alias_root = Path(args.images_root)
        if query_images_root is None:
            query_images_root = alias_root
        if retrieved_images_root is None:
            retrieved_images_root = alias_root

    output_dir.mkdir(parents=True, exist_ok=True)

    data = read_json(results_json)
    rows = data.get("results", [])
    if not rows:
        raise RuntimeError("No 'results' entries found in JSON.")

    basename_index = None
    if args.index_by_basename:
        roots_to_index = []
        if query_images_root is not None:
            roots_to_index.append(query_images_root)
        if retrieved_images_root is not None and retrieved_images_root != query_images_root:
            roots_to_index.append(retrieved_images_root)
        if not roots_to_index:
            raise RuntimeError("--index_by_basename requires --query_images_root and/or --retrieved_images_root (or --images_root)")

        basename_index = {}
        for root in roots_to_index:
            print(f"Building basename index under: {root}")
            partial = build_basename_index(root)
            for k, v in partial.items():
                basename_index.setdefault(k, v)
        print(f"Indexed {len(basename_index)} unique image basenames")

    cmap = get_colormap(args.colormap)

    saved = 0
    skipped = 0

    for row in rows:
        qid = str(row.get("query_image_id") or Path(str(row.get("query_image", "unknown"))).stem)
        query_image_raw = str(row.get("query_image", ""))
        retrieved = row.get("retrieved", []) or []
        query_keys = build_query_keys(row)

        if not retrieved:
            skipped += 1
            continue

        query_out = output_dir / qid
        query_out.mkdir(parents=True, exist_ok=True)

        if args.save_query_preview and query_image_raw:
            query_path = find_image_path(
                query_image_raw,
                [query_images_root] if query_images_root is not None else None,
                basename_index,
            )
            if query_path is not None:
                q_bgr = cv2.imread(str(query_path), cv2.IMREAD_COLOR)
                if q_bgr is not None:
                    cv2.imwrite(str(query_out / "query.jpg"), q_bgr)

        for idx, item in enumerate(retrieved, start=1):
            saliency_path = None
            for qkey in query_keys:
                candidate = saliency_root / qkey / args.rank_pattern.format(rank=idx)
                if candidate.exists():
                    saliency_path = candidate
                    break

                # Optional compatibility with zero-padded rank names.
                alt = saliency_root / qkey / f"rank{idx:02d}_saliency.npy"
                if alt.exists():
                    saliency_path = alt
                    break

            if saliency_path is None:
                print(f"[WARN] Missing saliency for query keys={query_keys}, rank={idx}")
                skipped += 1
                continue

            raw_img_path = extract_retrieved_image_path(item)
            img_path = find_image_path(
                raw_img_path,
                [retrieved_images_root] if retrieved_images_root is not None else None,
                basename_index,
            )

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
