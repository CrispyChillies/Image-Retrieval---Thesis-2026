"""
best_saliency_showcase.py — Pick & render the best SimAtt saliency examples

Builds on evaluate_PG_RSME.py's PG-hit / RMSE localization metrics. Two
things happen in one run:

 1. Full-dataset evaluation: every bbox-annotated image in the dataset is
    used as a query, retrieves --top_k similar images, and SimAtt saliency
    is scored against each retrieved image's ground-truth bbox (Pointing
    Game hit + pixel/normalized RMSE distance). This is the dataset-level
    number you quote in the report (saved to CSV + JSON).

 2. Showcase figures: query images are ranked by their own retrieval-set
    score (hit_rate desc, then normalized distance asc) — i.e. the queries
    whose saliency maps most reliably land on the bbox across all of their
    retrieved images — and the best --num_queries of them are rendered as
    large combined figures, one row per retrieved image (rank 1..top_k),
    3 columns per row:
      [1] query image (+ its own ground-truth bbox)
      [2] retrieved image + ground-truth bbox + max-saliency point (no heatmap)
      [3] retrieved image + saliency heatmap + ground-truth bbox + max-saliency point
    Border color is green/red per pointing-game hit/miss for that retrieved
    image. A top_k=3 run therefore produces 3 rows x 3 columns per figure.

    Use --hit_filter all_miss to instead showcase only "near-miss" cases:
    query groups where every retrieved image technically misses the bbox
    (red border), ranked so the smallest saliency-to-bbox distance comes
    first — useful to argue "the map doesn't hit, but lands very close",
    which is a case worth flagging for future research.

Works for --dataset tbx11k and --dataset vindr. Explainer is always SimAtt
(the same localization metric evaluate_PG_RSME.py uses), open for reuse by
future research on other datasets/models via the same CLI shape.

Usage:
    python best_saliency_showcase.py \
      --dataset tbx11k --data_dir /path/to/tbx11k/images --csv_file test.csv \
      --model_type convnextv2_sra --model_weights checkpoints/convnextv2_sra.pth \
      --embedding_dim 512 --top_k 3 --num_queries 10 \
      --output_dir ./results/tbx11k_saliency_showcase

    python best_saliency_showcase.py \
      --dataset vindr --data_dir /path/to/vindr/images \
      --csv_file annotations_rescaled_384.csv --bbox_coord_size 384 \
      --model_type convnextv2 --model_weights checkpoints/convnextv2.pth \
      --embedding_dim 512 --top_k 3 --num_queries 10 \
      --output_dir ./results/vindr_saliency_showcase
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from evaluate_PG_RSME import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_annotation_rows,
    load_excluded_image_ids,
    item_image_id,
    load_model,
    build_simatt,
    evaluate_retrieval_model,
    select_saliency_map,
    get_item_bboxes,
    get_item_bbox_classes,
    resize_saliency_to_image,
    summarize_results,
    save_csv,
    safe_stem,
)


# ---------------------------------------------------------------------------
# Ranking: which query images best show a high-PG-hit / low-RMSE saliency map
# ---------------------------------------------------------------------------

def group_by_query(per_image):
    groups = {}
    for r in per_image:
        groups.setdefault(r["query_fname"], []).append(r)
    for entries in groups.values():
        entries.sort(key=lambda r: r["retrieval_rank"])
    return groups


def rank_queries(groups, top_k, hit_filter="any"):
    """
    Rank query groups for the showcase.

    hit_filter:
      "any"      - default. Best = highest hit_rate, then lowest normalized distance.
      "all_hit"  - only keep groups where every retrieved image is a PG hit,
                   ranked by lowest normalized distance first.
      "all_miss" - only keep groups where every retrieved image is a PG miss,
                   ranked by lowest normalized distance first (i.e. the
                   "near-miss" cases: saliency doesn't land inside the bbox,
                   but is as close as possible to it).
    """
    scored = []
    for fname, entries in groups.items():
        if len(entries) < top_k:
            continue  # incomplete retrieval set (near end of a small dataset)
        hits = [bool(e["pg_hit"]) for e in entries]
        if hit_filter == "all_hit" and not all(hits):
            continue
        if hit_filter == "all_miss" and any(hits):
            continue
        hit_rate = float(np.mean(hits))
        avg_norm_dist = float(np.mean([e["normalized_distance"] for e in entries]))
        scored.append((hit_rate, avg_norm_dist, fname, entries))

    if hit_filter == "all_miss" or hit_filter == "all_hit":
        scored.sort(key=lambda t: t[1])  # nearest-to-bbox first, hit_rate is constant within the filter
    else:
        scored.sort(key=lambda t: (-t[0], t[1]))
    return scored


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def draw_border(ax, img_size, color, lw=8):
    w, h = img_size
    ax.add_patch(patches.Rectangle((0, 0), w - 1, h - 1, linewidth=lw, edgecolor=color, facecolor="none"))


def draw_gt_bboxes(ax, item, matched_idx=None):
    bboxes = get_item_bboxes(item)
    classes = get_item_bbox_classes(item)
    for idx, bbox in enumerate(bboxes):
        is_matched = matched_idx is None or idx == matched_idx
        rect = patches.Rectangle(
            (bbox["xmin"], bbox["ymin"]), bbox["width"], bbox["height"],
            linewidth=3.5 if is_matched else 2.0,
            edgecolor="lime" if is_matched else "yellow",
            facecolor="none",
            linestyle="-" if is_matched else ":",
        )
        ax.add_patch(rect)
        if classes[idx]:
            ax.text(
                bbox["xmin"], max(0, bbox["ymin"] - 4), classes[idx],
                color="lime" if is_matched else "yellow", fontsize=10, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=1.5),
            )


def plot_query_showcase(model_type, query_item, entries, saliencies, out_path, dpi):
    """
    One row per retrieved image (rank 1..k), 3 columns:
      [0] query image (+ its own GT bbox)
      [1] retrieved image + GT bbox + max-saliency point (no heatmap)
      [2] retrieved image + saliency heatmap + GT bbox + max-saliency point
    """
    k = len(entries)
    fig, axes = plt.subplots(k, 3, figsize=(21, 6.5 * k), squeeze=False)

    q_img = Image.open(query_item["image_path"]).convert("RGB")

    for i, (result, saliency) in enumerate(zip(entries, saliencies)):
        retrieved_item = result["_retrieved_item"]
        r_img = Image.open(retrieved_item["image_path"]).convert("RGB")
        overlay = resize_saliency_to_image(saliency, r_img.size)

        hit = result["pg_hit"]
        marker = "✓ HIT" if hit else "✗ MISS"
        border_color = "#2ecc40" if hit else "#ff4136"

        # --- Column 0: query image ---
        ax_q = axes[i][0]
        ax_q.imshow(q_img)
        draw_gt_bboxes(ax_q, query_item)
        draw_border(ax_q, q_img.size, "#0074d9")
        ax_q.set_title(
            "Query image" if i == 0 else "", fontsize=15, fontweight="bold",
        )
        ax_q.axis("off")

        # --- Column 1: retrieved image + bbox + peak (no heatmap) ---
        ax_r = axes[i][1]
        ax_r.imshow(r_img)
        draw_gt_bboxes(ax_r, retrieved_item, matched_idx=result["matched_bbox_index"])
        ax_r.scatter([result["peak_x"]], [result["peak_y"]], s=140, c="red", marker="x", linewidths=3)
        draw_border(ax_r, r_img.size, border_color)
        ax_r.set_title(
            f"Retrieved image — Top-{result['retrieval_rank']}  {marker}\n"
            f"sim={result['query_retrieved_similarity']:.3f}",
            fontsize=13, color=border_color, fontweight="bold",
        )
        ax_r.axis("off")

        # --- Column 2: retrieved image + saliency heatmap + bbox + peak ---
        ax_s = axes[i][2]
        ax_s.imshow(r_img)
        ax_s.imshow(overlay, cmap="jet", alpha=0.5)
        draw_gt_bboxes(ax_s, retrieved_item, matched_idx=result["matched_bbox_index"])
        ax_s.scatter([result["peak_x"]], [result["peak_y"]], s=140, c="red", marker="x", linewidths=3)
        draw_border(ax_s, r_img.size, border_color)
        ax_s.set_title(
            f"Saliency heatmap + lesion box\ndist={result['distance_px']:.1f}px",
            fontsize=13, color=border_color, fontweight="bold",
        )
        ax_s.axis("off")

    hits = sum(1 for r in entries if r["pg_hit"])
    mean_dist = float(np.mean([r["distance_px"] for r in entries]))
    fig.suptitle(
        f"[{model_type}]  Query: {query_item['fname']}  —  {hits}/{k} PG hits  "
        f"(mean dist={mean_dist:.1f}px)",
        fontsize=17, fontweight="bold",
    )

    legend_handles = [
        patches.Patch(edgecolor="#0074d9", facecolor="none", linewidth=3, label="Query"),
        patches.Patch(edgecolor="#2ecc40", facecolor="none", linewidth=3, label="Pointing-game HIT"),
        patches.Patch(edgecolor="#ff4136", facecolor="none", linewidth=3, label="Pointing-game MISS"),
        patches.Patch(edgecolor="lime", facecolor="none", label="Ground-truth bbox (matched)"),
        patches.Patch(edgecolor="yellow", facecolor="none", linestyle=":", label="Ground-truth bbox (other)"),
        plt.Line2D([0], [0], color="red", marker="x", linestyle="None", markersize=10, label="Max saliency point"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=12, framealpha=0.9)

    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_transform(img_size):
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def run(args):
    if args.top_k < 3:
        print(f"[WARN] --top_k={args.top_k} < 3; showcase figures will have fewer than 3 retrieved panels.")

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rows = load_annotation_rows(
        dataset=args.dataset, csv_file=args.csv_file, data_dir=args.data_dir,
        limit=args.limit, image_ext=args.image_ext, vindr_classes=args.vindr_classes,
        bbox_coord_size=args.bbox_coord_size,
    )
    if not rows:
        raise RuntimeError("No valid bbox rows found. Check --dataset/--csv_file/--data_dir.")
    print(f"Loaded {len(rows)} {args.dataset} bbox-annotated images from {args.csv_file}")

    if args.exclude_labels_csv:
        excluded_ids = load_excluded_image_ids(args.exclude_labels_csv, args.exclude_label)
        before = len(rows)
        rows = [r for r in rows if item_image_id(r) not in excluded_ids]
        print(f"Excluded {before - len(rows)} images labeled '{args.exclude_label}'; {len(rows)} remain")

    rows_by_fname = {r["fname"]: r for r in rows}
    transform = build_transform(args.img_size)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_pair_json_path = output_dir / f"{args.dataset}_{args.model_type}_per_pair.json"

    print(f"Loading {args.model_type} from {args.model_weights} ...")
    model = load_model(
        args.model_type, args.model_weights, device,
        embedding_dim=args.embedding_dim, sra_num_heads=args.sra_num_heads, sra_lam=args.sra_lam,
    )
    explainer = build_simatt(model, args.model_type).to(device)

    reuse_path = args.reuse_eval_json or (per_pair_json_path if per_pair_json_path.exists() and args.auto_reuse else None)
    if reuse_path:
        print(f"\nReusing cached per-pair evaluation from {reuse_path} (skipping the full SimAtt sweep) ...")
        with open(reuse_path, "r", encoding="utf-8") as f:
            per_image = json.load(f)
    else:
        print(f"\nEvaluating SimAtt retrieval PG-hit/RMSE across all {len(rows)} images (top_k={args.top_k}) ...")
        per_image = evaluate_retrieval_model(
            model_name=args.model_type, model=model, rows=rows, transform=transform, device=device,
            top_k=args.top_k, embedding_batch_size=args.batch_size, saliency_index=-1,
            visualization_dir=None, max_visualizations=0,
        )

        summary = summarize_results(per_image)
        print("\nDataset-level SimAtt localization summary:")
        print(f"  Samples:      {summary['num_samples']}")
        print(f"  PG hit rate:  {summary['pg_hit_rate']:.4f}")
        print(f"  RMSE (px):    {summary['rmse_distance_px']:.2f}")
        print(f"  RMSE (norm):  {summary['rmse_normalized_distance']:.4f}")

        save_csv(output_dir / f"{args.dataset}_{args.model_type}_pg_rmse_per_pair.csv", per_image)
        with open(output_dir / f"{args.dataset}_{args.model_type}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(per_pair_json_path, "w", encoding="utf-8") as f:
            json.dump(per_image, f)
        print(f"  Saved dataset-level CSV/JSON to: {output_dir}")
        print(f"  Cached full per-pair results to {per_pair_json_path} "
              f"-> pass --reuse_eval_json {per_pair_json_path} (or --auto_reuse) next time to skip re-evaluation.")

    groups = group_by_query(per_image)
    ranked = rank_queries(groups, top_k=args.top_k, hit_filter=args.hit_filter)
    print(f"\n{len(ranked)} / {len(groups)} query groups match --hit_filter={args.hit_filter} "
          f"with a full top-{args.top_k} retrieval set.")

    selected = ranked[args.offset: args.offset + args.num_queries]
    if not selected:
        print(f"[WARN] No query groups left at offset={args.offset} "
              f"(only {len(ranked)} ranked groups available).")
        return

    print(f"\nRendering {len(selected)} showcase figure(s) (rank {args.offset + 1}-{args.offset + len(selected)}) "
          f"to {output_dir} ...")
    for rank_pos, (hit_rate, avg_norm_dist, fname, entries) in enumerate(selected, start=args.offset + 1):
        query_item = rows_by_fname[fname]
        q_img = Image.open(query_item["image_path"]).convert("RGB")
        q_tensor = transform(q_img).unsqueeze(0).to(device)

        saliencies = []
        for e in entries:
            retrieved_item = rows_by_fname[e["retrieved_fname"]]
            e["_retrieved_item"] = retrieved_item
            r_img = Image.open(retrieved_item["image_path"]).convert("RGB")
            r_tensor = transform(r_img).unsqueeze(0).to(device)
            model.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(True):
                sal_tensor = explainer(q_tensor, r_tensor)
            saliencies.append(select_saliency_map(sal_tensor, saliency_index=-1))

        fname_stem = Path(fname).stem
        out_name = (
            f"{args.dataset}_{args.model_type}_showcase{rank_pos:02d}"
            f"_hit{int(round(hit_rate * 100))}_normdist{avg_norm_dist:.3f}_{safe_stem(fname_stem)}.png"
        )
        plot_query_showcase(
            model_type=args.model_type,
            query_item=query_item,
            entries=entries,
            saliencies=saliencies,
            out_path=output_dir / out_name,
            dpi=args.dpi,
        )

    print(f"\nDone. Dataset-level metrics + {len(selected)} showcase figures saved to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pick and render the best SimAtt saliency localization examples "
                     "(high PG-hit, low RMSE) for tbx11k / vindr."
    )
    parser.add_argument("--dataset", required=True, choices=["tbx11k", "vindr"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--csv_file", default="test.csv")
    parser.add_argument("--image_ext", default=".png")
    parser.add_argument("--vindr_classes", default=None,
                         help="Optional comma-separated VinDR class filter, e.g. 'Nodule/Mass,ILD'.")
    parser.add_argument("--bbox_coord_size", type=int, default=None,
                         help="Set to 384 for annotations_rescaled_384.csv-style fixed-coordinate bboxes.")
    parser.add_argument("--exclude_labels_csv", default=None)
    parser.add_argument("--exclude_label", default="No finding")
    parser.add_argument("--hit_filter", choices=["any", "all_hit", "all_miss"], default="any",
                         help="any (default) = rank by hit rate desc, then distance asc. "
                              "all_hit = only groups where every retrieved image is a PG hit. "
                              "all_miss = only groups where every retrieved image is a PG miss, "
                              "ranked nearest-miss-first (smallest normalized distance) — use this to "
                              "showcase saliency maps that land close to the bbox without a technical hit.")
    parser.add_argument("--model_type", default="convnextv2_sra",
                         choices=["convnextv2", "convnextv2_sra", "densenet121", "resnet50"])
    parser.add_argument("--model_weights", required=True)
    parser.add_argument("--embedding_dim", type=int, default=None)
    parser.add_argument("--sra_num_heads", type=int, default=8)
    parser.add_argument("--sra_lam", type=float, default=0.1)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--top_k", type=int, default=3,
                         help="Retrieved images per query (>=3 recommended; default 3)")
    parser.add_argument("--num_queries", type=int, default=10,
                         help="Number of best-scoring queries to render as showcase figures")
    parser.add_argument("--offset", type=int, default=0,
                         help="Skip this many top-ranked queries before rendering the next --num_queries "
                              "(e.g. --offset 10 to get ranks 11-20 after already rendering the top 10)")
    parser.add_argument("--reuse_eval_json", default=None,
                         help="Path to a previously saved '<dataset>_<model_type>_per_pair.json' "
                              "(from an earlier run) to skip the full SimAtt sweep entirely and just "
                              "re-rank/re-render showcase figures.")
    parser.add_argument("--auto_reuse", action="store_true",
                         help="If '<output_dir>/<dataset>_<model_type>_per_pair.json' from a prior run "
                              "already exists, reuse it automatically instead of --reuse_eval_json.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on dataset size for a quick run")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for showcase figures (report-quality export)")
    parser.add_argument("--output_dir", default="./results/saliency_showcase")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
