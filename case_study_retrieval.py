"""
case_study_retrieval.py — Same-query, 3-case retrieval figures

For each of --num_queries query images, this renders 3 retrieval-grid figures
using the *exact same query image*, differing only in how many of the top-k
retrieved slots are deliberately filled from the wrong class:

  Case 1 (0 wrong): top-k retrieved images, all real, all same label as the
                     query (ranked purely by cosine similarity within the
                     correct-label pool)
  Case 2 (1 wrong): same as Case 1, but the weakest correct slot is replaced
                     by the single most-similar different-label image
  Case 3 (2 wrong): same as Case 1, but the two weakest correct slots are
                     replaced by the two most-similar different-label images

Every image shown is a genuine gallery image ranked by real embedding
similarity — nothing is synthetic, only *which pool* (same-label vs.
different-label) each slot is drawn from is controlled. Rendering 10 queries
x 3 cases lets you compare how a 5/5, 4/5, and 3/5-correct figure reads for
the same query, and pick the clearest one for the thesis.

Visual conventions (reused from inference.py):
  - Query border: blue
  - Correct retrieval border: green
  - Wrong retrieval border: red
  - A blank spacer column widens the gap between the query panel and the
    retrieved-image panels.

Usage:
    python case_study_retrieval.py \
      --model_type convnextv2 --dataset isic \
      --data_dir /path/to/isic/images \
      --image_list /path/to/ISIC-2017_Test_v2_Part3_GroundTruth_balanced.csv \
      --model_weights /path/to/convnextv2.pth \
      --output_dir ./results/isic_case_studies \
      --num_queries 10 --top_k 5
"""

import argparse
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from inference import (
    MODEL_INPUT_SIZES,
    DATASET_LABEL_CODES,
    build_model,
    build_transform,
    load_dataset,
    filter_sick_only,
    extract_embeddings,
    label_caption,
    labels_match,
    load_display_image,
)


# ---------------------------------------------------------------------------
# Query selection — need >= top_k same-label and >= 2 different-label
# neighbours (excluding itself) to be able to build all 3 cases.
# ---------------------------------------------------------------------------

def _match_counts(labels):
    """Vectorised (n_same[i], n_diff[i]) neighbour counts, excluding self."""
    n = len(labels)
    if isinstance(labels[0], (list, np.ndarray)):
        L = np.stack([np.asarray(l) for l in labels])  # [N, C] multi-hot
        match = (L @ L.T) > 0
    else:
        arr = np.asarray([int(l) for l in labels])
        match = arr[:, None] == arr[None, :]
    np.fill_diagonal(match, False)
    n_same = match.sum(axis=1)
    n_diff = n - 1 - n_same
    return n_same, n_diff


def select_query_indices(labels, k, num_queries):
    n_same, n_diff = _match_counts(labels)
    valid = [i for i in range(len(labels)) if n_same[i] >= k and n_diff[i] >= 2]
    return valid[:num_queries]


# ---------------------------------------------------------------------------
# Retrieval with a controlled number of wrong-label slots
# ---------------------------------------------------------------------------

def retrieve_case(embeddings, labels, query_idx, k, num_wrong):
    """
    Rank the gallery by cosine similarity to `query_idx`, keep the top
    (k - num_wrong) same-label images plus the top `num_wrong`
    different-label images, then re-sort everything by similarity so
    "Top-1..Top-k" still reflects descending similarity.

    Returns (chosen_indices, scores) or None if this query doesn't have
    enough same/different-label neighbours to fill the case.
    """
    q_emb = embeddings[query_idx]
    sims = embeddings @ q_emb
    sims[query_idx] = -2.0
    order = sims.argsort(descending=True).tolist()

    q_label = labels[query_idx]
    correct_pool = [i for i in order if labels_match(q_label, labels[i])]
    wrong_pool = [i for i in order if not labels_match(q_label, labels[i])]

    num_correct = k - num_wrong
    if len(correct_pool) < num_correct or len(wrong_pool) < num_wrong:
        return None

    chosen = correct_pool[:num_correct] + wrong_pool[:num_wrong]
    chosen.sort(key=lambda i: sims[i].item(), reverse=True)
    scores = [sims[i].item() for i in chosen]
    return chosen, scores


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_case_figure(
    query_path, query_label, ret_paths, ret_labels, ret_scores,
    label_names, out_path, dataset, query_rank, case_num, num_wrong,
    label_codes=None,
):
    k = len(ret_paths)
    n_correct = k - num_wrong
    width_ratios = [1, 0.35] + [1] * k  # extra-wide gap after the query panel
    fig, axes = plt.subplots(
        1, k + 2, figsize=(4 * (k + 1) + 1, 5.5),
        gridspec_kw={"width_ratios": width_ratios},
    )
    fig.suptitle(
        f"[{dataset.upper()}]  Query #{query_rank + 1} — Case {case_num}: "
        f"{n_correct}/{k} correct retrievals",
        fontsize=13, fontweight="bold",
    )

    axes[1].axis("off")  # spacer column

    # --- Query ---
    ax = axes[0]
    q_img = load_display_image(query_path)
    ax.imshow(q_img)
    ax.set_title(
        f"QUERY\n• {label_caption(query_label, label_names, label_codes)}",
        fontsize=9, fontweight="bold",
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#0074d9")
        spine.set_linewidth(5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.text(
        0.5, 0.02, label_caption(query_label, label_names, label_codes),
        transform=ax.transAxes, fontsize=8, color="white", fontweight="bold",
        ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0074d9", alpha=0.8),
    )

    # --- Retrieved ---
    for i, (rp, rl, rs) in enumerate(zip(ret_paths, ret_labels, ret_scores)):
        match = labels_match(query_label, rl)
        marker = "✓" if match else "✗"
        border_color = "#2ecc40" if match else "#ff4136"

        ax = axes[i + 2]
        r_img = load_display_image(rp)
        ax.imshow(r_img)
        ax.set_title(
            f"Top-{i + 1}  {marker}\n• {label_caption(rl, label_names, label_codes)}\nsim={rs:.3f}",
            fontsize=9, color=border_color, fontweight="bold",
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(5)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.text(
            0.5, 0.02, f"{marker} {label_caption(rl, label_names, label_codes)}",
            transform=ax.transAxes, fontsize=8, color="white", fontweight="bold",
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=border_color, alpha=0.85),
        )

    legend_patches = [
        mpatches.Patch(color="#0074d9", label="Query image"),
        mpatches.Patch(color="#2ecc40", label="Correct retrieval (same class)"),
        mpatches.Patch(color="#ff4136", label="Incorrect retrieval (different class)"),
    ]
    fig.legend(
        handles=legend_patches, loc="lower center", ncol=len(legend_patches),
        fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    img_size = MODEL_INPUT_SIZES[args.model_type]
    print(f"Loading {args.model_type} from {args.model_weights} (input={img_size}x{img_size}) ...")
    model = build_model(args.model_type, args.model_weights, args.embedding_dim, device)
    transform = build_transform(img_size)

    print(f"Loading dataset '{args.dataset}' ...")
    image_paths, labels, label_names, bboxes = load_dataset(
        args.dataset, args.data_dir, args.image_list, transform
    )
    if args.sick_only:
        image_paths, labels, bboxes = filter_sick_only(args.dataset, image_paths, labels, bboxes)
    print(f"  {len(image_paths)} images found.")

    print("Extracting embeddings ...")
    embeddings = extract_embeddings(model, image_paths, transform, device, batch_size=args.batch_size)

    label_codes = DATASET_LABEL_CODES.get(args.dataset)

    query_indices = select_query_indices(labels, k=args.top_k, num_queries=args.num_queries)
    print(f"Selected {len(query_indices)} query image(s) with enough same/different-label neighbours "
          f"(need >= {args.top_k} same-label, >= 2 different-label).")

    os.makedirs(args.output_dir, exist_ok=True)

    for q_rank, q_idx in enumerate(query_indices):
        q_path = image_paths[q_idx]
        q_label = labels[q_idx]
        img_stem = os.path.splitext(os.path.basename(q_path))[0]
        print(f"\nQuery {q_rank + 1}/{len(query_indices)}: {os.path.basename(q_path)}")

        for case_num, num_wrong in enumerate([0, 1, 2], start=1):
            result = retrieve_case(embeddings, labels, q_idx, k=args.top_k, num_wrong=num_wrong)
            if result is None:
                print(f"  [SKIP] Case {case_num} ({num_wrong} wrong): not enough neighbours.")
                continue
            chosen, scores = result
            ret_paths = [image_paths[i] for i in chosen]
            ret_labels = [labels[i] for i in chosen]

            base = f"{args.dataset}_{args.model_type}_q{q_rank + 1:02d}_{img_stem}_case{case_num}"
            out_path = os.path.join(args.output_dir, f"{base}.png")

            plot_case_figure(
                query_path=q_path,
                query_label=q_label,
                ret_paths=ret_paths,
                ret_labels=ret_labels,
                ret_scores=scores,
                label_names=label_names,
                out_path=out_path,
                dataset=args.dataset,
                query_rank=q_rank,
                case_num=case_num,
                num_wrong=num_wrong,
                label_codes=label_codes,
            )

    print(f"\nDone. All case-study figures saved to: {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render 3 same-query retrieval cases (0/1/2 wrong) for N query images"
    )
    parser.add_argument(
        "--model_type", default="convnextv2",
        choices=["convnextv2", "convnextv2_sra", "densenet121", "resnet50"],
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=["covid", "tbx11k", "vindr", "isic"],
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--image_list", required=True)
    parser.add_argument("--model_weights", required=True)
    parser.add_argument("--output_dir", default="./results/case_studies")
    parser.add_argument("--embedding_dim", type=int, default=None)
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of retrieved images per figure (default: 5)",
    )
    parser.add_argument(
        "--num_queries", type=int, default=10,
        help="Number of distinct query images to run the 3 cases on (default: 10)",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sick_only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
