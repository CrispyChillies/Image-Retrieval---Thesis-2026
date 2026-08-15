"""
display_results.py — Notebook viewer for inference.py outputs (Kaggle/Jupyter)

Paste into a Kaggle notebook cell:

    from display_results import show_results
    show_results("/kaggle/working/results/covid_convnextv2_simcam")

Filter to a specific run (dataset/model/explainer) and cap how many queries
are rendered in one cell (a full-dataset inference.py run can produce
thousands of PNGs):

    show_results(
        "/kaggle/working/results",
        pattern="vindr_convnextv2_sra_simatt_*_retrieval.png",
        max_queries=5,
    )
"""

import glob
import os

import matplotlib.pyplot as plt
from PIL import Image


def show_results(output_dir, pattern="*_retrieval.png", max_queries=10, figsize_scale=3):
    """
    Display retrieval + saliency (XAI) figure pairs produced by inference.py.

    For each "<base>_retrieval.png" found in `output_dir` (matching `pattern`),
    also looks for the matching "<base>_saliency.png" and displays both,
    stacked vertically, inline in the notebook.

    Args:
        output_dir: directory containing inference.py PNG outputs.
        pattern: glob pattern (relative to output_dir) selecting retrieval
            figures, e.g. "covid_convnextv2_simcam_*_retrieval.png" to filter
            to one dataset/model/explainer combination.
        max_queries: max number of query results to render (<=0 = no limit).
        figsize_scale: inches of height per stacked row.
    """
    retrieval_paths = sorted(glob.glob(os.path.join(output_dir, pattern)))
    if not retrieval_paths:
        print(f"[WARN] No files matching '{pattern}' found in {output_dir}")
        return

    if max_queries > 0:
        retrieval_paths = retrieval_paths[:max_queries]

    print(f"Displaying {len(retrieval_paths)} result(s) from {output_dir}")

    for ret_path in retrieval_paths:
        sal_path = ret_path.replace("_retrieval.png", "_saliency.png")
        base_name = os.path.basename(ret_path).replace("_retrieval.png", "")
        has_sal = os.path.isfile(sal_path)
        n_rows = 2 if has_sal else 1

        fig, axes = plt.subplots(n_rows, 1, figsize=(14, figsize_scale * n_rows))
        axes = [axes] if n_rows == 1 else axes

        axes[0].imshow(Image.open(ret_path))
        axes[0].axis("off")
        axes[0].set_title(f"{base_name} — retrieval", fontsize=10)

        if has_sal:
            axes[1].imshow(Image.open(sal_path))
            axes[1].axis("off")
            axes[1].set_title(f"{base_name} — saliency (XAI)", fontsize=10)
        else:
            print(f"  [WARN] No saliency figure found for {base_name}")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Display inference.py results in a notebook/CLI")
    parser.add_argument("output_dir", help="Directory with inference.py PNG outputs")
    parser.add_argument("--pattern", default="*_retrieval.png", help="Glob pattern for retrieval figures")
    parser.add_argument("--max_queries", type=int, default=10, help="Max results to display (<=0 = all)")
    args = parser.parse_args()
    show_results(args.output_dir, pattern=args.pattern, max_queries=args.max_queries)
