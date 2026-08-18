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

Note: this embeds the already-saved PNG files directly via IPython.display,
rather than re-plotting them with matplotlib. That avoids a common Kaggle
gotcha where the active matplotlib backend (e.g. the ipympl/widget backend)
renders figures as a plain "Figure(WxH)" placeholder instead of an image.
"""

import glob
import os

from IPython.display import display, HTML, Image as IPyImage


def show_results(output_dir, pattern="*_retrieval.png", max_queries=10):
    """
    Display retrieval + saliency (XAI) figures produced by inference.py.

    For each "<base>_retrieval.png" found in `output_dir` (matching `pattern`),
    also looks for the matching "<base>_saliency.png" and displays both,
    inline in the notebook.

    Args:
        output_dir: directory containing inference.py PNG outputs.
        pattern: glob pattern (relative to output_dir) selecting retrieval
            figures, e.g. "covid_convnextv2_simcam_*_retrieval.png" to filter
            to one dataset/model/explainer combination.
        max_queries: max number of query results to render (<=0 = no limit).
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

        display(HTML(f"<b>{base_name} — retrieval</b>"))
        display(IPyImage(filename=ret_path))

        if os.path.isfile(sal_path):
            display(HTML(f"<b>{base_name} — saliency (XAI)</b>"))
            display(IPyImage(filename=sal_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="List (and optionally open) inference.py result PNGs")
    parser.add_argument("output_dir", help="Directory with inference.py PNG outputs")
    parser.add_argument("--pattern", default="*_retrieval.png", help="Glob pattern for retrieval figures")
    parser.add_argument("--max_queries", type=int, default=10, help="Max results to list/open (<=0 = all)")
    parser.add_argument("--open", action="store_true", help="Open each matching PNG in the system image viewer")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.output_dir, args.pattern)))
    if args.max_queries > 0:
        paths = paths[: args.max_queries]
    if not paths:
        print(f"[WARN] No files matching '{args.pattern}' found in {args.output_dir}")
    for p in paths:
        print(p)
        if args.open:
            from PIL import Image
            Image.open(p).show()

