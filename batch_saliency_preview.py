"""
Batch saliency preview: for N query images, retrieve top-K images and generate
SimAtt saliency visualizations so you can browse and pick the best pair.

Usage example:
    python batch_saliency_preview.py \
      --results ./results/isic_densenet121_embed_256_seed_0_epoch_20_ckpt.npz \
      --csv ISIC-2017_Test_v2_Part3_GroundTruth_balanced.csv \
      --image_dir /kaggle/input/isic-2017/ISIC-2017_Test_v2_Data/ISIC-2017_Test_v2_Data \
      --model_weights /kaggle/input/isic-backbone-resnet50-densenet121/pytorch/default/1/isic_densenet121_embed_256_seed_0_epoch_20_ckpt.pth \
      --model_type densenet121 \
      --embedding_dim 256 \
      --num_queries 10 \
      --top_k 5 \
      --output_dir ./saliency_preview
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import ConvNeXtV2, DenseNet121, ResNet50
from explanations import SimAtt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(model_type, model_weights, embedding_dim, device):
    if model_type == "densenet121":
        model = DenseNet121(embedding_dim=embedding_dim)
    elif model_type == "resnet50":
        model = ResNet50(embedding_dim=embedding_dim)
    elif model_type == "convnextv2":
        model = ConvNeXtV2(embedding_dim=embedding_dim)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    checkpoint = torch.load(model_weights, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    return model


def build_simatt(model, model_type):
    if model_type == "densenet121":
        target_layer = model.densenet121[0]
    elif model_type == "resnet50":
        target_layer = model.resnet50[7]
    elif model_type == "convnextv2":
        target_layer = model.convnext.stages[-1]
    else:
        raise ValueError(f"Unsupported model_type for SimAtt: {model_type}")
    explainer = SimAtt(model, target_layer, target_layers=None)
    explainer.eval()
    return explainer


def get_transform(model_type):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img_size = 384 if model_type == "convnextv2" else 224
    resize_size = 432 if img_size == 384 else 256
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])


def normalize_saliency(sal):
    sal = np.nan_to_num(sal.astype(np.float32))
    lo, hi = sal.min(), sal.max()
    return (sal - lo) / (hi - lo) if hi > lo else np.zeros_like(sal)


def generate_saliency(explainer, query_tensor, retrieved_tensor, device):
    model_zero = next(iter(explainer.parameters() if hasattr(explainer, "parameters") else []), None)
    with torch.set_grad_enabled(True):
        sal = explainer(query_tensor.to(device), retrieved_tensor.to(device))
    sal = sal.detach().squeeze().cpu().numpy()
    if sal.ndim == 3:
        sal = sal[-1]
    return normalize_saliency(sal)


def save_grid(query_img, retrieved_img, saliency, query_name, ret_name,
              similarity, rank, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(query_img)
    axes[0].set_title(f"Query\n{query_name}", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(retrieved_img)
    axes[1].set_title(f"Retrieved (rank {rank})\n{ret_name}\nsim={similarity:.4f}", fontsize=9)
    axes[1].axis("off")

    sal_resized = np.array(
        Image.fromarray((saliency * 255).astype(np.uint8)).resize(
            retrieved_img.size, Image.BILINEAR
        )
    ) / 255.0
    axes[2].imshow(retrieved_img)
    axes[2].imshow(sal_resized, cmap="jet", alpha=0.5)
    axes[2].set_title("SimAtt saliency overlay", fontsize=9)
    axes[2].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="NPZ from test.py with dists + embeds")
    parser.add_argument("--csv", required=True, help="Test CSV with image_id column")
    parser.add_argument("--id_col", default="image_id", help="Column name for image IDs")
    parser.add_argument("--image_dir", required=True, help="Directory containing test images")
    parser.add_argument("--image_ext", default=".jpg", help="Image file extension")
    parser.add_argument("--model_weights", required=True)
    parser.add_argument("--model_type", default="densenet121",
                        choices=["densenet121", "resnet50", "convnextv2"])
    parser.add_argument("--embedding_dim", type=int, default=None)
    parser.add_argument("--num_queries", type=int, default=10,
                        help="Number of query images to preview")
    parser.add_argument("--query_indices", type=str, default=None,
                        help="Comma-separated specific query indices, e.g. 0,3,7 (overrides --num_queries)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top retrieved images per query")
    parser.add_argument("--output_dir", default="./saliency_preview")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load distances
    print(f"Loading results from {args.results}")
    data = np.load(args.results)
    dists = data["dists"].copy()           # lower = more similar (test.py saves -cosine)
    np.fill_diagonal(dists, np.nan)

    # Load image ID mapping
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from CSV. Columns: {list(df.columns)}")
    image_ids = df[args.id_col].tolist()
    assert len(image_ids) == dists.shape[0], (
        f"CSV rows ({len(image_ids)}) != dists dimension ({dists.shape[0]})"
    )

    # Determine which queries to evaluate
    if args.query_indices:
        query_idxs = [int(x) for x in args.query_indices.split(",")]
    else:
        query_idxs = list(range(min(args.num_queries, len(image_ids))))

    # Load model + explainer once
    print(f"\nLoading {args.model_type} model...")
    model = load_model(args.model_type, args.model_weights, args.embedding_dim, device)
    explainer = build_simatt(model, args.model_type).to(device)
    transform = get_transform(args.model_type)

    output_dir = Path(args.output_dir)
    summary_rows = []

    for q_idx in query_idxs:
        query_id = image_ids[q_idx]
        query_path = Path(args.image_dir) / f"{query_id}{args.image_ext}"
        if not query_path.exists():
            print(f"  [skip] Query image not found: {query_path}")
            continue

        # Top-K retrieved
        row_dists = dists[q_idx].copy()
        top_k_idxs = np.argsort(row_dists)[:args.top_k]

        print(f"\nQuery [{q_idx}] {query_id}  →  top-{args.top_k} retrieved:")

        query_img = Image.open(query_path).convert("RGB")
        query_tensor = transform(query_img).unsqueeze(0).to(device)

        for rank, r_idx in enumerate(top_k_idxs, start=1):
            ret_id = image_ids[r_idx]
            sim = float(-row_dists[r_idx])   # stored as negative cosine
            ret_path = Path(args.image_dir) / f"{ret_id}{args.image_ext}"
            if not ret_path.exists():
                print(f"  rank {rank}: {ret_id}  [image missing, skip]")
                continue

            print(f"  rank {rank}: {ret_id}  sim={sim:.4f}")

            ret_img = Image.open(ret_path).convert("RGB")
            ret_tensor = transform(ret_img).unsqueeze(0).to(device)

            # Generate saliency
            model.zero_grad(set_to_none=True)
            saliency = generate_saliency(explainer, query_tensor, ret_tensor, device)

            # Save .npy
            npy_path = output_dir / "npy" / f"q{q_idx}_{query_id}__r{rank}_{ret_id}.npy"
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy_path, saliency)

            # Save visualization
            viz_path = output_dir / "viz" / f"q{q_idx}_rank{rank}__{query_id}_vs_{ret_id}.png"
            save_grid(query_img, ret_img, saliency, query_id, ret_id, sim, rank, viz_path)

            summary_rows.append({
                "query_index": q_idx,
                "query_id": query_id,
                "rank": rank,
                "retrieved_index": int(r_idx),
                "retrieved_id": ret_id,
                "similarity": sim,
                "viz_path": str(viz_path),
                "npy_path": str(npy_path),
            })

            del ret_tensor
            if device.type == "cuda":
                torch.cuda.empty_cache()

        del query_tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nDone. {len(summary_rows)} saliency maps saved.")
    print(f"Summary CSV : {summary_csv}")
    print(f"Visualizations: {output_dir / 'viz'}")
    print(f"NPY files   : {output_dir / 'npy'}")


if __name__ == "__main__":
    main()
