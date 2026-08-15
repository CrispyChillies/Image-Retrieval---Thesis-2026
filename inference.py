"""
inference.py — Multi-model retrieval + XAI saliency inference

Supported models  : convnextv2 | convnextv2_sra | densenet121 | resnet50
Supported XAI     : simcam | simatt
Supported datasets: covid | tbx11k | vindr

For each run:
  - Selects 3 query images (one per class when possible)
  - Retrieves top-5 similar images via embedding cosine similarity
  - Generates saliency maps (query + each retrieved image) with the chosen XAI method
  - Produces two output figures per query:
      Figure 1: 1 query + 5 retrieved images (labels, similarity, match/mismatch borders)
      Figure 2: saliency overlay on query + saliency overlay on 5 retrieved images

Usage examples:
    # ConvNeXtV2 + SimCAM on COVID CXR
    python inference.py \
      --model_type convnextv2 --explainer simcam \
      --dataset covid \
      --data_dir /path/to/covid/images \
      --image_list /path/to/test_split.txt \
      --model_weights /path/to/convnextv2.pth \
      --output_dir ./results/covid_convnextv2_simcam

    # DenseNet121 + SimAtt on TBX11k
    python inference.py \
      --model_type densenet121 --explainer simatt \
      --dataset tbx11k \
      --data_dir /path/to/tbx11k/images \
      --image_list /path/to/test.csv \
      --model_weights /path/to/densenet121.pth \
      --output_dir ./results/tbx11k_densenet121_simatt

    # ConvNeXtV2-SRA + SimAtt on VinDR
    python inference.py \
      --model_type convnextv2_sra --explainer simatt \
      --dataset vindr \
      --data_dir /path/to/vindr/images \
      --image_list /path/to/image_labels_test_vindr.csv \
      --model_weights /path/to/convnextv2_sra.pth \
      --output_dir ./results/vindr_convnextv2sra_simatt
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model import ConvNeXtV2, ConvNeXtV2_SRA, DenseNet121, ResNet50
from explanations import SimCAM, SimAtt
from read_data import ChestXrayDataSet, TBX11kDataSet, VINDRDataSet

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_LABEL_NAMES = {
    "covid":  {0: "Normal", 1: "Pneumonia", 2: "COVID-19"},
    "tbx11k": {0: "TB", 1: "Healthy", 2: "Sick-No-TB"},
    "vindr":  {
        0: "COPD", 1: "Lung tumor", 2: "Pneumonia",
        3: "Tuberculosis", 4: "Other diseases", 5: "No finding",
    },
}

# VinDR 6 disease label columns (as in VINDRDataSet)
VINDR_DISEASE_COLS = [
    "COPD", "Lung tumor", "Pneumonia",
    "Tuberculosis", "Other diseases", "No finding",
]

# ---------------------------------------------------------------------------
# Model + transform + explainer builders
# ---------------------------------------------------------------------------

# Input resolution for each model family
MODEL_INPUT_SIZES = {
    "convnextv2":     384,
    "convnextv2_sra": 384,
    "densenet121":    224,
    "resnet50":       224,
}


def build_model(model_type: str, weights_path: str, embedding_dim, device):
    """Instantiate and load a model from checkpoint."""
    if model_type == "convnextv2":
        model = ConvNeXtV2(embedding_dim=embedding_dim)
    elif model_type == "convnextv2_sra":
        model = ConvNeXtV2_SRA(embedding_dim=embedding_dim)
    elif model_type == "densenet121":
        model = DenseNet121(embedding_dim=embedding_dim)
    elif model_type == "resnet50":
        model = ResNet50(embedding_dim=embedding_dim)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. "
                         "Choose: convnextv2, convnextv2_sra, densenet121, resnet50")

    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict):
        state = (checkpoint.get("state-dict")
                 or checkpoint.get("state_dict")
                 or checkpoint.get("model_state_dict")
                 or checkpoint)
    else:
        state = checkpoint
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(device)
    return model


def build_transform(img_size: int):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    resize_size = 432 if img_size == 384 else 256
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])


def _simcam_target(model, model_type: str):
    """Return (backbone, target_layer) for SimCAM per model type."""
    if model_type in ("convnextv2", "convnextv2_sra"):
        backbone = model.convnext
        target_layer = model.convnext.stages[3].blocks[2]
    elif model_type == "densenet121":
        # features block (Sequential); hook on last dense block before norm
        backbone = model.densenet121[0]
        target_layer = model.densenet121[0].denseblock4
    elif model_type == "resnet50":
        # model.resnet50 is nn.Sequential ending at avgpool; index 7 = layer4
        backbone = model.resnet50
        target_layer = model.resnet50[7]
    else:
        raise ValueError(f"SimCAM not configured for model_type: {model_type}")
    return backbone, target_layer


def _simatt_target(model, model_type: str):
    """Return the feature_module (target layer) for SimAtt per model type."""
    if model_type in ("convnextv2", "convnextv2_sra"):
        return model.convnext.stages[-1]
    elif model_type == "densenet121":
        return model.densenet121[0]          # entire DenseNet features block
    elif model_type == "resnet50":
        return model.resnet50[7]             # layer4
    else:
        raise ValueError(f"SimAtt not configured for model_type: {model_type}")


def build_explainer(model, model_type: str, explainer_type: str):
    """Build the XAI explainer for the given model and explainer type."""
    if explainer_type == "simcam":
        backbone, target_layer = _simcam_target(model, model_type)
        explainer = SimCAM(model=backbone, target_layer=target_layer, fc=None)
        explainer.eval()
        return explainer
    elif explainer_type == "simatt":
        target_layer = _simatt_target(model, model_type)
        # target_layers=None → use forward hook (supports nested backbone layers)
        explainer = SimAtt(model=model, feature_module=target_layer, target_layers=None)
        explainer.eval()
        return explainer
    else:
        raise ValueError(f"Unsupported explainer: {explainer_type}. Choose: simcam, simatt")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_dataset(dataset: str, data_dir: str, image_list: str, transform):
    if dataset == "covid":
        ds = ChestXrayDataSet(
            data_dir=data_dir,
            image_list_file=image_list,
            transform=transform,
        )
        # image_names are full paths; labels are ints 0/1/2
        image_paths = ds.image_names
        labels = ds.labels
        label_names = DATASET_LABEL_NAMES["covid"]

    elif dataset == "tbx11k":
        ds = TBX11kDataSet(
            data_dir=data_dir,
            csv_file=image_list,
            transform=transform,
        )
        image_paths = ds.image_names
        labels = ds.labels
        label_names = DATASET_LABEL_NAMES["tbx11k"]

    elif dataset == "vindr":
        import pandas as pd
        df = pd.read_csv(image_list)
        if "Other disease" in df.columns and "Other diseases" not in df.columns:
            df = df.rename(columns={"Other disease": "Other diseases"})

        # Build image_paths and single-label (dominant disease) for display
        image_paths, labels = [], []
        for _, row in df.iterrows():
            img_id = row["image_id"]
            img_path = os.path.join(data_dir, f"{img_id}.png")
            if not os.path.isfile(img_path):
                continue
            image_paths.append(img_path)
            # Dominant label = first active disease column (or "No finding")
            lbl = 5  # default No finding
            for idx, col in enumerate(VINDR_DISEASE_COLS):
                if col in df.columns and row.get(col, 0) == 1:
                    lbl = idx
                    break
            labels.append(lbl)
        label_names = DATASET_LABEL_NAMES["vindr"]

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return image_paths, labels, label_names


def select_query_indices(labels, num_queries: int = -1):
    """
    Select which dataset indices to use as queries.

    If num_queries is None or <= 0, every image in the dataset is used as a
    query (full-dataset retrieval evaluation). Otherwise, only the first
    `num_queries` images are used (useful for quick smoke tests).
    """
    if num_queries is None or num_queries <= 0:
        return list(range(len(labels)))
    return list(range(min(num_queries, len(labels))))


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(model, image_paths, transform, device, batch_size=32):
    """Extract L2-normalised embeddings for a list of image paths."""
    embeddings = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start: start + batch_size]
        tensors = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(transform(img))
            except Exception as exc:
                print(f"[WARN] Could not load {p}: {exc}")
                tensors.append(torch.zeros(3, 384, 384))
        batch = torch.stack(tensors).to(device)
        emb = model(batch)
        emb = F.normalize(emb, p=2, dim=1)
        embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)  # [N, D]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_top_k(query_emb, gallery_embs, query_idx, k=5):
    """
    Cosine similarity retrieval.
    Excludes the query itself from results.

    Returns:
        top_indices: list[int]  — gallery indices of top-k results
        top_scores:  list[float]
    """
    sims = (gallery_embs @ query_emb).squeeze(-1)  # [N]
    sims[query_idx] = -2.0  # exclude self

    top_indices = sims.argsort(descending=True)[:k].tolist()
    top_scores = sims[top_indices].tolist()
    return top_indices, top_scores


# ---------------------------------------------------------------------------
# Saliency generation
# ---------------------------------------------------------------------------

def normalize_sal(sal: np.ndarray) -> np.ndarray:
    sal = np.nan_to_num(sal.astype(np.float32))
    lo, hi = sal.min(), sal.max()
    if hi > lo:
        return (sal - lo) / (hi - lo)
    return np.zeros_like(sal)


def compute_simcam(simcam, query_tensor, retrieved_tensors, device):
    """
    Run SimCAM for one query vs. a batch of retrieved images.
    SimCAM processes all K retrievals in a single forward pass.

    Returns:
        query_sal:  np.ndarray [H, W]  — averaged query map across all K pairs
        ret_sals:   list[np.ndarray]   — per-retrieved saliency [H, W]
    """
    simcam.eval()
    x_q = query_tensor.unsqueeze(0).to(device)        # [1, 3, H, W]
    x_r = torch.stack(retrieved_tensors).to(device)   # [K, 3, H, W]

    # SimCAM returns [K, 2, H, W]: ch-0 = query map, ch-1 = retrieval map
    with torch.set_grad_enabled(False):
        maps = simcam(x_q, x_r)  # [K, 2, H, W]

    maps = maps.cpu().numpy()
    query_sal = normalize_sal(maps[:, 0, :, :].mean(axis=0))
    ret_sals = [normalize_sal(maps[i, 1, :, :]) for i in range(maps.shape[0])]
    return query_sal, ret_sals


def compute_simatt(simatt, query_tensor, retrieved_tensors, device):
    """
    Run SimAtt for one query vs each retrieved image (pairwise loop).
    SimAtt is gradient-based — requires grad enabled and eval mode.

    Returns:
        query_sal:  np.ndarray [H, W]  — averaged query map across all K pairs
        ret_sals:   list[np.ndarray]   — per-retrieved saliency [H, W]
    """
    simatt.eval()
    x_q = query_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]

    query_sal_list, ret_sals = [], []
    for r_tensor in retrieved_tensors:
        x_r = r_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]
        # SimAtt(x_q, x_p) returns M: [2, H, W]  — index 0=query, 1=retrieved
        with torch.set_grad_enabled(True):
            M = simatt(x_q, x_r)
        M_np = M.detach().cpu().numpy()
        query_sal_list.append(normalize_sal(M_np[0]))
        ret_sals.append(normalize_sal(M_np[1]))

    query_sal = normalize_sal(np.stack(query_sal_list).mean(axis=0))
    return query_sal, ret_sals


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def load_display_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def overlay_saliency(img: Image.Image, sal: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a PIL image with a jet-coloured saliency overlay."""
    img_np = np.array(img.resize((sal.shape[1], sal.shape[0]), Image.BILINEAR)).astype(np.float32) / 255.0
    cmap = plt.get_cmap("jet")
    sal_rgb = cmap(sal)[..., :3]  # [H, W, 3]
    blended = (1 - alpha) * img_np + alpha * sal_rgb
    return np.clip(blended, 0.0, 1.0)


def label_str(lbl, label_names: dict) -> str:
    if isinstance(lbl, (list, np.ndarray)):
        active = [label_names.get(i, str(i)) for i, v in enumerate(lbl) if v]
        return ", ".join(active) if active else "No finding"
    return label_names.get(int(lbl), str(lbl))


def labels_match(q_lbl, r_lbl) -> bool:
    """Return True when query and retrieved image share at least one class."""
    if isinstance(q_lbl, (list, np.ndarray)) and isinstance(r_lbl, (list, np.ndarray)):
        return bool(np.any(np.asarray(q_lbl) & np.asarray(r_lbl)))
    return int(q_lbl) == int(r_lbl)


def _add_border(ax, match: bool, lw: float = 5):
    """Draw a green (match) or red (mismatch) border around an Axes."""
    color = "#2ecc40" if match else "#ff4136"   # green / red
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)


# ---------------------------------------------------------------------------
# Core plotting
# ---------------------------------------------------------------------------

def plot_retrieval_figure(
    query_path: str,
    query_label,
    ret_paths: list,
    ret_labels: list,
    ret_scores: list,
    label_names: dict,
    out_path: str,
    dataset: str,
    query_rank: int,
):
    """
    Figure 1: query image (left) + 5 retrieved images.
    Layout: 1 row × 6 columns.
    """
    n_ret = len(ret_paths)
    fig, axes = plt.subplots(1, n_ret + 1, figsize=(4 * (n_ret + 1), 5.5))
    fig.suptitle(
        f"[{dataset.upper()}]  Query #{query_rank + 1} — Top-{n_ret} Retrieval",
        fontsize=13, fontweight="bold",
    )

    # --- Query ---
    ax = axes[0]
    ax.imshow(load_display_image(query_path))
    ax.set_title(
        f"QUERY\n• {label_str(query_label, label_names)}",
        fontsize=9, fontweight="bold",
    )
    # Neutral blue border for query
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#0074d9")
        spine.set_linewidth(5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Add query label as text box on the image
    ax.text(
        0.5, 0.02, label_str(query_label, label_names),
        transform=ax.transAxes, fontsize=8, color="white", fontweight="bold",
        ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0074d9", alpha=0.8),
    )

    # --- Retrieved ---
    for i, (rp, rl, rs) in enumerate(zip(ret_paths, ret_labels, ret_scores)):
        match = labels_match(query_label, rl)
        marker = "\u2713" if match else "\u2717"   # ✓ / ✗
        border_color = "#2ecc40" if match else "#ff4136"
        label_color = "#2ecc40" if match else "#ff4136"

        ax = axes[i + 1]
        ax.imshow(load_display_image(rp))
        ax.set_title(
            f"Rank {i + 1}  {marker}\n• {label_str(rl, label_names)}\nsim={rs:.3f}",
            fontsize=9,
            color=border_color,
            fontweight="bold",
        )
        _add_border(ax, match)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Add retrieved label as text box on the image
        ax.text(
            0.5, 0.02, f"{marker} {label_str(rl, label_names)}",
            transform=ax.transAxes, fontsize=8, color="white", fontweight="bold",
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=border_color, alpha=0.85),
        )

    # Legend: match / mismatch
    legend_patches = [
        mpatches.Patch(color="#2ecc40", label="Correct retrieval (same class)"),
        mpatches.Patch(color="#ff4136", label="Incorrect retrieval (different class)"),
        mpatches.Patch(color="#0074d9", label="Query image"),
    ]
    fig.legend(
        handles=legend_patches, loc="lower center",
        ncol=3, fontsize=8, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig-1] Saved → {out_path}")


def plot_saliency_figure(
    query_path: str,
    query_sal: np.ndarray,
    ret_paths: list,
    ret_sals: list,
    label_names: dict,
    query_label,
    ret_labels: list,
    ret_scores: list,
    out_path: str,
    dataset: str,
    query_rank: int,
    model_type: str = "convnextv2",
    explainer_name: str = "SimCAM",
):
    """
    Figure 2: saliency overlay on query (left) + saliency overlay on each retrieved image.
    Layout: 1 row × 6 columns.
    """
    n_ret = len(ret_paths)
    fig, axes = plt.subplots(1, n_ret + 1, figsize=(4 * (n_ret + 1), 5.5))
    fig.suptitle(
        f"[{dataset.upper()}]  Query #{query_rank + 1} — {explainer_name} Saliency  [{model_type}]",
        fontsize=13, fontweight="bold",
    )

    q_img = load_display_image(query_path)
    q_sal_resized = np.array(
        Image.fromarray((query_sal * 255).astype(np.uint8)).resize(
            q_img.size, Image.BILINEAR
        )
    ) / 255.0

    # --- Query saliency ---
    ax = axes[0]
    ax.imshow(overlay_saliency(q_img, q_sal_resized))
    ax.set_title(
        f"QUERY (saliency)\n• {label_str(query_label, label_names)}",
        fontsize=9, fontweight="bold",
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#0074d9")
        spine.set_linewidth(5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.text(
        0.5, 0.02, label_str(query_label, label_names),
        transform=ax.transAxes, fontsize=8, color="white", fontweight="bold",
        ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0074d9", alpha=0.8),
    )

    # --- Retrieved saliency ---
    for i, (rp, rsal, rl, rs) in enumerate(zip(ret_paths, ret_sals, ret_labels, ret_scores)):
        match = labels_match(query_label, rl)
        marker = "\u2713" if match else "\u2717"   # ✓ / ✗
        border_color = "#2ecc40" if match else "#ff4136"

        r_img = load_display_image(rp)
        r_sal_resized = np.array(
            Image.fromarray((rsal * 255).astype(np.uint8)).resize(
                r_img.size, Image.BILINEAR
            )
        ) / 255.0
        ax = axes[i + 1]
        ax.imshow(overlay_saliency(r_img, r_sal_resized))
        ax.set_title(
            f"Rank {i + 1}  {marker}  (saliency)\n• {label_str(rl, label_names)}\nsim={rs:.3f}",
            fontsize=9, color=border_color, fontweight="bold",
        )
        _add_border(ax, match)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.text(
            0.5, 0.02, f"{marker} {label_str(rl, label_names)}",
            transform=ax.transAxes, fontsize=8, color="white", fontweight="bold",
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=border_color, alpha=0.85),
        )

    # Colour-bar legend for jet scale
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.01, pad=0.01)
    cbar.set_label("Saliency", fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig-2] Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Model ----
    img_size = MODEL_INPUT_SIZES[args.model_type]
    print(f"Loading {args.model_type} from {args.model_weights}  (input={img_size}x{img_size}) ...")
    model = build_model(args.model_type, args.model_weights, args.embedding_dim, device)
    transform = build_transform(img_size)
    print(f"Building {args.explainer.upper()} explainer ...")
    explainer = build_explainer(model, args.model_type, args.explainer)

    # ---- Dataset ----
    print(f"Loading dataset '{args.dataset}' ...")
    image_paths, labels, label_names = load_dataset(
        args.dataset, args.data_dir, args.image_list, transform
    )
    print(f"  {len(image_paths)} images found.")

    if len(image_paths) == 0:
        print("[ERROR] No images found. Check --data_dir and --image_list.")
        sys.exit(1)

    # ---- Embeddings ----
    print("Extracting embeddings ...")
    embeddings = extract_embeddings(model, image_paths, transform, device, batch_size=args.batch_size)
    print(f"  Embeddings shape: {embeddings.shape}")

    # ---- Query selection ----
    query_indices = select_query_indices(labels, num_queries=args.num_queries)
    print(f"  Running inference on {len(query_indices)} / {len(image_paths)} images as queries.")

    # ---- Per-query inference ----
    os.makedirs(args.output_dir, exist_ok=True)

    for q_rank, q_idx in enumerate(query_indices):
        q_path = image_paths[q_idx]
        q_label = labels[q_idx]
        q_emb = embeddings[q_idx].unsqueeze(-1)  # [D, 1]

        print(f"\nQuery {q_rank + 1}/{len(query_indices)}: {os.path.basename(q_path)}"
              f"  label={label_str(q_label, label_names)}")

        # Retrieval
        top_indices, top_scores = retrieve_top_k(
            q_emb, embeddings, query_idx=q_idx, k=args.top_k
        )
        ret_paths = [image_paths[i] for i in top_indices]
        ret_labels = [labels[i] for i in top_indices]

        print(f"  Retrieved: {[os.path.basename(p) for p in ret_paths]}")

        # Load tensors for saliency
        q_tensor = transform(Image.open(q_path).convert("RGB"))
        ret_tensors = []
        for rp in ret_paths:
            try:
                ret_tensors.append(transform(Image.open(rp).convert("RGB")))
            except Exception as exc:
                print(f"  [WARN] {rp}: {exc}")
                ret_tensors.append(torch.zeros(3, img_size, img_size))

        # Saliency
        print(f"  Computing {args.explainer.upper()} saliency ...")
        if args.explainer == "simcam":
            query_sal, ret_sals = compute_simcam(explainer, q_tensor, ret_tensors, device)
        else:
            query_sal, ret_sals = compute_simatt(explainer, q_tensor, ret_tensors, device)

        # Figure base name includes model + explainer + source image, so runs
        # over the full dataset don't overwrite each other.
        img_stem = os.path.splitext(os.path.basename(q_path))[0]
        base = f"{args.dataset}_{args.model_type}_{args.explainer}_{img_stem}"

        # Figure 1 — retrieval grid
        plot_retrieval_figure(
            query_path=q_path,
            query_label=q_label,
            ret_paths=ret_paths,
            ret_labels=ret_labels,
            ret_scores=top_scores,
            label_names=label_names,
            out_path=os.path.join(args.output_dir, f"{base}_retrieval.png"),
            dataset=args.dataset,
            query_rank=q_rank,
        )

        # Figure 2 — saliency grid
        plot_saliency_figure(
            query_path=q_path,
            query_sal=query_sal,
            ret_paths=ret_paths,
            ret_sals=ret_sals,
            label_names=label_names,
            query_label=q_label,
            ret_labels=ret_labels,
            ret_scores=top_scores,
            out_path=os.path.join(args.output_dir, f"{base}_saliency.png"),
            dataset=args.dataset,
            query_rank=q_rank,
            model_type=args.model_type,
            explainer_name=args.explainer.upper(),
        )

    print(f"\nDone. All results saved to: {args.output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-model retrieval + XAI saliency inference"
    )
    parser.add_argument(
        "--model_type", default="convnextv2",
        choices=["convnextv2", "convnextv2_sra", "densenet121", "resnet50"],
        help="Model architecture (default: convnextv2)",
    )
    parser.add_argument(
        "--explainer", default="simcam",
        choices=["simcam", "simatt"],
        help="XAI method: simcam (no-grad, batch) or simatt (gradient-based, pairwise).",
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=["covid", "tbx11k", "vindr"],
        help="Dataset to run inference on",
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Root directory containing the images",
    )
    parser.add_argument(
        "--image_list", required=True,
        help=(
            "For covid: text file with '<id> <path> <label>' lines. "
            "For tbx11k: CSV with fname/image_type columns. "
            "For vindr: CSV with image_id and disease label columns."
        ),
    )
    parser.add_argument(
        "--model_weights", required=True,
        help="Path to the model .pth checkpoint",
    )
    parser.add_argument(
        "--output_dir", default="./inference_results",
        help="Directory where output PNG figures are saved",
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=None,
        help="Optional projection embedding dimension (None = backbone default)",
    )
    parser.add_argument(
        "--num_queries", type=int, default=-1,
        help=(
            "Number of query images to process. Default: -1, meaning every "
            "image in the dataset is used as a query (full retrieval + XAI "
            "evaluation). Set to a positive integer to limit to the first N "
            "images (useful for quick smoke tests)."
        ),
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of images to retrieve per query (default: 5)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for embedding extraction",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
