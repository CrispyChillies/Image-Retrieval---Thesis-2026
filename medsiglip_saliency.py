"""
Grad-CAM saliency visualization for image retrieval with MedSigLIP.

Uses Grad-CAM to highlight which regions of a retrieved image contribute
most to its cosine similarity with the query image embedding.
Results are retrieved from a Zilliz Milvus vector index.

Usage:
    python medsiglip_saliency.py \
        --query_image path/to/query.jpg \
        --model_weights model.pth \
        --milvus_uri "https://..." \
        --milvus_token "..." \
        --local_data_base_path /path/to/local/images \
        --top_k 5
"""

import argparse
import gc
import json
import math
import os

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import CosineSimilarityTarget
except ImportError:
    GradCAM = None
    CosineSimilarityTarget = None

from model import MedSigLIP
from evaluation import gkern, auc
from milvus_setup import MilvusManager, MODEL_CONFIGS
from path_mapper import PathMapper


# ---------------------------------------------------------------------------
# Wrapper model that exposes internal conv-like feature maps for Grad-CAM
# ---------------------------------------------------------------------------

class MedSigLIPGradCAMWrapper(nn.Module):
    """Wraps a MedSigLIP model so that Grad-CAM can operate on it.

    pytorch-grad-cam expects a CNN-style target layer (B, C, H, W).
    MedSigLIP's ViT backbone outputs patch tokens (B, N, D).  This wrapper
    reshapes the last encoder layer output to (B, D, grid, grid) so that
    standard Grad-CAM can produce a spatial heatmap.
    """

    def __init__(self, model: MedSigLIP):
        super().__init__()
        self.model = model
        # The layer whose output we will reshape for Grad-CAM
        self.target_layer = _ReshapedPatchTokenLayer(model)

    def forward(self, x):
        return self.model(x)


class _ReshapedPatchTokenLayer(nn.Module):
    """Hooks into the last encoder layer and reshapes tokens to 2-D feature map."""

    def __init__(self, medsiglip_model: MedSigLIP):
        super().__init__()
        self.backbone = medsiglip_model.backbone
        # Last encoder layer
        self._target = self.backbone.encoder.layers[-1]

        self._features: torch.Tensor | None = None
        self._handle = None

    # -- hook management --
    def register(self):
        self._handle = self._target.register_forward_hook(self._hook)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, module, inp, out):
        # out is (B, N, D) or a tuple
        tokens = out[0] if isinstance(out, (tuple, list)) else out
        B, N, D = tokens.shape
        grid = int(N ** 0.5)
        # Reshape to (B, D, grid, grid)
        self._features = tokens.permute(0, 2, 1).reshape(B, D, grid, grid)

    def forward(self, x):
        # This is never called directly; Grad-CAM reads self._features via hook.
        return x


class _SimilarityModelForGradCAM(nn.Module):
    """A thin wrapper that, given a *retrieved* image tensor, returns its
    cosine similarity with a pre-computed query embedding.

    Grad-CAM will backprop through this to find the gradient of the
    similarity w.r.t. intermediate feature maps.
    """

    def __init__(self, medsiglip_model: MedSigLIP, query_embedding: torch.Tensor):
        super().__init__()
        self.model = medsiglip_model
        # (1, D)
        self.query_embedding = query_embedding.detach()
        self.reshape_layer = _ReshapedPatchTokenLayer(medsiglip_model)

    def get_target_layer(self):
        return self.reshape_layer

    def forward(self, x):
        # Register hook to capture reshaped features
        self.reshape_layer.register()
        emb = self.model(x)  # (B, D) normalised
        self.reshape_layer.remove()

        # Cosine similarity per sample
        sim = F.cosine_similarity(emb, self.query_embedding.to(emb.device), dim=1)
        return sim


# ---------------------------------------------------------------------------
# Grad-CAM computation
# ---------------------------------------------------------------------------

def compute_gradcam_saliency(
    model: MedSigLIP,
    query_tensor: torch.Tensor,
    retrieved_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Compute Grad-CAM heatmap for each retrieved image w.r.t. query.

    Args:
        model: MedSigLIP model (eval mode, on device).
        query_tensor: (1, 3, H, W) preprocessed query image.
        retrieved_tensor: (K, 3, H, W) preprocessed retrieved images.
        device: torch device.

    Returns:
        heatmaps: np.ndarray of shape (K, H_img, W_img) in [0, 1].
    """
    model.eval()

    # Compute query embedding (no grad needed)
    with torch.no_grad():
        query_emb = model(query_tensor.to(device))  # (1, D)

    sim_model = _SimilarityModelForGradCAM(model, query_emb)
    sim_model.to(device)
    sim_model.eval()

    target_layer = sim_model.get_target_layer()

    # We use pytorch_grad_cam's GradCAM.  It expects:
    #   - a model that returns a tensor
    #   - a list of target layers with spatial feature maps (B,C,H,W)
    #
    # _ReshapedPatchTokenLayer produces (B,D,grid,grid) via its hook.
    # We need a thin adapter so GradCAM can see it as a regular layer.

    heatmaps = []
    for i in range(retrieved_tensor.shape[0]):
        img_tensor = retrieved_tensor[i : i + 1].to(device)
        img_tensor.requires_grad = True

        # Register hook
        target_layer.register()

        sim = sim_model(img_tensor)  # scalar per sample
        model.zero_grad()
        sim.backward()

        # Grad-CAM: get gradients and activations
        grads = target_layer._features.grad  # may be None if not retained
        # Since _features is computed inside hook, we need to use an alternate approach.
        # We'll manually compute Grad-CAM from the hook.
        target_layer.remove()

        # Recompute with gradient tracking
        heatmap = _compute_single_gradcam(model, query_emb, img_tensor, device)
        heatmaps.append(heatmap)

    return np.stack(heatmaps, axis=0)


def _compute_single_gradcam(
    model: MedSigLIP,
    query_emb: torch.Tensor,
    img_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Compute Grad-CAM for a single retrieved image.

    Manually hooks into the last encoder layer, captures activations and
    gradients, and computes the Grad-CAM heatmap.
    """
    target_layer = model.backbone.encoder.layers[-1]

    activations = []
    gradients = []

    def fwd_hook(module, inp, out):
        tokens = out[0] if isinstance(out, (tuple, list)) else out
        activations.append(tokens)

    def bwd_hook(module, grad_in, grad_out):
        grad = grad_out[0] if isinstance(grad_out, (tuple, list)) else grad_out
        gradients.append(grad)

    fwd_handle = target_layer.register_forward_hook(fwd_hook)
    bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

    # Forward
    img = img_tensor.detach().requires_grad_(True)
    emb = model(img)  # (1, D)
    sim = F.cosine_similarity(emb, query_emb.detach(), dim=1).sum()

    # Backward
    model.zero_grad()
    sim.backward()

    fwd_handle.remove()
    bwd_handle.remove()

    # activations: (1, N, D),  gradients: (1, N, D)
    act = activations[0].detach()  # (1, N, D)
    grad = gradients[0].detach()   # (1, N, D)

    # Grad-CAM weights: global-average-pool the gradients over spatial dim
    weights = grad.mean(dim=1, keepdim=True)  # (1, 1, D)

    # Weighted combination
    cam = (act * weights).sum(dim=-1)  # (1, N)
    cam = F.relu(cam)

    # Reshape to spatial grid
    N = cam.shape[1]
    grid = int(N ** 0.5)
    cam = cam.view(1, 1, grid, grid)

    # Upsample to image resolution
    H, W = img_tensor.shape[2], img_tensor.shape[3]
    cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()

    # Normalize to [0, 1]
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    return cam


# ---------------------------------------------------------------------------
# CausalMetric (insertion / deletion)
# ---------------------------------------------------------------------------

class CausalMetric:
    """Compute insertion or deletion AUC for a query-retrieved image pair."""

    def __init__(self, model, mode, step, substrate_fn, input_size=448):
        assert mode in ('del', 'ins')
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn
        self.hw = input_size * input_size

    def evaluate(self, query_tensor, retrieved_tensor, saliency):
        device = query_tensor.device
        n_steps = (self.hw + self.step - 1) // self.step

        with torch.no_grad():
            q_feat = self.model(query_tensor)

            if self.mode == 'del':
                start = retrieved_tensor.clone()
                finish = self.substrate_fn(retrieved_tensor)
            else:
                start = self.substrate_fn(retrieved_tensor)
                finish = retrieved_tensor.clone()

            start = start.reshape(1, 3, self.hw)
            finish = finish.reshape(1, 3, self.hw)

            scores = np.empty(n_steps + 1)
            salient_order = torch.from_numpy(
                np.flip(np.argsort(saliency.flatten())).copy()
            ).unsqueeze(0).to(device)

            zero_counter = 0
            for i in range(n_steps + 1):
                r_feat = self.model(
                    start.reshape(1, 3,
                                  int(self.hw ** 0.5),
                                  int(self.hw ** 0.5))
                )
                cosine_sim = F.cosine_similarity(q_feat, r_feat)[0]

                if cosine_sim < 0:
                    cosine_sim = torch.clamp(cosine_sim, min=0, max=1)
                    zero_counter += 1

                scores[i] = cosine_sim.item()
                del r_feat, cosine_sim

                if i < n_steps:
                    coords = salient_order[:, self.step * i: self.step * (i + 1)]
                    start[0, :, coords] = finish[0, :, coords]

        del start, finish, q_feat, salient_order
        return auc(scores), scores, zero_counter


# ---------------------------------------------------------------------------
# Milvus retrieval helpers
# ---------------------------------------------------------------------------

def retrieve_similar_images(
    query_image_path: str,
    model: MedSigLIP,
    transform,
    milvus_uri: str,
    milvus_token: str,
    top_k: int = 5,
    model_type: str = "medsiglip",
) -> tuple:
    """Query Milvus and return top-K results.

    Returns:
        results: list of dicts with 'image_path', 'label', 'similarity'.
        query_embedding: (1, D) tensor.
    """
    manager = MilvusManager(uri=milvus_uri, token=milvus_token)
    manager.connect()

    from milvus_retrieval import MilvusRetriever

    retriever = MilvusRetriever(manager, model_type, model, transform)
    retriever.load_collection()

    results, query_embedding = retriever.search(
        query_image_path, top_k=top_k, metric_type="COSINE"
    )

    manager.disconnect()
    return results, query_embedding


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay a heatmap on an image.

    Args:
        image: RGB uint8 array (H, W, 3).
        heatmap: float array (H, W) in [0, 1].
        alpha: blending factor.

    Returns:
        Blended RGB uint8 array.
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    blended = np.uint8(alpha * colormap + (1 - alpha) * image)
    return blended


def _to_hwc(tensor):
    """Convert a MedSigLIP-normalised CHW tensor to displayable HWC [0,1] array."""
    arr = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    arr = 0.5 * arr + 0.5  # MedSigLIP uses mean=0.5, std=0.5
    return np.clip(arr, 0, 1)


def visualize_top_k(
    query_image_path: str,
    results: list,
    heatmaps: np.ndarray,
    metrics: list | None = None,
    local_data_base_path: str | None = None,
    save_path: str | None = None,
):
    """Visualize query + top-K results with saliency maps and metric curves.

    Layout per row:
        [Query]  |  [Retrieved]  |  [Saliency Overlay]  |  [Ins/Del Curves]

    Args:
        query_image_path: path to the query image.
        results: list of dicts from Milvus search.
        heatmaps: (K, H, W) array from compute_gradcam_saliency.
        metrics: list of dicts with 'del_auc', 'del_scores', 'ins_auc', 'ins_scores'.
        local_data_base_path: if set, remap /kaggle/ paths to local.
        save_path: if set, save figure to this path instead of showing.
    """
    path_mapper = None
    if local_data_base_path:
        path_mapper = PathMapper(local_base_path=local_data_base_path)

    K = len(results)
    has_metrics = metrics is not None and len(metrics) == K
    ncols = 4 if has_metrics else 3
    fig, axes = plt.subplots(K, ncols, figsize=(5 * ncols, 5 * K))
    if K == 1:
        axes = axes[np.newaxis, :]

    query_img = Image.open(query_image_path).convert("RGB")

    for i, result in enumerate(results):
        # Resolve retrieved image path
        ret_path = result["image_path"]
        if path_mapper and ret_path.startswith("/kaggle/"):
            ret_path = path_mapper.remap_path(ret_path)

        similarity = result.get("similarity", result.get("distance", 0.0))
        label = result.get("label", "N/A")

        # Load retrieved image
        try:
            ret_img = Image.open(ret_path).convert("RGB")
        except FileNotFoundError:
            print(f"WARNING: could not open {ret_path}, skipping row {i+1}")
            for j in range(ncols):
                axes[i, j].axis("off")
            axes[i, 0].set_title(f"Top {i+1}: image not found")
            continue

        ret_img_np = np.array(ret_img)

        # Resize heatmap to match the original retrieved image size
        h, w = ret_img_np.shape[:2]
        heatmap_resized = cv2.resize(heatmaps[i], (w, h), interpolation=cv2.INTER_LINEAR)
        saliency_overlay = overlay_heatmap(ret_img_np, heatmap_resized)

        # Column 0: Query image
        axes[i, 0].imshow(query_img)
        axes[i, 0].set_title("Query" if i == 0 else "")
        axes[i, 0].axis("off")

        # Column 1: Retrieved image
        axes[i, 1].imshow(ret_img)
        axes[i, 1].set_title(f"Top {i+1} | {label} | sim={similarity:.4f}")
        axes[i, 1].axis("off")

        # Column 2: Saliency overlay
        axes[i, 2].imshow(saliency_overlay)
        axes[i, 2].set_title(f"Grad-CAM Saliency (Top {i+1})")
        axes[i, 2].axis("off")

        # Column 3: Insertion / Deletion curves
        if has_metrics:
            m = metrics[i]
            ax3 = axes[i, 3]
            del_scores = m['del_scores']
            ins_scores = m['ins_scores']
            x = np.linspace(0, 1, len(del_scores))
            ax3.plot(x, del_scores, 'r-', linewidth=2,
                     label=f"Del AUC={m['del_auc']:.4f}")
            ax3.plot(x, ins_scores, 'b--', linewidth=2,
                     label=f"Ins AUC={m['ins_auc']:.4f}")
            ax3.fill_between(x, 0, del_scores, alpha=0.15, color='red')
            ax3.fill_between(x, 0, ins_scores, alpha=0.15, color='blue')
            ax3.set_xlabel('Fraction of pixels')
            ax3.set_ylabel('Cosine similarity')
            ax3.set_title(f'Insertion / Deletion (Top {i+1})')
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)

    plt.suptitle("Grad-CAM Saliency for Image Retrieval (MedSigLIP)", fontsize=16)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Grad-CAM saliency for MedSigLIP image retrieval via Milvus"
    )
    parser.add_argument("--query_image", type=str, required=True,
                        help="Path to the query image")
    parser.add_argument("--model_weights", type=str, default="model.pth",
                        help="Path to MedSigLIP model weights")
    parser.add_argument("--embedding_dim", type=int, default=512,
                        help="Embedding dimension")
    parser.add_argument("--milvus_uri", type=str, required=True,
                        help="Zilliz Milvus URI")
    parser.add_argument("--milvus_token", type=str, required=True,
                        help="Zilliz Milvus token")
    parser.add_argument("--local_data_base_path", type=str, default=None,
                        help="Local base path for image files (remaps /kaggle/ paths)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top results to display")
    parser.add_argument("--step_size", type=int, default=1000,
                        help="Pixel step size for insertion/deletion sweep")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save the output figure (shows interactively if omitted)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save per-rank figures, saliency .npy, and summary JSON")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: auto-detect)")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading MedSigLIP model...")
    model = MedSigLIP(embed_dim=args.embedding_dim)
    checkpoint = torch.load(args.model_weights, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    model.to(device)

    # Transform (MedSigLIP uses SigLIP normalisation)
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(512),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Retrieve similar images from Milvus
    print(f"Searching Milvus for top-{args.top_k} similar images...")
    results, query_emb = retrieve_similar_images(
        query_image_path=args.query_image,
        model=model,
        transform=transform,
        milvus_uri=args.milvus_uri,
        milvus_token=args.milvus_token,
        top_k=args.top_k,
    )

    if not results:
        print("No results returned from Milvus.")
        return

    print(f"Retrieved {len(results)} results.")

    # Prepare tensors for Grad-CAM
    query_tensor = transform(Image.open(args.query_image)).unsqueeze(0)

    path_mapper = None
    if args.local_data_base_path:
        path_mapper = PathMapper(local_base_path=args.local_data_base_path)

    retrieved_tensors = []
    valid_results = []
    for r in results:
        ret_path = r["image_path"]
        if path_mapper and ret_path.startswith("/kaggle/"):
            ret_path = path_mapper.remap_path(ret_path)
        try:
            ret_img = Image.open(ret_path).convert("RGB")
            retrieved_tensors.append(transform(ret_img))
            valid_results.append(r)
        except FileNotFoundError:
            print(f"WARNING: could not load {ret_path}, skipping")

    if not retrieved_tensors:
        print("No valid retrieved images found.")
        return

    retrieved_tensor = torch.stack(retrieved_tensors, dim=0)

    # Compute Grad-CAM saliency
    print("Computing Grad-CAM saliency maps...")
    heatmaps = compute_gradcam_saliency(model, query_tensor, retrieved_tensor, device)

    # ------------------------------------------------------------------
    # Insertion / Deletion metrics
    # ------------------------------------------------------------------
    img_size = 448
    klen = 51
    ksig = math.sqrt(50)
    kern = gkern(klen, ksig).to(device)
    blur_fn = lambda x: F.conv2d(x, kern, padding=klen // 2)

    del_metric = CausalMetric(model, 'del', args.step_size, torch.zeros_like, img_size)
    ins_metric = CausalMetric(model, 'ins', args.step_size, blur_fn, img_size)

    all_metrics = []
    for rank, (r, hm) in enumerate(zip(valid_results, heatmaps), 1):
        ret_path = r["image_path"]
        if path_mapper and ret_path.startswith("/kaggle/"):
            ret_path = path_mapper.remap_path(ret_path)

        ret_tensor_i = retrieved_tensor[rank - 1 : rank].to(device)

        del_auc, del_scores, del_zeros = del_metric.evaluate(
            query_tensor.to(device), ret_tensor_i, hm
        )
        ins_auc, ins_scores, ins_zeros = ins_metric.evaluate(
            query_tensor.to(device), ret_tensor_i, hm
        )

        print(f"  Rank {rank}: Del AUC={del_auc:.4f}  Ins AUC={ins_auc:.4f}")

        all_metrics.append({
            'rank': rank,
            'retrieved_image': os.path.basename(ret_path),
            'similarity': float(r.get('similarity', 0.0)),
            'del_auc': float(del_auc),
            'ins_auc': float(ins_auc),
            'del_scores': del_scores,
            'ins_scores': ins_scores,
        })

        del ret_tensor_i
        torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------------
    # Save per-rank figures + JSON if output_dir is set
    # ------------------------------------------------------------------
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for m in all_metrics:
            rank = m['rank']
            # Save raw saliency
            np.save(os.path.join(args.output_dir, f'rank{rank:02d}_saliency.npy'),
                    heatmaps[rank - 1])

        # Save JSON summary (scores as lists for serialization)
        summary = {
            'query_image': os.path.abspath(args.query_image),
            'model': 'medsiglip',
            'explainer': 'gradcam',
            'top_k': args.top_k,
            'step_size': args.step_size,
            'avg_del_auc': float(np.mean([m['del_auc'] for m in all_metrics])),
            'avg_ins_auc': float(np.mean([m['ins_auc'] for m in all_metrics])),
            'results': [
                {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in m.items()}
                for m in all_metrics
            ],
        }
        json_path = os.path.join(args.output_dir, 'summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {json_path}")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    for m in all_metrics:
        print(f"  Rank {m['rank']:2d}  sim={m['similarity']:.4f}  "
              f"Del={m['del_auc']:.4f}  Ins={m['ins_auc']:.4f}")
    print(f"  Avg Del AUC: {np.mean([m['del_auc'] for m in all_metrics]):.4f}")
    print(f"  Avg Ins AUC: {np.mean([m['ins_auc'] for m in all_metrics]):.4f}")
    print('=' * 60)

    # Visualize
    print("Generating visualization...")
    visualize_top_k(
        query_image_path=args.query_image,
        results=valid_results,
        heatmaps=heatmaps,
        metrics=all_metrics,
        local_data_base_path=args.local_data_base_path,
        save_path=args.save_path,
    )
    print("Done.")


if __name__ == "__main__":
    main()
