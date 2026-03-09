"""
Evaluate a single query image: retrieve top-k similar images from Milvus,
generate saliency maps, compute insertion/deletion metrics, and save visualizations.

Usage example:
    python evaluate_single_image.py \
        --image path/to/query.jpg \
        --model_type densenet121 \
        --model_weights model.pth \
        --output_dir results/single \
        --top_k 5 \
        --explainer simatt
"""

import gc
import os
import sys
import math
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

import argparse

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(__file__))
from model import ConvNeXtV2, DenseNet121, ResNet50
from explanations import SimAtt, SimCAM, SBSMBatch
from evaluation import gkern, auc
from milvus_setup import MilvusManager
from milvus_retrieval import MilvusRetriever, get_model_and_transform
from milvus_retrieval_patched import MilvusRetrieverPatched


# ---------------------------------------------------------------------------
# CausalMetric (insertion / deletion)
# ---------------------------------------------------------------------------

class CausalMetric:
    """Compute insertion or deletion AUC for a query–retrieved image pair."""

    def __init__(self, model, mode, step, substrate_fn, input_size=224):
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
                start  = retrieved_tensor.clone()
                finish = self.substrate_fn(retrieved_tensor)
            else:
                start  = self.substrate_fn(retrieved_tensor)
                finish = retrieved_tensor.clone()

            start  = start.reshape(1, 3, self.hw)
            finish = finish.reshape(1, 3, self.hw)

            scores = np.empty(n_steps + 1)
            salient_order = torch.from_numpy(
                np.flip(np.argsort(saliency.flatten())).copy()
            ).unsqueeze(0).to(device)

            for i in range(n_steps + 1):
                r_feat = self.model(
                    start.reshape(1, 3,
                                  int(self.hw ** 0.5),
                                  int(self.hw ** 0.5))
                )
                sim = F.cosine_similarity(q_feat, r_feat)[0]
                scores[i] = max(sim.item(), 0.0)

                if i < n_steps:
                    coords = salient_order[:, self.step * i: self.step * (i + 1)]
                    start[0, :, coords] = finish[0, :, coords]

        return auc(scores), scores


# ---------------------------------------------------------------------------
# Saliency generation
# ---------------------------------------------------------------------------

def generate_saliency(query_tensor, retrieved_tensor, explainer, explainer_type):
    with torch.set_grad_enabled(explainer_type != 'sbsm'):
        saliency = explainer(query_tensor, retrieved_tensor)
    return saliency.squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _to_hwc(tensor):
    """Convert a normalised CHW tensor to a displayable HWC numpy array."""
    arr = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    arr  = std * arr + mean
    return np.clip(arr, 0, 1)


def save_saliency_figure(query_tensor, ret_tensor, saliency, rank, similarity,
                         del_auc, del_scores, ins_auc, ins_scores, out_dir):
    """Save a combined figure: query | retrieved | saliency overlay | metric curves."""

    query_np = _to_hwc(query_tensor)
    ret_np   = _to_hwc(ret_tensor)

    fig = plt.figure(figsize=(20, 5))
    gs  = gridspec.GridSpec(1, 4, figure=fig)

    # --- Query image ---
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(query_np)
    ax0.set_title('Query', fontsize=12, fontweight='bold')
    ax0.axis('off')

    # --- Retrieved image ---
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(ret_np)
    ax1.set_title(f'Retrieved (rank {rank})\nSim: {similarity:.4f}', fontsize=12)
    ax1.axis('off')

    # --- Saliency overlay ---
    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(ret_np)
    sal_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    im = ax2.imshow(sal_norm, cmap='jet', alpha=0.5)
    ax2.set_title('Saliency overlay', fontsize=12)
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # --- Metric curves ---
    ax3 = fig.add_subplot(gs[3])
    x = np.linspace(0, 1, len(del_scores))
    ax3.plot(x, del_scores, 'r-',  linewidth=2, label=f'Deletion  AUC={del_auc:.4f}')
    ax3.plot(x, ins_scores, 'b--', linewidth=2, label=f'Insertion AUC={ins_auc:.4f}')
    ax3.fill_between(x, 0, del_scores, alpha=0.15, color='red')
    ax3.fill_between(x, 0, ins_scores, alpha=0.15, color='blue')
    ax3.set_xlabel('Fraction of pixels')
    ax3.set_ylabel('Cosine similarity')
    ax3.set_title('Insertion / Deletion', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'rank{rank:02d}_result.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a single query image with Milvus retrieval + XAI metrics'
    )
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the query image')
    parser.add_argument('--model_type', type=str, default='densenet121',
                        choices=['densenet121', 'resnet50', 'convnextv2'],
                        help='Model architecture')
    parser.add_argument('--model_weights', type=str, required=True,
                        help='Path to model weights (.pth)')
    parser.add_argument('--embedding_dim', type=int, default=None,
                        help='Embedding dimension (leave blank for model default)')
    parser.add_argument('--explainer', type=str, default='simatt',
                        choices=['simatt', 'simcam', 'sbsm'],
                        help='Explanation method')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of retrieved images to analyse')
    parser.add_argument('--step_size', type=int, default=1000,
                        help='Pixel step size for insertion/deletion sweep')
    parser.add_argument('--output_dir', type=str, default='single_image_results',
                        help='Directory to write output images and JSON')
    parser.add_argument('--uri', type=str, default=None,
                        help='Zilliz Cloud / Milvus URI')
    parser.add_argument('--token', type=str, default=None,
                        help='Zilliz Cloud token')
    parser.add_argument('--metric_type', type=str, default='COSINE',
                        choices=['COSINE', 'L2', 'IP'],
                        help='Distance metric used in Milvus index')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Compute device (cuda / cpu)')
    parser.add_argument('--local_data_base_path', type=str, default=None,
                        help='Remap /kaggle/input paths to a local directory')
    parser.add_argument('--gpu_batch', type=int, default=50,
                        help='GPU batch size for SBSM (lower = less VRAM)')

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if not os.path.exists(args.image):
        print(f'ERROR: Image not found: {args.image}')
        return

    img_size = 384 if args.model_type == 'convnextv2' else 224
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Connect to Milvus
    # ------------------------------------------------------------------
    print('\n' + '='*60)
    print('Connecting to Milvus...')
    manager = MilvusManager(uri=args.uri, token=args.token)
    if not manager.connect():
        print('ERROR: Failed to connect to Milvus')
        return

    try:
        # ------------------------------------------------------------------
        # Load model
        # ------------------------------------------------------------------
        print('Loading model...')
        model, transform = get_model_and_transform(
            args.model_type, args.model_weights, args.embedding_dim, device
        )
        print(f'  Model ready: {args.model_type}')

        # ------------------------------------------------------------------
        # Build retriever
        # ------------------------------------------------------------------
        if args.local_data_base_path:
            print(f'  Using path remapping: /kaggle/... -> {args.local_data_base_path}')
            retriever = MilvusRetrieverPatched(
                manager, args.model_type, model, transform,
                local_data_base_path=args.local_data_base_path
            )
        else:
            retriever = MilvusRetriever(manager, args.model_type, model, transform)
        retriever.load_collection()
        print('  Retriever ready')

        # ------------------------------------------------------------------
        # Build explainer
        # ------------------------------------------------------------------
        print(f'Setting up explainer: {args.explainer}')
        if args.explainer == 'simatt':
            if args.model_type == 'densenet121':
                model_seq = nn.Sequential(*list(model.children()))
                explainer = SimAtt(model_seq, model_seq[0], target_layers=['relu'])
            elif args.model_type == 'resnet50':
                target_layer = model.resnet50[7][-1].conv3
                explainer = SimAtt(model, target_layer, target_layers=None)
            elif args.model_type == 'convnextv2':
                target_layer = model.convnext.stages[-1]
                explainer = SimAtt(model, target_layer, target_layers=None)

        elif args.explainer == 'simcam':
            if args.model_type != 'densenet121':
                raise NotImplementedError('SimCAM only supports DenseNet121')
            model_seq = nn.Sequential(*list(model.children())[0],
                                       *list(model.children())[1:])
            explainer = SimCAM(model_seq, model_seq[0], target_layers=['relu'],
                               fc=model_seq[2] if args.embedding_dim else None)

        elif args.explainer == 'sbsm':
            explainer = SBSMBatch(model, input_size=(img_size, img_size),
                                   gpu_batch=args.gpu_batch)
            maskspath = f'masks_{img_size}x{img_size}.npy'
            if os.path.isfile(maskspath):
                try:
                    existing = np.load(maskspath)
                    if existing.shape[-2:] == (img_size, img_size):
                        explainer.load_masks(maskspath)
                        print(f'  Masks loaded from {maskspath}')
                    else:
                        raise ValueError('Shape mismatch')
                except Exception as e:
                    print(f'  Regenerating masks ({e})')
                    explainer.generate_masks(window_size=24, stride=5,
                                             savepath=maskspath)
            else:
                print(f'  Generating masks at size {img_size}...')
                explainer.generate_masks(window_size=24, stride=5,
                                          savepath=maskspath)

        explainer.to(device)
        print('  Explainer ready')

        # ------------------------------------------------------------------
        # Insertion / deletion metrics setup
        # ------------------------------------------------------------------
        klen    = 51
        ksig    = math.sqrt(50)
        kern    = gkern(klen, ksig).to(device)
        blur_fn = lambda x: F.conv2d(x, kern, padding=klen // 2)

        del_metric = CausalMetric(model, 'del', args.step_size,
                                   torch.zeros_like, img_size)
        ins_metric = CausalMetric(model, 'ins', args.step_size,
                                   blur_fn,           img_size)

        # ------------------------------------------------------------------
        # Load query image
        # ------------------------------------------------------------------
        print('\n' + '='*60)
        print(f'Query image: {args.image}')
        query_img    = Image.open(args.image).convert('RGB')
        query_tensor = transform(query_img).unsqueeze(0).to(device)

        # ------------------------------------------------------------------
        # Retrieve top-k
        # ------------------------------------------------------------------
        print(f'Retrieving top-{args.top_k} from Milvus...')
        results, query_emb = retriever.search(
            args.image, top_k=args.top_k, metric_type=args.metric_type
        )
        print(f'  Got {len(results)} results')

        # ------------------------------------------------------------------
        # Process each retrieved image
        # ------------------------------------------------------------------
        all_metrics = []

        for rank, result in enumerate(results, 1):
            ret_path   = result['image_path']
            similarity = result['similarity']

            print(f'\n  Rank {rank}: {os.path.basename(ret_path)}  '
                  f'(similarity={similarity:.4f})')

            ret_img    = Image.open(ret_path).convert('RGB')
            ret_tensor = transform(ret_img).unsqueeze(0).to(device)

            # Saliency
            saliency = generate_saliency(query_tensor, ret_tensor,
                                          explainer, args.explainer)

            # Metrics
            del_auc, del_scores = del_metric.evaluate(query_tensor, ret_tensor, saliency)
            ins_auc, ins_scores = ins_metric.evaluate(query_tensor, ret_tensor, saliency)

            print(f'    Deletion  AUC: {del_auc:.4f}')
            print(f'    Insertion AUC: {ins_auc:.4f}')

            # Visualisation
            save_saliency_figure(
                query_tensor, ret_tensor, saliency,
                rank, similarity,
                del_auc, del_scores,
                ins_auc, ins_scores,
                args.output_dir
            )

            # Save raw saliency
            np.save(os.path.join(args.output_dir, f'rank{rank:02d}_saliency.npy'),
                    saliency)

            all_metrics.append({
                'rank':             rank,
                'retrieved_image':  os.path.basename(ret_path),
                'retrieved_path':   ret_path,
                'similarity':       float(similarity),
                'del_auc':          float(del_auc),
                'ins_auc':          float(ins_auc),
                'del_start':        float(del_scores[0]),
                'del_end':          float(del_scores[-1]),
                'ins_start':        float(ins_scores[0]),
                'ins_end':          float(ins_scores[-1]),
                'saliency_std':     float(saliency.std()),
                'saliency_min':     float(saliency.min()),
                'saliency_max':     float(saliency.max()),
            })

            del ret_tensor, saliency
            torch.cuda.empty_cache()
            gc.collect()

        # ------------------------------------------------------------------
        # Save JSON summary
        # ------------------------------------------------------------------
        summary = {
            'query_image':  os.path.abspath(args.image),
            'model_type':   args.model_type,
            'explainer':    args.explainer,
            'top_k':        args.top_k,
            'metric_type':  args.metric_type,
            'results':      all_metrics,
            'avg_del_auc':  float(np.mean([m['del_auc'] for m in all_metrics])),
            'avg_ins_auc':  float(np.mean([m['ins_auc'] for m in all_metrics])),
        }

        json_path = os.path.join(args.output_dir, 'summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # ------------------------------------------------------------------
        # Print summary
        # ------------------------------------------------------------------
        print('\n' + '='*60)
        print('SUMMARY')
        print('='*60)
        print(f'  Query:         {os.path.basename(args.image)}')
        print(f'  Model:         {args.model_type}')
        print(f'  Explainer:     {args.explainer}')
        print(f'  Avg Del AUC:   {summary["avg_del_auc"]:.4f}')
        print(f'  Avg Ins AUC:   {summary["avg_ins_auc"]:.4f}')
        print(f'  Output dir:    {os.path.abspath(args.output_dir)}')
        print(f'  JSON summary:  {json_path}')
        print('='*60)

    finally:
        manager.disconnect()
        del query_tensor
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
