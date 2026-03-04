"""
Fast debug pipeline using Milvus for retrieval
This version is much faster because it uses pre-computed embeddings
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from model import ConvNeXtV2, DenseNet121, ResNet50
from explanations import SimAtt, SimCAM, SBSMBatch
from evaluation import gkern, auc
import argparse
import matplotlib.pyplot as plt

# Add milvus to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'milvus'))
from milvus_setup import MilvusManager
from milvus_retrieval import MilvusRetriever, get_model_and_transform


class CausalMetric():
    """Simplified causal metric for debugging"""
    
    def __init__(self, model, mode, step, substrate_fn, input_size=224):
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn
        self.hw = input_size * input_size
        
    def evaluate(self, img_tensor, retrieved_tensor, explanation):
        """Evaluate insertion/deletion metric"""
        q_feat = self.model(img_tensor.cuda())
        n_steps = (self.hw + self.step - 1) // self.step
        
        if self.mode == 'del':
            start = retrieved_tensor.clone()
            finish = self.substrate_fn(retrieved_tensor)
        else:
            start = self.substrate_fn(retrieved_tensor)
            finish = retrieved_tensor.clone()
        
        scores = np.empty(n_steps + 1)
        salient_order = np.flip(np.argsort(explanation.flatten())).copy()
        salient_order = torch.from_numpy(salient_order).unsqueeze(0)
        
        zero_counter = 0
        for i in range(n_steps + 1):
            r_feat = self.model(start.cuda())
            cosine_sim = F.cosine_similarity(q_feat, r_feat)[0]
            
            if cosine_sim < 0:
                cosine_sim = torch.clamp(cosine_sim, min=0, max=1)
                zero_counter += 1
            
            scores[i] = cosine_sim.item()
            
            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                start_reshaped = start.reshape(1, 3, self.hw)
                finish_reshaped = finish.reshape(1, 3, self.hw)
                start_reshaped[0, :, coords] = finish_reshaped[0, :, coords]
                start = start_reshaped.reshape(start.shape)
        
        return auc(scores), scores, zero_counter


def generate_saliency(query_tensor, retrieved_tensor, explainer, args):
    """Generate saliency map for query-retrieved pair"""
    with torch.set_grad_enabled(args.explainer != 'sbsm'):
        if args.explainer == 'sbsm':
            saliency = explainer(query_tensor, retrieved_tensor)
        else:
            saliency = explainer(query_tensor, retrieved_tensor)
    
    return saliency.squeeze().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Fast debug pipeline with Milvus')
    parser.add_argument('--query_image', type=str, required=True,
                       help='Path to query image')
    parser.add_argument('--model_type', type=str, default='densenet121',
                       choices=['densenet121', 'resnet50', 'convnextv2'],
                       help='Model architecture')
    parser.add_argument('--model_weights', type=str, required=True,
                       help='Path to model weights')
    parser.add_argument('--embedding_dim', type=int, default=None,
                       help='Embedding dimension')
    parser.add_argument('--explainer', type=str, default='simatt',
                       choices=['simatt', 'simcam', 'sbsm'],
                       help='Explanation method')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of retrieved images to analyze')
    parser.add_argument('--output_dir', type=str, default='./debug_milvus_output',
                       help='Output directory')
    parser.add_argument('--step_size', type=int, default=1000,
                       help='Step size for insertion/deletion')
    parser.add_argument('--uri', type=str, default=None,
                       help='Zilliz Cloud URI')
    parser.add_argument('--token', type=str, default=None,
                       help='Zilliz Cloud token')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Determine image size based on model
    if args.model_type == 'convnextv2':
        img_size = 384
    else:
        img_size = 224
    
    # Connect to Milvus
    print(f"\n{'='*70}")
    print(f"CONNECTING TO MILVUS")
    print(f"{'='*70}")
    manager = MilvusManager(uri=args.uri, token=args.token)
    if not manager.connect():
        print("❌ Failed to connect to Milvus")
        return
    
    try:
        # Load model
        print(f"\n{'='*70}")
        print(f"Loading {args.model_type} model...")
        print(f"{'='*70}")
        
        model, transform = get_model_and_transform(
            args.model_type, args.model_weights, args.embedding_dim, device
        )
        print(f"✅ Model loaded successfully")
        
        # Create retriever
        print(f"\n{'='*70}")
        print(f"[Step 1] Initializing Milvus Retriever...")
        print(f"{'='*70}")
        retriever = MilvusRetriever(manager, args.model_type, model, transform)
        retriever.load_collection()
        print(f"✅ Retriever ready")
        
        # Retrieve top-k similar images
        print(f"\n{'='*70}")
        print(f"[Step 2] Retrieving top-{args.top_k} similar images...")
        print(f"{'='*70}")
        print(f"Query: {os.path.basename(args.query_image)}")
        
        results, query_emb = retriever.search(args.query_image, top_k=args.top_k)
        
        print(f"\nTop-{args.top_k} Retrieved Images:")
        retrieved_paths = []
        similarities = []
        for i, result in enumerate(results, 1):
            print(f"  {i}. {os.path.basename(result['image_path'])} (similarity: {result['similarity']:.4f})")
            retrieved_paths.append(result['image_path'])
            similarities.append(result['similarity'])
        
        # Setup explainer
        print(f"\n{'='*70}")
        print(f"[Step 3] Setting up {args.explainer} explainer...")
        print(f"{'='*70}")
        
        if args.explainer == 'simatt':
            if args.model_type == 'densenet121':
                model_seq = nn.Sequential(*list(model.children()))
                explainer = SimAtt(model_seq, model_seq[0], target_layers=["relu"])
            elif args.model_type == 'resnet50':
                target_layer = model.resnet50[7][-1].conv3
                explainer = SimAtt(model, target_layer, target_layers=None)
            elif args.model_type == 'convnextv2':
                target_layer = model.convnext.stages[-1]
                explainer = SimAtt(model, target_layer, target_layers=None)
        
        elif args.explainer == 'simcam':
            if args.model_type != 'densenet121':
                raise NotImplementedError('SimCAM currently only supports DenseNet121')
            model_seq = nn.Sequential(*list(model.children())[0], *list(model.children())[1:])
            explainer = SimCAM(model_seq, model_seq[0], target_layers=["relu"],
                              fc=model_seq[2] if args.embedding_dim else None)
        
        elif args.explainer == 'sbsm':
            explainer = SBSMBatch(model, input_size=(img_size, img_size), gpu_batch=250)
            maskspath = 'masks.npy'
            if not os.path.isfile(maskspath):
                print("Generating masks for SBSM...")
                explainer.generate_masks(window_size=24, stride=5, savepath=maskspath)
            else:
                explainer.load_masks(maskspath)
                print('Masks loaded.')
        
        explainer.to(device)
        
        # Load query image
        query_img = Image.open(args.query_image).convert('RGB')
        query_tensor = transform(query_img).unsqueeze(0).to(device)
        
        # Setup insertion/deletion evaluators
        klen = 51
        ksig = math.sqrt(50)
        kern = gkern(klen, ksig).to(device)
        blur_fn = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)
        
        deletion_metric = CausalMetric(model, 'del', args.step_size, torch.zeros_like, img_size)
        insertion_metric = CausalMetric(model, 'ins', args.step_size, blur_fn, img_size)
        
        # Process each retrieved image
        os.makedirs(args.output_dir, exist_ok=True)
        all_results = []
        
        print(f"\n{'='*70}")
        print(f"[Step 4] Generating saliency and computing metrics...")
        print(f"{'='*70}")
        
        for i, (ret_path, sim) in enumerate(zip(retrieved_paths, similarities), 1):
            print(f"\n{'#'*70}")
            print(f"Processing Retrieved Image {i}/{args.top_k}")
            print(f"{'#'*70}")
            print(f"Retrieved: {os.path.basename(ret_path)}")
            print(f"Similarity: {sim:.4f}")
            
            # Load retrieved image
            ret_img = Image.open(ret_path).convert('RGB')
            ret_tensor = transform(ret_img).unsqueeze(0).to(device)
            
            # Generate saliency
            print(f"\n  Generating saliency map...")
            saliency = generate_saliency(query_tensor, ret_tensor, explainer, args)
            print(f"  Saliency shape: {saliency.shape}")
            print(f"  Saliency range: [{saliency.min():.4f}, {saliency.max():.4f}]")
            print(f"  Saliency std: {saliency.std():.4f}")
            
            # Save saliency
            sal_dir = os.path.join(args.output_dir, f'saliency_rank{i}')
            os.makedirs(sal_dir, exist_ok=True)
            np.save(os.path.join(sal_dir, 'saliency.npy'), saliency)
            
            # Visualize saliency
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(query_img)
            axes[0].set_title('Query')
            axes[0].axis('off')
            axes[1].imshow(ret_img)
            axes[1].set_title(f'Retrieved (Rank {i})\nSim: {sim:.4f}')
            axes[1].axis('off')
            im = axes[2].imshow(saliency, cmap='jet')
            axes[2].set_title('Saliency Map')
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2])
            plt.tight_layout()
            plt.savefig(os.path.join(sal_dir, 'saliency_viz.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Compute deletion metric
            print(f"\n  Computing DELETION metric...")
            del_auc, del_scores, del_zeros = deletion_metric.evaluate(query_tensor, ret_tensor, saliency)
            print(f"    DEL AUC: {del_auc:.4f}")
            print(f"    DEL Start: {del_scores[0]:.4f}, End: {del_scores[-1]:.4f}")
            
            # Compute insertion metric
            print(f"\n  Computing INSERTION metric...")
            ins_auc, ins_scores, ins_zeros = insertion_metric.evaluate(query_tensor, ret_tensor, saliency)
            print(f"    INS AUC: {ins_auc:.4f}")
            print(f"    INS Start: {ins_scores[0]:.4f}, End: {ins_scores[-1]:.4f}")
            
            # Plot curves
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Deletion curve
            x = np.arange(len(del_scores)) / (len(del_scores) - 1)
            axes[0].plot(x, del_scores, 'r-', linewidth=2)
            axes[0].fill_between(x, 0, del_scores, alpha=0.3, color='red')
            axes[0].set_xlim(-0.05, 1.05)
            axes[0].set_ylim(0, 1.05)
            axes[0].set_xlabel('Fraction of pixels removed')
            axes[0].set_ylabel('Cosine Similarity')
            axes[0].set_title(f'Deletion (AUC = {del_auc:.4f})')
            axes[0].grid(True, alpha=0.3)
            
            # Insertion curve
            axes[1].plot(x, ins_scores, 'b-', linewidth=2)
            axes[1].fill_between(x, 0, ins_scores, alpha=0.3, color='blue')
            axes[1].set_xlim(-0.05, 1.05)
            axes[1].set_ylim(0, 1.05)
            axes[1].set_xlabel('Fraction of pixels added')
            axes[1].set_ylabel('Cosine Similarity')
            axes[1].set_title(f'Insertion (AUC = {ins_auc:.4f})')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(sal_dir, 'metrics.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Store results
            all_results.append({
                'rank': i,
                'retrieved_image': os.path.basename(ret_path),
                'similarity': sim,
                'del_auc': del_auc,
                'ins_auc': ins_auc,
                'del_start': del_scores[0],
                'del_end': del_scores[-1],
                'ins_start': ins_scores[0],
                'ins_end': ins_scores[-1],
                'saliency_std': saliency.std()
            })
        
        # Print summary
        print(f"\n{'='*70}")
        print("SUMMARY RESULTS")
        print(f"{'='*70}")
        print(f"Query: {os.path.basename(args.query_image)}")
        print(f"\n{'Rank':<6} {'Retrieved Image':<40} {'Sim':<8} {'DEL':<8} {'INS':<8}")
        print('-' * 70)
        for r in all_results:
            print(f"{r['rank']:<6} {r['retrieved_image']:<40} {r['similarity']:<8.4f} {r['del_auc']:<8.4f} {r['ins_auc']:<8.4f}")
        
        # Calculate averages
        avg_del = np.mean([r['del_auc'] for r in all_results])
        avg_ins = np.mean([r['ins_auc'] for r in all_results])
        
        print(f"\n{'='*70}")
        print(f"Average DEL AUC: {avg_del:.4f}")
        print(f"Average INS AUC: {avg_ins:.4f}")
        print(f"{'='*70}")
        
        # Diagnostic
        print(f"\n{'='*70}")
        print("DIAGNOSTIC ANALYSIS")
        print(f"{'='*70}")
        
        if avg_del > 0.7:
            print(f"⚠️  HIGH DELETION AUC ({avg_del:.4f})")
            print("    Problem: Removing salient pixels doesn't significantly reduce similarity")
        else:
            print(f"✅ GOOD DELETION AUC ({avg_del:.4f})")
            print("    Removing salient pixels effectively reduces similarity")
        
        if avg_ins < 0.7:
            print(f"⚠️  LOW INSERTION AUC ({avg_ins:.4f})")
            print("    Problem: Adding salient pixels doesn't increase similarity much")
        else:
            print(f"✅ GOOD INSERTION AUC ({avg_ins:.4f})")
            print("    Adding salient pixels effectively increases similarity")
        
        # Check saliency quality
        avg_sal_std = np.mean([r['saliency_std'] for r in all_results])
        print(f"\nAverage saliency std: {avg_sal_std:.4f}")
        if avg_sal_std < 0.1:
            print("    ⚠️  Low saliency variance - maps may be nearly uniform!")
        
        print(f"\nAll outputs saved to: {args.output_dir}")
        print(f"  - Saliency visualizations: saliency_rank*/saliency_viz.png")
        print(f"  - Metric curves: saliency_rank*/metrics.png")
        
    finally:
        manager.disconnect()


if __name__ == '__main__':
    main()
