"""
Evaluate insertion/deletion metrics on entire test dataset using Milvus
Scales up debug_pipeline_with_milvus.py to process all test images
"""

import os
import sys
import json
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
from tqdm import tqdm
from datetime import datetime

# Add milvus to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'milvus'))
from milvus_setup import MilvusManager
from milvus_retrieval import MilvusRetriever, get_model_and_transform


class CausalMetric():
    """Simplified causal metric for evaluation"""
    
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


def generate_saliency(query_tensor, retrieved_tensor, explainer, explainer_type):
    """Generate saliency map for query-retrieved pair"""
    with torch.set_grad_enabled(explainer_type != 'sbsm'):
        if explainer_type == 'sbsm':
            saliency = explainer(query_tensor, retrieved_tensor)
        else:
            saliency = explainer(query_tensor, retrieved_tensor)
    
    return saliency.squeeze().cpu().numpy()


def load_image_list(image_list_file, data_dir):
    """Load list of test images"""
    images = []
    with open(image_list_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                img_path = os.path.join(data_dir, parts[0])
                if os.path.exists(img_path):
                    images.append({
                        'path': img_path,
                        'filename': parts[0],
                        'label': parts[1] if len(parts) > 1 else 'unknown'
                    })
    return images


def main():
    parser = argparse.ArgumentParser(description='Evaluate test dataset with Milvus')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--image_list', type=str, required=True,
                       help='File containing list of test images')
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
                       help='Number of retrieved images to analyze per query')
    parser.add_argument('--output_dir', type=str, default='./test_evaluation_results',
                       help='Output directory')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output JSON file (default: results_{model}_{explainer}.json)')
    parser.add_argument('--step_size', type=int, default=1000,
                       help='Step size for insertion/deletion')
    parser.add_argument('--uri', type=str, default=None,
                       help='Zilliz Cloud URI')
    parser.add_argument('--token', type=str, default=None,
                       help='Zilliz Cloud token')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--metric_type', type=str, default='COSINE',
                       choices=['COSINE', 'L2', 'IP'],
                       help='Distance metric used in Milvus index')
    parser.add_argument('--save_saliency', action='store_true',
                       help='Save saliency maps (increases storage)')
    parser.add_argument('--save_visualizations', action='store_true',
                       help='Save visualization images (slow, increases storage)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of test images to process (for testing)')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip already processed images')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Determine image size based on model
    if args.model_type == 'convnextv2':
        img_size = 384
    else:
        img_size = 224
    
    # Setup output
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_file = f'results_{args.model_type}_{args.explainer}_{timestamp}.json'
    
    output_path = os.path.join(args.output_dir, args.output_file)
    
    # Load existing results if skip_existing is enabled
    processed_queries = set()
    if args.skip_existing and os.path.exists(output_path):
        print(f"Loading existing results from {output_path}")
        with open(output_path, 'r') as f:
            existing_data = json.load(f)
            processed_queries = set(r['query_image'] for r in existing_data.get('results', []))
        print(f"Found {len(processed_queries)} already processed queries")
    
    # Load test images
    print(f"\n{'='*70}")
    print(f"LOADING TEST DATASET")
    print(f"{'='*70}")
    test_images = load_image_list(args.image_list, args.data_dir)
    
    if args.limit:
        test_images = test_images[:args.limit]
        print(f"Limited to {args.limit} images for testing")
    
    # Filter out already processed
    if args.skip_existing:
        test_images = [img for img in test_images if img['filename'] not in processed_queries]
        print(f"Remaining images to process: {len(test_images)}")
    
    print(f"Total test images to process: {len(test_images)}")
    
    if len(test_images) == 0:
        print("No images to process!")
        return
    
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
        print(f"LOADING MODEL AND EXPLAINER")
        print(f"{'='*70}")
        
        model, transform = get_model_and_transform(
            args.model_type, args.model_weights, args.embedding_dim, device
        )
        print(f"✅ Model loaded: {args.model_type}")
        
        # Create retriever
        retriever = MilvusRetriever(manager, args.model_type, model, transform)
        retriever.load_collection()
        print(f"✅ Retriever ready")
        
        # Setup explainer
        print(f"Setting up {args.explainer} explainer...")
        
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
        print(f"✅ Explainer ready: {args.explainer}")
        
        # Setup insertion/deletion evaluators
        klen = 51
        ksig = math.sqrt(50)
        kern = gkern(klen, ksig).to(device)
        blur_fn = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)
        
        deletion_metric = CausalMetric(model, 'del', args.step_size, torch.zeros_like, img_size)
        insertion_metric = CausalMetric(model, 'ins', args.step_size, blur_fn, img_size)
        
        print(f"✅ Metrics ready (step size: {args.step_size})")
        
        # Process all test images
        print(f"\n{'='*70}")
        print(f"PROCESSING TEST DATASET")
        print(f"{'='*70}")
        
        all_results = []
        
        for test_img_info in tqdm(test_images, desc="Processing queries"):
            query_path = test_img_info['path']
            query_filename = test_img_info['filename']
            
            try:
                # Load query image
                query_img = Image.open(query_path).convert('RGB')
                query_tensor = transform(query_img).unsqueeze(0).to(device)
                
                # Retrieve top-k similar images
                results, query_emb = retriever.search(
                    query_path, 
                    top_k=args.top_k, 
                    metric_type=args.metric_type
                )
                
                query_result = {
                    'query_image': query_filename,
                    'query_label': test_img_info['label'],
                    'model_type': args.model_type,
                    'explainer': args.explainer,
                    'top_k': args.top_k,
                    'retrieved': []
                }
                
                # Process each retrieved image
                for rank, result in enumerate(results, 1):
                    ret_path = result['image_path']
                    similarity = result['similarity']
                    
                    # Load retrieved image
                    ret_img = Image.open(ret_path).convert('RGB')
                    ret_tensor = transform(ret_img).unsqueeze(0).to(device)
                    
                    # Generate saliency
                    saliency = generate_saliency(query_tensor, ret_tensor, explainer, args.explainer)
                    
                    # Compute metrics
                    del_auc, del_scores, del_zeros = deletion_metric.evaluate(
                        query_tensor, ret_tensor, saliency
                    )
                    ins_auc, ins_scores, ins_zeros = insertion_metric.evaluate(
                        query_tensor, ret_tensor, saliency
                    )
                    
                    retrieved_result = {
                        'rank': rank,
                        'retrieved_image': os.path.basename(ret_path),
                        'similarity': float(similarity),
                        'del_auc': float(del_auc),
                        'ins_auc': float(ins_auc),
                        'del_start': float(del_scores[0]),
                        'del_end': float(del_scores[-1]),
                        'ins_start': float(ins_scores[0]),
                        'ins_end': float(ins_scores[-1]),
                        'saliency_std': float(saliency.std()),
                        'saliency_min': float(saliency.min()),
                        'saliency_max': float(saliency.max())
                    }
                    
                    # Optionally save saliency maps
                    if args.save_saliency:
                        sal_dir = os.path.join(args.output_dir, 'saliency_maps', 
                                              query_filename.replace('.', '_'))
                        os.makedirs(sal_dir, exist_ok=True)
                        np.save(os.path.join(sal_dir, f'rank{rank}_saliency.npy'), saliency)
                    
                    # Optionally save visualizations
                    if args.save_visualizations:
                        vis_dir = os.path.join(args.output_dir, 'visualizations',
                                              query_filename.replace('.', '_'))
                        os.makedirs(vis_dir, exist_ok=True)
                        
                        # Saliency visualization
                        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                        axes[0].imshow(query_img)
                        axes[0].set_title('Query')
                        axes[0].axis('off')
                        axes[1].imshow(ret_img)
                        axes[1].set_title(f'Rank {rank} (Sim: {similarity:.4f})')
                        axes[1].axis('off')
                        im = axes[2].imshow(saliency, cmap='jet')
                        axes[2].set_title('Saliency')
                        axes[2].axis('off')
                        plt.colorbar(im, ax=axes[2])
                        plt.tight_layout()
                        plt.savefig(os.path.join(vis_dir, f'rank{rank}_saliency.png'), 
                                   dpi=100, bbox_inches='tight')
                        plt.close()
                        
                        # Metrics curves
                        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                        x = np.arange(len(del_scores)) / (len(del_scores) - 1)
                        
                        axes[0].plot(x, del_scores, 'r-', linewidth=2)
                        axes[0].fill_between(x, 0, del_scores, alpha=0.3, color='red')
                        axes[0].set_title(f'Deletion (AUC={del_auc:.4f})')
                        axes[0].set_xlabel('Fraction removed')
                        axes[0].set_ylabel('Similarity')
                        axes[0].grid(True, alpha=0.3)
                        
                        axes[1].plot(x, ins_scores, 'b-', linewidth=2)
                        axes[1].fill_between(x, 0, ins_scores, alpha=0.3, color='blue')
                        axes[1].set_title(f'Insertion (AUC={ins_auc:.4f})')
                        axes[1].set_xlabel('Fraction added')
                        axes[1].set_ylabel('Similarity')
                        axes[1].grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(vis_dir, f'rank{rank}_metrics.png'),
                                   dpi=100, bbox_inches='tight')
                        plt.close()
                    
                    query_result['retrieved'].append(retrieved_result)
                
                # Compute query-level statistics
                query_result['avg_del_auc'] = float(np.mean([r['del_auc'] for r in query_result['retrieved']]))
                query_result['avg_ins_auc'] = float(np.mean([r['ins_auc'] for r in query_result['retrieved']]))
                query_result['avg_similarity'] = float(np.mean([r['similarity'] for r in query_result['retrieved']]))
                
                all_results.append(query_result)
                
                # Save incrementally (every 10 queries)
                if len(all_results) % 10 == 0:
                    save_results(output_path, all_results, args)
                
            except Exception as e:
                print(f"\n❌ Error processing {query_filename}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Final save
        save_results(output_path, all_results, args)
        
        # Print summary statistics
        print_summary(all_results, args)
        
    finally:
        manager.disconnect()


def save_results(output_path, results, args):
    """Save results to JSON file"""
    output_data = {
        'metadata': {
            'model_type': args.model_type,
            'explainer': args.explainer,
            'top_k': args.top_k,
            'step_size': args.step_size,
            'metric_type': args.metric_type,
            'num_queries': len(results),
            'timestamp': datetime.now().isoformat()
        },
        'results': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)


def print_summary(results, args):
    """Print summary statistics"""
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_type}")
    print(f"  Explainer: {args.explainer}")
    print(f"  Top-K: {args.top_k}")
    print(f"  Total queries processed: {len(results)}")
    
    # Overall statistics
    all_del_aucs = []
    all_ins_aucs = []
    all_sims = []
    
    for query_result in results:
        for retrieved in query_result['retrieved']:
            all_del_aucs.append(retrieved['del_auc'])
            all_ins_aucs.append(retrieved['ins_auc'])
            all_sims.append(retrieved['similarity'])
    
    print(f"\nOverall Statistics (across all {len(all_del_aucs)} query-retrieved pairs):")
    print(f"  Deletion AUC:  {np.mean(all_del_aucs):.4f} ± {np.std(all_del_aucs):.4f}")
    print(f"  Insertion AUC: {np.mean(all_ins_aucs):.4f} ± {np.std(all_ins_aucs):.4f}")
    print(f"  Similarity:    {np.mean(all_sims):.4f} ± {np.std(all_sims):.4f}")
    
    # Per-query averages
    query_del_aucs = [r['avg_del_auc'] for r in results]
    query_ins_aucs = [r['avg_ins_auc'] for r in results]
    
    print(f"\nPer-Query Averages (across {len(results)} queries):")
    print(f"  Deletion AUC:  {np.mean(query_del_aucs):.4f} ± {np.std(query_del_aucs):.4f}")
    print(f"  Insertion AUC: {np.mean(query_ins_aucs):.4f} ± {np.std(query_ins_aucs):.4f}")
    
    # Quality assessment
    print(f"\nQuality Assessment:")
    good_del = sum(1 for auc in all_del_aucs if auc < 0.6)
    good_ins = sum(1 for auc in all_ins_aucs if auc > 0.7)
    
    print(f"  Good deletion (AUC < 0.6): {good_del}/{len(all_del_aucs)} ({100*good_del/len(all_del_aucs):.1f}%)")
    print(f"  Good insertion (AUC > 0.7): {good_ins}/{len(all_ins_aucs)} ({100*good_ins/len(all_ins_aucs):.1f}%)")
    
    if np.mean(all_del_aucs) < 0.6 and np.mean(all_ins_aucs) > 0.7:
        print(f"\n  ✅ GOOD: Saliency maps are high quality")
    elif np.mean(all_del_aucs) > 0.7:
        print(f"\n  ⚠️  WARNING: High deletion AUC suggests weak saliency")
    elif np.mean(all_ins_aucs) < 0.7:
        print(f"\n  ⚠️  WARNING: Low insertion AUC suggests weak saliency")
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {os.path.join(args.output_dir, args.output_file)}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
