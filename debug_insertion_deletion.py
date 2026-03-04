"""
Debug script to understand insertion/deletion metric calculation for ConvNeXtV2.
This script runs on a single image and shows detailed step-by-step calculation.
"""

import os
import math
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from model import ConvNeXtV2
from evaluation import gkern, auc
import argparse


class DebugCausalMetric():
    """Modified CausalMetric class with detailed debugging output"""

    def __init__(self, model, mode, step, substrate_fn, input_size=384):
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn
        self.hw = input_size * input_size  # image area
        
        print(f"\n{'='*60}")
        print(f"Initialized {mode.upper()} Metric")
        print(f"{'='*60}")
        print(f"Mode: {mode}")
        print(f"Step size: {step} pixels")
        print(f"Image size: {input_size}x{input_size} = {self.hw} pixels")

    def debug_single_run(self, img_tensor, retrieved_tensor, explanation, save_dir='./debug_output'):
        """Debug version with detailed logging"""
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Starting {self.mode.upper()} calculation")
        print(f"{'='*60}")
        
        # Get query feature embedding
        print("\n[Step 1] Computing query image feature...")
        q_feat = self.model(img_tensor.cuda())
        print(f"  Query feature shape: {q_feat.shape}")
        print(f"  Query feature norm: {torch.norm(q_feat).item():.4f}")
        
        # Calculate number of steps
        n_steps = (self.hw + self.step - 1) // self.step
        print(f"\n[Step 2] Calculating number of iterations...")
        print(f"  Total pixels: {self.hw}")
        print(f"  Pixels per step: {self.step}")
        print(f"  Number of steps: {n_steps}")
        
        # Initialize start and finish images
        print(f"\n[Step 3] Initializing start and finish images...")
        if self.mode == 'del':
            print("  DELETION: Start with original, end with blurred")
            start = retrieved_tensor.clone()
            finish = self.substrate_fn(retrieved_tensor)
        elif self.mode == 'ins':
            print("  INSERTION: Start with blurred, end with original")
            start = self.substrate_fn(retrieved_tensor)
            finish = retrieved_tensor.clone()
        
        # Save start and finish images for visualization
        self._save_tensor_image(start[0], os.path.join(save_dir, f'{self.mode}_start.png'))
        self._save_tensor_image(finish[0], os.path.join(save_dir, f'{self.mode}_finish.png'))
        
        # Initialize scores array
        scores = np.empty(n_steps + 1)
        
        # Sort pixels by saliency
        print(f"\n[Step 4] Sorting pixels by saliency...")
        print(f"  Saliency map shape: {explanation.shape}")
        print(f"  Saliency map range: [{explanation.min():.4f}, {explanation.max():.4f}]")
        
        salient_order = np.flip(np.argsort(explanation.flatten())).copy()  # .copy() fixes negative stride
        salient_order = torch.from_numpy(salient_order).unsqueeze(0)
        
        # Show top salient pixels
        top_5_indices = salient_order[0, :5].numpy()
        top_5_values = explanation.flatten()[top_5_indices]
        print(f"  Top 5 most salient pixels:")
        for i, (idx, val) in enumerate(zip(top_5_indices, top_5_values)):
            row, col = idx // explanation.shape[1], idx % explanation.shape[1]
            print(f"    {i+1}. Position ({row}, {col}), Index {idx}, Saliency: {val:.4f}")
        
        # Run iterations
        print(f"\n[Step 5] Running {n_steps+1} iterations...")
        print(f"{'='*60}")
        
        zero_counter = 0
        
        for i in range(n_steps + 1):
            # Compute feature and similarity
            r_feat = self.model(start.cuda())
            cosine_sim = torch.nn.functional.cosine_similarity(q_feat, r_feat)[0]
            
            # Handle negative values
            if cosine_sim < 0:
                cosine_sim = torch.clamp(cosine_sim, min=0, max=1)
                zero_counter += 1
            
            scores[i] = cosine_sim.item()
            
            # Detailed logging for first few and last few steps
            if i < 3 or i > n_steps - 3 or i % (n_steps // 10) == 0:
                pixels_modified = min(i * self.step, self.hw)
                percent_modified = 100 * pixels_modified / self.hw
                print(f"\nIteration {i:3d}/{n_steps}:")
                print(f"  Pixels modified: {pixels_modified:6d} ({percent_modified:5.1f}%)")
                print(f"  Cosine similarity: {scores[i]:.6f}")
                print(f"  Retrieved feature norm: {torch.norm(r_feat).item():.4f}")
                
                # Save intermediate image every 10% or at key points
                if i % max(1, n_steps // 10) == 0 or i == n_steps:
                    img_path = os.path.join(save_dir, f'{self.mode}_step_{i:03d}.png')
                    self._save_tensor_image(start[0], img_path)
            
            # Modify pixels for next iteration
            if i < n_steps:
                # Get pixel coordinates to modify
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                
                # Reshape and modify
                start_reshaped = start.reshape(1, 3, self.hw)
                finish_reshaped = finish.reshape(1, 3, self.hw)
                start_reshaped[0, :, coords] = finish_reshaped[0, :, coords]
                start = start_reshaped.reshape(start.shape)
        
        # Calculate AUC
        auc_score = auc(scores)
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Mode: {self.mode.upper()}")
        print(f"AUC Score: {auc_score:.6f}")
        print(f"Negative values encountered: {zero_counter}")
        print(f"Score at start: {scores[0]:.6f}")
        print(f"Score at end: {scores[-1]:.6f}")
        print(f"Score difference: {scores[-1] - scores[0]:.6f}")
        
        # Plot results
        self._plot_results(scores, save_dir)
        
        return auc_score, zero_counter, scores
    
    def _save_tensor_image(self, tensor, path):
        """Save tensor as image"""
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = tensor.cpu() * std + mean
        img = torch.clamp(img, 0, 1)
        
        # Convert to PIL and save
        img_pil = transforms.ToPILImage()(img)
        img_pil.save(path)
    
    def _plot_results(self, scores, save_dir):
        """Plot the insertion/deletion curve"""
        n_steps = len(scores) - 1
        x = np.arange(len(scores)) / n_steps
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, scores, 'b-', linewidth=2, label=f'{self.mode.upper()} Score')
        plt.fill_between(x, 0, scores, alpha=0.3)
        plt.xlim(-0.05, 1.05)
        plt.ylim(0, 1.05)
        plt.xlabel('Fraction of pixels modified', fontsize=12)
        plt.ylabel('Cosine Similarity', fontsize=12)
        plt.title(f'{self.mode.upper()} Metric (AUC = {auc(scores):.4f})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        # Add annotations
        plt.annotate(f'Start: {scores[0]:.4f}', 
                    xy=(0, scores[0]), xytext=(0.1, scores[0]+0.1),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10)
        plt.annotate(f'End: {scores[-1]:.4f}', 
                    xy=(1, scores[-1]), xytext=(0.85, scores[-1]+0.1),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{self.mode}_curve.png'), dpi=150)
        plt.close()
        print(f"\nPlot saved to: {os.path.join(save_dir, f'{self.mode}_curve.png')}")


def main():
    parser = argparse.ArgumentParser(description='Debug insertion/deletion metrics for ConvNeXtV2')
    parser.add_argument('--query_image', type=str, required=True, 
                       help='Path to query image')
    parser.add_argument('--retrieved_image', type=str, default=None,
                       help='Path to retrieved image (if None, uses query image)')
    parser.add_argument('--saliency_map', type=str, default=None,
                       help='Path to saliency map (.npy file). If None, generates random saliency for demo')
    parser.add_argument('--model_weights', type=str, required=True,
                       help='Path to ConvNeXtV2 model weights')
    parser.add_argument('--embedding_dim', type=int, default=None,
                       help='Embedding dimension (None for no projection)')
    parser.add_argument('--output_dir', type=str, default='./debug_output',
                       help='Directory to save debug outputs')
    parser.add_argument('--step_size', type=int, default=1000,
                       help='Number of pixels to modify per iteration')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading ConvNeXtV2 model...")
    model = ConvNeXtV2(pretrained=False, embedding_dim=args.embedding_dim)
    
    if os.path.exists(args.model_weights):
        checkpoint = torch.load(args.model_weights, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded weights from: {args.model_weights}")
    else:
        print(f"WARNING: Model weights not found at {args.model_weights}")
        print("Using randomly initialized model (for testing only)")
    
    model = model.eval()
    model = model.to(device)
    
    # Prepare transform
    img_size = 384  # ConvNeXtV2 uses 384x384
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load query image
    print(f"\nLoading query image from: {args.query_image}")
    query_img = Image.open(args.query_image).convert('RGB')
    query_tensor = transform(query_img).unsqueeze(0).to(device)
    print(f"Query image shape: {query_tensor.shape}")
    
    # Load retrieved image (or use query image)
    if args.retrieved_image and os.path.exists(args.retrieved_image):
        print(f"Loading retrieved image from: {args.retrieved_image}")
        retrieved_img = Image.open(args.retrieved_image).convert('RGB')
        retrieved_tensor = transform(retrieved_img).unsqueeze(0).to(device)
    else:
        print("Using query image as retrieved image (self-similarity)")
        retrieved_tensor = query_tensor.clone()
    
    # Load or generate saliency map
    if args.saliency_map and os.path.exists(args.saliency_map):
        print(f"\nLoading saliency map from: {args.saliency_map}")
        saliency = np.load(args.saliency_map)
    else:
        print("\nWARNING: No saliency map provided, generating random saliency for demo")
        saliency = np.random.rand(img_size, img_size).astype(np.float32)
    
    print(f"Saliency map shape: {saliency.shape}")
    
    # Ensure saliency map matches image size
    if saliency.shape[0] != img_size or saliency.shape[1] != img_size:
        from scipy.ndimage import zoom
        zoom_factor = img_size / saliency.shape[0]
        saliency = zoom(saliency, zoom_factor, order=1)
        print(f"Resized saliency map to: {saliency.shape}")
    
    # Setup blur function for deletion/insertion
    klen = 51
    ksig = math.sqrt(50)
    kern = gkern(klen, ksig).to(device)
    blur_fn = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)
    
    # Run DELETION metric
    print(f"\n{'#'*60}")
    print("RUNNING DELETION METRIC")
    print(f"{'#'*60}")
    deletion_metric = DebugCausalMetric(
        model, 'del', args.step_size, torch.zeros_like, input_size=img_size
    )
    del_auc, del_zeros, del_scores = deletion_metric.debug_single_run(
        query_tensor, retrieved_tensor, saliency, 
        save_dir=os.path.join(args.output_dir, 'deletion')
    )
    
    # Run INSERTION metric
    print(f"\n{'#'*60}")
    print("RUNNING INSERTION METRIC")
    print(f"{'#'*60}")
    insertion_metric = DebugCausalMetric(
        model, 'ins', args.step_size, blur_fn, input_size=img_size
    )
    ins_auc, ins_zeros, ins_scores = insertion_metric.debug_single_run(
        query_tensor, retrieved_tensor, saliency,
        save_dir=os.path.join(args.output_dir, 'insertion')
    )
    
    # Summary
    print(f"\n{'#'*60}")
    print("SUMMARY")
    print(f"{'#'*60}")
    print(f"Deletion AUC:  {del_auc:.6f}")
    print(f"Insertion AUC: {ins_auc:.6f}")
    print(f"\nAll debug outputs saved to: {args.output_dir}")
    print(f"  - Deletion:  {os.path.join(args.output_dir, 'deletion')}")
    print(f"  - Insertion: {os.path.join(args.output_dir, 'insertion')}")


if __name__ == '__main__':
    main()
