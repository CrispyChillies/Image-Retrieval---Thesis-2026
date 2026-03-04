"""
Generate saliency map for a single query image (or query-retrieved pair).
This creates the saliency map needed for the debug script.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from model import ConvNeXtV2, DenseNet121, ResNet50
from explanations import SimAtt, SimCAM, SBSMBatch
import argparse


def main():
    parser = argparse.ArgumentParser(description='Generate saliency map for a single image')
    parser.add_argument('--query_image', type=str, required=True,
                       help='Path to query image')
    parser.add_argument('--retrieved_image', type=str, default=None,
                       help='Path to retrieved image (if None, uses query for self-saliency)')
    parser.add_argument('--model_type', type=str, default='convnextv2',
                       choices=['densenet121', 'resnet50', 'convnextv2'],
                       help='Model architecture')
    parser.add_argument('--model_weights', type=str, required=True,
                       help='Path to model weights')
    parser.add_argument('--embedding_dim', type=int, default=None,
                       help='Embedding dimension (None for no projection)')
    parser.add_argument('--explainer', type=str, default='simatt',
                       choices=['simatt', 'simcam', 'sbsm'],
                       help='Explanation method')
    parser.add_argument('--output_path', type=str, default='./saliency_map.npy',
                       help='Output path for saliency map (.npy file)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading {args.model_type} model...")
    if args.model_type == 'densenet121':
        model = DenseNet121(embedding_dim=args.embedding_dim)
        img_size = 224
    elif args.model_type == 'resnet50':
        model = ResNet50(embedding_dim=args.embedding_dim)
        img_size = 224
    elif args.model_type == 'convnextv2':
        model = ConvNeXtV2(embedding_dim=args.embedding_dim)
        img_size = 384
    
    if os.path.exists(args.model_weights):
        checkpoint = torch.load(args.model_weights, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded weights from: {args.model_weights}")
    else:
        raise FileNotFoundError(f"Model weights not found: {args.model_weights}")
    
    model.eval()
    
    # Setup explainer
    print(f"\nSetting up {args.explainer} explainer...")
    if args.explainer == 'simatt':
        if args.model_type == 'densenet121':
            model_seq = nn.Sequential(*list(model.children()))
            explainer = SimAtt(model_seq, model_seq[0], target_layers=["relu"])
        elif args.model_type == 'resnet50':
            target_layer = model.resnet50[7][-1].conv3
            explainer = SimAtt(model, target_layer, target_layers=None)
        elif args.model_type == 'convnextv2':
            # For ConvNeXtV2, use the last stage
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
    
    explainer = explainer.to(device)
    
    # Prepare transforms
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Load and transform images
    print(f"\nLoading query image from: {args.query_image}")
    query_img = Image.open(args.query_image).convert('RGB')
    query_tensor = transform(query_img).unsqueeze(0).to(device)
    
    if args.retrieved_image and os.path.exists(args.retrieved_image):
        print(f"Loading retrieved image from: {args.retrieved_image}")
        retrieved_img = Image.open(args.retrieved_image).convert('RGB')
        retrieved_tensor = transform(retrieved_img).unsqueeze(0).to(device)
    else:
        print("Using query image as retrieved image (self-saliency)")
        retrieved_tensor = query_tensor.clone()
    
    # Generate saliency map
    print(f"\nGenerating saliency map using {args.explainer}...")
    with torch.set_grad_enabled(args.explainer != 'sbsm'):
        if args.explainer == 'sbsm':
            saliency = explainer(retrieved_tensor)
        else:
            saliency = explainer(query_tensor, retrieved_tensor)
    
    # Convert to numpy
    saliency = saliency.squeeze().cpu().numpy()
    
    print(f"Saliency map shape: {saliency.shape}")
    print(f"Saliency map range: [{saliency.min():.4f}, {saliency.max():.4f}]")
    
    # Save saliency map
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    np.save(args.output_path, saliency)
    print(f"\nSaliency map saved to: {args.output_path}")
    
    # Also save a visualization
    import matplotlib.pyplot as plt
    viz_path = args.output_path.replace('.npy', '_visualization.png')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original query image
    axes[0].imshow(query_img)
    axes[0].set_title('Query Image')
    axes[0].axis('off')
    
    # Retrieved image (or query if self-saliency)
    if args.retrieved_image:
        axes[1].imshow(retrieved_img)
        axes[1].set_title('Retrieved Image')
    else:
        axes[1].imshow(query_img)
        axes[1].set_title('Query Image (Self-saliency)')
    axes[1].axis('off')
    
    # Saliency map
    im = axes[2].imshow(saliency, cmap='jet')
    axes[2].set_title('Saliency Map')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {viz_path}")
    
    print("\n" + "="*60)
    print("Success! You can now use this saliency map with:")
    print(f"  --saliency_map {args.output_path}")
    print("="*60)


if __name__ == '__main__':
    main()
