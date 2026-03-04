"""
Test script to get embedding dimension for Milvus setup.
This script loads a model and processes a single image to show embedding shape.
"""

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from model import DenseNet121, ResNet50, ConvNeXtV2, SwinV2, MedSigLIP, conceptCLIP
import argparse


def get_embedding_dimension(model_type, model_weights=None, test_image=None, embedding_dim=None):
    """Test embedding dimension for a given model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    print(f"{'='*70}")
    print(f"Loading {model_type} model...")
    print(f"{'='*70}")
    
    if model_type == 'densenet121':
        model = DenseNet121(embedding_dim=embedding_dim)
        img_size = 224
    elif model_type == 'resnet50':
        model = ResNet50(embedding_dim=embedding_dim)
        img_size = 224
    elif model_type == 'convnextv2':
        model = ConvNeXtV2(embedding_dim=embedding_dim)
        img_size = 384
    elif model_type == 'swinv2':
        model = SwinV2(embedding_dim=embedding_dim)
        img_size = 384
    elif model_type == 'medsiglip':
        model = MedSigLIP(embed_dim=embedding_dim if embedding_dim else 512)
        img_size = 448
    elif model_type == 'conceptclip':
        model = conceptCLIP(embedding_dim=embedding_dim)
        img_size = 384
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights if provided
    if model_weights:
        try:
            checkpoint = torch.load(model_weights, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            print(f"✅ Loaded weights from: {model_weights}")
        except Exception as e:
            print(f"⚠️  Could not load weights: {e}")
            print(f"   Using randomly initialized model (for dimension testing only)")
    else:
        print(f"⚠️  No weights provided, using randomly initialized model")
    
    model.eval()
    model.to(device)
    
    # Setup transform
    print(f"\n{'='*70}")
    print(f"Image preprocessing settings:")
    print(f"{'='*70}")
    print(f"Input size: {img_size}x{img_size}")
    
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    if model_type == 'medsiglip':
        # MedSigLIP uses specific preprocessing
        transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.Resize(256 if img_size == 224 else img_size),
            transforms.CenterCrop(img_size) if img_size == 224 else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            normalize
        ])
    
    # Create or load test image
    if test_image:
        try:
            print(f"\nLoading test image: {test_image}")
            img = Image.open(test_image).convert('RGB')
            print(f"✅ Image loaded: {img.size}")
        except Exception as e:
            print(f"⚠️  Could not load image: {e}")
            print(f"   Creating random test image...")
            img = Image.new('RGB', (img_size, img_size), color='red')
    else:
        print(f"\n⚠️  No test image provided, creating random test image...")
        img = Image.new('RGB', (img_size, img_size), color='red')
    
    # Transform and add batch dimension
    img_tensor = transform(img).unsqueeze(0).to(device)
    print(f"Tensor shape: {img_tensor.shape}")
    
    # Get embedding
    print(f"\n{'='*70}")
    print(f"Computing embedding...")
    print(f"{'='*70}")
    
    with torch.no_grad():
        embedding = model(img_tensor)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Model type: {model_type}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding dimension: {embedding.shape[1]}")
    print(f"Embedding dtype: {embedding.dtype}")
    print(f"Embedding device: {embedding.device}")
    print(f"\nEmbedding statistics:")
    print(f"  Min value: {embedding.min().item():.6f}")
    print(f"  Max value: {embedding.max().item():.6f}")
    print(f"  Mean: {embedding.mean().item():.6f}")
    print(f"  Std: {embedding.std().item():.6f}")
    print(f"  L2 norm: {torch.norm(embedding).item():.6f}")
    
    # Milvus configuration
    print(f"\n{'='*70}")
    print(f"MILVUS CONFIGURATION")
    print(f"{'='*70}")
    print(f"For your Milvus schema, use:")
    print(f"")
    print(f"fields = [")
    print(f"    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),")
    print(f"    FieldSchema(name='image_path', dtype=DataType.VARCHAR, max_length=512),")
    print(f"    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim={embedding.shape[1]}),")
    print(f"]")
    print(f"")
    print(f"index_params = {{")
    print(f"    'metric_type': 'COSINE',  # or 'L2' or 'IP'")
    print(f"    'index_type': 'IVF_FLAT',")
    print(f"    'params': {{'nlist': 1024}}")
    print(f"}}")
    
    return embedding.shape[1]


def main():
    parser = argparse.ArgumentParser(description='Test embedding dimension for Milvus setup')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['densenet121', 'resnet50', 'convnextv2', 'swinv2', 'medsiglip', 'conceptclip'],
                       help='Model architecture')
    parser.add_argument('--model_weights', type=str, default=None,
                       help='Path to model weights (optional)')
    parser.add_argument('--test_image', type=str, default=None,
                       help='Path to test image (optional, will create random if not provided)')
    parser.add_argument('--embedding_dim', type=int, default=None,
                       help='Custom embedding dimension if model uses projection layer')
    
    args = parser.parse_args()
    
    dim = get_embedding_dimension(
        model_type=args.model_type,
        model_weights=args.model_weights,
        test_image=args.test_image,
        embedding_dim=args.embedding_dim
    )
    
    print(f"\n✅ Embedding dimension for {args.model_type.upper()}: {dim}")


if __name__ == '__main__':
    main()
