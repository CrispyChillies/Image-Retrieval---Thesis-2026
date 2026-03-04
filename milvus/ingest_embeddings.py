"""
Ingest image embeddings into Milvus collections
Supports batch processing for efficient insertion
"""

import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ..model import DenseNet121, ResNet50, ConvNeXtV2
from milvus_setup import MilvusManager, MODEL_CONFIGS
from tqdm import tqdm
import argparse


def get_model_and_transform(model_type, model_weights, embedding_dim, device):
    """Load model and get appropriate transform"""
    
    # Load model
    if model_type == 'densenet121':
        model = DenseNet121(embedding_dim=embedding_dim)
        img_size = 224
    elif model_type == 'resnet50':
        model = ResNet50(embedding_dim=embedding_dim)
        img_size = 224
    elif model_type == 'convnextv2':
        model = ConvNeXtV2(embedding_dim=embedding_dim)
        img_size = 384
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    checkpoint = torch.load(model_weights, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    model.to(device)
    
    # Setup transform
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.Resize(256 if img_size == 224 else img_size),
        transforms.CenterCrop(img_size) if img_size == 224 else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        normalize
    ])
    
    return model, transform


def load_image_list(image_list_file, data_dir):
    """Load image paths and labels from file"""
    images = []
    
    with open(image_list_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                image_path = os.path.join(data_dir, parts[1])
                label = parts[2] if len(parts) > 2 else 'unknown'
                images.append((image_path, label))
    
    return images


def compute_embeddings_batch(model, image_paths, labels, transform, device, batch_size=32):
    """Compute embeddings for a batch of images"""
    all_embeddings = []
    all_paths = []
    all_labels = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        batch_tensors = []
        valid_paths = []
        valid_labels = []
        
        for img_path, label in zip(batch_paths, batch_labels):
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                batch_tensors.append(img_tensor)
                valid_paths.append(img_path)
                valid_labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        if batch_tensors:
            batch = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                embeddings = model(batch)
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_paths.extend(valid_paths)
            all_labels.extend(valid_labels)
    
    import numpy as np
    return np.concatenate(all_embeddings, axis=0), all_paths, all_labels


def ingest_embeddings(manager, model_type, embeddings, image_paths, labels, batch_size=100):
    """Insert embeddings into Milvus collection"""
    
    if model_type not in manager.collections:
        manager.load_collection(model_type)
    
    collection = manager.collections[model_type]
    
    total = len(embeddings)
    print(f"\nInserting {total} embeddings into {model_type} collection...")
    
    for i in tqdm(range(0, total, batch_size), desc="Ingesting"):
        batch_end = min(i + batch_size, total)
        
        batch_data = [
            image_paths[i:batch_end],           # image_path
            labels[i:batch_end],                # label
            embeddings[i:batch_end].tolist()    # embedding
        ]
        
        collection.insert(batch_data)
    
    # Flush to persist data
    collection.flush()
    print(f"✅ Inserted {total} embeddings")
    
    return total


def main():
    parser = argparse.ArgumentParser(description='Ingest image embeddings into Milvus')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['densenet121', 'resnet50', 'convnextv2'],
                       help='Model type to use')
    parser.add_argument('--model_weights', type=str, required=True,
                       help='Path to model weights')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--image_list', type=str, required=True,
                       help='Text file with image list (format: idx filename label)')
    parser.add_argument('--embedding_dim', type=int, default=None,
                       help='Custom embedding dimension if using projection')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for embedding computation')
    parser.add_argument('--insert_batch_size', type=int, default=100,
                       help='Batch size for Milvus insertion')
    parser.add_argument('--uri', type=str, default=None,
                       help='Zilliz Cloud URI')
    parser.add_argument('--token', type=str, default=None,
                       help='Zilliz Cloud token')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Connect to Milvus
    print("\n" + "="*70)
    print("CONNECTING TO MILVUS")
    print("="*70)
    manager = MilvusManager(uri=args.uri, token=args.token)
    if not manager.connect():
        return
    
    try:
        # Ensure collection exists and is loaded
        print("\n" + "="*70)
        print("PREPARING COLLECTION")
        print("="*70)
        manager.create_collection(args.model_type, drop_old=False)
        manager.create_index(args.model_type)
        manager.load_collection(args.model_type)
        
        # Load model
        print("\n" + "="*70)
        print("LOADING MODEL")
        print("="*70)
        print(f"Model: {args.model_type}")
        print(f"Weights: {args.model_weights}")
        model, transform = get_model_and_transform(
            args.model_type, args.model_weights, args.embedding_dim, device
        )
        print("✅ Model loaded")
        
        # Load image list
        print("\n" + "="*70)
        print("LOADING IMAGE LIST")
        print("="*70)
        image_data = load_image_list(args.image_list, args.data_dir)
        image_paths = [path for path, _ in image_data]
        labels = [label for _, label in image_data]
        print(f"Found {len(image_paths)} images")
        
        # Compute embeddings
        print("\n" + "="*70)
        print("COMPUTING EMBEDDINGS")
        print("="*70)
        embeddings, valid_paths, valid_labels = compute_embeddings_batch(
            model, image_paths, labels, transform, device, args.batch_size
        )
        print(f"✅ Computed {len(embeddings)} embeddings")
        print(f"   Embedding shape: {embeddings.shape}")
        
        # Ingest into Milvus
        print("\n" + "="*70)
        print("INGESTING INTO MILVUS")
        print("="*70)
        total_inserted = ingest_embeddings(
            manager, args.model_type, embeddings, 
            valid_paths, valid_labels, args.insert_batch_size
        )
        
        # Show collection info
        info = manager.get_collection_info(args.model_type)
        print("\n" + "="*70)
        print("COLLECTION INFO")
        print("="*70)
        print(f"Collection: {info['name']}")
        print(f"Total entities: {info['num_entities']}")
        print(f"Embeddings ingested: {total_inserted}")
        
        print("\n✅ INGESTION COMPLETE!")
        
    finally:
        manager.disconnect()


if __name__ == '__main__':
    main()
