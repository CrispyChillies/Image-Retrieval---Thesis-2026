"""
Fast retrieval using Milvus vector database
Replaces slow embedding-based retrieval in debug_single_image_pipeline.py
"""

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pymilvus import Collection
from milvus_setup import MilvusManager, MODEL_CONFIGS
import numpy as np


class MilvusRetriever:
    """Fast image retrieval using Milvus"""
    
    def __init__(self, manager, model_type, model, transform):
        """
        Args:
            manager: MilvusManager instance
            model_type: 'densenet121', 'resnet50', or 'convnextv2'
            model: Pre-loaded PyTorch model
            transform: Image transform pipeline
        """
        self.manager = manager
        self.model_type = model_type
        self.model = model
        self.transform = transform
        self.collection = None
        
    def load_collection(self):
        """Load collection into memory"""
        if self.collection is None:
            self.manager.load_collection(self.model_type)
            self.collection = self.manager.collections[self.model_type]
        return self.collection
    
    def search(self, query_image_path, top_k=10, search_params=None, metric_type='COSINE'):
        """
        Search for similar images
        
        Args:
            query_image_path: Path to query image or PIL Image
            top_k: Number of results to return
            search_params: Optional search parameters
            metric_type: 'COSINE', 'L2', or 'IP' - should match index metric
            
        Returns:
            results: List of dicts with keys: 'image_path', 'label', 'distance', 'similarity'
        """
        # Load image and compute embedding
        if isinstance(query_image_path, str):
            img = Image.open(query_image_path).convert('RGB')
        else:
            img = query_image_path
        
        img_tensor = self.transform(img).unsqueeze(0).to(self.model.device if hasattr(self.model, 'device') else 'cuda')
        
        with torch.no_grad():
            query_embedding = self.model(img_tensor)
            # Always normalize for consistent behavior
            query_embedding = F.normalize(query_embedding, p=2, dim=1)
        
        # Convert to list for Milvus
        query_vector = query_embedding.cpu().numpy().tolist()[0]
        
        # Default search params
        if search_params is None:
            search_params = {
                "metric_type": metric_type,
                "params": {"nprobe": 10}
            }
        
        # Ensure collection is loaded
        if self.collection is None:
            self.load_collection()
        
        # Search in Milvus
        search_results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["image_path", "label"]
        )
        
        # Format results and convert distance to similarity
        results = []
        for hits in search_results:
            for hit in hits:
                distance = hit.distance
                
                # Convert distance to similarity based on metric type
                if metric_type == 'COSINE':
                    # For COSINE: distance = 1 - cosine_similarity
                    # So: cosine_similarity = 1 - distance
                    similarity = distance
                elif metric_type == 'IP':
                    # For IP (Inner Product) with normalized vectors: IP = cosine_similarity
                    similarity = distance
                elif metric_type == 'L2':
                    # For L2: smaller distance = more similar
                    # Convert to similarity: exp(-distance) or 1/(1+distance)
                    # Since L2 dist for normalized vectors: sqrt(2 - 2*cos_sim)
                    # We can recover: cos_sim = 1 - (L2^2)/2
                    similarity = 1.0 - (distance * distance) / 2.0
                else:
                    similarity = None
                
                result = {
                    'id': hit.id,
                    'image_path': hit.entity.get('image_path'),
                    'label': hit.entity.get('label'),
                    'distance': distance,
                    'similarity': similarity
                }
                results.append(result)
        
        return results, query_embedding
    
    def batch_search(self, query_image_paths, top_k=10, search_params=None):
        """
        Search for multiple query images
        
        Args:
            query_image_paths: List of image paths
            top_k: Number of results per query
            search_params: Optional search parameters
            
        Returns:
            List of results for each query
        """
        all_results = []
        
        for query_path in query_image_paths:
            results, _ = self.search(query_path, top_k, search_params)
            all_results.append(results)
        
        return all_results


def get_model_and_transform(model_type, model_weights, embedding_dim, device):
    """Load model and get appropriate transform"""
    from model import DenseNet121, ResNet50, ConvNeXtV2, MedSigLIP

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
    elif model_type == 'medsiglip':
        embed_dim = embedding_dim if embedding_dim is not None else 512
        model = MedSigLIP(embed_dim=embed_dim)
        img_size = 448
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    checkpoint = torch.load(model_weights, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    model.to(device)
    
    # For MedSigLIP, verify attention output works (needed for explainability)
    if model_type == 'medsiglip':
        model.ensure_eager_attention()
        if not model.verify_attention_output(device):
            print("WARNING: MedSigLIP attention output may not work correctly. "
                  "Attention rollout explainer may fail.")
    
    # Setup transform
    # MedSigLIP uses SigLIP normalisation; all other models use ImageNet stats.
    if model_type == 'medsiglip':
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    else:
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Always enforce fixed output size so tensors can be stacked and explained safely.
    if img_size == 448:
        resize_size = 512
    elif img_size == 384:
        resize_size = 432
    else:
        resize_size = 256

    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize
    ])
    
    return model, transform


def main():
    """Example usage and testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Milvus retrieval')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['densenet121', 'resnet50', 'convnextv2', 'medsiglip'],
                       help='Model type')
    parser.add_argument('--model_weights', type=str, required=True,
                       help='Path to model weights')
    parser.add_argument('--query_image', type=str, required=True,
                       help='Path to query image')
    parser.add_argument('--embedding_dim', type=int, default=None,
                       help='Custom embedding dimension')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of results to retrieve')
    parser.add_argument('--uri', type=str, default=None,
                       help='Zilliz Cloud URI')
    parser.add_argument('--token', type=str, default=None,
                       help='Zilliz Cloud token')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Connect to Milvus
    print("\nConnecting to Milvus...")
    manager = MilvusManager(uri=args.uri, token=args.token)
    if not manager.connect():
        return
    
    try:
        # Load model
        print(f"\nLoading {args.model_type} model...")
        model, transform = get_model_and_transform(
            args.model_type, args.model_weights, args.embedding_dim, device
        )
        print("✅ Model loaded")
        
        # Create retriever
        print("\nInitializing retriever...")
        retriever = MilvusRetriever(manager, args.model_type, model, transform)
        retriever.load_collection()
        print("✅ Retriever ready")
        
        # Perform search
        print(f"\n{'='*70}")
        print(f"SEARCHING FOR SIMILAR IMAGES")
        print(f"{'='*70}")
        print(f"Query: {args.query_image}")
        print(f"Top-K: {args.top_k}")
        
        results, query_emb = retriever.search(args.query_image, top_k=args.top_k)
        
        print(f"\n{'='*70}")
        print(f"RESULTS")
        print(f"{'='*70}")
        print(f"Query embedding shape: {query_emb.shape}")
        print(f"\nTop-{args.top_k} Retrieved Images:")
        print(f"{'Rank':<6} {'Image Path':<50} {'Label':<15} {'Similarity':<12}")
        print("-" * 85)
        
        for i, result in enumerate(results, 1):
            img_name = result['image_path'].split('/')[-1]
            sim = result['similarity'] if result['similarity'] is not None else result['distance']
            print(f"{i:<6} {img_name:<50} {result['label']:<15} {sim:<12.4f}")
        
        print(f"\n✅ Retrieval complete!")
        
    finally:
        manager.disconnect()


if __name__ == '__main__':
    main()
