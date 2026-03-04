"""
Quick test to verify Milvus similarity calculation is working correctly
"""

import sys
sys.path.append('.')
from milvus_setup import MilvusManager
from milvus_retrieval import MilvusRetriever, get_model_and_transform
import torch
import argparse


def test_retrieval():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_image', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='densenet121')
    parser.add_argument('--model_weights', type=str, required=True)
    parser.add_argument('--uri', type=str, required=True)
    parser.add_argument('--token', type=str, required=True)
    parser.add_argument('--metric_type', type=str, default='COSINE')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Connect to Milvus
    print("\n" + "="*70)
    print("CONNECTING TO MILVUS")
    print("="*70)
    manager = MilvusManager(uri=args.uri, token=args.token)
    if not manager.connect():
        print("❌ Failed to connect")
        return
    
    # Load model
    print(f"\nLoading {args.model_type}...")
    model, transform = get_model_and_transform(
        args.model_type, args.model_weights, None, device
    )
    
    # Create retriever
    print("Creating retriever...")
    retriever = MilvusRetriever(manager, args.model_type, model, transform)
    retriever.load_collection()
    
    # Search
    print(f"\n" + "="*70)
    print(f"SEARCHING WITH METRIC: {args.metric_type}")
    print("="*70)
    print(f"Query: {args.query_image}")
    
    results, query_emb = retriever.search(
        args.query_image, 
        top_k=5,
        metric_type=args.metric_type
    )
    
    print(f"\n{'Rank':<6} {'Image':<50} {'Distance':<12} {'Similarity':<12}")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        import os
        image_name = os.path.basename(result['image_path'])
        print(f"{i:<6} {image_name:<50} {result['distance']:<12.6f} {result['similarity']:<12.6f}")
    
    # Verify conversion
    print(f"\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    print(f"For COSINE metric:")
    print(f"  Distance = 1 - cosine_similarity")
    print(f"  Similarity = 1 - distance")
    print(f"\nExample from rank 1:")
    dist = results[0]['distance']
    sim = results[0]['similarity']
    expected_sim = 1.0 - dist
    print(f"  Distance from Milvus: {dist:.6f}")
    print(f"  Calculated similarity: {sim:.6f}")
    print(f"  Expected similarity: {expected_sim:.6f}")
    print(f"  Match: {'✓' if abs(sim - expected_sim) < 1e-5 else '✗'}")
    
    if sim > 0.9:
        print(f"\n✅ CORRECT: Similarity values are in expected range (0.9+)")
    elif sim < 0.1:
        print(f"\n❌ ERROR: Similarity values are too low ({sim:.4f})")
        print(f"   This suggests distances are not being converted properly")
        print(f"   Raw distance: {dist:.6f}")
    else:
        print(f"\n⚠️  WARNING: Similarity {sim:.4f} is unusual")
    
    manager.disconnect()


if __name__ == '__main__':
    test_retrieval()
