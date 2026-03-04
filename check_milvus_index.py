"""
Diagnostic script to check Milvus index configuration
Helps identify the metric type used in your Milvus collections
"""

import argparse
from pymilvus import Collection, connections
from milvus_setup import MilvusManager, MODEL_CONFIGS


def check_index_info(collection_name, uri=None, token=None):
    """Check index configuration for a collection"""
    
    # Connect
    manager = MilvusManager(uri=uri, token=token)
    if not manager.connect():
        print("❌ Failed to connect to Milvus")
        return
    
    try:
        # Get collection
        collection = Collection(collection_name)
        
        print(f"\n{'='*70}")
        print(f"Collection: {collection_name}")
        print(f"{'='*70}")
        
        # Get collection info
        print(f"\n📊 Collection Info:")
        print(f"  Name: {collection.name}")
        print(f"  Description: {collection.description}")
        print(f"  Number of entities: {collection.num_entities}")
        
        # Get schema info
        print(f"\n📋 Schema:")
        for field in collection.schema.fields:
            if field.dtype.name == 'FLOAT_VECTOR':
                print(f"  {field.name}: {field.dtype.name} (dim={field.params.get('dim', 'N/A')})")
            else:
                print(f"  {field.name}: {field.dtype.name}")
        
        # Get index info
        print(f"\n🔍 Index Configuration:")
        indexes = collection.indexes
        
        if not indexes:
            print("  ⚠️  No index found!")
            print("  You need to create an index before searching.")
        else:
            for idx in indexes:
                print(f"  Field: {idx.field_name}")
                print(f"  Index Type: {idx.params.get('index_type', 'N/A')}")
                print(f"  Metric Type: {idx.params.get('metric_type', 'N/A')}")
                
                metric_type = idx.params.get('metric_type', 'UNKNOWN')
                
                # Provide guidance based on metric
                print(f"\n  💡 Usage Guidance:")
                if metric_type == 'COSINE':
                    print(f"     Your index uses COSINE similarity")
                    print(f"     ✓ Distance values: 0 (most similar) to 2 (least similar)")
                    print(f"     ✓ Similarity = 1 - distance")
                    print(f"     ✓ Run with: --metric_type COSINE")
                elif metric_type == 'L2':
                    print(f"     Your index uses L2 (Euclidean) distance")
                    print(f"     ✓ Distance values: 0 (identical) to ∞")
                    print(f"     ✓ For normalized vectors: similarity = 1 - (distance²/2)")
                    print(f"     ⚠️  Run with: --metric_type L2")
                    print(f"     ⚠️  Your similarity values will be LOW (~0.03)")
                    print(f"     ⚠️  Consider recreating index with COSINE metric")
                elif metric_type == 'IP':
                    print(f"     Your index uses IP (Inner Product)")
                    print(f"     ✓ For normalized vectors: IP = cosine similarity")
                    print(f"     ✓ Distance values: -1 to 1")
                    print(f"     ✓ Run with: --metric_type IP")
                else:
                    print(f"     ⚠️  Unknown metric type: {metric_type}")
                
                # Show other params
                other_params = {k: v for k, v in idx.params.items() 
                              if k not in ['index_type', 'metric_type']}
                if other_params:
                    print(f"\n  Other parameters:")
                    for k, v in other_params.items():
                        print(f"     {k}: {v}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        manager.disconnect()


def main():
    parser = argparse.ArgumentParser(description='Check Milvus index configuration')
    parser.add_argument('--model_type', type=str, default='all',
                       choices=['densenet121', 'resnet50', 'convnextv2', 'all'],
                       help='Model type to check')
    parser.add_argument('--uri', type=str, default=None,
                       help='Zilliz Cloud URI')
    parser.add_argument('--token', type=str, default=None,
                       help='Zilliz Cloud token')
    
    args = parser.parse_args()
    
    if args.model_type == 'all':
        model_types = ['densenet121', 'resnet50', 'convnextv2']
    else:
        model_types = [args.model_type]
    
    for model_type in model_types:
        config = MODEL_CONFIGS.get(model_type)
        if config:
            collection_name = config['collection_name']
            check_index_info(collection_name, args.uri, args.token)
            print("\n")


if __name__ == '__main__':
    main()
