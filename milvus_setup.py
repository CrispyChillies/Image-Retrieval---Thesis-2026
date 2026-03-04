"""
Milvus/Zilliz Setup and Management for Multiple Models
Supports: DenseNet121 (1024-d), ResNet50 (2048-d), ConvNeXtV2 (1024-d)
"""

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import os


# Model configurations
MODEL_CONFIGS = {
    'densenet121': {
        'embedding_dim': 1024,
        'collection_name': 'image_retrieval_densenet121',
        'description': 'DenseNet121 image embeddings'
    },
    'resnet50': {
        'embedding_dim': 2048,
        'collection_name': 'image_retrieval_resnet50',
        'description': 'ResNet50 image embeddings'
    },
    'convnextv2': {
        'embedding_dim': 1024,
        'collection_name': 'image_retrieval_convnextv2',
        'description': 'ConvNeXtV2 image embeddings'
    }
}


class MilvusManager:
    """Manager for Milvus/Zilliz vector database"""
    
    def __init__(self, uri=None, token=None, user='', password=''):
        """
        Initialize Milvus connection
        
        Args:
            uri: Zilliz Cloud URI (e.g., 'https://your-cluster.api.gcp-us-west1.zillizcloud.com')
            token: Zilliz Cloud API token
            user: Username for self-hosted Milvus
            password: Password for self-hosted Milvus
        """
        self.uri = uri or os.getenv('ZILLIZ_URI')
        self.token = token or os.getenv('ZILLIZ_TOKEN')
        self.user = user
        self.password = password
        self.collections = {}
        
    def connect(self):
        """Connect to Milvus/Zilliz"""
        try:
            if self.uri and self.token:
                # Zilliz Cloud connection
                print(f"Connecting to Zilliz Cloud...")
                connections.connect(
                    alias="default",
                    uri=self.uri,
                    token=self.token
                )
            else:
                # Local Milvus connection
                print(f"Connecting to local Milvus...")
                connections.connect(
                    alias="default",
                    host='localhost',
                    port='19530',
                    user=self.user,
                    password=self.password
                )
            print("✅ Connected to Milvus successfully")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Milvus"""
        connections.disconnect("default")
        print("Disconnected from Milvus")
    
    def create_collection(self, model_type, drop_old=False):
        """
        Create collection for a specific model
        
        Args:
            model_type: 'densenet121', 'resnet50', or 'convnextv2'
            drop_old: Whether to drop existing collection
        """
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}")
        
        config = MODEL_CONFIGS[model_type]
        collection_name = config['collection_name']
        embedding_dim = config['embedding_dim']
        
        # Drop old collection if exists
        if drop_old and utility.has_collection(collection_name):
            print(f"Dropping existing collection: {collection_name}")
            utility.drop_collection(collection_name)
        
        # Check if collection already exists
        if utility.has_collection(collection_name):
            print(f"Collection {collection_name} already exists, loading...")
            collection = Collection(collection_name)
            self.collections[model_type] = collection
            return collection
        
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description=config['description']
        )
        
        # Create collection
        print(f"Creating collection: {collection_name}")
        collection = Collection(
            name=collection_name,
            schema=schema,
            using='default'
        )
        
        print(f"✅ Collection created: {collection_name}")
        print(f"   - Embedding dimension: {embedding_dim}")
        print(f"   - Fields: id, image_path, label, embedding")
        
        self.collections[model_type] = collection
        return collection
    
    def create_index(self, model_type, index_type='IVF_FLAT', metric_type='COSINE', nlist=1024):
        """
        Create index for collection
        
        Args:
            model_type: 'densenet121', 'resnet50', or 'convnextv2'
            index_type: 'IVF_FLAT', 'IVF_SQ8', 'IVF_PQ', 'HNSW', 'FLAT'
            metric_type: 'COSINE', 'L2', or 'IP'
            nlist: Number of cluster units (for IVF indices)
        """
        if model_type not in self.collections:
            raise ValueError(f"Collection for {model_type} not loaded")
        
        collection = self.collections[model_type]
        
        # Define index parameters
        index_params = {
            'metric_type': metric_type,
            'index_type': index_type,
            'params': {'nlist': nlist} if 'IVF' in index_type else {}
        }
        
        print(f"Creating index for {model_type}...")
        print(f"  Index type: {index_type}")
        print(f"  Metric type: {metric_type}")
        
        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        print(f"✅ Index created for {model_type}")
        return True
    
    def load_collection(self, model_type):
        """Load collection into memory"""
        if model_type not in self.collections:
            config = MODEL_CONFIGS[model_type]
            collection_name = config['collection_name']
            if not utility.has_collection(collection_name):
                raise ValueError(f"Collection {collection_name} does not exist")
            self.collections[model_type] = Collection(collection_name)
        
        collection = self.collections[model_type]
        collection.load()
        print(f"✅ Collection loaded: {model_type}")
        return collection
    
    def get_collection_info(self, model_type):
        """Get collection statistics"""
        if model_type not in self.collections:
            self.load_collection(model_type)
        
        collection = self.collections[model_type]
        
        info = {
            'name': collection.name,
            'num_entities': collection.num_entities,
            'description': collection.description,
            'schema': collection.schema,
        }
        
        return info
    
    def setup_all_collections(self, drop_old=False):
        """Setup collections for all models"""
        print("\n" + "="*70)
        print("SETTING UP ALL COLLECTIONS")
        print("="*70)
        
        for model_type in MODEL_CONFIGS.keys():
            print(f"\n[{model_type.upper()}]")
            collection = self.create_collection(model_type, drop_old=drop_old)
            self.create_index(model_type)
            self.load_collection(model_type)
        
        print("\n" + "="*70)
        print("✅ ALL COLLECTIONS READY")
        print("="*70)


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup Milvus collections')
    parser.add_argument('--uri', type=str, default=None,
                       help='Zilliz Cloud URI (or set ZILLIZ_URI env var)')
    parser.add_argument('--token', type=str, default=None,
                       help='Zilliz Cloud token (or set ZILLIZ_TOKEN env var)')
    parser.add_argument('--drop_old', action='store_true',
                       help='Drop existing collections')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'densenet121', 'resnet50', 'convnextv2'],
                       help='Which model collection to setup')
    
    args = parser.parse_args()
    
    # Create manager
    manager = MilvusManager(uri=args.uri, token=args.token)
    
    # Connect
    if not manager.connect():
        return
    
    try:
        # Setup collections
        if args.model == 'all':
            manager.setup_all_collections(drop_old=args.drop_old)
        else:
            manager.create_collection(args.model, drop_old=args.drop_old)
            manager.create_index(args.model)
            manager.load_collection(args.model)
        
        # Show collection info
        print("\n" + "="*70)
        print("COLLECTION INFORMATION")
        print("="*70)
        
        models = MODEL_CONFIGS.keys() if args.model == 'all' else [args.model]
        for model_type in models:
            info = manager.get_collection_info(model_type)
            print(f"\n{model_type.upper()}:")
            print(f"  Collection: {info['name']}")
            print(f"  Entities: {info['num_entities']}")
            print(f"  Description: {info['description']}")
    
    finally:
        manager.disconnect()


if __name__ == '__main__':
    main()
