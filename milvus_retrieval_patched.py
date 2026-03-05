"""
Modified MilvusRetriever with Kaggle path remapping for local VM usage
Drop-in replacement for milvus_retrieval.py search method
"""

from path_mapper import PathMapper


class MilvusRetrieverPatched:
    """Extends MilvusRetriever to handle Kaggle -> VM path remapping"""
    
    def __init__(self, manager, model_type, model, transform, 
                 local_data_base_path=None, enable_path_mapping=True):
        """
        Args:
            local_data_base_path: e.g., "/media/vhviet03/datasets/covidx-cxr/data/train"
            enable_path_mapping: If True, automatically remap /kaggle/ paths
        """
        self.manager = manager
        self.model_type = model_type
        self.model = model
        self.transform = transform
        self.collection = None
        self.enable_path_mapping = enable_path_mapping
        
        if enable_path_mapping and local_data_base_path:
            self.path_mapper = PathMapper(local_base_path=local_data_base_path)
        else:
            self.path_mapper = None
    
    def _remap_image_path(self, kaggle_path):
        """Convert Kaggle path to local VM path if needed"""
        if not self.path_mapper:
            return kaggle_path
        
        # If path starts with /kaggle/, remap it
        if kaggle_path.startswith('/kaggle/'):
            remapped = self.path_mapper.remap_path(kaggle_path)
            print(f"  Path remapped: {kaggle_path} -> {remapped}")
            return remapped
        
        return kaggle_path
    
    def search(self, query_image_path, top_k=10, search_params=None, metric_type='COSINE'):
        """
        Search for similar images (with Kaggle path remapping)
        
        Args:
            query_image_path: Path to query image
            top_k: Number of results to return
            search_params: Optional search parameters
            metric_type: Distance metric
            
        Returns:
            results: List of dicts with remapped image paths
        """
        import torch
        import torch.nn.functional as F
        from PIL import Image
        
        # Remap query path if needed
        query_image_path = self._remap_image_path(query_image_path)
        
        # Load image and compute embedding
        if isinstance(query_image_path, str):
            img = Image.open(query_image_path).convert('RGB')
        else:
            img = query_image_path
        
        img_tensor = self.transform(img).unsqueeze(0).to(
            self.model.device if hasattr(self.model, 'device') else 'cuda'
        )
        
        with torch.no_grad():
            query_embedding = self.model(img_tensor)
            query_embedding = F.normalize(query_embedding, p=2, dim=1)
        
        query_vector = query_embedding.cpu().numpy().tolist()[0]
        
        if search_params is None:
            search_params = {
                "metric_type": metric_type,
                "params": {"nprobe": 10}
            }
        
        if self.collection is None:
            self.load_collection()
        
        search_results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["image_path", "label"]
        )
        
        # Format results with path remapping
        results = []
        for hits in search_results:
            for hit in hits:
                distance = hit.distance
                image_path = hit.entity.get('image_path')
                
                # Remap retrieved image path
                image_path = self._remap_image_path(image_path)
                
                # Convert distance to similarity
                if metric_type == 'COSINE':
                    similarity = distance
                elif metric_type == 'IP':
                    similarity = distance
                elif metric_type == 'L2':
                    similarity = 1.0 - (distance * distance) / 2.0
                else:
                    similarity = None
                
                result = {
                    'id': hit.id,
                    'image_path': image_path,
                    'label': hit.entity.get('label'),
                    'distance': distance,
                    'similarity': similarity
                }
                results.append(result)
        
        return results, query_embedding
    
    def load_collection(self):
        """Load collection into memory"""
        if self.collection is None:
            self.manager.load_collection(self.model_type)
            self.collection = self.manager.collections[self.model_type]
        return self.collection


# Usage example
if __name__ == '__main__':
    print("""
USAGE IN YOUR EVALUATION SCRIPT:
================================

Instead of:
    retriever = MilvusRetriever(manager, args.model_type, model, transform)

Use:
    from milvus_retrieval_patched import MilvusRetrieverPatched
    retriever = MilvusRetrieverPatched(
        manager, 
        args.model_type, 
        model, 
        transform,
        local_data_base_path="/media/vhviet03/datasets/covidx-cxr/data/train"
    )

Then use retriever.search() as normal - paths will be automatically remapped!
    """)
