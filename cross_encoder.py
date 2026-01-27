"""
Cross-Encoder Re-ranking Module
Uses pre-trained embedding model for initial retrieval,
then re-ranks top-K candidates using joint encoding.

No retraining required - works with existing checkpoints.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEncoderReranker(nn.Module):
    """
    Cross-encoder for re-ranking retrieved images.
    Processes query-candidate pairs jointly for more accurate similarity.
    
    Architecture:
    - Uses frozen pre-trained backbone for feature extraction
    - Concatenates query and candidate features
    - Passes through MLP to compute pairwise similarity score
    """
    def __init__(self, backbone_model, embedding_dim=1024, hidden_dim=512):
        super(CrossEncoderReranker, self).__init__()
        
        # Use pre-trained backbone (frozen)
        self.backbone = backbone_model
        
        # Freeze backbone to use pre-trained features
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Cross-encoder head for joint encoding
        # Takes concatenated features from query and candidate
        self.cross_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)  # Single similarity score
        )
        
    def forward_pairs(self, query_imgs, candidate_imgs):
        """
        Compute cross-encoder scores for query-candidate pairs.
        
        Args:
            query_imgs: Query images [B, C, H, W]
            candidate_imgs: Candidate images [B, C, H, W]
            
        Returns:
            Similarity scores [B]
        """
        # Extract features using frozen backbone
        with torch.no_grad():
            query_features = self.backbone(query_imgs)
            candidate_features = self.backbone(candidate_imgs)
        
        # Concatenate features for joint encoding
        joint_features = torch.cat([query_features, candidate_features], dim=1)
        
        # Compute similarity score via cross-encoder MLP
        scores = self.cross_encoder(joint_features).squeeze(-1)
        
        return scores
    
    def rerank_top_k(self, query_img, candidate_imgs, initial_scores, top_k=20, batch_size=32):
        """
        Re-rank top-K candidates using cross-encoder.
        Processes in batches for memory efficiency.
        
        Args:
            query_img: Single query image [1, C, H, W]
            candidate_imgs: All candidate images [N, C, H, W]
            initial_scores: Initial similarity scores from embedding retrieval [N]
            top_k: Number of candidates to re-rank
            batch_size: Batch size for processing pairs
            
        Returns:
            reranked_indices: Indices of re-ranked candidates in original order
            reranked_scores: Cross-encoder scores for re-ranked candidates
        """
        device = query_img.device
        
        # Select top-K candidates from initial retrieval
        _, topk_indices = torch.topk(initial_scores, min(top_k, len(initial_scores)))
        topk_candidates = candidate_imgs[topk_indices]
        
        # Process in batches to avoid OOM
        cross_scores = []
        num_candidates = len(topk_candidates)
        
        for i in range(0, num_candidates, batch_size):
            end_idx = min(i + batch_size, num_candidates)
            batch_candidates = topk_candidates[i:end_idx]
            
            # Expand query to match batch size
            query_batch = query_img.expand(len(batch_candidates), -1, -1, -1)
            
            # Compute cross-encoder scores for this batch
            batch_scores = self.forward_pairs(query_batch, batch_candidates)
            cross_scores.append(batch_scores)
        
        cross_scores = torch.cat(cross_scores)
        
        # Re-rank based on cross-encoder scores
        reranked_indices_local = torch.argsort(cross_scores, descending=True)
        reranked_topk_indices = topk_indices[reranked_indices_local]
        reranked_scores = cross_scores[reranked_indices_local]
        
        return reranked_topk_indices, reranked_scores, topk_indices


def create_cross_encoder_from_checkpoint(checkpoint_path, model_class, embedding_dim=1024, **model_kwargs):
    """
    Create cross-encoder re-ranker from existing checkpoint.
    Uses pre-trained backbone without any additional training.
    
    Args:
        checkpoint_path: Path to pre-trained model checkpoint
        model_class: Model class (e.g., HybridConvNeXtViT, ConvNeXtV2)
        embedding_dim: Embedding dimension of the model
        **model_kwargs: Additional model arguments
        
    Returns:
        CrossEncoderReranker with frozen backbone
    """
    # Load pre-trained backbone
    backbone = model_class(embedding_dim=embedding_dim, **model_kwargs)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    
    backbone.load_state_dict(checkpoint, strict=False)
    
    # Create cross-encoder with frozen backbone
    cross_encoder = CrossEncoderReranker(backbone, embedding_dim)
    
    return cross_encoder
