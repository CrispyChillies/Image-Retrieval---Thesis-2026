"""
Ensemble method for combining predictions from multiple models.
Supports ConvNeXtV2 and SwinV2 model ensembling.
"""
import torch
import torch.nn.functional as F


def ensemble_embeddings(embeddings_list, method='average'):
    """
    Ensemble embeddings from multiple models.
    
    Args:
        embeddings_list: List of embedding tensors from different models
        method: Ensemble method ('average', 'concatenate', or 'weighted')
    
    Returns:
        Combined embeddings tensor
    """
    if method == 'average':
        # Average the embeddings and re-normalize
        combined = torch.stack(embeddings_list, dim=0).mean(dim=0)
        combined = F.normalize(combined, dim=1)
        
    elif method == 'concatenate':
        # Concatenate embeddings and normalize
        combined = torch.cat(embeddings_list, dim=1)
        combined = F.normalize(combined, dim=1)
        
    elif method == 'weighted':
        # Weighted average (can be extended to learned weights)
        # Currently using equal weights, but this can be customized
        weights = [1.0 / len(embeddings_list) for _ in embeddings_list]
        combined = sum(w * emb for w, emb in zip(weights, embeddings_list))
        combined = F.normalize(combined, dim=1)
        
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    return combined


def load_ensemble_models(model_class1, model_class2, checkpoint_path1, checkpoint_path2, device):
    """
    Load two models from checkpoints for ensemble.
    
    Args:
        model_class1: First model class (e.g., ConvNeXtV2)
        model_class2: Second model class (e.g., SwinV2)
        checkpoint_path1: Path to first model checkpoint
        checkpoint_path2: Path to second model checkpoint
        device: Device to load models on
    
    Returns:
        Tuple of (model1, model2)
    """
    # Load first model
    model1 = model_class1
    checkpoint1 = torch.load(checkpoint_path1, map_location=device)
    if 'state_dict' in checkpoint1:
        checkpoint1 = checkpoint1['state_dict']
    model1.load_state_dict(checkpoint1, strict=False)
    model1.to(device)
    model1.eval()
    
    # Load second model
    model2 = model_class2
    checkpoint2 = torch.load(checkpoint_path2, map_location=device)
    if 'state_dict' in checkpoint2:
        checkpoint2 = checkpoint2['state_dict']
    model2.load_state_dict(checkpoint2, strict=False)
    model2.to(device)
    model2.eval()
    
    return model1, model2


@torch.no_grad()
def get_ensemble_embeddings(models, data_loader, device, ensemble_method='average'):
    """
    Extract embeddings from multiple models and ensemble them.
    
    Args:
        models: List of models
        data_loader: DataLoader for the dataset
        device: Device to run inference on
        ensemble_method: Method to combine embeddings
    
    Returns:
        Tuple of (ensemble_embeddings, labels)
    """
    # Set all models to eval mode
    for model in models:
        model.eval()
    
    all_embeddings = [[] for _ in models]
    all_labels = []
    
    for data in data_loader:
        samples = data[0].to(device)
        labels = data[1].to(device)
        
        # Get embeddings from each model
        for i, model in enumerate(models):
            embeddings = model(samples)
            all_embeddings[i].append(embeddings)
        
        all_labels.append(labels)
    
    # Concatenate embeddings from all batches
    for i in range(len(models)):
        all_embeddings[i] = torch.cat(all_embeddings[i], dim=0)
    
    all_labels = torch.cat(all_labels, dim=0)
    
    # Ensemble embeddings batch by batch to save memory
    batch_size = 1000
    num_samples = all_embeddings[0].size(0)
    ensemble_results = []
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_embeddings = [emb[start_idx:end_idx] for emb in all_embeddings]
        ensemble_batch = ensemble_embeddings(batch_embeddings, method=ensemble_method)
        ensemble_results.append(ensemble_batch)
    
    ensemble_embeddings = torch.cat(ensemble_results, dim=0)
    
    return ensemble_embeddings, all_labels