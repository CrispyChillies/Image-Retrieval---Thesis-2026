import os
import numpy as np
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import torch
from torch.utils.data import DataLoader


import torchvision.transforms as transforms
from read_data import ISICDataSet, ChestXrayDataSet, TBX11kDataSet


def conceptclip_collate_fn(batch):
    """Custom collate function for ConceptCLIP that keeps PIL images as-is."""
    images = [item[0] for item in batch]
    labels = torch.stack([item[1] if isinstance(item[1], torch.Tensor) else torch.tensor(item[1]) for item in batch])
    return images, labels

from model import ResNet50, DenseNet121, ConvNeXtV2, SwinV2

try:
    from transformers import AutoModel, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. ConceptCLIP will not be available.")


def retrieval_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.cpu()
        target = target.cpu()
        pred = target[pred].t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].any(dim=0).sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
    return res


# Source: https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/utils/evaluate.py
def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.

    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images

    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.
         Usage: 
           mAP = compute_map (ranks, gnd) 
                 computes mean average precsion (mAP) only

           mAP, aps, pr, prs = compute_map (ranks, gnd, kappas) 
                 computes mean average precision (mAP), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query

         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) If there are no positive images for some query, that query is excluded from the evaluation
    """

    mAP = 0.
    nq = len(gnd)  # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.where(gnd == gnd[i])[0]

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        # sorted positions of positive images (0 based)
        pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgnd)]

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        mAP = mAP + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1  # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j])
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    mAP = mAP / (nq - nempty)
    pr = pr / (nq - nempty)

    return mAP, aps, pr, prs


def majority_vote(retrieved_labels):
    """Get the majority label from retrieved images.
    
    Args:
        retrieved_labels: array of labels from retrieved images
    
    Returns:
        predicted label based on majority vote
    """
    if len(retrieved_labels) == 0:
        return None
    counter = Counter(retrieved_labels)
    return counter.most_common(1)[0][0]


def compute_classification_metrics(labels, dists, k_values=[1, 5, 10, 15, 20]):
    """Compute Precision, Recall, F1, and Accuracy for different k values.
    
    Args:
        labels (Tensor): ground truth labels
        dists (Tensor): distance matrix (higher = more similar)
        k_values (list): list of k values for top-k retrieval
    
    Returns:
        dict: metrics for each k value
    """
    labels_np = labels.cpu().numpy()
    n_samples = labels.size(0)
    
    # Get sorted indices for each query (most similar to least similar)
    ranks = torch.argsort(dists, dim=0, descending=True).cpu().numpy()
    
    results = {}
    
    for k in k_values:
        predicted_labels = []
        true_labels = []
        
        # For each query image
        for i in range(n_samples):
            # Get top-k retrieved images (excluding self)
            top_k_indices = ranks[:k, i]
            retrieved_labels = labels_np[top_k_indices]
            
            # Get predicted label by majority vote
            pred_label = majority_vote(retrieved_labels)
            predicted_labels.append(pred_label)
            true_labels.append(labels_np[i])
        
        # Calculate metrics
        # Get unique labels for averaging
        unique_labels = np.unique(labels_np)
        
        # Calculate metrics with different averaging methods
        precision_macro = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
        recall_macro = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
        f1_macro = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
        
        precision_weighted = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        recall_weighted = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        f1_weighted = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        results[k] = {
            'precision_macro': precision_macro * 100.0,
            'recall_macro': recall_macro * 100.0,
            'f1_macro': f1_macro * 100.0,
            'precision_weighted': precision_weighted * 100.0,
            'recall_weighted': recall_weighted * 100.0,
            'f1_weighted': f1_weighted * 100.0,
            'accuracy': accuracy * 100.0
        }
    
    return results


@torch.no_grad()
def evaluate_with_text_reranking(img_model, text_model, text_processor, loader, conceptclip_loader, device, args, label_names, is_conceptclip_img=False):
    """Evaluate using image backbone for initial retrieval + ConceptCLIP text encoder for re-ranking.
    
    Args:
        img_model: Image backbone model (e.g., ConvNeXtV2, ResNet50)
        text_model: ConceptCLIP model for text encoding
        text_processor: ConceptCLIP processor
        loader: DataLoader for test data (with backbone transforms)
        conceptclip_loader: DataLoader with PIL images for ConceptCLIP
        device: torch device
        args: command line arguments
        label_names: list of class label names for text prompts
        is_conceptclip_img: whether the image model is ConceptCLIP
    """
    img_model.eval()
    text_model.eval()
    embeds, labels = [], []
    
    print(f"\n=== Two-Model Re-ranking Evaluation ===")
    print(f"Image Model: {args.model}")
    print(f"Text Model: ConceptCLIP")
    print(f"Using {len(label_names)} class labels: {label_names}")
    print(f"Re-ranking top-{args.rerank_k} results with text similarity")
    print(f"Image weight: {args.text_weight}, Text weight: {1-args.text_weight}\\n")
    
    print("Step 1: Extracting image embeddings from backbone model for initial retrieval...")
    for data in loader:
        if is_conceptclip_img:
            # ConceptCLIP as image backbone
            images = data[0]
            _labels = data[1].to(device)
            dummy_text = [""]
            inputs = text_processor(images=images, text=dummy_text, return_tensors='pt', padding=True).to(device)
            outputs = img_model(**inputs)
            embeds.append(outputs['image_features'])
        else:
            # Regular backbone (ConvNeXtV2, ResNet50, etc.)
            samples = data[0].to(device)
            _labels = data[1].to(device)
            out = img_model(samples)
            embeds.append(out)
        labels.append(_labels)
    
    embeds = torch.cat(embeds, dim=0)
    labels = torch.cat(labels, dim=0)
    
    # Normalize image embeddings
    embeds = embeds / embeds.norm(dim=-1, keepdim=True)
    
    print("Step 2: Extracting ConceptCLIP image embeddings for text similarity...")
    # Need to extract ConceptCLIP image embeddings for compatibility with text embeddings
    conceptclip_img_embeds = []
    
    if is_conceptclip_img:
        # Already have ConceptCLIP embeddings
        conceptclip_img_embeds = embeds
    else:
        # Need to extract ConceptCLIP image embeddings using PIL images
        for data in conceptclip_loader:
            images = data[0]  # PIL images
            dummy_text = [""]
            inputs = text_processor(images=images, text=dummy_text, return_tensors='pt', padding=True).to(device)
            outputs = text_model(**inputs)
            conceptclip_img_embeds.append(outputs['image_features'])
        
        conceptclip_img_embeds = torch.cat(conceptclip_img_embeds, dim=0)
        conceptclip_img_embeds = conceptclip_img_embeds / conceptclip_img_embeds.norm(dim=-1, keepdim=True)
    
    print("Step 3: Getting text embeddings from ConceptCLIP...")
    # Get text embeddings for each class using ConceptCLIP
    texts = [f'a medical image of {label}' for label in label_names]
    from PIL import Image
    dummy_image = Image.new('RGB', (224, 224), color='black')
    text_inputs = text_processor(
        images=[dummy_image],
        text=texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=77
    ).to(device)
    
    text_outputs = text_model(**text_inputs)
    text_embeds = text_outputs['text_features']
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    print("Step 4: Computing initial image-based retrieval...")
    # Initial retrieval with image similarity (using backbone model embeddings)
    img_sim = embeds @ embeds.t()
    
    print(f"Step 5: Re-ranking top-{args.rerank_k} results using text similarity...")
    # Compute image-to-text similarity using ConceptCLIP embeddings
    img_text_sim = conceptclip_img_embeds @ text_embeds.t()  # [N, num_classes]
    
    # Re-rank top-k for each query
    dists = img_sim.clone()
    alpha = args.text_weight  # weight for image similarity
    beta = 1.0 - alpha  # weight for text similarity
    
    for i in range(len(labels)):
        # Get top-k indices from initial retrieval
        top_k_scores, top_k_indices = torch.topk(img_sim[i], k=min(args.rerank_k, len(labels)), largest=True)
        
        # Re-score top-k using text similarity
        for j in top_k_indices:
            if i != j:
                # Text score based on whether retrieved image's class matches query
                text_score = img_text_sim[j, labels[i]]
                dists[i, j] = alpha * img_sim[i, j] + beta * text_score
    
    dists.fill_diagonal_(float('-inf'))
    
    print("\n=== Evaluation Results ===")
    # top-k accuracy (i.e. R@K)
    kappas = [1, 5, 10]
    accuracy = retrieval_accuracy(dists, labels, topk=kappas)
    accuracy = torch.stack(accuracy).cpu().numpy()
    print('>> R@K{}: {}%'.format(kappas, np.around(accuracy, 2)))

    # mean average precision and mean precision (i.e. mAP and pr)
    ranks = torch.argsort(dists, dim=0, descending=True)
    mAP, _, pr, _ = compute_map(ranks.cpu().numpy(), labels.cpu().numpy(), kappas)
    print('>> mAP: {:.2f}%'.format(mAP * 100.0))
    print('>> mP@K{}: {}%'.format(kappas, np.around(pr * 100.0, 2)))
    
    # Classification metrics with majority voting
    print('\n>> Classification Metrics (Majority Voting):')
    k_values = [1, 5, 10, 15, 20]
    classification_results = compute_classification_metrics(labels, dists, k_values)
    
    for k in k_values:
        metrics = classification_results[k]
        print(f'\n>> Top-{k} Retrieved Images:')
        print(f'   Accuracy: {metrics["accuracy"]:.2f}%')
        print(f'   Precision (macro): {metrics["precision_macro"]:.2f}%')
        print(f'   Recall (macro): {metrics["recall_macro"]:.2f}%')
        print(f'   F1 (macro): {metrics["f1_macro"]:.2f}%')
        print(f'   Precision (weighted): {metrics["precision_weighted"]:.2f}%')
        print(f'   Recall (weighted): {metrics["recall_weighted"]:.2f}%')
        print(f'   F1 (weighted): {metrics["f1_weighted"]:.2f}%')
    
    # Save results
    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        file_name = f'{args.model}_conceptclip_rerank'
        save_path = os.path.join(args.save_dir, file_name)
        
        classification_k_values = list(classification_results.keys())
        classification_metrics = {k: v for k, v in classification_results.items()}
        
        np.savez(save_path, embeds=embeds.cpu().numpy(),
                 labels=labels.cpu().numpy(), dists=-dists.cpu().numpy(),
                 kappas=kappas, acc=accuracy, mAP=mAP, pr=pr,
                 classification_k_values=classification_k_values,
                 text_embeds=text_embeds.cpu().numpy(),
                 label_names=label_names,
                 image_model=args.model,
                 rerank_k=args.rerank_k,
                 text_weight=args.text_weight,
                 **{f'classification_k{k}': np.array(list(v.values())) for k, v in classification_metrics.items()})
        print(f'\n>> Results saved to {save_path}.npz')


@torch.no_grad()
def evaluate_conceptclip_with_text(model, processor, loader, device, args, label_names):
    """Evaluate ConceptCLIP using text-enhanced retrieval.
    
    Args:
        model: ConceptCLIP model
        processor: ConceptCLIP processor
        loader: DataLoader for test data
        device: torch device
        args: command line arguments
        label_names: list of class label names for text prompts
    """
    model.eval()
    embeds, labels = [], []
    
    print(f"\nExtracting ConceptCLIP image embeddings for text-enhanced retrieval...")
    print(f"Using {len(label_names)} class labels: {label_names}")
    print(f"Fusion strategy: {args.text_fusion_strategy}")
    
    for data in loader:
        images = data[0]
        _labels = data[1].to(device)
        
        # Process images with dummy text to get image embeddings
        dummy_text = [""]  # Empty text to get image features
        inputs = processor(
            images=images,
            text=dummy_text,
            return_tensors='pt',
            padding=True
        ).to(device)
        
        # Get image embeddings from model outputs
        outputs = model(**inputs)
        embeds.append(outputs['image_features'])
        labels.append(_labels)
    
    embeds = torch.cat(embeds, dim=0)
    labels = torch.cat(labels, dim=0)
    
    # Normalize image embeddings
    embeds = embeds / embeds.norm(dim=-1, keepdim=True)
    
    # Get text embeddings for each class
    texts = [f'a medical image of {label}' for label in label_names]
    # Need to provide a dummy image for ConceptCLIP
    from PIL import Image
    dummy_image = Image.new('RGB', (224, 224), color='black')
    text_inputs = processor(
        images=[dummy_image],
        text=texts,
        return_tensors='pt',
        padding=True,
        truncation=True
    ).to(device)
    
    text_outputs = model(**text_inputs)
    text_embeds = text_outputs['text_features']
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    # Strategy 1: Hybrid Similarity (weighted combination)
    if args.text_fusion_strategy == 'hybrid':
        alpha = args.text_weight  # weight for image similarity
        beta = 1.0 - alpha  # weight for text similarity
        
        # Image-to-image similarity
        img_sim = embeds @ embeds.t()
        
        # Image-to-text similarity for each sample
        img_text_sim = embeds @ text_embeds.t()  # [N, num_classes]
        
        # For each query, get text similarity based on target labels
        text_sim = torch.zeros_like(img_sim)
        for i in range(len(labels)):
            for j in range(len(labels)):
                text_sim[i, j] = img_text_sim[j, labels[i]]
        
        # Combine similarities
        dists = alpha * img_sim + beta * text_sim
        print(f"   Using hybrid fusion (image weight={alpha:.2f}, text weight={beta:.2f})")
    
    # Strategy 2: Text-Guided Re-ranking
    elif args.text_fusion_strategy == 'rerank':
        k_initial = args.rerank_k  # number of top results to re-rank
        alpha = args.text_weight
        
        # Initial retrieval with image similarity
        img_sim = embeds @ embeds.t()
        img_text_sim = embeds @ text_embeds.t()
        
        # Re-rank top-k for each query
        dists = img_sim.clone()
        for i in range(len(labels)):
            # Get top-k indices
            top_k_scores, top_k_indices = torch.topk(img_sim[i], k=min(k_initial, len(labels)), largest=True)
            
            # Re-score top-k using text similarity
            for idx_pos, j in enumerate(top_k_indices):
                if i != j:
                    text_score = img_text_sim[j, labels[i]]
                    dists[i, j] = alpha * img_sim[i, j] + (1-alpha) * text_score
        
        print(f"   Using re-ranking fusion (top-{k_initial}, text weight={1-alpha:.2f})")
    
    # Strategy 3: Concatenated Embeddings
    elif args.text_fusion_strategy == 'concat':
        # For each image, concatenate its embedding with its class text embedding
        combined_embeds = []
        for i in range(len(embeds)):
            label_idx = labels[i]
            # Concatenate image embedding with corresponding text embedding
            combined = torch.cat([embeds[i], text_embeds[label_idx]], dim=0)
            combined_embeds.append(combined)
        
        combined_embeds = torch.stack(combined_embeds)
        combined_embeds = combined_embeds / combined_embeds.norm(dim=-1, keepdim=True)
        
        # Compute similarity with concatenated embeddings
        dists = combined_embeds @ combined_embeds.t()
        print(f"   Using concatenation fusion (image+text embeddings)")
    
    else:
        raise ValueError(f"Unknown fusion strategy: {args.text_fusion_strategy}")
    
    dists.fill_diagonal_(float('-inf'))
    
    # top-k accuracy (i.e. R@K)
    kappas = [1, 5, 10]
    accuracy = retrieval_accuracy(dists, labels, topk=kappas)
    accuracy = torch.stack(accuracy).cpu().numpy()
    print('>> R@K{}: {}%'.format(kappas, np.around(accuracy, 2)))

    # mean average precision and mean precision (i.e. mAP and pr)
    ranks = torch.argsort(dists, dim=0, descending=True)
    mAP, _, pr, _ = compute_map(ranks.cpu().numpy(), labels.cpu().numpy(), kappas)
    print('>> mAP: {:.2f}%'.format(mAP * 100.0))
    print('>> mP@K{}: {}%'.format(kappas, np.around(pr * 100.0, 2)))
    
    # Classification metrics with majority voting (same as other models)
    print('\n>> Classification Metrics (Majority Voting):')  
    k_values = [1, 5, 10, 15, 20]
    classification_results = compute_classification_metrics(labels, dists, k_values)
    
    for k in k_values:
        metrics = classification_results[k]
        print(f'\n>> Top-{k} Retrieved Images:')
        print(f'   Accuracy: {metrics["accuracy"]:.2f}%')
        print(f'   Precision (macro): {metrics["precision_macro"]:.2f}%')
        print(f'   Recall (macro): {metrics["recall_macro"]:.2f}%')
        print(f'   F1 (macro): {metrics["f1_macro"]:.2f}%')
        print(f'   Precision (weighted): {metrics["precision_weighted"]:.2f}%')
        print(f'   Recall (weighted): {metrics["recall_weighted"]:.2f}%')
        print(f'   F1 (weighted): {metrics["f1_weighted"]:.2f}%')
    
    # Save results
    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        file_name = f'conceptclip_text_{args.text_fusion_strategy}'
        save_path = os.path.join(args.save_dir, file_name)
        
        classification_k_values = list(classification_results.keys())
        classification_metrics = {k: v for k, v in classification_results.items()}
        
        np.savez(save_path, embeds=embeds.cpu().numpy(),
                 labels=labels.cpu().numpy(), dists=-dists.cpu().numpy(),
                 kappas=kappas, acc=accuracy, mAP=mAP, pr=pr,
                 classification_k_values=classification_k_values,
                 text_embeds=text_embeds.cpu().numpy(),
                 label_names=label_names,
                 fusion_strategy=args.text_fusion_strategy,
                 **{f'classification_k{k}': np.array(list(v.values())) for k, v in classification_metrics.items()})
        print(f'\n>> Results saved to {save_path}.npz')


@torch.no_grad()
def evaluate_conceptclip(model, processor, loader, device, args):
    """Evaluate ConceptCLIP model using image retrieval (same as other models).
    
    Args:
        model: ConceptCLIP model
        processor: ConceptCLIP processor
        loader: DataLoader for test data
        device: torch device
        args: command line arguments
    """
    model.eval()
    embeds, labels = [], []
    
    print(f"\nExtracting ConceptCLIP image embeddings for retrieval...")
    
    for data in loader:
        images = data[0]
        _labels = data[1].to(device)
        
        # Process images - ConceptCLIP needs text input, use dummy text
        dummy_text = [""]  # Empty text to get image features
        inputs = processor(
            images=images,
            text=dummy_text,
            return_tensors='pt',
            padding=True
        ).to(device)
        
        # Get image embeddings from model outputs
        outputs = model(**inputs)
        embeds.append(outputs['image_features'])
        labels.append(_labels)
    
    embeds = torch.cat(embeds, dim=0)
    labels = torch.cat(labels, dim=0)
    
    # Normalize embeddings for cosine similarity
    embeds = embeds / embeds.norm(dim=-1, keepdim=True)
    
    # Compute similarity matrix (cosine similarity via normalized dot product)
    dists = embeds @ embeds.t()
    dists.fill_diagonal_(float('-inf'))
    
    # top-k accuracy (i.e. R@K)
    kappas = [1, 5, 10]
    accuracy = retrieval_accuracy(dists, labels, topk=kappas)
    accuracy = torch.stack(accuracy).cpu().numpy()
    print('>> R@K{}: {}%'.format(kappas, np.around(accuracy, 2)))

    # mean average precision and mean precision (i.e. mAP and pr)
    ranks = torch.argsort(dists, dim=0, descending=True)
    mAP, _, pr, _ = compute_map(ranks.cpu().numpy(), labels.cpu().numpy(), kappas)
    print('>> mAP: {:.2f}%'.format(mAP * 100.0))
    print('>> mP@K{}: {}%'.format(kappas, np.around(pr * 100.0, 2)))
    
    # Classification metrics with majority voting (same as other models)
    print('\n>> Classification Metrics (Majority Voting):')
    k_values = [1, 5, 10, 15, 20]
    classification_results = compute_classification_metrics(labels, dists, k_values)
    
    for k in k_values:
        metrics = classification_results[k]
        print(f'\n>> Top-{k} Retrieved Images:')
        print(f'   Accuracy: {metrics["accuracy"]:.2f}%')
        print(f'   Precision (macro): {metrics["precision_macro"]:.2f}%')
        print(f'   Recall (macro): {metrics["recall_macro"]:.2f}%')
        print(f'   F1 (macro): {metrics["f1_macro"]:.2f}%')
        print(f'   Precision (weighted): {metrics["precision_weighted"]:.2f}%')
        print(f'   Recall (weighted): {metrics["recall_weighted"]:.2f}%')
        print(f'   F1 (weighted): {metrics["f1_weighted"]:.2f}%')
    
    # Save results
    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        file_name = 'conceptclip_retrieval'
        save_path = os.path.join(args.save_dir, file_name)
        
        classification_k_values = list(classification_results.keys())
        classification_metrics = {k: v for k, v in classification_results.items()}
        
        np.savez(save_path, embeds=embeds.cpu().numpy(),
                 labels=labels.cpu().numpy(), dists=-dists.cpu().numpy(),
                 kappas=kappas, acc=accuracy, mAP=mAP, pr=pr,
                 classification_k_values=classification_k_values,
                 **{f'classification_k{k}': np.array(list(v.values())) for k, v in classification_metrics.items()})
        print(f'\n>> Results saved to {save_path}.npz')


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    embeds, labels = [], []

    for data in loader:
        samples = data[0].to(device)
        _labels = data[1].to(device)
        out = model(samples)
        embeds.append(out)
        labels.append(_labels)

    embeds = torch.cat(embeds, dim=0)
    labels = torch.cat(labels, dim=0)

    dists = -torch.cdist(embeds, embeds)
    dists.fill_diagonal_(float('-inf'))

    # top-k accuracy (i.e. R@K)
    kappas = [1, 5, 10]
    accuracy = retrieval_accuracy(dists, labels, topk=kappas)
    accuracy = torch.stack(accuracy).cpu().numpy()
    print('>> R@K{}: {}%'.format(kappas, np.around(accuracy, 2)))

    # mean average precision and mean precision (i.e. mAP and pr)
    ranks = torch.argsort(dists, dim=0, descending=True)
    mAP, _, pr, _ = compute_map(ranks.cpu().numpy(),  labels.cpu().numpy(), kappas)
    print('>> mAP: {:.2f}%'.format(mAP * 100.0))
    print('>> mP@K{}: {}%'.format(kappas, np.around(pr * 100.0, 2)))
    
    # Classification metrics with majority voting
    print('\n>> Classification Metrics (Majority Voting):')
    k_values = [1, 5, 10, 15, 20]
    classification_results = compute_classification_metrics(labels, dists, k_values)
    
    for k in k_values:
        metrics = classification_results[k]
        print(f'\n>> Top-{k} Retrieved Images:')
        print(f'   Accuracy: {metrics["accuracy"]:.2f}%')
        print(f'   Precision (macro): {metrics["precision_macro"]:.2f}%')
        print(f'   Recall (macro): {metrics["recall_macro"]:.2f}%')
        print(f'   F1 (macro): {metrics["f1_macro"]:.2f}%')
        print(f'   Precision (weighted): {metrics["precision_weighted"]:.2f}%')
        print(f'   Recall (weighted): {metrics["recall_weighted"]:.2f}%')
        print(f'   F1 (weighted): {metrics["f1_weighted"]:.2f}%')

    # Save results
    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        file_name = args.resume.split('/')[-1].split('.')[0]

        save_path = os.path.join(args.save_dir, file_name)
        # Convert classification_results dict to numpy arrays for saving
        classification_k_values = list(classification_results.keys())
        classification_metrics = {k: v for k, v in classification_results.items()}
        
        np.savez(save_path, embeds=embeds.cpu().numpy(),
                 labels=labels.cpu().numpy(), dists=-dists.cpu().numpy(),
                 kappas=kappas, acc=accuracy, mAP=mAP, pr=pr,
                 classification_k_values=classification_k_values,
                 **{f'classification_k{k}': np.array(list(v.values())) for k, v in classification_metrics.items()})


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Two-model re-ranking: load both image backbone and ConceptCLIP
    if args.use_rerank_2models:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError('transformers library is required for ConceptCLIP. Install it with: pip install transformers')
        
        print("=== Loading Two Models for Re-ranking ===")
        
        # Load ConceptCLIP for text encoding
        print("Loading ConceptCLIP for text encoding...")
        text_model = AutoModel.from_pretrained('JerrryNie/ConceptCLIP', trust_remote_code=True)
        text_processor = AutoProcessor.from_pretrained('JerrryNie/ConceptCLIP', trust_remote_code=True)
        text_model.to(device)
        
        # Load image backbone model
        print(f"Loading {args.model} as image backbone...")
        if args.model == 'conceptclip':
            # Use ConceptCLIP for both (image features only)
            img_model = text_model
            is_conceptclip_img = True
        elif args.model == 'densenet121':
            img_model = DenseNet121(embedding_dim=args.embedding_dim)
            is_conceptclip_img = False
        elif args.model == 'resnet50':
            img_model = ResNet50(embedding_dim=args.embedding_dim)
            is_conceptclip_img = False
        elif args.model == 'convnextv2':
            img_model = ConvNeXtV2(embedding_dim=args.embedding_dim)
            is_conceptclip_img = False
        elif args.model == 'swinv2':
            img_model = SwinV2(embedding_dim=args.embedding_dim)
            is_conceptclip_img = False
        else:
            raise NotImplementedError('Model not supported!')
        
        # Load checkpoint for image model if not ConceptCLIP
        if not is_conceptclip_img:
            if os.path.isfile(args.resume):
                print("=> loading image model checkpoint")
                checkpoint = torch.load(args.resume)
                if 'state-dict' in checkpoint:
                    checkpoint = checkpoint['state-dict']
                img_model.load_state_dict(checkpoint, strict=False)
                print("=> loaded checkpoint")
            else:
                print("=> no checkpoint found for image model")
            img_model.to(device)
        
        is_conceptclip = False  # We're using two-model approach
        use_two_model_rerank = True
    else:
        # Single model approach (original behavior)
        use_two_model_rerank = False
        
        # Choose model
        if args.model == 'conceptclip':
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError('transformers library is required for ConceptCLIP. Install it with: pip install transformers')
            print("Loading ConceptCLIP model...")
            model = AutoModel.from_pretrained('JerrryNie/ConceptCLIP', trust_remote_code=True)
            processor = AutoProcessor.from_pretrained('JerrryNie/ConceptCLIP', trust_remote_code=True)
            model.to(device)
            is_conceptclip = True
        elif args.model == 'densenet121':
            model = DenseNet121(embedding_dim=args.embedding_dim)
            is_conceptclip = False
        elif args.model == 'resnet50':
            model = ResNet50(embedding_dim=args.embedding_dim)
            is_conceptclip = False
        elif args.model == 'convnextv2':
            model = ConvNeXtV2(embedding_dim=args.embedding_dim)
            is_conceptclip = False
        elif args.model == 'swinv2':
            model = SwinV2(embedding_dim=args.embedding_dim)
            is_conceptclip = False
        else:
            raise NotImplementedError('Model not supported!')

    if not use_two_model_rerank:
        if not is_conceptclip:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint")
                checkpoint = torch.load(args.resume)
                if 'state-dict' in checkpoint:
                    checkpoint = checkpoint['state-dict']
                model.load_state_dict(checkpoint, strict=False)
                print("=> loaded checkpoint")
            else:
                print("=> no checkpoint found")
            model.to(device)
        else:
            print("=> Using pre-trained ConceptCLIP (zero-shot), no checkpoint needed")

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    
    # ConceptCLIP uses PIL images directly (processor handles preprocessing)
    if (is_conceptclip and not use_two_model_rerank) or (use_two_model_rerank and is_conceptclip_img):
        test_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB'))
        ])
    else:
        # Use 384x384 for ConvNeXtV2 and SwinV2, 224x224 for other models
        img_size = 384 if args.model in ['convnextv2', 'swinv2'] else 224

        if args.model in ['convnextv2', 'swinv2']:
            test_transform = transforms.Compose([
                transforms.Lambda(lambda img: img.convert('RGB')),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                normalize
            ])
        else:
            test_transform = transforms.Compose([transforms.Lambda(lambda image: image.convert('RGB')),
                                             transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             normalize])

    # Set up dataset and dataloader
    if args.dataset == 'covid':
        test_dataset = ChestXrayDataSet(data_dir=args.test_dataset_dir,
                                        image_list_file=args.test_image_list,
                                        mask_dir=args.mask_dir,
                                        transform=test_transform)
    elif args.dataset == 'isic':
        test_dataset = ISICDataSet(data_dir=args.test_dataset_dir,
                                   image_list_file=args.test_image_list,
                                   mask_dir=args.mask_dir,
                                   transform=test_transform)
    elif args.dataset == 'tbx11k':
        test_dataset = TBX11kDataSet(data_dir=args.test_dataset_dir,
                                    csv_file=args.test_image_list,
                                    transform=test_transform)
    else:
        raise NotImplementedError('Dataset not supported!')

    # Use custom collate function for ConceptCLIP to handle PIL images
    use_conceptclip_collate = (is_conceptclip and not use_two_model_rerank) or (use_two_model_rerank and is_conceptclip_img)
    
    if use_conceptclip_collate:
        test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                                 shuffle=False,
                                 num_workers=args.workers,
                                 collate_fn=conceptclip_collate_fn)
    else:
        test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                                 shuffle=False,
                                 num_workers=args.workers)

    print('Evaluating...')
    
    if use_two_model_rerank:
        # Two-model re-ranking approach - need separate loader for ConceptCLIP with PIL images
        if args.dataset == 'covid':
            label_names = args.covid_labels.split(',') if args.covid_labels else ['normal', 'pneumonia', 'COVID-19']
        elif args.dataset == 'isic':
            label_names = args.isic_labels.split(',') if args.isic_labels else ['melanoma', 'nevus', 'seborrheic keratosis']
        elif args.dataset == 'tbx11k':
            label_names = args.tbx11k_labels.split(',') if args.tbx11k_labels else ['normal', 'active TB', 'latent TB', 'uncertain TB']
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        
        # Create separate dataset/loader with PIL images for ConceptCLIP
        if not is_conceptclip_img:
            pil_transform = transforms.Compose([
                transforms.Lambda(lambda img: img.convert('RGB'))
            ])
            
            if args.dataset == 'covid':
                conceptclip_dataset = ChestXrayDataSet(data_dir=args.test_dataset_dir,
                                                       image_list_file=args.test_image_list,
                                                       mask_dir=args.mask_dir,
                                                       transform=pil_transform)
            elif args.dataset == 'isic':
                conceptclip_dataset = ISICDataSet(data_dir=args.test_dataset_dir,
                                                  image_list_file=args.test_image_list,
                                                  mask_dir=args.mask_dir,
                                                  transform=pil_transform)
            elif args.dataset == 'tbx11k':
                conceptclip_dataset = TBX11kDataSet(data_dir=args.test_dataset_dir,
                                                   csv_file=args.test_image_list,
                                                   transform=pil_transform)
            
            conceptclip_loader = DataLoader(conceptclip_dataset, batch_size=args.eval_batch_size,
                                           shuffle=False,
                                           num_workers=args.workers,
                                           collate_fn=conceptclip_collate_fn)
        else:
            # Already using PIL images
            conceptclip_loader = test_loader
        
        evaluate_with_text_reranking(img_model, text_model, text_processor, test_loader, conceptclip_loader, device, args, label_names, is_conceptclip_img)
    elif is_conceptclip:
        if args.use_text:
            # Get label names for text-enhanced retrieval
            if args.dataset == 'covid':
                label_names = args.covid_labels.split(',') if args.covid_labels else ['normal', 'pneumonia', 'COVID-19']
            elif args.dataset == 'isic':
                label_names = args.isic_labels.split(',') if args.isic_labels else ['melanoma', 'nevus', 'seborrheic keratosis']
            elif args.dataset == 'tbx11k':
                label_names = args.tbx11k_labels.split(',') if args.tbx11k_labels else ['normal', 'active TB', 'latent TB', 'uncertain TB']
            else:
                raise ValueError(f"Unknown dataset: {args.dataset}")
            
            evaluate_conceptclip_with_text(model, processor, test_loader, device, args, label_names)
        else:
            evaluate_conceptclip(model, processor, test_loader, device, args)
    else:
        evaluate(model, test_loader, device, args)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Embedding Learning')

    parser.add_argument('--dataset', default='covid',
                        help='Dataset to use (covid or isic)')
    parser.add_argument('--test-dataset-dir', default='/data/brian.hu/COVID/data/test',
                        help='Test dataset directory path')
    parser.add_argument('--test-image-list', default='./test_COVIDx4.txt',
                        help='Test image list')
    parser.add_argument('--mask-dir', default=None,
                        help='Segmentation masks path (if used)')
    parser.add_argument('--model', default='densenet121',
                        help='Model to use (densenet121, resnet50, convnextv2, swinv2, or conceptclip)')
    parser.add_argument('--embedding-dim', default=None, type=int,
                        help='Embedding dimension of model')
    
    # ConceptCLIP text-enhanced retrieval options
    parser.add_argument('--use-text', action='store_true',
                        help='Enable text-enhanced retrieval for ConceptCLIP')
    parser.add_argument('--use-rerank-2models', action='store_true',
                        help='Use image backbone (e.g., ConvNeXtV2) for initial retrieval + ConceptCLIP text encoder for re-ranking')
    parser.add_argument('--text-fusion-strategy', default='hybrid', choices=['hybrid', 'rerank', 'concat'],
                        help='Text fusion strategy: hybrid (weighted combination), rerank (re-rank top-k), concat (concatenate embeddings)')
    parser.add_argument('--text-weight', default=0.5, type=float,
                        help='Weight for text similarity in hybrid/rerank fusion (0.0-1.0). For hybrid: image_weight=text_weight, text_weight=1-text_weight')
    parser.add_argument('--rerank-k', default=50, type=int,
                        help='Number of top results to re-rank when using rerank strategy')
    parser.add_argument('--covid-labels', default=None, type=str,
                        help='Comma-separated list of COVID dataset labels for ConceptCLIP (e.g., "normal,pneumonia,COVID-19")')
    parser.add_argument('--isic-labels', default=None, type=str,
                        help='Comma-separated list of ISIC dataset labels for ConceptCLIP')
    parser.add_argument('--tbx11k-labels', default=None, type=str,
                        help='Comma-separated list of TBX11K dataset labels for ConceptCLIP')
    parser.add_argument('--eval-batch-size', default=64, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='Number of data loading workers')
    parser.add_argument('--save-dir', default='./results',
                        help='Result save directory')
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
