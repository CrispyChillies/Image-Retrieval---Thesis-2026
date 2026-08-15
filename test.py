import csv
import json
import os
import numpy as np
from collections import Counter
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import timm
from timm.data import resolve_model_data_config
from read_data import (
    ISICDataSet,
    ChestXrayDataSet,
    TBX11kDataSet,
    NIHChestXrayRetrievalDataSet,
    NIH_U_LABELS,
)
from model import (
    ResNet50,
    DenseNet121,
    ConvNeXtV2,
    ConvNeXtV2_SRA,
    ConvNeXtV2_ATH,
    ConvNeXtV2_PCAM,
    ConvNeXtV2_LoFi,
    ConvNeXtV2_RRAVL,
    ConvNeXtV2_MSAtt,
    SwinV2,
    DinoV2,
    MedSigLIP,
)


class MIMICIRImageDataset(torch.utils.data.Dataset):
    """Image-only view of the official MIMIC-IR CSV.

    The row order is intentionally preserved because the official RaTEScore
    matrix uses the same order as ``val_caption.csv``.  The returned target is
    the CSV row index, not a diagnostic label.
    """

    def __init__(self, root, csv_file, transform=None, path_column='File Path'):
        self.root = root
        self.transform = transform
        with open(csv_file, 'r', encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f'MIMIC-IR CSV has no header: {csv_file}')
            by_lower = {name.strip().lower(): name for name in reader.fieldnames}
            selected = by_lower.get(path_column.strip().lower())
            if selected is None:
                candidates = ('file path', 'path', 'image_path', 'img_path')
                selected = next((by_lower[c] for c in candidates if c in by_lower), None)
            if selected is None:
                raise ValueError(
                    f'Cannot find image path column {path_column!r} in {reader.fieldnames}'
                )
            self.image_paths = [row[selected].strip() for row in reader]

        if not self.image_paths:
            raise ValueError(f'MIMIC-IR CSV contains no samples: {csv_file}')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        relative_path = self.image_paths[index].replace('\\', os.sep).replace('/', os.sep)
        image_path = relative_path if os.path.isabs(relative_path) else os.path.join(self.root, relative_path)
        with Image.open(image_path) as image:
            image = image.convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
        return image, torch.tensor(index, dtype=torch.long)

def conceptclip_collate_fn(batch):
    """Custom collate function for ConceptCLIP that keeps PIL images as-is."""
    images = [item[0] for item in batch]
    labels = torch.stack([item[1] if isinstance(item[1], torch.Tensor) else torch.tensor(item[1]) for item in batch])
    return images, labels



try:
    from transformers import AutoModel, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. ConceptCLIP will not be available.")


try:
    from open_clip import create_model_from_pretrained, get_tokenizer
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    print("Warning: open_clip library not available. BiomedCLIP will not be available.")


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


def get_dataset_label_names(args):
    """Get class label names for zero-shot text prompts by dataset."""
    if args.dataset == 'covid':
        return args.covid_labels.split(',') if args.covid_labels else ['normal', 'pneumonia', 'COVID-19']
    if args.dataset == 'isic':
        return args.isic_labels.split(',') if args.isic_labels else ['nevus', 'seborrheic keratosis', 'melanoma']
    if args.dataset == 'tbx11k':
        return args.tbx11k_labels.split(',') if args.tbx11k_labels else ['tuberculosis', 'healthy', 'sick but no tuberculosis']
    raise ValueError(f"Unknown dataset: {args.dataset}")


@torch.no_grad()
def evaluate_biomedclip_zeroshot(model, tokenizer, preprocess, loader, device, args):
    """Zero-shot evaluation with BiomedCLIP on image classification + retrieval."""
    model.eval()
    class_names = get_dataset_label_names(args)
    text_prompts = [args.biomedclip_prompt_template.format(label=label) for label in class_names]

    print("\n=== BiomedCLIP Zero-Shot Evaluation ===")
    print(f"Dataset: {args.dataset}")
    print(f"Class names: {class_names}")
    print(f"Prompt template: {args.biomedclip_prompt_template}")

    text_tokens = tokenizer(text_prompts).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    all_predictions = []
    all_labels = []
    embeds = []

    logit_scale = model.logit_scale.exp() if hasattr(model, 'logit_scale') else torch.tensor(100.0, device=device)

    for batch_idx, data in enumerate(loader):
        images = data[0]
        labels = data[1].to(device)

        image_tensor = torch.stack([preprocess(img) for img in images]).to(device)
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features @ text_features.t()
        predictions = torch.argmax(logits, dim=-1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        embeds.append(image_features)

        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {(batch_idx + 1) * args.eval_batch_size} images...")

    labels = torch.tensor(np.array(all_labels), dtype=torch.long, device=device)
    embeds = torch.cat(embeds, dim=0)

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Zero-shot classification metrics
    accuracy_cls = accuracy_score(all_labels, all_predictions) * 100.0
    precision_macro = precision_score(all_labels, all_predictions, average='macro', zero_division=0) * 100.0
    recall_macro = recall_score(all_labels, all_predictions, average='macro', zero_division=0) * 100.0
    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0) * 100.0

    print("\n>> Zero-shot Classification Metrics:")
    print(f"   Accuracy: {accuracy_cls:.2f}%")
    print(f"   Precision (macro): {precision_macro:.2f}%")
    print(f"   Recall (macro): {recall_macro:.2f}%")
    print(f"   F1 (macro): {f1_macro:.2f}%")

    # Retrieval metrics from image embeddings
    dists = embeds @ embeds.t()
    dists.fill_diagonal_(float('-inf'))

    kappas = [1, 5, 10]
    accuracy = retrieval_accuracy(dists, labels, topk=kappas)
    accuracy = torch.stack(accuracy).cpu().numpy()
    print('>> R@K{}: {}%'.format(kappas, np.around(accuracy, 2)))

    ranks = torch.argsort(dists, dim=0, descending=True)
    mAP, _, pr, _ = compute_map(ranks.cpu().numpy(), labels.cpu().numpy(), kappas)
    print('>> mAP: {:.2f}%'.format(mAP * 100.0))
    print('>> mP@K{}: {}%'.format(kappas, np.around(pr * 100.0, 2)))

    print('\n>> Retrieval Classification Metrics (Majority Voting):')
    k_values = [1, 5, 10, 15, 20]
    classification_results = compute_classification_metrics(labels, dists, k_values)

    for k in k_values:
        metrics = classification_results[k]
        print(f'\n>> Top-{k} Retrieved Images:')
        print(f'   Accuracy: {metrics["accuracy"]:.2f}%')
        print(f'   Precision (macro): {metrics["precision_macro"]:.2f}%')
        print(f'   Recall (macro): {metrics["recall_macro"]:.2f}%')
        print(f'   F1 (macro): {metrics["f1_macro"]:.2f}%')

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, 'biomedclip_zeroshot')

        classification_k_values = list(classification_results.keys())
        classification_metrics = {k: v for k, v in classification_results.items()}

        np.savez(
            save_path,
            embeds=embeds.cpu().numpy(),
            labels=labels.cpu().numpy(),
            dists=-dists.cpu().numpy(),
            predictions=all_predictions,
            class_names=np.array(class_names),
            text_prompts=np.array(text_prompts),
            zero_shot_accuracy=accuracy_cls,
            zero_shot_precision_macro=precision_macro,
            zero_shot_recall_macro=recall_macro,
            zero_shot_f1_macro=f1_macro,
            kappas=kappas,
            acc=accuracy,
            mAP=mAP,
            pr=pr,
            classification_k_values=classification_k_values,
            **{f'classification_k{k}': np.array(list(v.values())) for k, v in classification_metrics.items()}
        )
        print(f'\n>> Results saved to {save_path}.npz')


@torch.no_grad()
def evaluate_conceptclip_concept_retrieval(model, processor, loader, device, args, concept_list):
    """Evaluate using concept-based multi-label classification and retrieval.
    
    Each image is represented by confidence scores for multiple concepts.
    Retrieval is based on similarity of concept profiles between images.
    
    Args:
        model: ConceptCLIP model
        processor: ConceptCLIP processor
        loader: DataLoader for test data
        device: torch device
        args: command line arguments
        concept_list: list of concept descriptions (e.g., ["ground glass opacity", "consolidation", ...])
    """
    model.eval()
    labels = []
    
    print(f"\n=== Concept-Based Multi-Label Retrieval ===")
    print(f"Using {len(concept_list)} concepts:")
    for i, concept in enumerate(concept_list):
        print(f"  {i+1}. {concept}")
    print()
    
    # Step 1: Get text embeddings for all concepts
    print("Step 1: Extracting concept embeddings from text encoder...")
    
    
    # Create concept prompts
    concept_texts = [f'a medical image showing {concept}' for concept in concept_list]
    
    # Get a real image from the loader to use as dummy for text extraction
    first_batch = next(iter(loader))
    sample_image = first_batch[0][0]  # Get first PIL image from batch
    
    # Process all concept texts at once using the sample image
    text_inputs = processor(
        images=[sample_image],
        text=concept_texts,
        return_tensors='pt',
        padding=True,
        truncation=True
    ).to(device)
    
    text_outputs = model(**text_inputs)
    concept_embeds = text_outputs['text_features']
    concept_embeds = concept_embeds / concept_embeds.norm(dim=-1, keepdim=True)
    print(f"   Concept embeddings shape: {concept_embeds.shape}")
    
    # Step 2: Extract image embeddings and compute concept scores
    print("Step 2: Computing concept confidence scores for each image...")
    all_concept_scores = []
    
    for batch_idx, data in enumerate(loader):
        images = data[0]
        _labels = data[1].to(device)
        
        # Get image embeddings
        dummy_text = [""]
        inputs = processor(
            images=images,
            text=dummy_text,
            return_tensors='pt',
            padding=True
        ).to(device)
        
        outputs = model(**inputs)
        img_embeds = outputs['image_features']
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        
        # Compute image-to-concept similarities
        # Use logit_scale from ConceptCLIP model
        logit_scale = outputs.get('logit_scale', torch.tensor(100.0).to(device))
        similarities = logit_scale * (img_embeds @ concept_embeds.t())  # [batch_size, num_concepts]
        
        # Apply sigmoid to get confidence scores (0-1) for multi-label classification
        concept_scores = torch.sigmoid(similarities)
        all_concept_scores.append(concept_scores)
        labels.append(_labels)
        
        if (batch_idx + 1) % 5 == 0:
            print(f"   Processed {(batch_idx + 1) * args.eval_batch_size} images...")
    
    all_concept_scores = torch.cat(all_concept_scores, dim=0)  # [N, num_concepts]
    labels = torch.cat(labels, dim=0)
    
    print(f"   Concept scores shape: {all_concept_scores.shape}")
    print(f"   Mean concept scores: {all_concept_scores.mean(dim=0).cpu().numpy()}")
    print(f"   Min concept score: {all_concept_scores.min().item():.4f}, Max: {all_concept_scores.max().item():.4f}")
    
    # Step 3: Compute similarity based on concept profiles
    print("\nStep 3: Computing image similarity based on concept profiles...")
    # Use cosine similarity between concept score vectors
    # Normalize concept scores to unit vectors
    concept_scores_norm = all_concept_scores / (all_concept_scores.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Compute pairwise similarity matrix
    dists = concept_scores_norm @ concept_scores_norm.t()
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
    
    # Additional: Show top concepts for sample images
    print("\n>> Sample Concept Activations:")
    sample_indices = torch.randperm(len(all_concept_scores))[:min(5, len(all_concept_scores))]
    for idx in sample_indices:
        scores = all_concept_scores[idx]
        label = labels[idx].item()
        top_concepts = torch.topk(scores, k=min(5, len(concept_list)))
        print(f"\n   Image {idx.item()} (Label: {label}):")
        for score, concept_idx in zip(top_concepts.values, top_concepts.indices):
            print(f"      {concept_list[concept_idx]}: {score.item():.3f}")
    
    # Save results
    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        file_name = 'conceptclip_concept_retrieval'
        save_path = os.path.join(args.save_dir, file_name)
        
        classification_k_values = list(classification_results.keys())
        classification_metrics = {k: v for k, v in classification_results.items()}
        
        np.savez(save_path,
                 concept_scores=all_concept_scores.cpu().numpy(),
                 labels=labels.cpu().numpy(),
                 dists=-dists.cpu().numpy(),
                 kappas=kappas, acc=accuracy, mAP=mAP, pr=pr,
                 classification_k_values=classification_k_values,
                 concept_list=concept_list,
                 concept_embeds=concept_embeds.cpu().numpy(),
                 **{f'classification_k{k}': np.array(list(v.values())) for k, v in classification_metrics.items()})
        print(f'\n>> Results saved to {save_path}.npz')


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

def compute_map_multilabel(dists, labels, threshold=0.5):
    """
    Tính mAP cho dữ liệu đa nhãn.
    Args:
        dists: Ma trận khoảng cách (batch_size, batch_size)
        labels: Ma trận nhãn multi-hot (batch_size, num_classes)
        threshold: Ngưỡng Jaccard để coi là một kết quả 'đúng'
    """
    labels = labels.cpu().numpy()
    dists = dists.cpu().numpy()
    nq = labels.shape[0]
    aps = []

    # Tính toán ma trận Jaccard cho toàn bộ tập test
    # Intersection / Union
    intersection = np.dot(labels, labels.T)
    row_sums = labels.sum(axis=1).reshape(-1, 1)
    union = row_sums + row_sums.T - intersection
    jaccard_matrix = intersection / (union + 1e-8)

    # Lấy thứ tự ưu tiên từ khoảng cách (giá trị càng lớn càng gần)
    ranks = np.argsort(-dists, axis=0) 

    for i in range(nq):
        # Định nghĩa các ảnh liên quan là những ảnh có Jaccard > threshold
        # Loại bỏ chính nó (i)
        binary_relevance = (jaccard_matrix[i] > threshold).astype(float)
        binary_relevance[i] = 0 
        
        if np.sum(binary_relevance) > 0:
            # Sử dụng hàm chuẩn của sklearn để tính AP cho query i
            # ranks[:, i] là danh sách các index ảnh được sắp xếp theo độ gần với query i
            sorted_relevance = binary_relevance[ranks[:, i]]
            
            # Tính AP
            count_pos = 0
            ap = 0
            for rank, is_rel in enumerate(sorted_relevance):
                if is_rel > 0:
                    count_pos += 1
                    precision_at_rank = count_pos / (rank + 1)
                    ap += precision_at_rank
            aps.append(ap / np.sum(binary_relevance))

    return np.mean(aps) if aps else 0        


def extract_state_dict(checkpoint):
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif "state-dict" in checkpoint:
        checkpoint = checkpoint["state-dict"]
    if isinstance(checkpoint, dict) and checkpoint and all(key.startswith('module.') for key in checkpoint):
        checkpoint = {key[len('module.'):]: value for key, value in checkpoint.items()}
    return checkpoint


def load_state_dict_for_test(model, checkpoint, args, description='model'):
    incompatible = model.load_state_dict(checkpoint, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    print(f'=> loaded {description} checkpoint (missing={len(missing)}, unexpected={len(unexpected)})')
    fair_mode = args.fair_i2i_method != 'standard' or args.dataset == 'mimic_ir'
    if fair_mode and (missing or unexpected) and not args.allow_checkpoint_mismatch:
        details = []
        if missing:
            details.append(f'missing examples: {missing[:5]}')
        if unexpected:
            details.append(f'unexpected examples: {unexpected[:5]}')
        raise RuntimeError(
            'Checkpoint/model mismatch would invalidate the fair comparison; '
            + '; '.join(details)
            + '. Pass --allow-checkpoint-mismatch only after manually verifying these keys.'
        )
    return incompatible


def infer_num_labels_from_checkpoint(checkpoint, default_num_labels):
    if not isinstance(checkpoint, dict):
        return default_num_labels
    for key in ("classification_head.weight", "module.classification_head.weight"):
        weight = checkpoint.get(key)
        if weight is not None and hasattr(weight, "shape") and len(weight.shape) == 2:
            return int(weight.shape[0])
    return default_num_labels


def infer_pcam_num_classes(checkpoint, configured_num_classes=None):
    """Infer PCAM class-map count so the evaluation model matches its checkpoint."""
    if isinstance(checkpoint, dict):
        weight = checkpoint.get('pcam.classifier.weight')
        if weight is not None and hasattr(weight, 'shape') and len(weight.shape) == 4:
            inferred = int(weight.shape[0])
            if configured_num_classes is not None and configured_num_classes != inferred:
                raise ValueError(
                    f'--pcam-num-classes={configured_num_classes} but checkpoint contains '
                    f'{inferred} PCAM classes.'
                )
            return inferred
    if configured_num_classes is None:
        raise ValueError(
            'Cannot infer PCAM class count from checkpoint. Pass --pcam-num-classes '
            'with the class count used during PCAM training.'
        )
    return configured_num_classes


def validate_ath_checkpoint_config(checkpoint, hash_bits, num_classes):
    if not isinstance(checkpoint, dict):
        return
    hash_weight = checkpoint.get('hash_layer.weight')
    if hash_weight is not None and int(hash_weight.shape[0]) != hash_bits:
        raise ValueError(
            f'--ath-hash-bits={hash_bits}, but checkpoint uses {int(hash_weight.shape[0])} bits.'
        )
    class_weight = checkpoint.get('classification_head.weight')
    if class_weight is not None and int(class_weight.shape[0]) != num_classes:
        raise ValueError(
            f'--ath-num-classes={num_classes}, but checkpoint uses '
            f'{int(class_weight.shape[0])} classes.'
        )

@torch.no_grad()
def evaluate_multilabels(model, loader, device, args):
    model.eval()
    embeds, labels = [], []

    for data in loader:
        samples = data[0].to(device)
        _labels = data[1].to(device) 
        out = model(samples)
        
        if isinstance(out, dict):
            embedding = out["embedding"]
        else:
            embedding = out[0] if isinstance(out, tuple) else out
        embeds.append(embedding)
        labels.append(_labels)

    embeds = torch.cat(embeds, dim=0)
    labels = torch.cat(labels, dim=0)

    # Sử dụng Cosine Similarity
    embeds_norm = torch.nn.functional.normalize(embeds, p=2, dim=1)
    dists = torch.mm(embeds_norm, embeds_norm.t())
    dists.fill_diagonal_(-float('inf'))

    print('\n--- Multi-label Retrieval Results ---')
    
    # 1. Tính mAP (Giữ nguyên từ code trước)
    for t in [0.25, 0.5]:
        mAP_val = compute_map_multilabel(dists, labels, threshold=t)
        print(f'>> mAP (Jaccard > {t}): {mAP_val * 100.0:.2f}%')

    # 2. Tính Precision@K và Recall@K chuyên sâu
    k_values = [1, 5, 10, 15, 20]
    ranks = torch.argsort(dists, dim=1, descending=True)
    
    # Chuyển sang CPU numpy để tính toán nhanh hơn
    labels_np = labels.cpu().numpy()
    ranks_np = ranks.cpu().numpy()
    num_queries = labels_np.shape[0]

    print(f'\n{"K":<5} | {"Precision@K":<15} | {"Recall@K":<15}')
    print("-" * 40)

    for k in k_values:
        total_precision = 0
        total_recall = 0
        
        for i in range(num_queries):
            query_label = labels_np[i]
            # Lấy nhãn của top K ảnh được truy vấn
            top_k_labels = labels_np[ranks_np[i, :k]]
            
            # Trong đa nhãn:
            # - Precision@K: Tỉ lệ ảnh trong top K có ít nhất 1 nhãn trùng với Query
            # - Recall@K: Khả năng tìm thấy ít nhất 1 ảnh trùng nhãn trong top K (như code cũ)
            
            # Kiểm tra từng ảnh trong top K xem có "liên quan" không (chung ít nhất 1 nhãn)
            # (top_k_labels * query_label).sum(axis=1) > 0 trả về mảng Boolean [k]
            matches = (top_k_labels * query_label).sum(axis=1) > 0
            num_matches = np.sum(matches)
            
            # Precision@K cho query i
            total_precision += (num_matches / k)
            
            # Recall@K cho query i: 1 nếu có ít nhất 1 match, 0 nếu không
            if num_matches > 0:
                total_recall += 1

        avg_precision = (total_precision / num_queries) * 100
        avg_recall = (total_recall / num_queries) * 100
        
        print(f"{k:<5} | {avg_precision:<15.2f}% | {avg_recall:<15.2f}%")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, 'evaluation_results.npz')
        np.savez(save_path, embeds=embeds.cpu().numpy(), labels=labels.cpu().numpy())
        print(f'\n>> Results saved to {save_path}')


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


def _embedding_from_output(output):
    if isinstance(output, dict):
        if 'embedding' in output:
            return output['embedding']
        if 'image_features' in output:
            return output['image_features']
        raise KeyError(f'Model output has no embedding key: {list(output.keys())}')
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _label_relevance(labels):
    """Create a shared relevance target for fair image-to-image evaluation."""
    labels = labels.float()
    if labels.ndim == 1 or (labels.ndim == 2 and labels.shape[1] == 1):
        flat = labels.reshape(-1)
        return flat[:, None].eq(flat[None, :]).float()

    # Multi-label relevance is graded Jaccard similarity. This keeps evaluation
    # independent of the model and makes NDCG meaningful for partial overlap.
    labels = (labels > 0).float()
    intersection = labels @ labels.t()
    cardinality = labels.sum(dim=1)
    union = cardinality[:, None] + cardinality[None, :] - intersection
    return torch.where(union > 0, intersection / union.clamp_min(1.0), torch.zeros_like(union))


def _average_precision_from_binary(binary_relevance):
    if not np.any(binary_relevance):
        return None
    cumulative = np.cumsum(binary_relevance, dtype=np.float64)
    precision = cumulative / np.arange(1, len(binary_relevance) + 1)
    return float(np.sum(precision * binary_relevance) / np.sum(binary_relevance))


@torch.no_grad()
def evaluate_fair_image_to_image(model, loader, device, args):
    """Exact image-to-image retrieval shared by SRA, RadIR-CXR and LoFi.

    All methods use the same images, L2 normalization, cosine similarity,
    exhaustive gallery and complete sorting.  With MIMIC-IR, ground truth is
    the official graded RaTEScore matrix.  Otherwise it is derived solely from
    dataset labels and is therefore also identical across compared methods.
    """
    model.eval()
    embeddings, targets = [], []
    for samples, target in loader:
        samples = samples.to(device, non_blocking=True)
        output = model(samples)
        embedding = _embedding_from_output(output)
        embeddings.append(embedding.detach().float().cpu())
        targets.append(target.detach().cpu())

    embeddings = torch.nn.functional.normalize(torch.cat(embeddings, dim=0), p=2, dim=1)
    targets = torch.cat(targets, dim=0)
    sample_indices = targets.long().numpy() if args.dataset == 'mimic_ir' else np.arange(len(targets))

    relevance_path = args.i2i_relevance_npy
    if relevance_path:
        relevance_source = np.load(relevance_path, mmap_mode='r')
        if relevance_source.ndim != 2 or relevance_source.shape[0] != relevance_source.shape[1]:
            raise ValueError(f'Relevance matrix must be square, got {relevance_source.shape}')
        if sample_indices.max(initial=-1) >= relevance_source.shape[0]:
            raise ValueError(
                f'Sample index {sample_indices.max()} exceeds relevance matrix size '
                f'{relevance_source.shape[0]}'
            )
        relevance_kind = 'official graded RaTEScore'
    else:
        relevance_source = _label_relevance(targets)
        relevance_kind = 'label-derived relevance'

    ks = sorted(set(args.i2i_k))
    if not ks or min(ks) < 1:
        raise ValueError('--i2i-k values must be positive integers')
    max_k = min(max(ks), len(embeddings) - 1)
    if max_k < 1:
        raise ValueError('Image-to-image evaluation needs at least two samples')

    recall_hits = {k: [] for k in ks}
    ndcg_values = {k: [] for k in ks}
    average_precisions = []
    gallery = embeddings.to(device)
    chunk_size = max(1, args.i2i_query_chunk_size)

    print('\n=== Fair exhaustive image-to-image evaluation ===')
    print(f'Method tag: {args.fair_i2i_method}')
    print(f'Model: {args.model} | samples: {len(embeddings)} | relevance: {relevance_kind}')
    print('Similarity: L2-normalized cosine | self-match: excluded | ANN: disabled')

    for start in range(0, len(embeddings), chunk_size):
        stop = min(start + chunk_size, len(embeddings))
        similarities = embeddings[start:stop].to(device) @ gallery.t()
        local_rows = torch.arange(stop - start, device=device)
        global_rows = torch.arange(start, stop, device=device)
        similarities[local_rows, global_rows] = -float('inf')
        ranking = torch.argsort(similarities, dim=1, descending=True).cpu().numpy()

        for local_index, query_index in enumerate(range(start, stop)):
            if relevance_path:
                source_query_index = sample_indices[query_index]
                row = np.asarray(relevance_source[source_query_index, sample_indices], dtype=np.float32)
                if row.size and np.nanmax(row) > 1.0:
                    row = row / 100.0
            else:
                row = relevance_source[query_index].cpu().numpy().astype(np.float32, copy=False)

            row = np.nan_to_num(row, nan=0.0, posinf=1.0, neginf=0.0)
            row[query_index] = 0.0
            ordered_relevance = row[ranking[local_index]]
            binary_ordered = ordered_relevance > args.i2i_positive_threshold
            has_positive = bool(np.any(row > args.i2i_positive_threshold))

            if has_positive:
                ap = _average_precision_from_binary(binary_ordered)
                if ap is not None:
                    average_precisions.append(ap)
                for k in ks:
                    cutoff = min(k, len(binary_ordered))
                    recall_hits[k].append(float(np.any(binary_ordered[:cutoff])))

            ideal = np.sort(np.delete(row, query_index))[::-1]
            for k in ks:
                cutoff = min(k, len(ordered_relevance))
                discounts = np.log2(np.arange(2, cutoff + 2, dtype=np.float64))
                dcg = float(np.sum(ordered_relevance[:cutoff] / discounts))
                idcg = float(np.sum(ideal[:cutoff] / discounts))
                if idcg > 0:
                    ndcg_values[k].append(dcg / idcg)

    results = {
        'method': args.fair_i2i_method,
        'model': args.model,
        'num_samples': len(embeddings),
        'positive_threshold': args.i2i_positive_threshold,
        'mAP': float(np.mean(average_precisions)) if average_precisions else float('nan'),
        'recall_at_k': {str(k): float(np.mean(recall_hits[k])) if recall_hits[k] else float('nan') for k in ks},
        'ndcg_at_k': {str(k): float(np.mean(ndcg_values[k])) if ndcg_values[k] else float('nan') for k in ks},
        'valid_queries_recall': len(average_precisions),
    }

    print(f'>> mAP (relevance > {args.i2i_positive_threshold:.3f}): {results["mAP"] * 100.0:.2f}%')
    print('>> Recall@K: ' + ', '.join(f'{k}: {results["recall_at_k"][str(k)] * 100.0:.2f}%' for k in ks))
    print('>> NDCG@K: ' + ', '.join(f'{k}: {results["ndcg_at_k"][str(k)] * 100.0:.2f}%' for k in ks))
    print(f'>> Valid queries with at least one positive: {len(average_precisions)}/{len(embeddings)}')

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        stem = f'{args.fair_i2i_method}_{args.model}_fair_i2i'
        np.savez_compressed(
            os.path.join(args.save_dir, stem + '.npz'),
            embeddings=embeddings.numpy(),
            sample_indices=sample_indices,
        )
        with open(os.path.join(args.save_dir, stem + '.json'), 'w', encoding='utf-8') as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
        print(f'>> Saved embeddings and metrics to {args.save_dir}')

    return results


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    fair_i2i_enabled = args.fair_i2i_method != 'standard' or args.dataset == 'mimic_ir'
    if fair_i2i_enabled:
        expected_models = {
            'baseline': 'convnextv2',
            'sra': 'convnextv2_sra',
            'ath': 'convnextv2_ath',
            'pcam': 'convnextv2_pcam',
            'radir_cxr': 'convnextv2',
            'lofi': 'convnextv2_lofi',
            'rra_vl': 'convnextv2_rra_vl',
            'msatt': 'convnextv2_msatt',
        }
        expected_model = expected_models.get(args.fair_i2i_method)
        if expected_model is not None and args.model != expected_model:
            raise ValueError(
                f'Fair comparison requires --model {expected_model} for method '
                f'{args.fair_i2i_method!r}; got {args.model!r}.'
            )
        if expected_model is not None and not os.path.isfile(args.resume):
            raise FileNotFoundError(
                f'--fair-i2i-method {args.fair_i2i_method} requires its trained ConvNeXtV2 '
                f'checkpoint via --resume; refusing to evaluate ImageNet weights under that method name.'
            )
        if args.use_rerank_2models or args.use_text or args.use_concept_retrieval:
            raise ValueError('Fair image-to-image mode cannot use text fusion or re-ranking at inference time.')
        if args.dataset == 'mimic_ir' and not args.i2i_relevance_npy:
            raise ValueError('--dataset mimic_ir requires --i2i-relevance-npy from the official MIMIC-IR release.')

    use_two_model_rerank = False
    is_conceptclip = False
    is_biomedclip = False
    is_conceptclip_img = False
    model = None
    processor = None
    tokenizer = None
    preprocess = None
    img_model = None
    text_model = None
    text_processor = None
    checkpoint = None
    if os.path.isfile(args.resume):
        checkpoint = extract_state_dict(torch.load(args.resume, map_location=device))
    num_labels = (
        infer_num_labels_from_checkpoint(checkpoint, len(NIH_U_LABELS))
        if args.dataset == 'nih'
        else None
    )
    pcam_num_classes = (
        infer_pcam_num_classes(checkpoint, args.pcam_num_classes)
        if args.model == 'convnextv2_pcam'
        else None
    )
    if args.model == 'convnextv2_ath':
        validate_ath_checkpoint_config(checkpoint, args.ath_hash_bits, args.ath_num_classes)

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
            img_model = DenseNet121(embedding_dim=args.embedding_dim, num_labels=num_labels)
            is_conceptclip_img = False
        elif args.model == 'resnet50':
            img_model = ResNet50(embedding_dim=args.embedding_dim, num_labels=num_labels)
            is_conceptclip_img = False
        elif args.model == 'convnextv2':
            img_model = ConvNeXtV2(embedding_dim=args.embedding_dim, num_labels=num_labels)
            is_conceptclip_img = False
        elif args.model == 'convnextv2_sra':
            img_model = ConvNeXtV2_SRA(num_heads=args.sra_num_heads, lam=args.sra_lam, num_labels=num_labels)
            is_conceptclip_img = False
        elif args.model == 'convnextv2_ath':
            img_model = ConvNeXtV2_ATH(
                hash_bits=args.ath_hash_bits,
                num_classes=args.ath_num_classes,
            )
            is_conceptclip_img = False
        elif args.model == 'convnextv2_pcam':
            img_model = ConvNeXtV2_PCAM(
                num_classes=pcam_num_classes,
                lam=args.pcam_lam,
                embedding_dim=args.embedding_dim,
            )
            is_conceptclip_img = False
        elif args.model == 'swinv2':
            img_model = SwinV2(embedding_dim=args.embedding_dim)
            is_conceptclip_img = False
        elif args.model == 'dinov2':
            img_model = DinoV2(
                model_name=args.dinov2_model_name,
                embedding_dim=args.embedding_dim,
                unfreeze_blocks=args.unfreeze_blocks,
                num_labels=num_labels,
            )
            is_conceptclip_img = False
        elif args.model == 'medsiglip':
            img_model = MedSigLIP() 
            is_conceptclip_img = False
        else:
            raise NotImplementedError('Model not supported!')
        
        # Load checkpoint for image model if not ConceptCLIP
        if not is_conceptclip_img:
            if checkpoint is not None:
                print("=> loading image model checkpoint")
                load_state_dict_for_test(img_model, checkpoint, args, description='image model')
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
        elif args.model == 'biomedclip':
            if not OPEN_CLIP_AVAILABLE:
                raise ImportError('open_clip_torch is required for BiomedCLIP. Install it with: pip install open_clip_torch')
            print("Loading BiomedCLIP model...")
            loaded = create_model_from_pretrained(
                args.biomedclip_model_name,
                device=device
            )
            # open_clip versions differ: some return (model, preprocess), others
            # return (model, preprocess_train, preprocess_val)
            if isinstance(loaded, tuple):
                if len(loaded) == 2:
                    model, preprocess = loaded
                elif len(loaded) >= 3:
                    model = loaded[0]
                    preprocess = loaded[-1]
                else:
                    raise RuntimeError('Unexpected return from create_model_from_pretrained')
            else:
                raise RuntimeError('Unexpected return type from create_model_from_pretrained')
            tokenizer = get_tokenizer(args.biomedclip_model_name)
            model.eval()
            is_biomedclip = True
        elif args.model == 'dinov2':
            model = DinoV2(
                model_name=args.dinov2_model_name,
                embedding_dim=args.embedding_dim,
                unfreeze_blocks=args.unfreeze_blocks,
                num_labels=num_labels,
            )
            is_conceptclip = False
        elif args.model == 'densenet121':
            model = DenseNet121(embedding_dim=args.embedding_dim, num_labels=num_labels)
            is_conceptclip = False
        elif args.model == 'resnet50':
            model = ResNet50(embedding_dim=args.embedding_dim, num_labels=num_labels)
            is_conceptclip = False
        elif args.model == 'convnextv2':
            model = ConvNeXtV2(embedding_dim=args.embedding_dim, num_labels=num_labels)
            is_conceptclip = False
        elif args.model == 'convnextv2_sra':
            model = ConvNeXtV2_SRA(num_heads=args.sra_num_heads, lam=args.sra_lam, num_labels=num_labels)
            is_conceptclip = False
        elif args.model == 'convnextv2_lofi':
            model = ConvNeXtV2_LoFi(
                num_classes=args.lofi_num_classes,
                num_regions=args.lofi_num_regions,
                lam=args.lofi_lam,
            )
            is_conceptclip = False
        elif args.model == 'convnextv2_rra_vl':
            model = ConvNeXtV2_RRAVL(
                num_classes=args.rra_num_classes,
                num_regions=args.rra_num_regions,
                context_dim=args.rra_context_dim,
                num_heads=args.rra_num_heads,
                depth=args.rra_depth,
                lam=args.rra_lam,
            )
            is_conceptclip = False
        elif args.model == 'convnextv2_msatt':
            model = ConvNeXtV2_MSAtt(
                bottleneck_reduction=args.msatt_bottleneck_reduction,
                se_reduction=args.msatt_se_reduction,
            )
            is_conceptclip = False
        elif args.model == 'convnextv2_ath':
            model = ConvNeXtV2_ATH(
                hash_bits=args.ath_hash_bits,
                num_classes=args.ath_num_classes,
            )
            is_conceptclip = False
        elif args.model == 'convnextv2_pcam':
            model = ConvNeXtV2_PCAM(
                num_classes=pcam_num_classes,
                lam=args.pcam_lam,
                embedding_dim=args.embedding_dim,
            )
            is_conceptclip = False
        elif args.model == 'swinv2':
            model = SwinV2(embedding_dim=args.embedding_dim)
            is_conceptclip = False
        elif args.model == 'medsiglip':
            model = MedSigLIP()
            is_conceptclip = False
        else:
            raise NotImplementedError('Model not supported!')

    if not use_two_model_rerank:
        if not is_conceptclip and not is_biomedclip:
            if checkpoint is not None:
                print("=> loading checkpoint")
                load_state_dict_for_test(model, checkpoint, args)
            else:
                print("=> no checkpoint found")
            model.to(device)
        elif is_conceptclip:
            print("=> Using pre-trained ConceptCLIP (zero-shot), no checkpoint needed")
        elif is_biomedclip:
            print("=> Using pre-trained BiomedCLIP (zero-shot), no checkpoint needed")

    # ConceptCLIP uses PIL images directly (processor handles preprocessing)
    if (is_conceptclip and not use_two_model_rerank) or (use_two_model_rerank and is_conceptclip_img) or is_biomedclip:
        test_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB'))
        ])
    else:
        if args.model == 'dinov2':
            temp_model = timm.create_model(
                args.dinov2_model_name,
                pretrained=False,
                num_classes=0,
            )
            data_config = resolve_model_data_config(temp_model)
            img_size = data_config['input_size'][-1]
            normalize = transforms.Normalize(data_config['mean'], data_config['std'])
            test_transform = transforms.Compose([
                transforms.Lambda(lambda img: img.convert('RGB')),
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])

            # Use 384x384 for ConvNeXtV2 and SwinV2, 448x448 for MedSigLIP, 224x224 for other models
            if args.model == 'medsiglip':
                img_size = 448
            elif args.model in ['convnextv2', 'convnextv2_sra', 'convnextv2_ath', 'convnextv2_pcam', 'convnextv2_lofi', 'convnextv2_rra_vl', 'convnextv2_msatt', 'swinv2']:
                img_size = 384
            else:
                img_size = 224

            if args.dataset == 'nih' and args.model in ['convnextv2', 'convnextv2_sra', 'convnextv2_ath', 'convnextv2_pcam', 'convnextv2_lofi', 'convnextv2_rra_vl', 'convnextv2_msatt', 'swinv2']:
                test_transform = transforms.Compose([
                    transforms.Lambda(lambda img: img.convert('RGB')),
                    transforms.Resize(432),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    normalize
                ])
            elif args.model in ['convnextv2', 'convnextv2_sra', 'convnextv2_ath', 'convnextv2_pcam', 'convnextv2_lofi', 'convnextv2_rra_vl', 'convnextv2_msatt', 'swinv2', 'medsiglip']:
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
    elif args.dataset == 'nih':
        test_dataset = NIHChestXrayRetrievalDataSet(data_dir=args.test_dataset_dir,
                                                    image_list_file=args.test_image_list,
                                                    labels_csv_file=args.nih_labels_csv,
                                                    transform=test_transform)
    elif args.dataset == 'mimic_ir':
        if not args.mimic_ir_csv:
            raise ValueError('--dataset mimic_ir requires --mimic-ir-csv, e.g. val_caption.csv')
        test_dataset = MIMICIRImageDataset(
            root=args.test_dataset_dir,
            csv_file=args.mimic_ir_csv,
            transform=test_transform,
            path_column=args.mimic_ir_path_column,
        )
    else:
        raise NotImplementedError('Dataset not supported!')

    if args.eval_max_samples is not None and args.eval_max_samples > 0:
        if len(test_dataset) > args.eval_max_samples:
            test_dataset = Subset(test_dataset, range(args.eval_max_samples))

    # Use custom collate function for ConceptCLIP to handle PIL images
    use_conceptclip_collate = (is_conceptclip and not use_two_model_rerank) or (use_two_model_rerank and is_conceptclip_img) or is_biomedclip
    effective_workers = args.workers
    if use_conceptclip_collate and args.workers > 0 and not args.allow_pil_multiprocess:
        print("=> PIL-based loading detected. For stability in constrained environments, using num_workers=0.")
        print("=> Set --allow-pil-multiprocess to keep multiprocessing enabled.")
        effective_workers = 0
    
    if use_conceptclip_collate:
        test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                                 shuffle=False,
                                 num_workers=effective_workers,
                                 collate_fn=conceptclip_collate_fn)
    else:
        test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                                 shuffle=False,
                                 num_workers=effective_workers)

    print('Evaluating...')
    
    if fair_i2i_enabled:
        evaluate_fair_image_to_image(model, test_loader, device, args)
    elif use_two_model_rerank:
        # Two-model re-ranking approach - need separate loader for ConceptCLIP with PIL images
        label_names = get_dataset_label_names(args)
        
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
                                           num_workers=effective_workers,
                                           collate_fn=conceptclip_collate_fn)
        else:
            # Already using PIL images
            conceptclip_loader = test_loader
        
        evaluate_with_text_reranking(img_model, text_model, text_processor, test_loader, conceptclip_loader, device, args, label_names, is_conceptclip_img)
    elif is_conceptclip:
        if args.use_concept_retrieval:
            # Concept-based multi-label retrieval
            if args.concept_list:
                concept_list = args.concept_list.split(',')
            else:
                # Default concepts for each dataset
                if args.dataset == 'covid':
                    concept_list = [
                        'ground glass opacity',
                        'consolidation',
                        'pleural effusion',
                        'normal lung tissue',
                        'bilateral infiltrates',
                        'clear lung fields',
                        'pulmonary edema',
                        'pneumonia pattern'
                    ]
                elif args.dataset == 'isic':
                    concept_list = [
                        'irregular border',
                        'asymmetric shape',
                        'color variation',
                        'dark pigmentation',
                        'light brown color',
                        'smooth surface',
                        'uniform color',
                        'well-defined border'
                    ]
                elif args.dataset == 'tbx11k':
                    concept_list = [
                        'cavitation',
                        'infiltration',
                        'fibrosis',
                        'calcification',
                        'pleural thickening',
                        'normal lung',
                        'nodular pattern',
                        'miliary pattern'
                    ]
                else:
                    raise ValueError(f"Please provide --concept-list for dataset: {args.dataset}")
            
            evaluate_conceptclip_concept_retrieval(model, processor, test_loader, device, args, concept_list)
        elif args.use_text:
            # Get label names for text-enhanced retrieval
            label_names = get_dataset_label_names(args)
            
            evaluate_conceptclip_with_text(model, processor, test_loader, device, args, label_names)
        else:
            evaluate_conceptclip(model, processor, test_loader, device, args)
    elif is_biomedclip:
        evaluate_biomedclip_zeroshot(model, tokenizer, preprocess, test_loader, device, args)
    elif args.dataset == 'nih':
        evaluate_multilabels(model, test_loader, device, args)
    else:
        evaluate(model, test_loader, device, args)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Embedding Learning')

    parser.add_argument('--dataset', default='covid',
                        help='Dataset to use (covid, isic, tbx11k, nih, or mimic_ir)')
    parser.add_argument('--test-dataset-dir', default='/data/brian.hu/COVID/data/test',
                        help='Test dataset directory path')
    parser.add_argument('--test-image-list', default='./test_COVIDx4.txt',
                        help='Test image list')
    parser.add_argument('--nih-labels-csv', default=None,
                        help='NIH metadata CSV with image labels, e.g. Data_Entry_2017.csv')
    parser.add_argument('--mimic-ir-csv', default=None,
                        help='Official MIMIC-IR caption CSV whose row order matches the relevance matrix')
    parser.add_argument('--mimic-ir-path-column', default='File Path',
                        help='Image path column in --mimic-ir-csv')
    parser.add_argument('--mask-dir', default=None,
                        help='Segmentation masks path (if used)')
    parser.add_argument('--model', default='densenet121',
                        help='Model to use (densenet121, resnet50, convnextv2, convnextv2_sra, convnextv2_ath, convnextv2_pcam, convnextv2_lofi, convnextv2_rra_vl, convnextv2_msatt, swinv2, medsiglip, conceptclip, biomedclip, or dinov2)')
    parser.add_argument('--embedding-dim', default=None, type=int,
                        help='Embedding dimension of model')
    parser.add_argument('--dinov2-model-name', default='vit_base_patch14_dinov2.lvd142m', type=str,
                        help='timm model name for DINOv2 backbone')
    parser.add_argument('--unfreeze-blocks', default=3, type=int,
                        help='Number of final DINOv2 transformer blocks kept trainable when loading the model')
    parser.add_argument('--sra-num-heads', default=8, type=int,
                        help='Number of attention heads for SRA (ConvNeXtV2_SRA)')
    parser.add_argument('--sra-lam', default=0.1, type=float,
                        help='Lambda for residual attention in SRA (ConvNeXtV2_SRA)')
    parser.add_argument('--pcam-num-classes', default=None, type=int,
                        help='Number of PCAM class maps; inferred from the checkpoint when omitted')
    parser.add_argument('--pcam-lam', default=0.1, type=float,
                        help='Residual PCAM feature weight (must match training)')
    parser.add_argument('--lofi-num-classes', default=3, type=int)
    parser.add_argument('--lofi-num-regions', default=64, type=int)
    parser.add_argument('--lofi-lam', default=0.1, type=float,
                        help='Residual local-feature weight; must match LoFi adaptation training')
    parser.add_argument('--rra-num-classes', default=3, type=int)
    parser.add_argument('--rra-num-regions', default=8, type=int)
    parser.add_argument('--rra-context-dim', default=64, type=int)
    parser.add_argument('--rra-num-heads', default=8, type=int)
    parser.add_argument('--rra-depth', default=3, type=int)
    parser.add_argument('--rra-lam', default=0.1, type=float)
    parser.add_argument('--msatt-bottleneck-reduction', default=4, type=int)
    parser.add_argument('--msatt-se-reduction', default=16, type=int)
    parser.add_argument('--ath-hash-bits', default=1024, type=int,
                        help='ATH hash dimension; 1024 is capacity-matched to SRA, paper default is 36')
    parser.add_argument('--ath-num-classes', default=3, type=int,
                        help='Number of classes used by the ATH classification head')

    # Fair image-to-image comparison. RadIR-CXR and LoFi here denote the
    # training objective used to produce a ConvNeXtV2 checkpoint; inference is
    # deliberately identical to SRA (image embedding + exhaustive cosine).
    parser.add_argument(
        '--fair-i2i-method', default='standard',
        choices=['standard', 'baseline', 'sra', 'ath', 'pcam', 'radir_cxr', 'lofi', 'rra_vl', 'msatt'],
        help=(
            'Enable the shared exact image-to-image protocol and tag the checkpoint method. '
            'Use convnextv2 for baseline/radir_cxr, convnextv2_lofi for lofi, '
            'convnextv2_rra_vl for rra_vl, convnextv2_sra for sra, '
            'convnextv2_msatt for msatt, convnextv2_ath for ath, and convnextv2_pcam for pcam.'
        ),
    )
    parser.add_argument(
        '--i2i-relevance-npy', '--radir-relevance-npy', dest='i2i_relevance_npy', default=None,
        help='Square graded relevance matrix; for MIMIC-IR use the official val_ratescore.npy.',
    )
    parser.add_argument(
        '--i2i-positive-threshold', default=0.9, type=float,
        help='Binary positive threshold for Recall@K and mAP (RadIR protocol: > 0.9).',
    )
    parser.add_argument(
        '--i2i-k', default=[5, 10, 50, 100], type=int, nargs='+',
        help='Cutoffs shared by Recall@K and NDCG@K.',
    )
    parser.add_argument(
        '--i2i-query-chunk-size', default=256, type=int,
        help='Number of queries per exact cosine-ranking chunk; does not approximate ranking.',
    )
    
    # ConceptCLIP text-enhanced retrieval options
    parser.add_argument('--use-text', action='store_true',
                        help='Enable text-enhanced retrieval for ConceptCLIP')
    parser.add_argument('--use-concept-retrieval', action='store_true',
                        help='Enable concept-based multi-label retrieval using ConceptCLIP text encoder')
    parser.add_argument('--concept-list', default=None, type=str,
                        help='Comma-separated list of concepts for multi-label retrieval (e.g., "ground glass opacity,consolidation,pleural effusion")')
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
    parser.add_argument('--biomedclip-model-name', default='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', type=str,
                        help='BiomedCLIP model identifier for open_clip')
    parser.add_argument('--biomedclip-prompt-template', default='this is a medical image of {label}', type=str,
                        help='Prompt template for BiomedCLIP zero-shot classification. Must contain {label}.')
    parser.add_argument('--eval-batch-size', default=64, type=int)
    parser.add_argument('--eval-max-samples', default=None, type=int,
                        help='Evaluate at most this many samples from the test/eval dataset')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='Number of data loading workers')
    parser.add_argument('--allow-pil-multiprocess', action='store_true',
                        help='Allow num_workers>0 when loading PIL-image batches (can be unstable on low-memory environments)')
    parser.add_argument('--save-dir', default='./results',
                        help='Result save directory')
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint')
    parser.add_argument('--allow-checkpoint-mismatch', action='store_true',
                        help='Allow missing/unexpected checkpoint keys in fair I2I mode (unsafe unless audited)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
