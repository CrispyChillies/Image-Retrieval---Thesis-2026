import os
import numpy as np
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import torch
from torch.utils.data import DataLoader


import torchvision.transforms as transforms
from read_data import ISICDataSet, ChestXrayDataSet, TBX11kDataSet

from model import ConvNeXtV2, ResNet50, DenseNet121, HybridConvNeXtViT
from cross_encoder import create_cross_encoder_from_checkpoint

# MedCLIP imports
try:
    from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel
    from medclip import MedCLIPProcessor
    MEDCLIP_AVAILABLE = True
except ImportError:
    MEDCLIP_AVAILABLE = False
    print("Warning: MedCLIP not available. Install with: pip install medclip")


class MedCLIPWrapper(torch.nn.Module):
    """Wrapper for MedCLIP model to extract image embeddings for retrieval."""
    
    def __init__(self, vision_cls=None, embedding_dim=512):
        super(MedCLIPWrapper, self).__init__()
        if not MEDCLIP_AVAILABLE:
            raise ImportError("MedCLIP is not installed. Install with: pip install medclip")
        
        # Default to ViT if not specified
        if vision_cls is None:
            vision_cls = MedCLIPVisionModelViT
        
        self.model = MedCLIPModel(vision_cls=vision_cls)
        self.model.from_pretrained()
        self.embedding_dim = embedding_dim
        
        # Add projection layer if needed to match desired embedding_dim
        # MedCLIP default embedding is 512, so only add if different
        actual_dim = 512  # MedCLIP default
        if embedding_dim != actual_dim:
            self.projection = torch.nn.Linear(actual_dim, embedding_dim)
        else:
            self.projection = None
    
    def forward(self, x):
        """Extract image embeddings.
        
        Args:
            x: Input images (already preprocessed by MedCLIPProcessor)
        
        Returns:
            Normalized image embeddings
        """
        # MedCLIP expects dict with 'pixel_values' key
        if isinstance(x, dict):
            outputs = self.model(**x)
        else:
            # If tensor is passed directly
            outputs = self.model(pixel_values=x)
        
        # Extract image embeddings
        img_embeds = outputs['img_embeds']
        
        # Apply projection if needed
        if self.projection is not None:
            img_embeds = self.projection(img_embeds)
        
        # Normalize embeddings
        img_embeds = torch.nn.functional.normalize(img_embeds, p=2, dim=1)
        
        return img_embeds


class MedCLIPDatasetWrapper(torch.utils.data.Dataset):
    """Wrapper to apply MedCLIP processor to existing dataset."""
    
    def __init__(self, base_dataset, processor):
        self.base_dataset = base_dataset
        self.processor = processor
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get image and label from base dataset
        # The base dataset returns (transformed_image, label)
        # But we need raw PIL image for MedCLIP processor
        # So we need to get the raw image
        img_path = self.base_dataset.images[idx]
        label = self.base_dataset.labels[idx]
        
        # Load raw PIL image
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        
        # Apply MedCLIP processor
        inputs = self.processor(images=image, return_tensors="pt")
        # Remove batch dimension added by processor
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        return pixel_values, label


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
def evaluate(model, loader, device, args):
    """
    Evaluate model with optional cross-encoder re-ranking.
    """
    model.eval()
    embeds, labels, raw_images = [], [], []

    # Extract embeddings and optionally store raw images for cross-encoder
    for data in loader:
        samples = data[0].to(device)
        _labels = data[1].to(device)
        out = model(samples)
        embeds.append(out)
        labels.append(_labels)
        
        # Store raw images if cross-encoder is enabled
        if args.use_cross_encoder:
            raw_images.append(samples)

    embeds = torch.cat(embeds, dim=0)
    labels = torch.cat(labels, dim=0)
    if args.use_cross_encoder:
        raw_images = torch.cat(raw_images, dim=0)
 
    # Initial retrieval using embedding similarity
    dists = -torch.cdist(embeds, embeds)
    dists.fill_diagonal_(float('-inf'))

    print('\n' + '='*70)
    print('BASELINE: Embedding-based Retrieval')
    print('='*70)

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
        print(f'   F1-score (macro): {metrics["f1_macro"]:.2f}%')
    
    # Cross-encoder re-ranking
    if args.use_cross_encoder:
        print('\n' + '='*70)
        print(f'CROSS-ENCODER RE-RANKING (Top-{args.top_k_rerank} candidates)')
        print('='*70)
        
        # Get model class for creating cross-encoder
        model_classes = {
            'densenet121': DenseNet121,
            'resnet50': ResNet50,
            'convnextv2': ConvNeXtV2,
            'hybrid_convnext_vit': HybridConvNeXtViT
        }
        model_class = model_classes[args.model]
        
        # Create cross-encoder from checkpoint
        print(f'>> Loading cross-encoder from: {args.resume}')
        cross_encoder = create_cross_encoder_from_checkpoint(
            args.resume,
            model_class,
            embedding_dim=args.embedding_dim
        ).to(device)
        cross_encoder.eval()
        
        # Re-rank for each query
        dists_reranked = dists.clone()
        n_queries = len(raw_images)
        
        print(f'>> Re-ranking {n_queries} queries...')
        for query_idx in range(n_queries):
            query_img = raw_images[query_idx:query_idx+1]
            initial_scores = dists[:, query_idx]
            
            # Re-rank top-K using cross-encoder
            reranked_indices, reranked_scores, topk_indices = cross_encoder.rerank_top_k(
                query_img,
                raw_images,
                initial_scores,
                top_k=args.top_k_rerank
            )
            
            # Update distance matrix with re-ranked scores
            # Only update the top-K positions
            for rank_pos, (orig_idx, new_score) in enumerate(zip(reranked_indices, reranked_scores)):
                dists_reranked[orig_idx, query_idx] = new_score
            
            # Progress indicator
            if (query_idx + 1) % 50 == 0:
                print(f'   Progress: {query_idx + 1}/{n_queries} queries processed')
        
        print(f'>> Re-ranking completed!')
        
        # Evaluate re-ranked results
        print('\n' + '='*70)
        print('RESULTS AFTER CROSS-ENCODER RE-RANKING')
        print('='*70)
        
        accuracy_reranked = retrieval_accuracy(dists_reranked, labels, topk=kappas)
        accuracy_reranked = torch.stack(accuracy_reranked).cpu().numpy()
        print('>> R@K{}: {}%'.format(kappas, np.around(accuracy_reranked, 2)))
        
        ranks_reranked = torch.argsort(dists_reranked, dim=0, descending=True)
        mAP_reranked, _, pr_reranked, _ = compute_map(
            ranks_reranked.cpu().numpy(),
            labels.cpu().numpy(),
            kappas
        )
        print('>> mAP: {:.2f}%'.format(mAP_reranked * 100.0))
        print('>> mP@K{}: {}%'.format(kappas, np.around(pr_reranked * 100.0, 2)))
        
        # Re-ranked classification metrics
        print('\n>> Classification Metrics (Majority Voting, Re-ranked):')
        classification_results_reranked = compute_classification_metrics(labels, dists_reranked, k_values)
        
        for k in k_values:
            metrics = classification_results_reranked[k]
            print(f'\n>> Top-{k} Retrieved Images:')
            print(f'   Accuracy: {metrics["accuracy"]:.2f}%')
            print(f'   Precision (macro): {metrics["precision_macro"]:.2f}%')
            print(f'   Recall (macro): {metrics["recall_macro"]:.2f}%')
            print(f'   F1-score (macro): {metrics["f1_macro"]:.2f}%')
        
        # Improvement summary
        print('\n' + '='*70)
        print('IMPROVEMENT SUMMARY')
        print('='*70)
        print(f'>> mAP improvement: {(mAP_reranked - mAP) * 100:+.2f}% absolute')
        print(f'>> R@1 improvement: {(accuracy_reranked[0] - accuracy[0]):+.2f}% absolute')
        print(f'>> R@5 improvement: {(accuracy_reranked[1] - accuracy[1]):+.2f}% absolute')
        print(f'>> R@10 improvement: {(accuracy_reranked[2] - accuracy[2]):+.2f}% absolute')
        print('='*70)
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

    # Choose model
    if args.model == 'densenet121':
        model = DenseNet121(embedding_dim=args.embedding_dim)
    elif args.model == 'resnet50':
        model = ResNet50(embedding_dim=args.embedding_dim)
    elif args.model == 'convnextv2':
        model = ConvNeXtV2(embedding_dim=args.embedding_dim)
    elif args.model == 'hybrid_convnext_vit':
        model = HybridConvNeXtViT(embedding_dim=args.embedding_dim)
    elif args.model == 'medclip':
        if not MEDCLIP_AVAILABLE:
            raise ImportError("MedCLIP is not installed. Install with: pip install medclip")
        # Use ViT version by default (better performance)
        model = MedCLIPWrapper(
            vision_cls=MedCLIPVisionModelViT,
            embedding_dim=args.embedding_dim if args.embedding_dim else 512
        )
    elif args.model == 'medclip_resnet':
        if not MEDCLIP_AVAILABLE:
            raise ImportError("MedCLIP is not installed. Install with: pip install medclip")
        # ResNet50 version
        model = MedCLIPWrapper(
            vision_cls=MedCLIPVisionModel,
            embedding_dim=args.embedding_dim if args.embedding_dim else 512
        )
    else:
        raise NotImplementedError('Model not supported!')

    # Load checkpoint (skip for MedCLIP as it uses pretrained weights)
    if os.path.isfile(args.resume):
        if args.model not in ['medclip', 'medclip_resnet']:
            print("=> loading checkpoint")
            checkpoint = torch.load(args.resume)
            if 'state-dict' in checkpoint:
                checkpoint = checkpoint['state-dict']
            model.load_state_dict(checkpoint, strict=False)
            print("=> loaded checkpoint")
        else:
            print("=> MedCLIP uses pretrained weights, skipping checkpoint loading")
    else:
        if args.model not in ['medclip', 'medclip_resnet']:
            print("=> no checkpoint found")

    model.to(device)

    # MedCLIP uses its own processor, others use standard transforms
    if args.model in ['medclip', 'medclip_resnet']:
        # MedCLIP handles preprocessing internally
        medclip_processor = MedCLIPProcessor()
        
        # Create base dataset without transforms
        if args.dataset == 'covid':
            base_dataset = ChestXrayDataSet(data_dir=args.test_dataset_dir,
                                           image_list_file=args.test_image_list,
                                           mask_dir=args.mask_dir,
                                           transform=None)
        elif args.dataset == 'isic':
            base_dataset = ISICDataSet(data_dir=args.test_dataset_dir,
                                      image_list_file=args.test_image_list,
                                      mask_dir=args.mask_dir,
                                      transform=None)
        elif args.dataset == 'tbx11k':
            base_dataset = TBX11kDataSet(data_dir=args.test_dataset_dir,
                                        csv_file=args.test_image_list,
                                        transform=None)
        else:
            raise NotImplementedError('Dataset not supported!')
        
        # Wrap with MedCLIP processor
        test_dataset = MedCLIPDatasetWrapper(base_dataset, medclip_processor)
    else:
        # Standard preprocessing for other models
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])

        # Use 384x384 for ConvNeXtV2 and Hybrid model, 224x224 for other models
        img_size = 384 if args.model in ['convnextv2', 'hybrid_convnext_vit'] else 224

        test_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])

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

    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                             shuffle=False,
                             num_workers=args.workers)

    print('Evaluating...')
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
                        help='Model to use (densenet121, resnet50, convnextv2, hybrid_convnext_vit, medclip, or medclip_resnet)')
    parser.add_argument('--embedding-dim', default=None, type=int,
                        help='Embedding dimension of model')
    parser.add_argument('--eval-batch-size', default=64, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='Number of data loading workers')
    parser.add_argument('--save-dir', default='./results',
                        help='Result save directory')
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint')
    
    # Cross-encoder re-ranking arguments
    parser.add_argument('--use-cross-encoder', action='store_true',
                        help='Enable cross-encoder re-ranking for improved accuracy')
    parser.add_argument('--top-k-rerank', default=20, type=int,
                        help='Number of top candidates to re-rank with cross-encoder (default: 20)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
