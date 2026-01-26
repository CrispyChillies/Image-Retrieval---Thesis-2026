import os
import numpy as np
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import torch
from torch.utils.data import DataLoader


import torchvision.transforms as transforms
from read_data import ISICDataSet, ChestXrayDataSet, TBX11kDataSet

from model import ConvNeXtV2, ResNet50, DenseNet121




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

    covid_labels = ['No Finding', 'Pneumonia', 'COVID-19'] 
    prompts = [f'a radiographic representation assessing for {l}' for l in covid_labels]

    K = 50 # Number of rerank candidates
    alpha = 0.7 # Combination weights: alpha * ConvNeXt + (1-alpha) * ConceptCLIP

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
        

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize a tensor image with mean and standard deviation."""
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean


# @torch.no_grad()
# def evaluate(model, loader, device, args):
#     from transformers import AutoModel, AutoProcessor
    
#     # 1. Initialize ConceptCLIP (Ideally passed in or initialized once)
#     c_model = AutoModel.from_pretrained('JerrryNie/ConceptCLIP', trust_remote_code=True).to(device)
#     c_processor = AutoProcessor.from_pretrained('JerrryNie/ConceptCLIP', trust_remote_code=True)
#     c_model.eval()

#     model.eval()
#     embeds, labels, raw_samples = [], [], []

#     # Keep track of raw samples for ConceptCLIP processing
#     for data in loader:
#         samples = data[0].to(device)
#         _labels = data[1].to(device)
#         out = model(samples)
#         embeds.append(out)
#         labels.append(_labels)
#         raw_samples.append(samples)

#     embeds = torch.cat(embeds, dim=0)
#     labels = torch.cat(labels, dim=0)
#     raw_samples = torch.cat(raw_samples, dim=0)

#     # Calculate initial Backbone distances
#     dists = -torch.cdist(embeds, embeds)
#     dists.fill_diagonal_(float('-inf'))

#     # --- START RERANKING BLOCK ---
#     covid_labels = ['No Finding', 'Pneumonia', 'COVID-19'] 
#     prompts = [f'a radiographic representation assessing for {l}' for l in covid_labels]
    
#     K = 50 # Number of rerank candidates
#     alpha = 0.7 # Combination weights: alpha * ConvNeXt + (1-alpha) * ConceptCLIP
    
#     # Identify top-K candidates from backbone to rerank
#     _, top_k_indices = torch.topk(dists, K, dim=1)
#     new_dists = dists.clone()

#     print(f'>> Reranking top {K} candidates with ConceptCLIP...')
#     for i in range(embeds.size(0)):
#         # Denormalize query image for ConceptCLIP
#         query_img_denorm = denormalize(raw_samples[i])
        
#         # Extract ConceptCLIP features for Query i
#         query_input = c_processor(images=query_img_denorm, text=prompts, return_tensors='pt', padding=True).to(device)
#         query_out = c_model(**query_input)
#         # IT-Align: Global image-text alignment [cite: 18, 434]
#         query_sim = query_out['image_features'] @ query_out['text_features'].t()

#         for j in top_k_indices[i]:
#             # Denormalize candidate image for ConceptCLIP
#             cand_img_denorm = denormalize(raw_samples[j])
            
#             # Extract ConceptCLIP features for Candidate j
#             cand_input = c_processor(images=cand_img_denorm, text=prompts, return_tensors='pt', padding=True).to(device)
#             cand_out = c_model(**cand_input)
#             cand_sim = cand_out['image_features'] @ cand_out['text_features'].t()

#             # Calculate semantic consistency score
#             semantic_score = torch.nn.functional.cosine_similarity(query_sim, cand_sim, dim=1)
            
#             # Update distance with weighted fusion [cite: 490]
#             new_dists[i, j] = (alpha * dists[i, j]) + ((1 - alpha) * semantic_score)
            
#     dists = new_dists # Use reranked distances for all subsequent metrics
#     # --- END RERANKING BLOCK ---

#     # top-k accuracy (i.e. R@K)
#     kappas = [1, 5, 10]
#     accuracy = retrieval_accuracy(dists, labels, topk=kappas)
#     accuracy = torch.stack(accuracy).cpu().numpy()
#     print('>> R@K{}: {}%'.format(kappas, np.around(accuracy, 2)))

#     # mean average precision and mean precision (i.e. mAP and pr)
#     ranks = torch.argsort(dists, dim=1, descending=True) # Sort row-wise for distance matrix
#     mAP, _, pr, _ = compute_map(ranks.t().cpu().numpy(), labels.cpu().numpy(), kappas)
#     print('>> mAP: {:.2f}%'.format(mAP * 100.0))
#     print('>> mP@K{}: {}%'.format(kappas, np.around(pr * 100.0, 2)))
    
#     # Classification metrics with majority voting
#     print('\n>> Classification Metrics (Majority Voting):')
#     k_values = [1, 5, 10, 15, 20]
#     classification_results = compute_classification_metrics(labels, dists, k_values)
    
#     for k in k_values:
#         metrics = classification_results[k]
#         print(f'\n>> Top-{k} Retrieved Images:')
#         print(f'   Accuracy: {metrics["accuracy"]:.2f}%')
#         print(f'   Precision (macro): {metrics["precision_macro"]:.2f}%')
#         print(f'   Recall (macro): {metrics["recall_macro"]:.2f}%')
#         print(f'   F1 (macro): {metrics["f1_macro"]:.2f}%')
#         print(f'   Precision (weighted): {metrics["precision_weighted"]:.2f}%')
#         print(f'   Recall (weighted): {metrics["recall_weighted"]:.2f}%')
#         print(f'   F1 (weighted): {metrics["f1_weighted"]:.2f}%')

#     # Save results
#     if args.save_dir:
#         if not os.path.exists(args.save_dir):
#             os.makedirs(args.save_dir)
#         file_name = args.resume.split('/')[-1].split('.')[0]

#         save_path = os.path.join(args.save_dir, file_name)
#         # Convert classification_results dict to numpy arrays for saving
#         classification_k_values = list(classification_results.keys())
#         classification_metrics = {k: v for k, v in classification_results.items()}
        
#         np.savez(save_path, embeds=embeds.cpu().numpy(),
#                  labels=labels.cpu().numpy(), dists=-dists.cpu().numpy(),
#                  kappas=kappas, acc=accuracy, mAP=mAP, pr=pr,
#                  classification_k_values=classification_k_values,
#                  **{f'classification_k{k}': np.array(list(v.values())) for k, v in classification_metrics.items()})


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Choose model
    if args.model == 'densenet121':
        model = DenseNet121(embedding_dim=args.embedding_dim)
    elif args.model == 'resnet50':
        model = ResNet50(embedding_dim=args.embedding_dim)
    elif args.model == 'convnextv2':
        model = ConvNeXtV2(embedding_dim=args.embedding_dim)
    else:
        raise NotImplementedError('Model not supported!')

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

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    # test_transform = transforms.Compose([transforms.Lambda(lambda image: image.convert('RGB')),
    #                                      transforms.Resize(256),
    #                                      transforms.CenterCrop(224),
    #                                      transforms.ToTensor(),
    #                                      normalize])

    # Use 384x384 for ConvNeXtV2, 224x224 for other models
    img_size = 384 if args.model == 'convnextv2' else 224

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
                        help='Model to use (densenet121 or resnet50)')
    parser.add_argument('--embedding-dim', default=None, type=int,
                        help='Embedding dimension of model')
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
