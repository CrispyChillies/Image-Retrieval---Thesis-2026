import os
import numpy as np
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import torch
from torch.utils.data import DataLoader


import torchvision.transforms as transforms
from read_data import ISICDataSet, ChestXrayDataSet, TBX11kDataSet, VINDRDataSet

from model import ConvNeXtV2, ResNet50, DenseNet121, HybridConvNeXtViT, ConceptCLIPBackbone, Resnet50_with_Attention, MedSigLIPRetrieval


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
def evaluate_multilabels(model, loader, device, args):
    model.eval()
    embeds, labels = [], []

    for data in loader:
        samples = data[0].to(device)
        _labels = data[1].to(device) 
        out = model(samples)
        
        embedding = out[0] if isinstance(out, tuple) else out
        embeds.append(embedding)
        labels.append(_labels)

    embeds = torch.cat(embeds, dim=0)
    labels = torch.cat(labels, dim=0)

    embeds_norm = torch.nn.functional.normalize(embeds, p=2, dim=1)
    dists = torch.mm(embeds_norm, embeds_norm.t())
    dists.fill_diagonal_(-float('inf'))

    print('\n--- VinDr-CXR Retrieval Results ---')
    
    for t in [0.25, 0.5]:
        mAP_val = compute_map_multilabel(dists, labels, threshold=t)
        print(f'>> mAP (Jaccard > {t}): {mAP_val * 100.0:.2f}%')

    k_values = [1, 5, 10, 15, 20]
    ranks = torch.argsort(dists, dim=1, descending=True)

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
            top_k_labels = labels_np[ranks_np[i, :k]]
            matches = (top_k_labels * query_label).sum(axis=1) > 0
            num_matches = np.sum(matches)

            total_precision += (num_matches / k)
            
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
        if isinstance(out, tuple):
            embeds.append(out[0])
        else:
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
        os.makedirs(args.save_dir, exist_ok=True)
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
        

@torch.no_grad()
def evaluate_zeroshot(model, loader, device, args):
    """Evaluate model in zero-shot setting (no fine-tuning)"""
    model.eval()
    embeds, labels = [], []

    print('\n>> Extracting embeddings for zero-shot retrieval...')
    
    for data in loader:
        samples = data[0].to(device)
        _labels = data[1].to(device)
        out = model(samples)
        if isinstance(out, tuple):
            embeds.append(out[0])
        else:
            embeds.append(out)
        labels.append(_labels)

    embeds = torch.cat(embeds, dim=0)
    labels = torch.cat(labels, dim=0)

    # Compute similarity matrix
    dists = -torch.cdist(embeds, embeds)
    dists.fill_diagonal_(float('-inf'))

    print('\n' + '='*60)
    print(f'Zero-Shot Retrieval Results ({args.model.upper()})')
    print('='*60)

    # top-k accuracy (i.e. R@K)
    kappas = [1, 5, 10]
    accuracy = retrieval_accuracy(dists, labels, topk=kappas)
    accuracy = torch.stack(accuracy).cpu().numpy()
    print('\n>> Retrieval Accuracy:')
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

    # Save results
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        file_name = f'zeroshot_{args.model}'
        save_path = os.path.join(args.save_dir, file_name)
        
        np.savez(save_path, 
                 embeds=embeds.cpu().numpy(),
                 labels=labels.cpu().numpy(), 
                 dists=-dists.cpu().numpy(),
                 kappas=kappas, 
                 acc=accuracy, 
                 mAP=mAP, 
                 pr=pr,
                 classification_k_values=k_values,
                 **{f'classification_k{k}': np.array(list(v.values())) 
                    for k, v in classification_results.items()})
        print(f'\n>> Results saved to {save_path}.npz')
        print('='*60)
        

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
    elif args.model == 'conceptclip':
        model = ConceptCLIPBackbone(
            pretrained=True,
            embedding_dim=args.embedding_dim,
            freeze=True,
            processor_normalize=True
        )
    elif args.model == 'medsiglip':
        model = MedSigLIPRetrieval(
            model_name="google/medsiglip-448",
            embed_dim=args.embedding_dim,
            zero_shot=args.zero_shot if hasattr(args, 'zero_shot') else True
        )
    elif args.model == 'resnet50_attention':
        model = Resnet50_with_Attention(embedding_dim=args.embedding_dim)
    else:
        raise NotImplementedError(f'Model not supported: {args.model}')

    # Load checkpoint if provided (for fine-tuned models)
    if args.resume and os.path.isfile(args.resume):
        print("=> loading checkpoint")
        checkpoint = torch.load(args.resume)
        if 'state-dict' in checkpoint:
            checkpoint = checkpoint['state-dict']
        model.load_state_dict(checkpoint, strict=False)
        print("=> loaded checkpoint")
    elif args.zero_shot:
        print("=> running zero-shot evaluation with pretrained weights")
    else:
        print("=> no checkpoint found, using pretrained weights")

    model.to(device)

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    # Use 448x448 for MedSigLIP, 384x384 for ConvNeXtV2 and Hybrid model, 224x224 for other models
    if args.model == 'medsiglip':
        img_size = 448
        resize_size = 480
    elif args.model in ['convnextv2', 'hybrid_convnext_vit']:
        img_size = 384
        resize_size = 416
    else:
        img_size = 224
        resize_size = 256

    # test_transform = transforms.Compose([
    #     transforms.Lambda(lambda img: img.convert('RGB')),
    #     transforms.Resize(resize_size),
    #     transforms.CenterCrop(img_size),
    #     transforms.ToTensor(),
    #     normalize
    # ])
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
    elif args.dataset == 'vindr':
        test_dataset = VINDRDataSet(data_dir=args.test_dataset_dir,
                                   csv_file=args.test_image_list,
                                   transform=test_transform)
    else:
        raise NotImplementedError('Dataset not supported!')

    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                             shuffle=False,
                             num_workers=args.workers)

    print('Evaluating...')
    
    # Use zero-shot evaluation if flag is set
    if args.zero_shot:
        evaluate_zeroshot(model, test_loader, device, args)
    elif args.dataset == 'vindr':
        evaluate_multilabels(model, test_loader, device, args)
    else:   
        evaluate(model, test_loader, device, args)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Embedding Learning')

    parser.add_argument('--dataset', default='covid',
                        help='Dataset to use (covid, isic, tbx11k, or vindr)')
    parser.add_argument('--test-dataset-dir', default='/data/brian.hu/COVID/data/test',
                        help='Test dataset directory path')
    parser.add_argument('--test-image-list', default='./test_COVIDx4.txt',
                        help='Test image list')
    parser.add_argument('--mask-dir', default=None,
                        help='Segmentation masks path (if used)')
    parser.add_argument('--model', default='densenet121',
                        help='Model to use (densenet121, resnet50, convnextv2, medsiglip, hybrid_convnext_vit, medclip_vit, medclip_resnet, or conceptclip)')
    parser.add_argument('--embedding-dim', default=None, type=int,
                        help='Embedding dimension of model')
    parser.add_argument('--zero-shot', action='store_true',
                        help='Run zero-shot evaluation with pretrained weights only')
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
