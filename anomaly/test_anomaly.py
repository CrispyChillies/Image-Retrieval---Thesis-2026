import os
import numpy as np

import torch
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from read_data import ISICDataSet, ChestXrayDataSet

from model import ResNet50, DenseNet121
from anomaly import get_measures, print_measures
from scipy.spatial.distance import cdist


@torch.no_grad()
def evaluate(model, train_loader, test_loader, device, args):
    model.eval()

    # Train embeddings
    embeds, labels = [], []
    for data in train_loader:
        samples, _labels = data[0].to(device), data[1]
        out = model(samples)
        embeds.append(out)
        labels.append(_labels)

    embeds = torch.cat(embeds, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()

    # Get cluster centers
    class_0 = embeds[labels == 0].mean(axis=0)
    class_1 = embeds[labels == 1].mean(axis=0)

    # Test embeddings
    embeds, labels = [], []
    for data in test_loader:
        samples, _labels = data[0].to(device), data[1]
        out = model(samples)
        embeds.append(out)
        labels.append(_labels)

    embeds = torch.cat(embeds, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()

    # Compute distance of test embeddings to nearest cluster
    dists = cdist(embeds, np.stack((class_0, class_1)))
    dists = dists.min(axis=1)
    dists /= dists.max()

    # Added to save FPR, TPR, Prec, and Recall
    _pos = dists[labels == 2]
    _neg = dists[labels != 2]
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    from sklearn.metrics import roc_curve, precision_recall_curve
    fpr, tpr, _ = roc_curve(labels, examples)
    prec, recall, _ = precision_recall_curve(labels, examples)

    # Compute anomaly detection metrics
    auroc, aupr, fpr = get_measures(dists[labels == 2], dists[labels != 2])
    print_measures(auroc, aupr, fpr)

    # Save results
    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        file_name = args.resume.split('/')[-1].split('.')[0]

        save_path = os.path.join(args.save_dir, file_name)
        np.savez(save_path, auroc=auroc, aupr=aupr,
                 fpr=fpr, tpr=tpr, prec=prec, recall=recall)


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Choose model
    if args.model == 'densenet121':
        model = DenseNet121(embedding_dim=args.embedding_dim)
    elif args.model == 'resnet50':
        model = ResNet50(embedding_dim=args.embedding_dim)
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

    test_transform = transforms.Compose([transforms.Lambda(lambda image: image.convert('RGB')),
                                         transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize])

    # Set up dataset and dataloader
    if args.dataset == 'covid':
        train_dataset = ChestXrayDataSet(data_dir=os.path.join(args.dataset_dir, 'train'),
                                         image_list_file=args.train_image_list,
                                         use_covid=False,
                                         mask_dir=os.path.join(
                                             args.mask_dir, 'train') if args.mask_dir else None,
                                         transform=test_transform)
        test_dataset = ChestXrayDataSet(data_dir=os.path.join(args.dataset_dir, 'test'),
                                        image_list_file=args.test_image_list,
                                        mask_dir=os.path.join(
                                            args.mask_dir, 'test') if args.mask_dir else None,
                                        transform=test_transform)
    elif args.dataset == 'isic':
        train_dataset = ISICDataSet(data_dir=os.path.join(args.dataset_dir, 'ISIC-2017_Training_Data'),
                                    image_list_file=args.train_image_list,
                                    use_melanoma=False,
                                    mask_dir=os.path.join(
                                        args.mask_dir, 'train') if args.mask_dir else None,
                                    transform=test_transform)
        test_dataset = ISICDataSet(data_dir=os.path.join(args.dataset_dir, 'ISIC-2017_Test_v2_Data'),
                                   image_list_file=args.test_image_list,
                                   mask_dir=os.path.join(
                                       args.mask_dir, 'test') if args.mask_dir else None,
                                   transform=test_transform)
    else:
        raise NotImplementedError('Dataset not supported!')

    train_loader = DataLoader(train_dataset, batch_size=args.eval_batch_size,
                              shuffle=False,
                              num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                             shuffle=False,
                             num_workers=args.workers)

    print('Evaluating...')
    evaluate(model, train_loader, test_loader, device, args)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Embedding Learning')

    parser.add_argument('--dataset', default='covid',
                        help='Dataset to use (covid or isic)')
    parser.add_argument('--dataset-dir', default='/data/brian.hu/COVID/data',
                        help='Test dataset directory path')
    parser.add_argument('--train-image-list', default='./train_split.txt',
                        help='Train image list')
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
