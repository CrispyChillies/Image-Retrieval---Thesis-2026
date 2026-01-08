import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from read_data import ISICDataSet, ChestXrayDataSet

from model import ResNet50, DenseNet121
from explanations import SBSMBatch, SimAtt, SimCAM

from PIL import Image
from torch.utils.data import Dataset


def rank_retrieval(dists, labels, topk=1):
    """Finds top-k closest embeddings"""
    dists_copy = dists.copy()
    # fill diagonal with nan
    np.fill_diagonal(dists_copy, np.nan)

    # rank based on distance
    idx = np.argsort(dists_copy, axis=1)[:, :topk]
    pred = labels[idx]

    return pred, idx


class ImageListDataSet(Dataset):
    def __init__(self, image_dir, image_list, transform=None):
        """
        Args:
            image_dir: path to directory of images.
            image_list: list of image filenames.
            transform: optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.image_list = image_list
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its filename
        """
        image_name = self.image_list[index]
        image = Image.open(os.path.join(
            self.image_dir, image_name)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, image_name

    def __len__(self):
        return len(self.image_list)


def process(explainer, loader, device, args):
    if args.self_saliency:  # self-saliency
        dataset = ImageListDataSet(
            image_dir='', image_list=loader.dataset.image_names, transform=loader.dataset.transform)
        loader = DataLoader(dataset, batch_size=args.eval_batch_size *
                            torch.cuda.device_count(), num_workers=args.workers)

        for i, data in enumerate(loader):
            samples, paths = data[0].to(device), data[1]
            if args.explainer == 'sbsm':
                salmaps = explainer(samples)
            else:
                salmaps = explainer(samples, samples)

            # convert to numpy
            salmaps = salmaps.cpu().numpy()

            # save output map
            base_path = os.path.join(args.save_dir)
            if not os.path.exists(base_path):
                os.makedirs(base_path)

            for s, p in zip(salmaps, paths):
                np.save(os.path.join(base_path, p.split('/')[-1]), s)
    else:
        # Load results
        results = np.load(args.results)
        pred, idx = rank_retrieval(
            results['dists'], results['labels'], topk=args.topk)
        image_list = loader.dataset.image_names

        for img, ind in zip(image_list, idx):
            # Transform the query image
            x_q = loader.dataset.transform(
                Image.open(img)).unsqueeze(0).to(device)
            x_q = torch.cat([x_q]*torch.cuda.device_count())

            # Redefine loader here for each query image
            x_r = [image_list[i] for i in ind]
            dataset = ImageListDataSet(
                image_dir='', image_list=x_r, transform=loader.dataset.transform)
            loader = DataLoader(dataset, batch_size=args.eval_batch_size *
                                torch.cuda.device_count(), num_workers=args.workers)

            for i, data in enumerate(loader):
                samples, paths = data[0].to(device), data[1]
                salmaps = explainer(x_q, samples)

                # convert to numpy
                salmaps = salmaps.cpu().numpy()

                # save output map
                base_path = os.path.join(args.save_dir, img.split('/')[-1])
                if not os.path.exists(base_path):
                    os.makedirs(base_path)

                for s, p in zip(reversed(salmaps), reversed(paths)):
                    np.save(os.path.join(base_path, p.split('/')[-1]), s)


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

    model.eval()

    # Choose explanation method
    if args.explainer == 'sbsm':
        explainer = SBSMBatch(model, input_size=(
            224, 224), gpu_batch=args.gpu_batch)

        maskspath = 'masks.npy'
        if not os.path.isfile(maskspath):
            explainer.generate_masks(
                window_size=24, stride=5, savepath=maskspath)
        else:
            explainer.load_masks(maskspath)
            print('Masks are loaded.')
    elif args.explainer == 'simatt':
        model = nn.Sequential(*list(model.children())
                              [0], *list(model.children())[1:])
        # TODO: Currently DenseNet121-specific
        explainer = SimAtt(model, model[0], target_layers=["relu"])
    elif args.explainer == 'simcam':
        model = nn.Sequential(*list(model.children())
                              [0], *list(model.children())[1:])
        # TODO: Currently DenseNet121-specific
        explainer = SimCAM(model, model[0], target_layers=[
                           "relu"], fc=model[2] if args.embedding_dim else None)
    else:
        raise NotImplementedError('Explainer not supported!')

    # DataParallel over multiple GPUs
    explainer = nn.DataParallel(explainer)
    explainer.to(device)

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

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
    else:
        raise NotImplementedError('Dataset not supported!')

    test_loader = DataLoader(test_dataset,
                             batch_size=args.eval_batch_size*torch.cuda.device_count(),
                             shuffle=False, num_workers=args.workers)

    # Compute and save saliency maps
    print('Evaluating...')
    with torch.set_grad_enabled(args.explainer != 'sbsm'):
        process(explainer, test_loader, device, args)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Saliency Evaluation')

    parser.add_argument('--dataset', default='covid',
                        help='Dataset to use (covid or isic)')
    parser.add_argument('--test-dataset-dir', default='/data/brian.hu/COVID/data/test',
                        help='Test dataset directory path')
    parser.add_argument('--test-image-list', default='./test_COVIDx4.txt',
                        help='Test image list')
    parser.add_argument('--mask-dir', default=None,
                        help='Segmentation masks path (if used)')
    parser.add_argument('--results', default=None,
                        help='Results file to load')
    parser.add_argument('--model', default='densenet121',
                        help='Model to use (densenet121 or resnet50)')
    parser.add_argument('--embedding-dim', default=None, type=int,
                        help='Embedding dimension of model')
    parser.add_argument('--explainer', default='sbsm',
                        help='Explanation type (sbsm, simatt, or simcam)')
    parser.add_argument('--self-saliency', action='store_true',
                        help='Compute self-similarity saliency')
    parser.add_argument('--eval-batch-size', default=1, type=int)
    parser.add_argument('--gpu-batch', default=250, type=int,
                        help='Internal batch size (only used for sbsm)')
    parser.add_argument('--topk', default=5, type=int,
                        help='Number of top-k images to compute saliency')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='Number of data loading workers')
    parser.add_argument('--save-dir', default='./saliency',
                        help='Result save directory')
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
