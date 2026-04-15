import argparse
import os
import random

import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from loss import TripletMarginLoss, WeightedMultiLabelTripletLoss
from model import ConvNeXtV2_SRA
from read_data import ChestXrayDataSet, ISICDataSet, TBX11kDataSet, VINDRDataSet
from sampler import PKSampler
from train import evaluate, save, train_epoch


def build_datasets(args, train_transform, test_transform):
    if args.dataset == 'covid':
        train_dataset = ChestXrayDataSet(
            data_dir=os.path.join(args.dataset_dir, 'train'),
            image_list_file=args.train_image_list,
            use_covid=not args.anomaly,
            mask_dir=os.path.join(args.mask_dir, 'train') if args.mask_dir else None,
            transform=train_transform,
        )
        test_dataset = ChestXrayDataSet(
            data_dir=os.path.join(args.dataset_dir, 'test'),
            image_list_file=args.test_image_list,
            mask_dir=os.path.join(args.mask_dir, 'test') if args.mask_dir else None,
            transform=test_transform,
        )
    elif args.dataset == 'isic':
        train_dataset = ISICDataSet(
            data_dir=os.path.join(args.dataset_dir, 'ISIC-2017_Training_Data'),
            image_list_file=args.train_image_list,
            use_melanoma=not args.anomaly,
            mask_dir=os.path.join(args.mask_dir, 'train') if args.mask_dir else None,
            transform=train_transform,
        )
        test_dataset = ISICDataSet(
            data_dir=os.path.join(args.dataset_dir, 'ISIC-2017_Test_v2_Data'),
            image_list_file=args.test_image_list,
            mask_dir=os.path.join(args.mask_dir, 'test') if args.mask_dir else None,
            transform=test_transform,
        )
    elif args.dataset == 'tbx11k':
        train_dataset = TBX11kDataSet(
            data_dir=os.path.join(args.dataset_dir, 'train'),
            csv_file=args.train_image_list,
            transform=train_transform,
        )
        test_dataset = TBX11kDataSet(
            data_dir=os.path.join(args.dataset_dir, 'test'),
            csv_file=args.test_image_list,
            transform=test_transform,
        )
    elif args.dataset == 'vindr':
        train_dataset = VINDRDataSet(
            data_dir=os.path.join(args.dataset_dir, 'train/train'),
            csv_file=args.train_image_list,
            transform=train_transform,
        )
        test_dataset = VINDRDataSet(
            data_dir=os.path.join(args.dataset_dir, 'test/test'),
            csv_file=args.test_image_list,
            transform=test_transform,
        )
    else:
        raise NotImplementedError('Dataset not supported for SRA training.')

    return train_dataset, test_dataset


def build_loaders(args, train_dataset, test_dataset, rank, world_size, batch_size, per_gpu_batch_size):
    targets = train_dataset.labels
    p = args.labels_per_batch if not args.anomaly else args.labels_per_batch - 1
    k = args.samples_per_label

    if args.use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=per_gpu_batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=True,
            prefetch_factor=2 if args.workers > 0 else None,
            persistent_workers=True if args.workers > 0 else False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            sampler=test_sampler,
            num_workers=args.workers,
            pin_memory=True,
            prefetch_factor=2 if args.workers > 0 else None,
            persistent_workers=True if args.workers > 0 else False,
        )
    else:
        if args.use_random_sampler:
            train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = PKSampler(targets, p, k)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.workers,
        )

    return train_loader, test_loader


def main(args):
    rank = 0
    world_size = 1
    if args.use_ddp:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        if rank == 0:
            print(f'Using DDP with {world_size} GPUs')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    p = args.labels_per_batch if not args.anomaly else args.labels_per_batch - 1
    k = args.samples_per_label
    batch_size = args.batch_size if args.batch_size else p * k
    per_gpu_batch_size = batch_size

    if rank == 0 and args.use_ddp:
        effective_batch = per_gpu_batch_size * world_size
        print(f'Per-GPU batch size: {per_gpu_batch_size}, Effective batch size: {effective_batch}')

    model = ConvNeXtV2_SRA(num_heads=args.sra_num_heads, lam=args.sra_lam)

    if os.path.isfile(args.resume):
        if rank == 0:
            print('=> loading checkpoint')
        checkpoint = torch.load(args.resume, map_location=device)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        model.load_state_dict(checkpoint, strict=False)
        if rank == 0:
            print('=> loaded checkpoint')
    else:
        if rank == 0:
            print('=> no checkpoint found')

    model.to(device)

    if args.use_ddp:
        model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=False,
        )

    if args.dataset == 'vindr':
        criterion = WeightedMultiLabelTripletLoss(margin=args.margin)
    else:
        criterion = TripletMarginLoss(margin=args.margin)

    model_without_ddp = model.module if args.use_ddp else model
    backbone_params = []
    head_params = []
    for name, param in model_without_ddp.named_parameters():
        if 'sra' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = Adam([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': head_params, 'lr': args.lr},
    ])

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.Resize(432),
        transforms.RandomCrop(384, padding=4) if args.rand_resize else transforms.CenterCrop(384),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.Resize(432),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset, test_dataset = build_datasets(args, train_transform, test_transform)
    train_loader, test_loader = build_loaders(
        args,
        train_dataset,
        test_dataset,
        rank,
        world_size,
        batch_size,
        per_gpu_batch_size,
    )

    best_metric = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        if args.use_ddp:
            train_loader.sampler.set_epoch(epoch)

        if rank == 0:
            print(f'\n{"=" * 60}')
            print(f'SRA training epoch {epoch}/{args.epochs}...')
            print(f'{"=" * 60}')

        train_epoch(model, optimizer, criterion, train_loader, device, epoch, args.print_freq, rank=rank)

        if epoch % args.eval_freq == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if args.use_ddp:
                dist.barrier()

            if rank == 0:
                print(f'\n{"=" * 60}')
                print(f'Evaluating SRA epoch {epoch}...')
                print(f'{"=" * 60}')

            current_metric = evaluate(model, test_loader, device, rank=rank, world_size=world_size)

            if rank == 0:
                model_to_save = model.module if args.use_ddp else model
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_epoch = epoch
                    print(f'\nNew best SRA model: {current_metric:.3f}% (epoch {epoch})')
                    save(model_to_save, epoch, args.save_dir, args, is_best=True)
                else:
                    print(f'\nCurrent: {current_metric:.3f}%, Best: {best_metric:.3f}% (epoch {best_epoch})')

                if epoch % 10 == 0:
                    save(model_to_save, epoch, args.save_dir, args, is_best=False)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if rank == 0:
        print(f'\n{"=" * 60}')
        print('SRA training completed!')
        print(f'{"=" * 60}')
        print(f'Best model: Epoch {best_epoch} with metric: {best_metric:.3f}%')
        print(f'Best model saved in: {args.save_dir}')
        print(f'{"=" * 60}')

    if args.use_ddp:
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description='Train ConvNeXtV2 SRA retrieval model')
    parser.add_argument('--dataset', default='covid', help='Dataset to use: covid, isic, tbx11k, or vindr')
    parser.add_argument('--dataset-dir', default='/data/brian.hu/COVID/data/', help='Dataset directory path')
    parser.add_argument('--train-image-list', default='./train_split.txt', help='Train image list')
    parser.add_argument('--test-image-list', default='./test_COVIDx4.txt', help='Test image list')
    parser.add_argument('--mask-dir', default=None, help='Segmentation masks path if used')
    parser.add_argument('--rand-resize', action='store_true', help='Use random crop augmentation')
    parser.add_argument('--anomaly', action='store_true', help='Train without anomaly class')
    parser.add_argument('--model', default='convnextv2_sra', help='Fixed for checkpoint naming')
    parser.add_argument('--embedding-dim', default=None, type=int, help='Unused for SRA, kept for compatibility')
    parser.add_argument('--sra-num-heads', default=8, type=int, help='Number of attention heads in SRA')
    parser.add_argument('--sra-lam', default=0.1, type=float, help='Residual attention scaling factor')
    parser.add_argument('-p', '--labels-per-batch', default=3, type=int, help='Number of unique labels per batch')
    parser.add_argument('-k', '--samples-per-label', default=16, type=int, help='Number of samples per label')
    parser.add_argument('--eval-batch-size', default=64, type=int, help='Evaluation batch size')
    parser.add_argument('--epochs', default=20, type=int, help='Number of training epochs')
    parser.add_argument('--eval-freq', default=2, type=int, help='Evaluate every N epochs')
    parser.add_argument('-j', '--workers', default=4, type=int, help='Number of data loading workers')
    parser.add_argument('--lr', default=0.0001, type=float, help='Base learning rate')
    parser.add_argument('--margin', default=0.2, type=float, help='Triplet loss margin')
    parser.add_argument('--print-freq', default=5, type=int, help='Print frequency')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--save-dir', default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', default='', help='Checkpoint path to resume from')
    parser.add_argument('--batch-size', default=None, type=int, help='Overrides p*k batch size')
    parser.add_argument('--use-random-sampler', action='store_true', help='Use RandomSampler instead of PKSampler')
    parser.add_argument('--use-ddp', action='store_true', help='Use Distributed Data Parallel training')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
