"""Evaluate non-CLIP retrieval backbones with the same preprocessing and metrics as test.py.

Supported models:
- densenet121
- resnet50
- convnextv2
- convnextv2_sra
- swinv2
- dinov2
- medsiglip

Example:
python test_nonclip.py --dataset covid --model convnextv2 --resume path/to/checkpoint.pth --test-dataset-dir /path/to/test --test-image-list test.txt
"""

import os
from collections import Counter

import numpy as np
import timm
import torch
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from timm.data import resolve_model_data_config
from torch.utils.data import DataLoader

from model import ConvNeXtV2, ConvNeXtV2_SRA, DenseNet121, DinoV2, MedSigLIP, ResNet50, SwinV2
from read_data import ChestXrayDataSet, ISICDataSet, TBX11kDataSet


def retrieval_accuracy(output, target, topk=(1,)):
    """Computes retrieval accuracy at k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.cpu()
        target = target.cpu()
        pred = target[pred].t()
        correct = pred.eq(target[None])

        results = []
        for k in topk:
            correct_k = correct[:k].any(dim=0).sum(dtype=torch.float32)
            results.append(correct_k * (100.0 / batch_size))
    return results


def compute_ap(ranks, nres):
    """Compute average precision for ranked positives."""
    ap = 0.0
    recall_step = 1.0 / nres
    for j in np.arange(len(ranks)):
        rank = ranks[j]
        precision_0 = 1.0 if rank == 0 else float(j) / rank
        precision_1 = float(j + 1) / (rank + 1)
        ap += (precision_0 + precision_1) * recall_step / 2.0
    return ap


def compute_map(ranks, gnd, kappas=None):
    """Compute mAP and mean precision@k exactly as in test.py."""
    if kappas is None:
        kappas = []

    mean_ap = 0.0
    nq = len(gnd)
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.where(gnd == gnd[i])[0]
        if qgnd.shape[0] == 0:
            aps[i] = float("nan")
            prs[i, :] = float("nan")
            nempty += 1
            continue

        pos = np.arange(ranks.shape[0])[np.isin(ranks[:, i], qgnd)]
        ap = compute_ap(pos, len(qgnd))
        mean_ap += ap
        aps[i] = ap

        pos += 1
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j])
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    mean_ap = mean_ap / (nq - nempty)
    pr = pr / (nq - nempty)
    return mean_ap, aps, pr, prs


def majority_vote(retrieved_labels):
    """Return the majority label in retrieved labels."""
    if len(retrieved_labels) == 0:
        return None
    return Counter(retrieved_labels).most_common(1)[0][0]


def compute_classification_metrics(labels, dists, k_values=(1, 5, 10, 15, 20)):
    """Compute majority-vote classification metrics from retrieval results."""
    labels_np = labels.cpu().numpy()
    n_samples = labels.size(0)
    ranks = torch.argsort(dists, dim=0, descending=True).cpu().numpy()
    results = {}

    for k in k_values:
        predicted_labels = []
        true_labels = []
        for i in range(n_samples):
            top_k_indices = ranks[:k, i]
            retrieved_labels = labels_np[top_k_indices]
            predicted_labels.append(majority_vote(retrieved_labels))
            true_labels.append(labels_np[i])

        results[k] = {
            "precision_macro": precision_score(true_labels, predicted_labels, average="macro", zero_division=0) * 100.0,
            "recall_macro": recall_score(true_labels, predicted_labels, average="macro", zero_division=0) * 100.0,
            "f1_macro": f1_score(true_labels, predicted_labels, average="macro", zero_division=0) * 100.0,
            "precision_weighted": precision_score(true_labels, predicted_labels, average="weighted", zero_division=0) * 100.0,
            "recall_weighted": recall_score(true_labels, predicted_labels, average="weighted", zero_division=0) * 100.0,
            "f1_weighted": f1_score(true_labels, predicted_labels, average="weighted", zero_division=0) * 100.0,
            "accuracy": accuracy_score(true_labels, predicted_labels) * 100.0,
        }
    return results


@torch.no_grad()
def evaluate(model, loader, device, args):
    """Evaluate a non-CLIP retrieval model."""
    model.eval()
    embeds = []
    labels = []

    for data in loader:
        samples = data[0].to(device)
        batch_labels = data[1].to(device)
        output = model(samples)
        embedding = output[0] if isinstance(output, tuple) else output
        embeds.append(embedding.cpu())
        labels.append(batch_labels.cpu())

    embeds = torch.cat(embeds, dim=0)
    labels = torch.cat(labels, dim=0)

    # Keep default behavior consistent with train.py validation: negative Euclidean distance.
    dists = embeds @ embeds.t() if args.metric == "cosine" else -torch.cdist(embeds, embeds)
    dists.fill_diagonal_(float("-inf"))

    kappas = [1, 5, 10]
    accuracy = retrieval_accuracy(dists, labels, topk=kappas)
    accuracy = torch.stack(accuracy).cpu().numpy()
    print(f">> R@K{kappas}: {np.around(accuracy, 2)}%")

    ranks = torch.argsort(dists, dim=0, descending=True)
    mean_ap, _, pr, _ = compute_map(ranks.cpu().numpy(), labels.cpu().numpy(), kappas)
    print(f">> mAP: {mean_ap * 100.0:.2f}%")
    print(f">> mP@K{kappas}: {np.around(pr * 100.0, 2)}%")

    print("\n>> Retrieval Classification Metrics (Majority Voting):")
    classification_results = compute_classification_metrics(labels, dists)
    for k in [1, 5, 10]:
        metrics = classification_results[k]
        print(f"\n>> Top-{k} Retrieved Images:")
        print(f"   Accuracy: {metrics['accuracy']:.2f}%")
        print(f"   Precision (macro): {metrics['precision_macro']:.2f}%")
        print(f"   Recall (macro): {metrics['recall_macro']:.2f}%")
        print(f"   F1 (macro): {metrics['f1_macro']:.2f}%")


def build_model(args):
    """Build one non-CLIP retrieval model."""
    if args.model == "densenet121":
        return DenseNet121(embedding_dim=args.embedding_dim)
    if args.model == "resnet50":
        return ResNet50(embedding_dim=args.embedding_dim)
    if args.model == "convnextv2":
        return ConvNeXtV2(embedding_dim=args.embedding_dim)
    if args.model == "convnextv2_sra":
        return ConvNeXtV2_SRA(num_heads=args.sra_num_heads, lam=args.sra_lam)
    if args.model == "swinv2":
        return SwinV2(embedding_dim=args.embedding_dim)
    if args.model == "dinov2":
        return DinoV2(
            model_name=args.dinov2_model_name,
            embedding_dim=args.embedding_dim,
            unfreeze_blocks=args.unfreeze_blocks,
        )
    if args.model == "medsiglip":
        return MedSigLIP()
    raise NotImplementedError(f"Model not supported in test_nonclip.py: {args.model}")


def build_test_transform(args):
    """Build the exact non-CLIP evaluation preprocessing used in test.py."""
    if args.model == "dinov2":
        temp_model = timm.create_model(
            args.dinov2_model_name,
            pretrained=False,
            num_classes=0,
        )
        data_config = resolve_model_data_config(temp_model)
        img_size = data_config["input_size"][-1]
        normalize = transforms.Normalize(data_config["mean"], data_config["std"])
        return transforms.Compose(
            [
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize,
            ]
        )

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if args.model == "medsiglip":
        img_size = 448
    elif args.model in ["convnextv2", "convnextv2_sra", "swinv2"]:
        img_size = 384
    else:
        img_size = 224

    if args.model in ["convnextv2", "convnextv2_sra", "swinv2"]:
        return transforms.Compose(
            [
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize(432),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize,
            ]
        )

    if args.model == "medsiglip":
        return transforms.Compose(
            [
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )


def build_dataset(args, transform):
    """Build the requested evaluation dataset."""
    if args.dataset == "covid":
        return ChestXrayDataSet(
            data_dir=args.test_dataset_dir,
            image_list_file=args.test_image_list,
            mask_dir=args.mask_dir,
            transform=transform,
        )
    if args.dataset == "isic":
        return ISICDataSet(
            data_dir=args.test_dataset_dir,
            image_list_file=args.test_image_list,
            mask_dir=args.mask_dir,
            transform=transform,
        )
    if args.dataset == "tbx11k":
        return TBX11kDataSet(
            data_dir=args.test_dataset_dir,
            csv_file=args.test_image_list,
            transform=transform,
        )
    raise NotImplementedError(f"Dataset not supported: {args.dataset}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(args)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint")
            checkpoint = torch.load(args.resume, map_location=device)
            if "state-dict" in checkpoint:
                checkpoint = checkpoint["state-dict"]
            elif "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            model.load_state_dict(checkpoint, strict=False)
            print("=> loaded checkpoint")
        else:
            print("=> no checkpoint found")
    model.to(device)

    test_transform = build_test_transform(args)
    test_dataset = build_dataset(args, test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    print("Evaluating non-CLIP model...")
    evaluate(model, test_loader, device, args)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate non-CLIP retrieval models")
    parser.add_argument("--dataset", default="covid", help="Dataset to use (covid, isic, or tbx11k)")
    parser.add_argument("--test-dataset-dir", default="/data/brian.hu/COVID/data/test", help="Test dataset directory path")
    parser.add_argument("--test-image-list", default="./test_COVIDx4.txt", help="Test image list")
    parser.add_argument("--mask-dir", default=None, help="Segmentation masks path (if used)")
    parser.add_argument(
        "--model",
        default="densenet121",
        choices=["densenet121", "resnet50", "convnextv2", "convnextv2_sra", "swinv2", "dinov2", "medsiglip"],
        help="Non-CLIP model to evaluate",
    )
    parser.add_argument("--embedding-dim", default=None, type=int, help="Embedding dimension override")
    parser.add_argument("--dinov2-model-name", default="vit_base_patch14_dinov2.lvd142m", type=str, help="timm model name for DINOv2 backbone")
    parser.add_argument("--unfreeze-blocks", default=3, type=int, help="Number of final DINOv2 transformer blocks kept trainable when loading the model")
    parser.add_argument("--sra-num-heads", default=8, type=int, help="Number of attention heads for SRA")
    parser.add_argument("--sra-lam", default=0.1, type=float, help="Lambda for residual attention in SRA")
    parser.add_argument("--eval-batch-size", default=64, type=int)
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N", help="Number of data loading workers")
    parser.add_argument("--save-dir", default="./results", help="Result save directory")
    parser.add_argument("--resume", default="", help="Resume from checkpoint")
    parser.add_argument(
        "--metric",
        default="cdist",
        choices=["cosine", "cdist"],
        help="Retrieval similarity metric (default cdist to match train.py validation)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
