from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.amp import GradScaler

from nih_multilabel_retrieval import (
    AsymmetricLoss,
    ConvNeXtV2MultiLabelRetrievalModel,
    DINOv2MultiLabelRetrievalModel,
    MultiLabelContrastiveLoss,
    build_nih_dataloader,
    build_optimizer,
    build_nih_train_transform,
    build_nih_val_transform,
    get_backbone_image_config,
    train_step,
)


@dataclass
class BackboneSpec:
    name: str
    model_builder: Callable[[int, str, bool], torch.nn.Module]
    default_backbone_name: str


BACKBONE_SPECS: Dict[str, BackboneSpec] = {
    "dinov2": BackboneSpec(
        name="dinov2",
        model_builder=lambda num_labels, backbone_name, pretrained: DINOv2MultiLabelRetrievalModel(
            num_labels=num_labels,
            backbone_name=backbone_name,
            pretrained=pretrained,
        ),
        default_backbone_name="vit_base_patch14_dinov2.lvd142m",
    ),
    "convnextv2": BackboneSpec(
        name="convnextv2",
        model_builder=lambda num_labels, backbone_name, pretrained: ConvNeXtV2MultiLabelRetrievalModel(
            num_labels=num_labels,
            backbone_name=backbone_name,
            pretrained=pretrained,
        ),
        default_backbone_name="convnextv2_base.fcmae_ft_in22k_in1k_384",
    ),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_map(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    jaccard_threshold: float = 0.4,
) -> float:
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in data_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            embeddings.append(outputs["embedding"].cpu())
            labels.append(batch_labels.cpu())

    embeddings = F.normalize(torch.cat(embeddings, dim=0), dim=1)
    labels = torch.cat(labels, dim=0)
    similarity = embeddings @ embeddings.t()
    similarity.fill_diagonal_(-1)

    aps = []
    for i in range(labels.size(0)):
        intersection = (labels[i] * labels).sum(dim=1)
        union = (labels[i] + labels).clamp(max=1).sum(dim=1)
        jaccard = intersection / (union + 1e-8)
        relevance = (jaccard > jaccard_threshold).float().numpy()
        if relevance.sum() > 0:
            aps.append(average_precision_score(relevance, similarity[i].numpy()))

    if not aps:
        return 0.0
    return float(np.mean(aps) * 100.0)


def save_checkpoint(
    model: torch.nn.Module,
    save_dir: str,
    backbone_type: str,
    epoch: int,
    metric: float,
    is_best: bool,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    filename = f"nih_{backbone_type}_{'best' if is_best else f'epoch_{epoch}'}_ckpt.pth"
    save_path = os.path.join(save_dir, filename)
    torch.save(
        {
            "epoch": epoch,
            "metric": metric,
            "state_dict": model.state_dict(),
        },
        save_path,
    )
    return save_path


def run_training(args: argparse.Namespace) -> None:
    spec = BACKBONE_SPECS[args.backbone_type]
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_config = get_backbone_image_config(args.backbone_type)
    train_transform = build_nih_train_transform(
        image_size=image_config["image_size"],
        resize_size=image_config["resize_size"],
        use_random_resized_crop=not args.disable_rand_resize,
    )
    val_transform = build_nih_val_transform(
        image_size=image_config["image_size"],
        resize_size=image_config["resize_size"],
    )

    model = spec.model_builder(
        args.num_labels,
        args.backbone_name or spec.default_backbone_name,
        not args.no_pretrained,
    ).to(device)

    optimizer = build_optimizer(
        model,
        backbone_lr=args.backbone_lr,
        heads_lr=args.heads_lr,
        weight_decay=args.weight_decay,
    )
    contrastive_loss_fn = MultiLabelContrastiveLoss(
        temperature=args.temperature,
        use_jaccard_weight=not args.disable_jaccard_weight,
    )
    asl_loss_fn = AsymmetricLoss(
        gamma_pos=args.gamma_pos,
        gamma_neg=args.gamma_neg,
        clip=args.clip,
    )
    scaler = GradScaler(device="cuda", enabled=args.amp and device.type == "cuda")

    train_loader = build_nih_dataloader(
        data_dir=args.dataset_dir,
        image_list_file=args.train_image_list,
        batch_size=args.batch_size,
        transform=train_transform,
        num_workers=args.workers,
        label_aware_sampling=args.label_aware_sampling,
        labels_per_batch=args.labels_per_batch,
        samples_per_label=args.samples_per_label,
        shuffle=not args.label_aware_sampling,
        drop_last=True,
    )
    val_loader = build_nih_dataloader(
        data_dir=args.val_dataset_dir or args.dataset_dir,
        image_list_file=args.val_image_list,
        batch_size=args.eval_batch_size,
        transform=val_transform,
        num_workers=args.workers,
        label_aware_sampling=False,
        shuffle=False,
        drop_last=False,
    )

    best_map = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_contrastive = 0.0
        running_asl = 0.0

        for step, batch in enumerate(train_loader, start=1):
            metrics = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                contrastive_loss_fn=contrastive_loss_fn,
                asl_loss_fn=asl_loss_fn,
                device=device,
                scaler=scaler,
                alpha=args.alpha,
                amp_enabled=args.amp,
            )
            running_loss += metrics["loss"]
            running_contrastive += metrics["contrastive_loss"]
            running_asl += metrics["asl_loss"]

            if step % args.print_freq == 0:
                print(
                    f"[epoch {epoch} step {step}] "
                    f"loss={running_loss / args.print_freq:.4f} "
                    f"contrastive={running_contrastive / args.print_freq:.4f} "
                    f"asl={running_asl / args.print_freq:.4f}"
                )
                running_loss = 0.0
                running_contrastive = 0.0
                running_asl = 0.0

        if epoch % args.eval_freq != 0:
            continue

        current_map = evaluate_map(
            model=model,
            data_loader=val_loader,
            device=device,
            jaccard_threshold=args.jaccard_threshold,
        )
        print(f"[epoch {epoch}] val_mAP={current_map:.3f}%")

        if current_map > best_map:
            best_map = current_map
            ckpt_path = save_checkpoint(
                model=model,
                save_dir=args.save_dir,
                backbone_type=args.backbone_type,
                epoch=epoch,
                metric=current_map,
                is_best=True,
            )
            print(f"best checkpoint saved to {ckpt_path}")

        if epoch % args.save_freq == 0:
            ckpt_path = save_checkpoint(
                model=model,
                save_dir=args.save_dir,
                backbone_type=args.backbone_type,
                epoch=epoch,
                metric=current_map,
                is_best=False,
            )
            print(f"checkpoint saved to {ckpt_path}")


def build_parser(default_backbone_type: str) -> argparse.ArgumentParser:
    spec = BACKBONE_SPECS[default_backbone_type]
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-type", default=default_backbone_type, choices=list(BACKBONE_SPECS))
    parser.add_argument("--backbone-name", default=spec.default_backbone_name)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--val-dataset-dir", default=None)
    parser.add_argument("--train-image-list", default=None)
    parser.add_argument("--val-image-list", default=None)
    parser.add_argument("--save-dir", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-freq", type=int, default=2)
    parser.add_argument("--save-freq", type=int, default=10)
    parser.add_argument("--print-freq", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-labels", type=int, default=14)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--gamma-pos", type=float, default=1.0)
    parser.add_argument("--gamma-neg", type=float, default=4.0)
    parser.add_argument("--clip", type=float, default=0.05)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--heads-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--jaccard-threshold", type=float, default=0.4)
    parser.add_argument("--labels-per-batch", type=int, default=8)
    parser.add_argument("--samples-per-label", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--label-aware-sampling", action="store_true")
    parser.add_argument("--disable-jaccard-weight", action="store_true")
    parser.add_argument("--disable-rand-resize", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser
