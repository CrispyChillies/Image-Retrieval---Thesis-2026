from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, Sampler
import timm
import torchvision.transforms as transforms

from read_data import NIHChestXrayRetrievalDataSet, NIH_RETRIEVAL_PATHOLOGIES


def build_nih_train_transform(
    image_size: int,
    resize_size: int,
    use_random_resized_crop: bool = True,
) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize(resize_size),
            (
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.9, 1.0),
                    ratio=(0.95, 1.05),
                )
                if use_random_resized_crop
                else transforms.CenterCrop(image_size)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ]
    )


def build_nih_val_transform(image_size: int, resize_size: int) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def get_backbone_image_config(backbone_type: str) -> Dict[str, int]:
    if backbone_type == "dinov2":
        return {"image_size": 518, "resize_size": 518}
    if backbone_type == "convnextv2":
        return {"image_size": 384, "resize_size": 432}
    raise ValueError(f"Unsupported backbone_type: {backbone_type}")


def compute_multilabel_masks_and_weights(
    labels: torch.Tensor,
    use_jaccard_weight: bool = True,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = labels.float()
    intersection = labels @ labels.t()
    label_cardinality = labels.sum(dim=1, keepdim=True)
    union = label_cardinality + label_cardinality.t() - intersection
    jaccard = intersection / union.clamp_min(eps)

    batch_size = labels.size(0)
    eye_mask = torch.eye(batch_size, device=labels.device, dtype=torch.bool)
    positive_mask = (intersection > 0) & ~eye_mask
    negative_mask = (intersection == 0) & ~eye_mask

    if use_jaccard_weight:
        positive_weights = jaccard * positive_mask.float()
    else:
        positive_weights = positive_mask.float()

    return positive_mask, negative_mask, positive_weights


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_pos: float = 1.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        prob_pos = torch.sigmoid(logits)
        prob_neg = 1.0 - prob_pos

        if self.clip is not None and self.clip > 0:
            prob_neg = (prob_neg + self.clip).clamp(max=1.0)

        log_pos = torch.log(prob_pos.clamp_min(self.eps))
        log_neg = torch.log(prob_neg.clamp_min(self.eps))

        loss = targets * log_pos + (1.0 - targets) * log_neg

        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt = prob_pos * targets + prob_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            focal_weight = torch.pow(1.0 - pt, gamma)
            loss = loss * focal_weight

        return -loss.sum(dim=1).mean()


class MultiLabelContrastiveLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        use_jaccard_weight: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.use_jaccard_weight = use_jaccard_weight
        self.eps = eps

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=1)
        positive_mask, _, positive_weights = compute_multilabel_masks_and_weights(
            labels=labels,
            use_jaccard_weight=self.use_jaccard_weight,
            eps=self.eps,
        )

        logits = embeddings @ embeddings.t()
        logits = logits / self.temperature

        batch_size = embeddings.size(0)
        self_mask = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
        logits = logits.masked_fill(self_mask, float("-inf"))
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        positive_weight_sums = positive_weights.sum(dim=1)
        valid_anchors = positive_weight_sums > 0
        if not valid_anchors.any():
            return embeddings.new_tensor(0.0)

        weighted_log_prob = (positive_weights * log_prob).sum(dim=1)
        loss = -weighted_log_prob[valid_anchors] / positive_weight_sums[valid_anchors].clamp_min(self.eps)
        return loss.mean()


class DINOv2MultiLabelRetrievalModel(nn.Module):
    def __init__(
        self,
        num_labels: int = 14,
        backbone_name: str = "vit_base_patch14_dinov2.lvd142m",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
        )

        backbone_dim = getattr(self.backbone, "num_features", 768)
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )
        self.classification_head = nn.Linear(256, num_labels)

    def _extract_cls_token(self, features: torch.Tensor | Dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(features, dict):
            if "x_norm_clstoken" in features:
                return features["x_norm_clstoken"]
            if "cls_token" in features:
                return features["cls_token"]
            if "x_prenorm" in features:
                return features["x_prenorm"][:, 0]
            raise KeyError("Unsupported DINOv2 feature dict: missing CLS token.")

        if features.ndim == 3:
            return features[:, 0]
        if features.ndim == 2:
            return features

        raise ValueError(f"Unsupported DINOv2 feature shape: {tuple(features.shape)}")

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone.forward_features(images)
        cls_embedding = self._extract_cls_token(features)
        projection = self.projection_head(cls_embedding)
        logits = self.classification_head(projection)
        embedding = F.normalize(projection, dim=1)

        return {
            "cls_embedding": cls_embedding,
            "projection": projection,
            "embedding": embedding,
            "logits": logits,
        }


class ConvNeXtV2MultiLabelRetrievalModel(nn.Module):
    def __init__(
        self,
        num_labels: int = 14,
        backbone_name: str = "convnextv2_base.fcmae_ft_in22k_in1k_384",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
        )

        backbone_dim = getattr(self.backbone, "num_features", 1024)
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )
        self.classification_head = nn.Linear(256, num_labels)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(images)
        projection = self.projection_head(features)
        logits = self.classification_head(projection)
        embedding = F.normalize(projection, dim=1)

        return {
            "backbone_embedding": features,
            "projection": projection,
            "embedding": embedding,
            "logits": logits,
        }


def build_optimizer(
    model: DINOv2MultiLabelRetrievalModel,
    backbone_lr: float = 1e-5,
    heads_lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> AdamW:
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [
        p
        for module in (model.projection_head, model.classification_head)
        for p in module.parameters()
        if p.requires_grad
    ]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": heads_lr})

    return AdamW(param_groups, weight_decay=weight_decay)


def unpack_batch(batch: Sequence[torch.Tensor] | Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        return batch["images"], batch["labels"]
    if len(batch) < 2:
        raise ValueError("Batch must provide image and label tensors.")
    return batch[0], batch[1]


def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    contrastive_loss_fn: nn.Module,
    asl_loss_fn: nn.Module,
    alpha: float = 1.0,
) -> Dict[str, torch.Tensor]:
    contrastive_loss = contrastive_loss_fn(outputs["embedding"], labels)
    asl_loss = asl_loss_fn(outputs["logits"], labels)
    total_loss = contrastive_loss + alpha * asl_loss
    return {
        "total_loss": total_loss,
        "contrastive_loss": contrastive_loss,
        "asl_loss": asl_loss,
    }


def train_step(
    model: nn.Module,
    batch: Sequence[torch.Tensor] | Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    contrastive_loss_fn: nn.Module,
    asl_loss_fn: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    alpha: float = 1.0,
    amp_enabled: bool = True,
) -> Dict[str, float]:
    images, labels = unpack_batch(batch)
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True).float()

    optimizer.zero_grad(set_to_none=True)

    use_amp = amp_enabled and device.type == "cuda"
    with autocast(device_type=device.type, enabled=use_amp):
        outputs = model(images)
        losses = compute_total_loss(
            outputs=outputs,
            labels=labels,
            contrastive_loss_fn=contrastive_loss_fn,
            asl_loss_fn=asl_loss_fn,
            alpha=alpha,
        )

    if scaler is not None and use_amp:
        scaler.scale(losses["total_loss"]).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        losses["total_loss"].backward()
        optimizer.step()

    return {
        "loss": float(losses["total_loss"].detach().item()),
        "contrastive_loss": float(losses["contrastive_loss"].detach().item()),
        "asl_loss": float(losses["asl_loss"].detach().item()),
    }


class MultiLabelBalancedBatchSampler(BatchSampler):
    def __init__(
        self,
        multi_hot_labels: Sequence[Sequence[float]],
        batch_size: int = 32,
        labels_per_batch: int = 8,
        samples_per_label: int = 4,
        drop_last: bool = True,
    ) -> None:
        self.multi_hot_labels = torch.as_tensor(multi_hot_labels, dtype=torch.float32)
        self.batch_size = batch_size
        self.labels_per_batch = labels_per_batch
        self.samples_per_label = samples_per_label
        self.drop_last = drop_last

        self.label_to_indices: Dict[int, List[int]] = {}
        for label_idx in range(self.multi_hot_labels.size(1)):
            indices = torch.nonzero(self.multi_hot_labels[:, label_idx] > 0, as_tuple=False).squeeze(1)
            if indices.numel() > 0:
                self.label_to_indices[label_idx] = indices.tolist()

        if not self.label_to_indices:
            raise ValueError("Label-aware sampling requires at least one positive label.")

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.multi_hot_labels) // self.batch_size
        return math.ceil(len(self.multi_hot_labels) / self.batch_size)

    def __iter__(self) -> Iterable[List[int]]:
        label_ids = list(self.label_to_indices.keys())
        dataset_indices = list(range(len(self.multi_hot_labels)))

        for _ in range(len(self)):
            batch: List[int] = []
            chosen_labels = random.sample(
                label_ids,
                k=min(self.labels_per_batch, len(label_ids)),
            )

            for label_id in chosen_labels:
                candidates = self.label_to_indices[label_id]
                if not candidates:
                    continue
                if len(candidates) >= self.samples_per_label:
                    sampled = random.sample(candidates, k=self.samples_per_label)
                else:
                    sampled = random.choices(candidates, k=self.samples_per_label)
                batch.extend(sampled)

            batch = list(dict.fromkeys(batch))
            if len(batch) < self.batch_size:
                remaining = [idx for idx in dataset_indices if idx not in batch]
                needed = self.batch_size - len(batch)
                if len(remaining) >= needed:
                    batch.extend(random.sample(remaining, k=needed))
                else:
                    batch.extend(random.choices(dataset_indices, k=needed))

            random.shuffle(batch)
            yield batch[: self.batch_size]


def build_nih_dataloader(
    data_dir: str,
    image_list_file: Optional[str] = None,
    batch_size: int = 32,
    transform: Optional[transforms.Compose] = None,
    num_workers: int = 4,
    label_aware_sampling: bool = False,
    labels_per_batch: int = 8,
    samples_per_label: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    dataset = NIHChestXrayRetrievalDataSet(
        data_dir=data_dir,
        image_list_file=image_list_file,
        transform=transform,
        pathology_names=NIH_RETRIEVAL_PATHOLOGIES,
    )

    if label_aware_sampling:
        batch_sampler: Sampler[List[int]] = MultiLabelBalancedBatchSampler(
            multi_hot_labels=dataset.labels,
            batch_size=batch_size,
            labels_per_batch=labels_per_batch,
            samples_per_label=samples_per_label,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def example_batch_processing(
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DINOv2MultiLabelRetrievalModel(num_labels=len(NIH_RETRIEVAL_PATHOLOGIES)).to(device)
    optimizer = build_optimizer(model)
    contrastive_loss_fn = MultiLabelContrastiveLoss(temperature=0.07, use_jaccard_weight=True)
    asl_loss_fn = AsymmetricLoss(gamma_pos=1.0, gamma_neg=4.0, clip=0.05)
    scaler = GradScaler(device="cuda", enabled=device.type == "cuda")

    batch = next(iter(data_loader))
    return train_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        contrastive_loss_fn=contrastive_loss_fn,
        asl_loss_fn=asl_loss_fn,
        device=device,
        scaler=scaler,
        alpha=1.0,
        amp_enabled=True,
    )
