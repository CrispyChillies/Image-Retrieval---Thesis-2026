"""BBox-supervised ConceptCLIP fine-tuning for VinDr-CXR localization.

The original ConceptCLIP PC/RC alignment only requires one patch somewhere in
an image to match a concept. This script adds explicit VinDr box supervision:

1. spatial distribution loss moves concept evidence into the matching box;
2. peak ranking loss makes the strongest in-box patch beat every out-of-box
   patch, directly matching the Pointing Game objective;
3. background and concept-competition losses sharpen and disambiguate maps.

Model selection uses class-aware validation PGHit after the same competitive
calibration used by test_vindr_crar.py. Never train on the test bbox CSV.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from test_vindr_crar import (
    ALL_CONCEPTS,
    LESION_CONCEPTS,
    anatomy_roi_mask,
    load_conceptclip,
    load_vindr_bboxes,
    multilabel_metrics_at_threshold,
    negative_prompts,
    output_value,
    patch_grid,
    positive_prompts,
    threshold_free_classification_metrics,
)


@dataclass
class LocalizationRecord:
    image_id: str
    image_path: Path
    boxes: dict[int, list[tuple[float, float, float, float]]]
    labels: np.ndarray | None = None
    teacher_embedding: torch.Tensor | None = None


def resolve_image(image_dir: Path, image_id: str) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = image_dir / f"{image_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def build_records(
    image_dir: str | os.PathLike[str],
    bbox_csv: str | os.PathLike[str],
    bbox_coord_size: float | None,
    labels_csv: str | os.PathLike[str] | None = None,
) -> tuple[list[LocalizationRecord], dict[str, Any]]:
    root = Path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    grouped, skipped_classes, loader_stats = load_vindr_bboxes(
        bbox_csv, LESION_CONCEPTS
    )
    concept_to_index = {name: index for index, name in enumerate(LESION_CONCEPTS)}
    label_by_id: dict[str, np.ndarray] = {}
    if labels_csv:
        labels_path = Path(labels_csv)
        if not labels_path.is_file():
            raise FileNotFoundError(f"VinDr labels CSV does not exist: {labels_path}")
        label_frame = pd.read_csv(labels_path)
        if "Other disease" in label_frame.columns and "Other diseases" not in label_frame.columns:
            label_frame = label_frame.rename(columns={"Other disease": "Other diseases"})
        missing_columns = [
            column for column in ["image_id", *ALL_CONCEPTS] if column not in label_frame.columns
        ]
        if missing_columns:
            raise ValueError(f"VinDr labels CSV is missing columns: {missing_columns}")
        label_frame[ALL_CONCEPTS] = (
            label_frame[ALL_CONCEPTS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        )
        label_frame = label_frame.groupby("image_id", as_index=False)[ALL_CONCEPTS].max()
        label_by_id = {
            str(row["image_id"]): row[ALL_CONCEPTS].to_numpy(dtype=np.float32)
            for _, row in label_frame.iterrows()
        }

    records: list[LocalizationRecord] = []
    missing_images: list[str] = []
    clipped_boxes = 0

    candidate_ids = list(label_by_id) if label_by_id else list(grouped)
    for image_id in candidate_ids:
        image_annotations = grouped.get(image_id, {})
        image_path = resolve_image(root, image_id)
        if image_path is None:
            missing_images.append(image_id)
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        coordinate_width = bbox_coord_size or float(width)
        coordinate_height = bbox_coord_size or float(height)
        normalized: dict[int, list[tuple[float, float, float, float]]] = {}
        for concept, boxes in image_annotations.items():
            concept_index = concept_to_index[concept]
            for x_min, y_min, x_max, y_max in boxes:
                box = (
                    max(0.0, min(1.0, x_min / coordinate_width)),
                    max(0.0, min(1.0, y_min / coordinate_height)),
                    max(0.0, min(1.0, x_max / coordinate_width)),
                    max(0.0, min(1.0, y_max / coordinate_height)),
                )
                if box[2] <= box[0] or box[3] <= box[1]:
                    clipped_boxes += 1
                    continue
                normalized.setdefault(concept_index, []).append(box)
        if normalized or image_id in label_by_id:
            records.append(
                LocalizationRecord(
                    image_id,
                    image_path,
                    normalized,
                    label_by_id.get(image_id),
                )
            )

    stats = {
        "bbox_loader": loader_stats,
        "skipped_classes": skipped_classes,
        "total_images": len(records),
        "localization_images": sum(bool(record.boxes) for record in records),
        "classification_images": sum(record.labels is not None for record in records),
        "missing_images": len(missing_images),
        "missing_image_examples": missing_images[:10],
        "boxes_removed_after_clipping": clipped_boxes,
    }
    if not records:
        raise ValueError("No bbox-annotated images could be loaded")
    return records, stats


class VinDrLocalizationDataset(Dataset):
    def __init__(self, records: Sequence[LocalizationRecord], flip_probability: float):
        self.records = list(records)
        self.flip_probability = flip_probability

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.image_path) as handle:
            image = handle.convert("RGB")
        boxes = {key: list(value) for key, value in record.boxes.items()}
        if self.flip_probability > 0.0 and random.random() < self.flip_probability:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            boxes = {
                concept: [(1.0 - x_max, y_min, 1.0 - x_min, y_max)
                          for x_min, y_min, x_max, y_max in concept_boxes]
                for concept, concept_boxes in boxes.items()
            }
        return {
            "image_id": record.image_id,
            "image": image,
            "boxes": boxes,
            "labels": torch.from_numpy(record.labels.copy())
            if record.labels is not None
            else None,
            "teacher_embedding": record.teacher_embedding,
        }


def collate_samples(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labels = [item["labels"] for item in batch]
    teacher_embeddings = [item["teacher_embedding"] for item in batch]
    return {
        "image_ids": [item["image_id"] for item in batch],
        "images": [item["image"] for item in batch],
        "boxes": [item["boxes"] for item in batch],
        "labels": torch.stack(labels) if all(label is not None for label in labels) else None,
        "teacher_embeddings": torch.stack(teacher_embeddings)
        if all(embedding is not None for embedding in teacher_embeddings)
        else None,
    }


def split_records(
    records: Sequence[LocalizationRecord], validation_fraction: float, seed: int
) -> tuple[list[LocalizationRecord], list[LocalizationRecord]]:
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, round(len(indices) * validation_fraction))
    validation_indices = set(indices[:validation_count])
    train = [record for index, record in enumerate(records) if index not in validation_indices]
    validation = [record for index, record in enumerate(records) if index in validation_indices]
    if not train:
        raise ValueError("Validation split consumed every localization image")
    return train, validation


def concept_counts(records: Sequence[LocalizationRecord]) -> dict[str, int]:
    counts = {name: 0 for name in LESION_CONCEPTS}
    for record in records:
        for concept_index in record.boxes:
            counts[LESION_CONCEPTS[concept_index]] += 1
    return {name: count for name, count in counts.items() if count}


@torch.inference_mode()
def encode_prompt_ensemble(
    model,
    processor,
    device: torch.device,
    prompt_builder,
) -> torch.Tensor:
    flat_prompts = [prompt for concept in ALL_CONCEPTS for prompt in prompt_builder(concept)]
    inputs = processor(
        text=flat_prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    text_inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if key in {"input_ids", "attention_mask", "token_type_ids"}
    }
    features = output_value(model(**text_inputs), "text_features").float()
    prompts_per_concept = len(prompt_builder(ALL_CONCEPTS[0]))
    features = features.view(len(ALL_CONCEPTS), prompts_per_concept, -1).mean(dim=1)
    return F.normalize(features, dim=-1)


def spatial_tokens(outputs: Any) -> torch.Tensor:
    try:
        tokens = output_value(outputs, "image_token_features")
    except KeyError:
        tokens = output_value(outputs, "last_hidden_state")
    if tokens.ndim != 3:
        raise ValueError(f"Expected [batch,patch,dim] image tokens, got {tokens.shape}")
    token_count = tokens.shape[1]
    side = math.isqrt(token_count)
    if side * side != token_count:
        side_without_cls = math.isqrt(token_count - 1)
        if side_without_cls * side_without_cls == token_count - 1:
            tokens = tokens[:, 1:]
    return F.normalize(tokens.float(), dim=-1)


def bbox_overlap_targets(
    annotations: Sequence[dict[int, list[tuple[float, float, float, float]]]],
    grid_height: int,
    grid_width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(annotations)
    patch_count = grid_height * grid_width
    targets = torch.zeros(
        (batch_size, len(LESION_CONCEPTS), patch_count), device=device
    )
    valid = torch.zeros((batch_size, len(LESION_CONCEPTS)), dtype=torch.bool, device=device)
    y0 = torch.arange(grid_height, device=device).float() / grid_height
    y1 = (torch.arange(grid_height, device=device).float() + 1.0) / grid_height
    x0 = torch.arange(grid_width, device=device).float() / grid_width
    x1 = (torch.arange(grid_width, device=device).float() + 1.0) / grid_width
    cell_area = 1.0 / (grid_height * grid_width)

    for batch_index, image_annotations in enumerate(annotations):
        for concept_index, boxes in image_annotations.items():
            mask = torch.zeros((grid_height, grid_width), device=device)
            for box_x0, box_y0, box_x1, box_y1 in boxes:
                overlap_x = (
                    torch.minimum(x1, torch.tensor(box_x1, device=device))
                    - torch.maximum(x0, torch.tensor(box_x0, device=device))
                ).clamp_min(0.0)
                overlap_y = (
                    torch.minimum(y1, torch.tensor(box_y1, device=device))
                    - torch.maximum(y0, torch.tensor(box_y0, device=device))
                ).clamp_min(0.0)
                coverage = overlap_y[:, None] * overlap_x[None, :] / cell_area
                mask = torch.maximum(mask, coverage.clamp(0.0, 1.0))
            if mask.max() > 0:
                targets[batch_index, concept_index] = mask.flatten()
                valid[batch_index, concept_index] = True
    return targets, valid


def localization_loss(
    tokens: torch.Tensor,
    lesion_text: torch.Tensor,
    annotations: Sequence[dict[int, list[tuple[float, float, float, float]]]],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    grid_height, grid_width = patch_grid(tokens.shape[1])
    targets, valid = bbox_overlap_targets(
        annotations, grid_height, grid_width, tokens.device
    )
    scores = tokens @ F.normalize(lesion_text.float(), dim=-1).t()
    losses = {"distribution": [], "ranking": [], "background": [], "concept": []}

    for batch_index, concept_index in valid.nonzero(as_tuple=False).tolist():
        target = targets[batch_index, concept_index]
        concept_scores = scores[batch_index, :, concept_index]
        positive = target > 0
        negative = ~positive
        target_distribution = target / target.sum().clamp_min(1e-8)
        distribution = -(
            target_distribution
            * F.log_softmax(concept_scores / args.spatial_temperature, dim=0)
        ).sum()
        inside_peak = concept_scores[positive].max()
        outside_peak = concept_scores[negative].max() if negative.any() else inside_peak
        ranking = F.relu(args.ranking_margin + outside_peak - inside_peak)
        foreground = (
            target[positive] * F.softplus(-concept_scores[positive] / args.spatial_temperature)
        ).sum() / target[positive].sum().clamp_min(1e-8)
        background = F.softplus(
            concept_scores[negative] / args.spatial_temperature
        ).mean() if negative.any() else concept_scores.new_zeros(())

        pooled_patch = F.normalize(
            (target_distribution[:, None] * tokens[batch_index]).sum(dim=0), dim=0
        )
        concept_logits = pooled_patch @ F.normalize(lesion_text.float(), dim=-1).t()
        concept_loss = F.cross_entropy(
            concept_logits[None] / args.concept_temperature,
            torch.tensor([concept_index], device=tokens.device),
        )
        losses["distribution"].append(distribution)
        losses["ranking"].append(ranking)
        losses["background"].append(0.5 * (foreground + background))
        losses["concept"].append(concept_loss)

    if not losses["distribution"]:
        zero = tokens.sum() * 0.0
        return zero, {
            "distribution": 0.0,
            "ranking": 0.0,
            "background": 0.0,
            "concept": 0.0,
            "localization_pairs": 0.0,
        }
    means = {name: torch.stack(values).mean() for name, values in losses.items()}
    total = (
        args.distribution_weight * means["distribution"]
        + args.ranking_weight * means["ranking"]
        + args.background_weight * means["background"]
        + args.concept_weight * means["concept"]
    )
    components = {name: float(value.detach()) for name, value in means.items()}
    components["localization_pairs"] = float(valid.sum())
    return total, components


def concept_classification_logits(
    image_features: torch.Tensor,
    positive_text: torch.Tensor,
    negative_text: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    image_features = F.normalize(image_features.float(), dim=-1)
    positive_text = F.normalize(positive_text.float(), dim=-1)
    negative_text = F.normalize(negative_text.float(), dim=-1)
    return (
        image_features @ positive_text.t() - image_features @ negative_text.t()
    ) / temperature


def asymmetric_multilabel_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_positive: float,
    gamma_negative: float,
    probability_clip: float,
) -> torch.Tensor:
    """ASL for imbalanced VinDr concepts using prompt-derived global logits."""
    targets = targets.float()
    positive_probability = torch.sigmoid(logits)
    negative_probability = 1.0 - positive_probability
    if probability_clip > 0.0:
        negative_probability = (negative_probability + probability_clip).clamp(max=1.0)
    log_likelihood = (
        targets * torch.log(positive_probability.clamp_min(1e-8))
        + (1.0 - targets) * torch.log(negative_probability.clamp_min(1e-8))
    )
    probability_correct = (
        targets * positive_probability + (1.0 - targets) * negative_probability
    )
    gamma = targets * gamma_positive + (1.0 - targets) * gamma_negative
    asymmetric_weight = (1.0 - probability_correct).clamp_min(0.0).pow(gamma)
    return -(asymmetric_weight.detach() * log_likelihood).mean()


@torch.inference_mode()
def cache_pretrained_embeddings(
    model,
    processor,
    records: Sequence[LocalizationRecord],
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """Cache the frozen starting embedding for low-memory feature preservation."""
    model.eval()
    loader = DataLoader(
        VinDrLocalizationDataset(records, 0.0),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_samples,
    )
    record_by_id = {record.image_id: record for record in records}
    for batch_index, batch in enumerate(loader, start=1):
        inputs = processor(images=batch["images"], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.amp and device.type == "cuda",
        ):
            outputs = model(pixel_values=pixel_values)
            embeddings = F.normalize(
                output_value(outputs, "image_features").float(), dim=-1
            ).cpu()
        for image_id, embedding in zip(batch["image_ids"], embeddings):
            record_by_id[image_id].teacher_embedding = embedding
        if batch_index % args.print_freq == 0:
            print(f"[preservation cache] {batch_index}/{len(loader)} batches")


def calibrated_scores(raw_alignment: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    concept_count = raw_alignment.shape[1]
    other_mean = (raw_alignment.sum(dim=1, keepdim=True) - raw_alignment) / max(
        1, concept_count - 1
    )
    specific = raw_alignment - args.eval_common_mode_strength * other_mean
    center = specific.median(dim=1, keepdim=True).values
    scale = 1.4826 * (specific - center).abs().median(dim=1, keepdim=True).values
    discriminative = (specific - center) / scale.clamp_min(1e-4)
    competition = torch.softmax(
        discriminative / args.eval_calibration_temperature, dim=1
    )
    return F.relu(discriminative) * competition


def heatmap_peak_hit(
    patch_values: torch.Tensor,
    boxes: Sequence[tuple[float, float, float, float]],
    grid_height: int,
    grid_width: int,
    roi: torch.Tensor,
    activation_quantile: float,
) -> bool:
    cutoff = torch.quantile(patch_values, activation_quantile)
    values = F.relu(patch_values - cutoff)
    upper = torch.quantile(values, 0.99)
    if float(upper) <= 1e-8:
        upper = values.max()
    values = values / upper if float(upper) > 1e-8 else torch.zeros_like(values)
    heatmap = F.interpolate(
        values.reshape(1, 1, grid_height, grid_width),
        size=roi.shape,
        mode="bicubic",
        align_corners=False,
    ).clamp(0.0, 1.0)[0, 0]
    heatmap = (heatmap * roi).clamp(0.0, 1.0)
    y_peak, x_peak = np.unravel_index(int(heatmap.argmax()), heatmap.shape)
    x = (float(x_peak) + 0.5) / heatmap.shape[1]
    y = (float(y_peak) + 0.5) / heatmap.shape[0]
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in boxes)


@torch.inference_mode()
def evaluate_pghit(
    model,
    processor,
    loader: DataLoader,
    positive_text: torch.Tensor,
    negative_text: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model.eval()
    positive_text = F.normalize(positive_text.float(), dim=-1)
    negative_text = F.normalize(negative_text.float(), dim=-1)
    roi = anatomy_roi_mask((args.eval_heatmap_size, args.eval_heatmap_size))
    variants = {"before_calibration": [], "after_calibration": []}
    per_concept = {
        variant: {name: [] for name in LESION_CONCEPTS} for variant in variants
    }
    classification_probabilities: list[torch.Tensor] = []
    classification_targets: list[torch.Tensor] = []

    for batch_index, batch in enumerate(loader, start=1):
        inputs = processor(images=batch["images"], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.amp and device.type == "cuda",
        ):
            outputs = model(pixel_values=pixel_values)
            tokens = spatial_tokens(outputs)
            image_features = output_value(outputs, "image_features")
        raw = tokens.float() @ positive_text.t()
        if batch["labels"] is not None:
            logits = concept_classification_logits(
                image_features,
                positive_text,
                negative_text,
                args.global_prompt_temperature,
            )
            classification_probabilities.append(torch.sigmoid(logits).cpu())
            classification_targets.append(batch["labels"].cpu())
        for image_index, image_annotations in enumerate(batch["boxes"]):
            grid_height, grid_width = patch_grid(tokens.shape[1])
            calibrated = calibrated_scores(raw[image_index], args)
            for concept_index, boxes in image_annotations.items():
                for variant, patch_values in (
                    ("before_calibration", raw[image_index, :, concept_index]),
                    ("after_calibration", calibrated[:, concept_index]),
                ):
                    hit = heatmap_peak_hit(
                        patch_values.detach().cpu(),
                        boxes,
                        grid_height,
                        grid_width,
                        roi,
                        args.eval_activation_quantile,
                    )
                    variants[variant].append(hit)
                    per_concept[variant][LESION_CONCEPTS[concept_index]].append(hit)
        if batch_index % args.print_freq == 0:
            print(f"[validation] {batch_index}/{len(loader)} batches")

    result: dict[str, Any] = {}
    for variant, hits in variants.items():
        class_metrics = {
            name: {"samples": len(values), "pg_hit": float(np.mean(values))}
            for name, values in per_concept[variant].items()
            if values
        }
        result[variant] = {
            "samples": len(hits),
            "hits": int(sum(hits)),
            "micro_pg_hit": float(np.mean(hits)) if hits else None,
            "macro_pg_hit": float(
                np.mean([metric["pg_hit"] for metric in class_metrics.values()])
            ) if class_metrics else None,
            "per_concept": class_metrics,
        }
    if classification_targets:
        probabilities = torch.cat(classification_probabilities).numpy()
        targets = torch.cat(classification_targets).numpy()
        result["classification"] = {
            "at_threshold": multilabel_metrics_at_threshold(
                probabilities,
                targets,
                ALL_CONCEPTS,
                args.validation_classification_threshold,
            ),
            "threshold_free": threshold_free_classification_metrics(
                probabilities,
                targets,
                ALL_CONCEPTS,
            ),
        }
    else:
        result["classification"] = None
    return result


def configure_trainable_parameters(model, args: argparse.Namespace) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    named_parameters = list(model.named_parameters())

    if args.trainable_pattern:
        patterns = [re.compile(pattern) for pattern in args.trainable_pattern]
        for name, parameter in named_parameters:
            if any(pattern.search(name) for pattern in patterns):
                parameter.requires_grad = True
    else:
        block_pattern = re.compile(
            r"(?:visual|vision)[^.]*.*?(?:resblocks|blocks|layers|layer)\.(\d+)(?:\.|$)",
            re.IGNORECASE,
        )
        block_indices = {
            int(match.group(1))
            for name, _ in named_parameters
            if (match := block_pattern.search(name))
        }
        if block_indices:
            first_trainable = max(block_indices) - args.unfreeze_vision_blocks + 1
            for name, parameter in named_parameters:
                match = block_pattern.search(name)
                if match and int(match.group(1)) >= first_trainable:
                    parameter.requires_grad = True
        projection_patterns = (
            "visual.proj", "visual.ln_post", "visual.head", "vision_model.post_layernorm",
            "vision_model.head", "vision_model.projection", "visual.trunk.norm",
        )
        for name, parameter in named_parameters:
            if any(pattern in name for pattern in projection_patterns):
                parameter.requires_grad = True

    trainable = [name for name, parameter in named_parameters if parameter.requires_grad]
    if not trainable:
        examples = "\n".join(name for name, _ in named_parameters[:30])
        raise RuntimeError(
            "Could not identify ConceptCLIP vision layers to train. Pass one or more "
            f"--trainable-pattern regex values. First parameter names:\n{examples}"
        )
    return trainable


def save_delta_checkpoint(
    path: Path,
    model,
    trainable_names: Sequence[str],
    epoch: int,
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    state = model.state_dict()
    delta = {name: state[name].detach().cpu() for name in trainable_names}
    torch.save(
        {
            "state_dict": delta,
            "epoch": epoch,
            "validation": metrics,
            "base_model": args.conceptclip_model,
            "base_revision": args.conceptclip_revision,
            "checkpoint_type": "trainable_parameter_delta",
            "trainable_names": list(trainable_names),
        },
        path,
    )


def train_one_epoch(
    model,
    processor,
    loader: DataLoader,
    positive_text: torch.Tensor,
    negative_text: torch.Tensor,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals: dict[str, float] = {
        name: 0.0
        for name in (
            "total",
            "localization",
            "global_classification",
            "preservation",
            "distribution",
            "ranking",
            "background",
            "concept",
            "localization_pairs",
        )
    }

    for batch_index, batch in enumerate(loader, start=1):
        inputs = processor(images=batch["images"], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.amp and device.type == "cuda",
        ):
            outputs = model(pixel_values=pixel_values)
            tokens = spatial_tokens(outputs)
            image_features = output_value(outputs, "image_features")
            localization, components = localization_loss(
                tokens,
                positive_text[: len(LESION_CONCEPTS)],
                batch["boxes"],
                args,
            )
            if batch["labels"] is not None:
                global_logits = concept_classification_logits(
                    image_features,
                    positive_text,
                    negative_text,
                    args.global_prompt_temperature,
                )
                global_classification = asymmetric_multilabel_loss(
                    global_logits,
                    batch["labels"].to(device, non_blocking=True),
                    args.global_asl_gamma_positive,
                    args.global_asl_gamma_negative,
                    args.global_asl_clip,
                )
            else:
                global_classification = image_features.sum() * 0.0
            if batch["teacher_embeddings"] is not None:
                teacher = batch["teacher_embeddings"].to(device, non_blocking=True)
                preservation = (
                    1.0
                    - F.cosine_similarity(
                        F.normalize(image_features.float(), dim=-1),
                        F.normalize(teacher.float(), dim=-1),
                        dim=-1,
                    )
                ).mean()
            else:
                preservation = image_features.sum() * 0.0
            loss = (
                localization
                + args.global_classification_weight * global_classification
                + args.preservation_weight * preservation
            )
            scaled_loss = loss / args.gradient_accumulation
        scaler.scale(scaled_loss).backward()

        should_step = (
            batch_index % args.gradient_accumulation == 0 or batch_index == len(loader)
        )
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        totals["total"] += float(loss.detach())
        totals["localization"] += float(localization.detach())
        totals["global_classification"] += float(global_classification.detach())
        totals["preservation"] += float(preservation.detach())
        for name, value in components.items():
            totals[name] += value
        if batch_index % args.print_freq == 0:
            print(
                f"[train] {batch_index}/{len(loader)} loss={loss.item():.4f} "
                f"rank={components['ranking']:.4f} "
                f"global={global_classification.item():.4f} "
                f"preserve={preservation.item():.4f}"
            )
    return {name: value / len(loader) for name, value in totals.items()}


def main(args: argparse.Namespace) -> None:
    named_test_files = [
        path
        for path in (
            args.train_bbox_csv,
            args.val_bbox_csv,
            args.train_labels_csv,
            args.val_labels_csv,
        )
        if path and "test" in Path(path).name.casefold()
    ]
    if named_test_files and not args.allow_test_named_training_data:
        raise ValueError(
            f"A train/validation bbox filename contains 'test': {named_test_files}. "
            "Fine-tuning or selecting a checkpoint on test boxes is leakage. "
            "Use VinDr training boxes, or explicitly pass --allow-test-named-training-data "
            "only for a non-reportable debugging run."
        )
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in (0, 1)")
    if args.gradient_accumulation < 1 or args.batch_size < 1:
        raise ValueError("Batch size and gradient accumulation must be positive")
    if args.unfreeze_vision_blocks < 1:
        raise ValueError("--unfreeze-vision-blocks must be at least 1")
    if args.bbox_coord_size is not None and args.bbox_coord_size <= 0:
        raise ValueError("--bbox-coord-size must be positive")
    if not 0.0 <= args.flip_probability <= 1.0:
        raise ValueError("--flip-probability must be in [0, 1]")
    if args.spatial_temperature <= 0 or args.concept_temperature <= 0:
        raise ValueError("Loss temperatures must be positive")
    if args.global_prompt_temperature <= 0:
        raise ValueError("--global-prompt-temperature must be positive")
    if not 0.0 <= args.eval_activation_quantile < 1.0:
        raise ValueError("--eval-activation-quantile must be in [0, 1)")
    if not 0.0 <= args.validation_classification_threshold <= 1.0:
        raise ValueError("--validation-classification-threshold must be in [0, 1]")
    if args.global_asl_gamma_positive < 0 or args.global_asl_gamma_negative < 0:
        raise ValueError("ASL gamma values must be non-negative")
    if not 0.0 <= args.global_asl_clip < 1.0:
        raise ValueError("--global-asl-clip must be in [0, 1)")
    if args.classification_selection_weight < 0 or args.maximum_pghit_drop < 0:
        raise ValueError("Selection weight and maximum PGHit drop must be non-negative")
    loss_weights = (
        args.distribution_weight,
        args.ranking_weight,
        args.background_weight,
        args.concept_weight,
        args.global_classification_weight,
        args.preservation_weight,
    )
    if any(weight < 0 for weight in loss_weights) or not any(loss_weights):
        raise ValueError("Loss weights must be non-negative and at least one must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records, train_stats = build_records(
        args.train_image_dir,
        args.train_bbox_csv,
        args.bbox_coord_size,
        args.train_labels_csv,
    )
    if args.val_bbox_csv:
        train_records = all_records
        val_records, val_stats = build_records(
            args.val_image_dir or args.train_image_dir,
            args.val_bbox_csv,
            args.bbox_coord_size,
            args.val_labels_csv,
        )
        overlap = {record.image_id for record in train_records} & {
            record.image_id for record in val_records
        }
        if overlap:
            raise ValueError(f"Train/validation image leakage detected: {len(overlap)} IDs")
    else:
        train_records, val_records = split_records(
            all_records, args.validation_fraction, args.seed
        )
        val_stats = {"source": "grouped split from training bbox CSV"}

    if args.global_classification_weight > 0.0:
        missing_train_labels = sum(record.labels is None for record in train_records)
        missing_val_labels = sum(record.labels is None for record in val_records)
        if missing_train_labels or missing_val_labels:
            raise ValueError(
                "Global classification loss requires --train-labels-csv and labels "
                f"for every split image; missing train={missing_train_labels}, "
                f"validation={missing_val_labels}"
            )

    print(f"Device: {device}")
    print(f"Train images: {len(train_records)} | validation images: {len(val_records)}")
    print(f"Train concepts: {concept_counts(train_records)}")
    print(f"Validation concepts: {concept_counts(val_records)}")

    train_loader = DataLoader(
        VinDrLocalizationDataset(train_records, args.flip_probability),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_samples,
    )
    val_loader = DataLoader(
        VinDrLocalizationDataset(val_records, 0.0),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_samples,
    )

    load_args = argparse.Namespace(
        conceptclip_model=args.conceptclip_model,
        conceptclip_revision=args.conceptclip_revision,
        conceptclip_checkpoint=None,
    )
    print(f"Loading {args.conceptclip_model} at {args.conceptclip_revision}...")
    model, processor = load_conceptclip(load_args, device)
    positive_text = encode_prompt_ensemble(
        model, processor, device, positive_prompts
    ).detach()
    negative_text = encode_prompt_ensemble(
        model, processor, device, negative_prompts
    ).detach()
    if args.preservation_weight > 0.0:
        print("Caching pretrained global embeddings for preservation loss...")
        cache_pretrained_embeddings(
            model, processor, train_records, device, args
        )
    trainable_names = configure_trainable_parameters(model, args)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    print(
        f"Trainable: {sum(parameter.numel() for parameter in trainable_parameters):,} / "
        f"{sum(parameter.numel() for parameter in model.parameters()):,} parameters"
    )
    print("Trainable parameter examples:")
    for name in trainable_names[:20]:
        print(f"  {name}")

    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = round(args.warmup_ratio * total_steps)

    def learning_rate_multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-3, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp and device.type == "cuda"
    )

    print("Evaluating pretrained localization/classification baseline...")
    baseline = evaluate_pghit(
        model,
        processor,
        val_loader,
        positive_text,
        negative_text,
        device,
        args,
    )
    baseline_pg = baseline["after_calibration"]["micro_pg_hit"]
    baseline_map = baseline["classification"]["threshold_free"][
        "macro_average_precision"
    ]
    print(
        f"Pretrained validation PGHit: {baseline_pg:.4f} | "
        f"classification macro-mAP: {baseline_map:.4f}"
    )
    history: list[dict[str, Any]] = []
    best_pg = baseline_pg
    best_map = baseline_map
    best_score = baseline_pg + args.classification_selection_weight * baseline_map
    best_epoch = 0
    patience = 0
    best_path = output_dir / "conceptclip_vindr_localization_classification_best.pth"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = train_one_epoch(
            model, processor, train_loader, positive_text, negative_text,
            optimizer, scheduler, scaler, device, args
        )
        validation = evaluate_pghit(
            model,
            processor,
            val_loader,
            positive_text,
            negative_text,
            device,
            args,
        )
        validation_pg = validation["after_calibration"]["micro_pg_hit"]
        validation_map = validation["classification"]["threshold_free"][
            "macro_average_precision"
        ]
        validation_f1 = validation["classification"]["at_threshold"]["macro_f1"]
        selection_score = (
            validation_pg + args.classification_selection_weight * validation_map
        )
        row = {"epoch": epoch, "train": train_metrics, "validation": validation}
        history.append(row)
        print(
            f"Validation PGHit={validation_pg:.4f}, macro-mAP={validation_map:.4f}, "
            f"macro-F1={validation_f1:.4f} | target PGHit>{args.target_pghit:.4f}"
        )
        pghit_is_preserved = validation_pg >= baseline_pg - args.maximum_pghit_drop
        if selection_score > best_score and pghit_is_preserved:
            best_score = selection_score
            best_pg = validation_pg
            best_map = validation_map
            best_epoch = epoch
            patience = 0
            save_delta_checkpoint(
                best_path, model, trainable_names, epoch, validation, args
            )
            print(f"Saved improved delta checkpoint: {best_path}")
        else:
            patience += 1
        if args.early_stop_patience and patience >= args.early_stop_patience:
            print("Early stopping: validation PGHit did not improve")
            break

    report = {
        "base_model": args.conceptclip_model,
        "base_revision": args.conceptclip_revision,
        "train_data": train_stats,
        "validation_data": val_stats,
        "train_images": len(train_records),
        "validation_images": len(val_records),
        "train_concept_counts": concept_counts(train_records),
        "validation_concept_counts": concept_counts(val_records),
        "pretrained_validation": baseline,
        "best_epoch": best_epoch,
        "best_validation_pghit": best_pg,
        "best_validation_classification_macro_map": best_map,
        "best_selection_score": best_score,
        "target_pghit": args.target_pghit,
        "target_passed": bool(best_pg > args.target_pghit),
        "checkpoint": str(best_path) if best_epoch else None,
        "history": history,
        "arguments": vars(args),
    }
    report_path = output_dir / "localization_classification_training_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"\nBest validation PGHit: {best_pg:.4f} at epoch {best_epoch}")
    print(f"Target passed: {best_pg > args.target_pghit}")
    print(f"Saved report: {report_path}")
    if best_epoch == 0:
        print("No balanced checkpoint saved because fine-tuning never beat the pretrained score.")
    if args.require_target and best_pg <= args.target_pghit:
        raise RuntimeError(
            f"Best validation PGHit {best_pg:.4f} did not exceed target {args.target_pghit:.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BBox-supervised ConceptCLIP fine-tuning with validation PGHit selection"
    )
    parser.add_argument("--train-image-dir", required=True)
    parser.add_argument("--train-bbox-csv", required=True)
    parser.add_argument(
        "--train-labels-csv",
        default="vindr/image_labels_train.csv",
        help="VinDr 28-label CSV used by the global multi-label loss",
    )
    parser.add_argument("--val-image-dir", default=None)
    parser.add_argument("--val-bbox-csv", default=None)
    parser.add_argument("--val-labels-csv", default=None)
    parser.add_argument("--bbox-coord-size", default=384.0, type=float)
    parser.add_argument("--validation-fraction", default=0.20, type=float)
    parser.add_argument("--conceptclip-model", default="JerrryNie/ConceptCLIP")
    parser.add_argument(
        "--conceptclip-revision",
        default="8120d7f1e07b590a7dce35bd2a01126b0e42b6c3",
    )
    parser.add_argument("--output-dir", default="./checkpoints/conceptclip_vindr_balanced")
    parser.add_argument("--epochs", default=12, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--eval-batch-size", default=4, type=int)
    parser.add_argument("--gradient-accumulation", default=8, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--learning-rate", default=1e-6, type=float)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--warmup-ratio", default=0.10, type=float)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--unfreeze-vision-blocks", default=2, type=int)
    parser.add_argument(
        "--trainable-pattern",
        action="append",
        default=None,
        help="Regex selecting trainable model parameters; repeat as needed",
    )
    parser.add_argument("--distribution-weight", default=1.0, type=float)
    parser.add_argument("--ranking-weight", default=2.0, type=float)
    parser.add_argument("--background-weight", default=0.25, type=float)
    parser.add_argument("--concept-weight", default=0.25, type=float)
    parser.add_argument("--global-classification-weight", default=2.0, type=float)
    parser.add_argument("--preservation-weight", default=1.0, type=float)
    parser.add_argument("--global-prompt-temperature", default=0.07, type=float)
    parser.add_argument("--global-asl-gamma-positive", default=1.0, type=float)
    parser.add_argument("--global-asl-gamma-negative", default=4.0, type=float)
    parser.add_argument("--global-asl-clip", default=0.05, type=float)
    parser.add_argument("--spatial-temperature", default=0.07, type=float)
    parser.add_argument("--concept-temperature", default=0.07, type=float)
    parser.add_argument("--ranking-margin", default=0.10, type=float)
    parser.add_argument("--flip-probability", default=0.50, type=float)
    parser.add_argument("--eval-common-mode-strength", default=1.0, type=float)
    parser.add_argument("--eval-calibration-temperature", default=0.75, type=float)
    parser.add_argument("--eval-activation-quantile", default=0.75, type=float)
    parser.add_argument("--eval-heatmap-size", default=384, type=int)
    parser.add_argument("--target-pghit", default=0.25, type=float)
    parser.add_argument("--validation-classification-threshold", default=0.50, type=float)
    parser.add_argument(
        "--classification-selection-weight",
        default=0.50,
        type=float,
        help="Weight of validation classification macro-mAP in checkpoint selection",
    )
    parser.add_argument(
        "--maximum-pghit-drop",
        default=0.0,
        type=float,
        help="Maximum PGHit decrease allowed when saving a balanced checkpoint",
    )
    parser.add_argument("--require-target", action="store_true")
    parser.add_argument("--early-stop-patience", default=4, type=int)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--print-freq", default=25, type=int)
    parser.add_argument(
        "--allow-test-named-training-data",
        action="store_true",
        help="Allow a test-named bbox CSV for debugging only; results are invalid for reporting",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
