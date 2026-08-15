"""Evaluate ConvNeXtV2-SRA retrieval on VinDr-CXR with CRAR reranking.

CRAR follows the supplied Hybrid Concept-Visual Ranking algorithm exactly:

    S_vis = (1 + cosine(e_q, e_n)) / 2
    J = C_q intersection C_n
    S_concept = sum_j(w_j * min(s_qj, s_nj)) / sum_j(w_j), j in J
    S_total = (1 - gamma) * S_vis + gamma * S_concept

The base model supplies visual embeddings. ConceptCLIP and VinDr-specific prompt
ensembles supply concept confidences. Ground-truth labels are used only for
evaluation and never by CRAR.
"""

from __future__ import annotationsz 

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModel, AutoProcessor, PreTrainedModel

from model import ConvNeXtV2_SRA


LESION_CONCEPTS = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Clavicle fracture",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Enlarged PA",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Lung cavity",
    "Lung cyst",
    "Mediastinal shift",
    "Nodule/Mass",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
    "Rib fracture",
    "Other lesion",
]

DISEASE_CONCEPTS = [
    "COPD",
    "Lung tumor",
    "Pneumonia",
    "Tuberculosis",
    "Other diseases",
    "No finding",
]

ALL_CONCEPTS = LESION_CONCEPTS + DISEASE_CONCEPTS

# Descriptions are intentionally radiographic and match the labels present in
# image_labels_test.csv. They are inserted into multiple prompt templates.
CONCEPT_DESCRIPTIONS = {
    "Aortic enlargement": "an enlarged aortic contour and widened upper mediastinum",
    "Atelectasis": "atelectasis with focal or linear lung volume loss",
    "Calcification": "a calcified thoracic opacity",
    "Cardiomegaly": "cardiomegaly with an enlarged cardiac silhouette",
    "Clavicle fracture": "a clavicle fracture with cortical discontinuity",
    "Consolidation": "air-space consolidation with dense pulmonary opacity",
    "Edema": "pulmonary edema with bilateral interstitial or perihilar opacity",
    "Emphysema": "emphysema with hyperinflation and reduced vascular markings",
    "Enlarged PA": "an enlarged pulmonary artery suggesting pulmonary hypertension",
    "ILD": "interstitial lung disease with reticular or ground-glass opacity",
    "Infiltration": "an ill-defined pulmonary infiltrate",
    "Lung Opacity": "a focal or diffuse abnormal lung opacity",
    "Lung cavity": "a pulmonary cavity with a visible wall and central lucency",
    "Lung cyst": "a thin-walled air-filled lung cyst",
    "Mediastinal shift": "displacement of the mediastinum from the midline",
    "Nodule/Mass": "a pulmonary nodule or lung mass",
    "Pleural effusion": "pleural fluid with costophrenic angle blunting",
    "Pleural thickening": "focal or diffuse pleural thickening",
    "Pneumothorax": "a pleural line with absent peripheral lung markings",
    "Pulmonary fibrosis": "pulmonary fibrosis with reticulation or fibrotic scarring",
    "Rib fracture": "a rib fracture with cortical disruption",
    "Other lesion": "another abnormal thoracic radiographic lesion",
    "COPD": "chronic obstructive pulmonary disease with hyperinflation",
    "Lung tumor": "a lung tumor or pulmonary malignancy",
    "Pneumonia": "pneumonia with infectious air-space opacity",
    "Tuberculosis": "pulmonary tuberculosis with typical parenchymal abnormalities",
    "Other diseases": "another clinically significant thoracic disease",
    "No finding": "a normal chest radiograph without significant abnormality",
}


def positive_prompts(concept: str) -> list[str]:
    description = CONCEPT_DESCRIPTIONS[concept]
    if concept == "No finding":
        return [
            "a normal frontal chest x-ray without significant pathological findings",
            "a chest radiograph with clear lungs and no acute cardiopulmonary abnormality",
            "a VinDr-CXR image showing no abnormal radiographic finding",
        ]
    return [
        f"a frontal chest x-ray showing {description}",
        f"a chest radiograph demonstrating {description}",
        f"a VinDr-CXR image with evidence of {description}",
    ]


def negative_prompts(concept: str) -> list[str]:
    if concept == "No finding":
        return [
            "an abnormal chest x-ray with a pathological thoracic finding",
            "a chest radiograph showing one or more cardiopulmonary abnormalities",
            "a VinDr-CXR image with an abnormal lesion",
        ]
    description = CONCEPT_DESCRIPTIONS[concept]
    return [
        f"a frontal chest x-ray without {description}",
        f"a chest radiograph showing no evidence of {description}",
        f"a VinDr-CXR image negative for {concept.lower()}",
    ]


class VinDrCRARDataset(Dataset):
    """VinDr test set with aligned tensor/PIL views and multi-hot labels."""

    def __init__(
        self,
        image_dir: str | os.PathLike[str],
        labels_csv: str | os.PathLike[str],
        transform: Any,
        label_columns: Sequence[str],
    ) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.label_columns = list(label_columns)

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"VinDr image directory does not exist: {self.image_dir}")
        csv_path = Path(labels_csv)
        if not csv_path.is_file():
            raise FileNotFoundError(f"VinDr labels CSV does not exist: {csv_path}")

        frame = pd.read_csv(csv_path)
        if "Other disease" in frame.columns and "Other diseases" not in frame.columns:
            frame = frame.rename(columns={"Other disease": "Other diseases"})
        if "image_id" not in frame.columns:
            raise ValueError("VinDr labels CSV must contain an image_id column")
        missing_columns = [name for name in self.label_columns if name not in frame.columns]
        if missing_columns:
            raise ValueError(f"VinDr labels CSV is missing columns: {missing_columns}")

        frame[self.label_columns] = (
            frame[self.label_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        )
        # VinDr training CSV may contain multiple radiologists per image. max()
        # produces one multi-hot target per image; it is a no-op for the test CSV.
        frame = frame.groupby("image_id", as_index=False)[self.label_columns].max()

        self.image_ids: list[str] = []
        self.image_paths: list[Path] = []
        labels: list[np.ndarray] = []
        missing_images: list[str] = []
        for _, row in frame.iterrows():
            image_id = str(row["image_id"])
            image_path = self._resolve_image(image_id)
            if image_path is None:
                missing_images.append(image_id)
                continue
            self.image_ids.append(image_id)
            self.image_paths.append(image_path)
            labels.append(row[self.label_columns].to_numpy(dtype=np.float32))

        if missing_images:
            examples = ", ".join(missing_images[:5])
            raise FileNotFoundError(
                f"Missing {len(missing_images)} VinDr images under {self.image_dir}. "
                f"Examples: {examples}. Expected <image_id>.png (or jpg/jpeg)."
            )
        if not self.image_paths:
            raise ValueError("No VinDr images were loaded")
        self.labels = torch.from_numpy(np.stack(labels))

    def _resolve_image(self, image_id: str) -> Path | None:
        for suffix in (".png", ".jpg", ".jpeg"):
            candidate = self.image_dir / f"{image_id}{suffix}"
            if candidate.is_file():
                return candidate
        return None

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Image.Image, torch.Tensor, str]:
        with Image.open(self.image_paths[index]) as handle:
            image = handle.convert("RGB")
        tensor = self.transform(image)
        return tensor, image, self.labels[index], self.image_ids[index]

def collate_vindr(batch: Iterable[tuple[torch.Tensor, Image.Image, torch.Tensor, str]]):
    tensors, pil_images, labels, image_ids = zip(*batch)
    return torch.stack(tensors), list(pil_images), torch.stack(labels), list(image_ids)


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must contain a state-dict-like mapping")
    for key in ("state_dict", "state-dict", "model_state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            payload = candidate
            break
    state = {}
    for key, value in payload.items():
        if not torch.is_tensor(value):
            continue
        normalized = key[7:] if key.startswith("module.") else key
        state[normalized] = value
    if not state:
        raise ValueError("No tensor parameters were found in the checkpoint")
    return state


def infer_num_labels(state: dict[str, torch.Tensor]) -> int | None:
    weight = state.get("classification_head.weight")
    return int(weight.shape[0]) if weight is not None and weight.ndim == 2 else None


def load_base_model(args: argparse.Namespace, device: torch.device) -> ConvNeXtV2_SRA:
    checkpoint_path = Path(args.base_checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Base checkpoint does not exist: {checkpoint_path}")
    state = extract_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = ConvNeXtV2_SRA(
        num_heads=args.sra_num_heads,
        lam=args.sra_lam,
        num_labels=infer_num_labels(state),
    )
    incompatible = model.load_state_dict(state, strict=False)
    critical_missing = [key for key in incompatible.missing_keys if not key.startswith("classification_head.")]
    if critical_missing:
        raise RuntimeError(
            "Checkpoint is not compatible with ConvNeXtV2-SRA; missing keys include: "
            + ", ".join(critical_missing[:8])
        )
    if incompatible.unexpected_keys:
        print(f"[warning] Ignored {len(incompatible.unexpected_keys)} unexpected checkpoint keys")
    model.to(device).eval()
    return model


def load_conceptclip(args: argparse.Namespace, device: torch.device):
    processor = AutoProcessor.from_pretrained(
        args.conceptclip_model,
        trust_remote_code=True,
        revision=args.conceptclip_revision,
        # Transformers recently changed SigLIP to the fast processor by default.
        # Keep the preprocessing used by the released ConceptCLIP checkpoint.
        use_fast=False,
    )

    # Some recent Transformers releases assume every custom PreTrainedModel ran
    # the new post_init(), which creates all_tied_weights_keys. The released
    # ConceptCLIP remote class predates that API and does not create the field.
    # Add an empty per-instance mapping before Transformers finalizes loading.
    original_mark_tied = getattr(PreTrainedModel, "mark_tied_weights_as_initialized", None)
    if original_mark_tied is not None:
        def mark_tied_weights_compat(model_self, *method_args, **method_kwargs):
            if not hasattr(model_self, "all_tied_weights_keys"):
                model_self.all_tied_weights_keys = {}
            return original_mark_tied(model_self, *method_args, **method_kwargs)

        PreTrainedModel.mark_tied_weights_as_initialized = mark_tied_weights_compat
    try:
        model = AutoModel.from_pretrained(
            args.conceptclip_model,
            trust_remote_code=True,
            revision=args.conceptclip_revision,
        )
    finally:
        if original_mark_tied is not None:
            PreTrainedModel.mark_tied_weights_as_initialized = original_mark_tied
    if args.conceptclip_checkpoint:
        path = Path(args.conceptclip_checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"ConceptCLIP checkpoint does not exist: {path}")
        wrapper_state = extract_state_dict(torch.load(path, map_location="cpu"))
        # train.py saves the conceptCLIP wrapper, whose remote model is under model.*
        state = {
            (key[6:] if key.startswith("model.") else key): value
            for key, value in wrapper_state.items()
            if not key.startswith("fc.")
        }
        incompatible = model.load_state_dict(state, strict=False)
        print(
            f"[ConceptCLIP checkpoint] missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
    model.to(device).eval()
    return model, processor


def output_value(outputs: Any, name: str) -> torch.Tensor:
    if isinstance(outputs, dict) and name in outputs:
        return outputs[name]
    value = getattr(outputs, name, None)
    if value is None:
        raise KeyError(f"ConceptCLIP output does not contain '{name}'")
    return value


@torch.inference_mode()
def encode_texts(model, processor, texts: Sequence[str], device: torch.device) -> torch.Tensor:
    inputs = processor(text=list(texts), padding=True, truncation=True, return_tensors="pt")
    text_kwargs = {key: value.to(device) for key, value in inputs.items() if key in {"input_ids", "attention_mask", "token_type_ids"}}
    outputs = model(**text_kwargs)
    return F.normalize(output_value(outputs, "text_features").float(), dim=1)


@torch.inference_mode()
def build_prompt_embeddings(model, processor, concepts: Sequence[str], device: torch.device):
    positive_flat = [prompt for concept in concepts for prompt in positive_prompts(concept)]
    negative_flat = [prompt for concept in concepts for prompt in negative_prompts(concept)]
    positive = encode_texts(model, processor, positive_flat, device)
    negative = encode_texts(model, processor, negative_flat, device)
    prompts_per_concept = len(positive_prompts(concepts[0]))
    positive = positive.view(len(concepts), prompts_per_concept, -1).mean(dim=1)
    negative = negative.view(len(concepts), prompts_per_concept, -1).mean(dim=1)
    return F.normalize(positive, dim=1), F.normalize(negative, dim=1)


@torch.inference_mode()
def extract_features(
    base_model,
    conceptclip_model,
    processor,
    loader,
    positive_text: torch.Tensor,
    negative_text: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
):
    base_embeddings, clip_embeddings, confidence_batches, labels, image_ids = [], [], [], [], []
    use_amp = args.amp and device.type == "cuda"
    for batch_index, (base_images, pil_images, batch_labels, batch_ids) in enumerate(loader, start=1):
        base_images = base_images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            base_output = base_model(base_images)
            if isinstance(base_output, dict):
                base_output = base_output["embedding"]
        base_embedding = F.normalize(base_output.float(), dim=1)

        clip_inputs = processor(images=pil_images, return_tensors="pt")
        pixel_values = clip_inputs["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            clip_output = conceptclip_model(pixel_values=pixel_values)
        clip_embedding = F.normalize(output_value(clip_output, "image_features").float(), dim=1)

        positive_logits = clip_embedding @ positive_text.t() / args.prompt_temperature
        negative_logits = clip_embedding @ negative_text.t() / args.prompt_temperature
        # Binary positive-vs-negative prompt probability for each concept.
        concept_confidence = torch.softmax(
            torch.stack((negative_logits, positive_logits), dim=-1), dim=-1
        )[..., 1]

        base_embeddings.append(base_embedding.cpu())
        clip_embeddings.append(clip_embedding.cpu())
        confidence_batches.append(concept_confidence.cpu())
        labels.append(batch_labels.cpu())
        image_ids.extend(batch_ids)
        if batch_index % args.print_freq == 0:
            print(f"[features] processed {min(batch_index * args.batch_size, len(loader.dataset))}/{len(loader.dataset)} images")

    return (
        torch.cat(base_embeddings),
        torch.cat(clip_embeddings),
        torch.cat(confidence_batches),
        torch.cat(labels),
        image_ids,
    )


def load_concept_weights(args: argparse.Namespace, concepts: Sequence[str], confidences: torch.Tensor) -> torch.Tensor:
    if args.concept_weights_json:
        path = Path(args.concept_weights_json)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        unknown = sorted(set(payload) - set(concepts))
        if unknown:
            raise ValueError(f"Unknown concepts in weights JSON: {unknown}")
        weights = torch.tensor([float(payload.get(name, 1.0)) for name in concepts])
    elif args.concept_weighting == "idf":
        prevalence = (confidences >= args.concept_threshold).float().mean(dim=0)
        weights = torch.log((1.0 + len(confidences)) / (1.0 + prevalence * len(confidences))) + 1.0
    else:
        weights = torch.ones(len(concepts))
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("All CRAR concept weights must be finite and greater than zero")
    return weights.float()


def crar_rerank(
    visual_embeddings: torch.Tensor,
    concept_confidences: torch.Tensor,
    concept_weights: torch.Tensor,
    gamma: float,
    threshold: float,
    rerank_top_k: int,
):
    """Apply Algorithm 2 to the base top-K candidates of every query."""
    visual_embeddings = F.normalize(visual_embeddings.float(), dim=1)
    cosine = visual_embeddings @ visual_embeddings.t()
    visual_similarity = (1.0 + cosine) / 2.0
    visual_similarity.fill_diagonal_(-float("inf"))

    count = len(visual_embeddings)
    base_ranks = torch.argsort(visual_similarity, dim=1, descending=True)[:, : max(0, count - 1)]
    reranked = base_ranks.clone()
    effective_k = min(rerank_top_k, max(0, count - 1))
    active = concept_confidences >= threshold

    overlap_pairs = 0
    concept_score_sum = 0.0
    for query_index in range(count):
        candidates = base_ranks[query_index, :effective_k]
        if candidates.numel() == 0:
            continue
        shared = active[query_index].unsqueeze(0) & active[candidates]
        alignment = torch.minimum(
            concept_confidences[query_index].unsqueeze(0), concept_confidences[candidates]
        )
        weighted_shared = shared.float() * concept_weights.unsqueeze(0)
        normalization = weighted_shared.sum(dim=1)
        concept_score = (weighted_shared * alignment).sum(dim=1) / normalization.clamp_min(1e-12)
        concept_score = torch.where(normalization > 0, concept_score, torch.zeros_like(concept_score))

        total_score = (
            (1.0 - gamma) * visual_similarity[query_index, candidates]
            + gamma * concept_score
        )
        order = torch.argsort(total_score, descending=True, stable=True)
        reranked[query_index, :effective_k] = candidates[order]
        overlap_pairs += int((normalization > 0).sum().item())
        concept_score_sum += float(concept_score.sum().item())

    stats = {
        "rerank_top_k": effective_k,
        "candidate_pairs": count * effective_k,
        "pairs_with_shared_concepts": overlap_pairs,
        "shared_concept_pair_rate": overlap_pairs / max(1, count * effective_k),
        "mean_concept_score": concept_score_sum / max(1, count * effective_k),
    }
    return base_ranks, reranked, stats


def crar_pair_scores(
    query_index: int,
    candidate_indices: torch.Tensor,
    visual_embeddings: torch.Tensor,
    concept_confidences: torch.Tensor,
    concept_weights: torch.Tensor,
    gamma: float,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Svis, Sconcept and Stotal for selected query-candidate pairs."""
    visual_embeddings = F.normalize(visual_embeddings.float(), dim=1)
    cosine = visual_embeddings[candidate_indices] @ visual_embeddings[query_index]
    visual_score = (1.0 + cosine) / 2.0

    query_active = concept_confidences[query_index] >= threshold
    candidate_active = concept_confidences[candidate_indices] >= threshold
    shared = candidate_active & query_active.unsqueeze(0)
    alignment = torch.minimum(
        concept_confidences[candidate_indices],
        concept_confidences[query_index].unsqueeze(0),
    )
    weighted_shared = shared.float() * concept_weights.unsqueeze(0)
    normalization = weighted_shared.sum(dim=1)
    concept_score = (weighted_shared * alignment).sum(dim=1) / normalization.clamp_min(1e-12)
    concept_score = torch.where(normalization > 0, concept_score, torch.zeros_like(concept_score))
    total_score = (1.0 - gamma) * visual_score + gamma * concept_score
    return visual_score, concept_score, total_score


def patch_grid(token_count: int) -> tuple[int, int]:
    """Find a near-square patch grid for the number of spatial tokens."""
    height = int(math.sqrt(token_count))
    while height > 1 and token_count % height != 0:
        height -= 1
    return height, token_count // height


def remove_cls_token_if_present(tokens: torch.Tensor) -> torch.Tensor:
    """Remove a leading CLS token only when N-1 forms a cleaner square grid."""
    token_count = tokens.shape[0]
    side = int(math.sqrt(token_count))
    if side * side == token_count:
        return tokens
    side_without_cls = int(math.sqrt(token_count - 1))
    if side_without_cls * side_without_cls == token_count - 1:
        return tokens[1:]
    # ConceptCLIP normally returns 27x27 tokens (or CLS + 27x27). For an
    # unusual rectangular grid, prefer keeping all tokens over dropping data.
    return tokens


@torch.inference_mode()
def extract_patch_tokens(
    model,
    processor,
    images: Sequence[Image.Image],
    device: torch.device,
    use_amp: bool,
) -> list[torch.Tensor]:
    """Extract normalized ConceptCLIP spatial tokens for a small image list."""
    result: list[torch.Tensor] = []
    for image in images:
        inputs = processor(images=[image], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            outputs = model(pixel_values=pixel_values)
        try:
            tokens = output_value(outputs, "image_token_features")
        except KeyError:
            tokens = output_value(outputs, "last_hidden_state")
        tokens = remove_cls_token_if_present(tokens[0].float())
        result.append(F.normalize(tokens, dim=-1).cpu())
    return result


def anatomy_roi_mask(
    output_size: tuple[int, int], edge_softness: float = 0.08
) -> torch.Tensor:
    """Soft union of neck/shoulder, thorax and upper-abdomen regions.

    This geometric prior removes corners, borders and acquisition artifacts
    without pretending to be a pathology or lung segmentation model.
    """
    height, width = output_size
    y = torch.linspace(0.0, 1.0, height).view(height, 1)
    x = torch.linspace(0.0, 1.0, width).view(1, width)

    def soft_ellipse(cx: float, cy: float, rx: float, ry: float) -> torch.Tensor:
        distance = ((x - cx) / rx).square() + ((y - cy) / ry).square()
        return torch.sigmoid((1.0 - distance) / edge_softness)

    shoulder_neck = soft_ellipse(0.50, 0.19, 0.48, 0.20)
    thorax = soft_ellipse(0.50, 0.50, 0.44, 0.40)
    upper_abdomen = soft_ellipse(0.50, 0.87, 0.40, 0.24)
    mask = torch.maximum(torch.maximum(shoulder_neck, thorax), upper_abdomen)

    # Smoothly suppress the extreme top/bottom strips where burned-in markers,
    # collimation borders and table hardware commonly appear.
    vertical_window = torch.sigmoid((y - 0.025) / 0.015) * torch.sigmoid((0.99 - y) / 0.015)
    return (mask * vertical_window).clamp(0.0, 1.0)


def calibrated_concept_heatmaps(
    patch_tokens: torch.Tensor,
    all_concept_embeddings: torch.Tensor,
    selected_concept_indices: Sequence[int],
    output_size: tuple[int, int],
    args: argparse.Namespace,
) -> list[np.ndarray]:
    """Build concept-specific maps calibrated jointly within the same image.

    Independent min-max normalization makes weak/shared maps look equally
    salient. Here we remove the response shared by other concepts at each patch,
    robustly standardize across the concept dictionary, and use a softmax to
    make concepts compete for spatial evidence before applying a sparse ROI.
    """
    # Patch tokens are deliberately returned on CPU to keep visualization and
    # full-dataset PGHit memory bounded after the 2.17 GB model forward pass.
    concept_embeddings = F.normalize(all_concept_embeddings.detach().float().cpu(), dim=1)
    raw_alignment = patch_tokens.float() @ concept_embeddings.t()  # [patch, concept]
    concept_count = raw_alignment.shape[1]

    if args.heatmap_calibration == "competitive" and concept_count > 1:
        other_mean = (
            raw_alignment.sum(dim=1, keepdim=True) - raw_alignment
        ) / (concept_count - 1)
        specific_alignment = (
            raw_alignment
            - args.heatmap_common_mode_strength * other_mean
        )

        # Per-patch robust z-score across concepts: patches activated for nearly
        # every label receive little discriminative evidence.
        center = specific_alignment.median(dim=1, keepdim=True).values
        absolute_deviation = (specific_alignment - center).abs()
        scale = 1.4826 * absolute_deviation.median(dim=1, keepdim=True).values
        discriminative = (specific_alignment - center) / scale.clamp_min(1e-4)
        competition = torch.softmax(
            discriminative / args.heatmap_calibration_temperature,
            dim=1,
        )
        patch_scores = F.relu(discriminative) * competition
    else:
        # Diagnostic fallback matching the original independent cosine maps.
        patch_scores = raw_alignment

    height, width = output_size
    roi = (
        anatomy_roi_mask(output_size)
        if not args.no_anatomy_mask
        else torch.ones((height, width))
    )
    heatmaps: list[np.ndarray] = []
    for concept_index in selected_concept_indices:
        values = patch_scores[:, concept_index]
        # Keep only the most discriminative patches. This is performed after
        # cross-concept calibration, not independently on raw cosine values.
        cutoff = torch.quantile(values, args.heatmap_activation_quantile)
        values = F.relu(values - cutoff)
        upper = torch.quantile(values, 0.99)
        # A highly specific map may activate fewer than 1% of patches. In that
        # case q99 is exactly zero even though a valid positive peak exists;
        # falling back to max preserves the sparse localization instead of
        # turning the entire map into zeros (whose argmax is the top-left).
        if float(upper) <= 1e-8:
            upper = values.max()
        if float(upper) > 1e-8:
            values = (values / upper).clamp(0.0, 1.0)
        else:
            values = torch.zeros_like(values)

        grid_h, grid_w = patch_grid(len(values))
        heatmap = values.reshape(1, 1, grid_h, grid_w)
        heatmap = F.interpolate(
            heatmap,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 1.0)[0, 0]
        heatmaps.append((heatmap * roi).clamp(0.0, 1.0).numpy())
    return heatmaps


def canonical_concept_name(name: str, concepts: Sequence[str]) -> str | None:
    """Map harmless CSV spelling/case variants to the configured concept name."""
    normalized = " ".join(str(name).strip().split()).casefold()
    aliases = {
        "other disease": "other diseases",
        "lung opacity": "lung opacity",
        "nodule/mass": "nodule/mass",
    }
    normalized = aliases.get(normalized, normalized)
    return next(
        (concept for concept in concepts if concept.casefold() == normalized),
        None,
    )


def load_vindr_bboxes(
    csv_path: str | os.PathLike[str],
    concepts: Sequence[str],
) -> tuple[
    dict[str, dict[str, list[tuple[float, float, float, float]]]],
    dict[str, int],
    dict[str, Any],
]:
    """Load VinDr boxes as image -> concept -> [(xmin, ymin, xmax, ymax)]."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"VinDr bbox CSV does not exist: {path}")

    grouped: dict[str, dict[str, list[tuple[float, float, float, float]]]] = {}
    skipped_classes: dict[str, int] = {}
    loader_stats: dict[str, Any] = {
        "total_rows": 0,
        "valid_bbox_rows": 0,
        "blank_coordinate_rows": 0,
        "invalid_numeric_rows": 0,
        "degenerate_bbox_rows": 0,
        "rows_without_bbox_by_class": {},
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        normalized_columns = {
            str(column).strip().lstrip("\ufeff").casefold(): column
            for column in (reader.fieldnames or [])
        }
        required = ("image_id", "class_name", "x_min", "y_min", "x_max", "y_max")
        missing = [column for column in required if column not in normalized_columns]
        if missing:
            raise ValueError(
                "VinDr bbox CSV must contain image_id,class_name,x_min,y_min,x_max,y_max; "
                f"missing {missing}"
            )

        keys = {column: normalized_columns[column] for column in required}
        for row in reader:
            loader_stats["total_rows"] += 1
            image_id = str(row.get(keys["image_id"], "")).strip()
            raw_class = str(row.get(keys["class_name"], "")).strip()
            if not image_id or not raw_class:
                continue
            concept = canonical_concept_name(raw_class, concepts)
            if concept is None:
                skipped_classes[raw_class] = skipped_classes.get(raw_class, 0) + 1
                continue
            coordinate_values = [
                str(row.get(keys[column], "") or "").strip()
                for column in ("x_min", "y_min", "x_max", "y_max")
            ]
            if any(not value for value in coordinate_values):
                # VinDr represents No finding as an image-level row with four
                # empty coordinates. It has no localization target and must be
                # excluded from PGHit rather than counted as a miss.
                loader_stats["blank_coordinate_rows"] += 1
                by_class = loader_stats["rows_without_bbox_by_class"]
                by_class[raw_class] = by_class.get(raw_class, 0) + 1
                continue
            try:
                box = tuple(float(value) for value in coordinate_values)
            except (TypeError, ValueError):
                loader_stats["invalid_numeric_rows"] += 1
                continue
            if not all(math.isfinite(value) for value in box):
                loader_stats["invalid_numeric_rows"] += 1
                continue
            x_min, y_min, x_max, y_max = box
            if x_max <= x_min or y_max <= y_min:
                loader_stats["degenerate_bbox_rows"] += 1
                continue
            grouped.setdefault(image_id, {}).setdefault(concept, []).append(box)
            loader_stats["valid_bbox_rows"] += 1
    return grouped, skipped_classes, loader_stats


def pointing_game_peak(
    heatmap: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[float, float, int, int, float]:
    """Return the max-saliency pixel centre in original-image coordinates."""
    saliency = np.nan_to_num(np.asarray(heatmap, dtype=np.float32))
    if saliency.ndim != 2 or saliency.size == 0:
        raise ValueError(f"PGHit expects a non-empty 2D heatmap, got {saliency.shape}")
    map_height, map_width = saliency.shape
    y_map, x_map = np.unravel_index(int(np.argmax(saliency)), saliency.shape)
    x_image = (float(x_map) + 0.5) * float(image_width) / float(map_width)
    y_image = (float(y_map) + 0.5) * float(image_height) / float(map_height)
    return x_image, y_image, int(x_map), int(y_map), float(saliency[y_map, x_map])


def scale_and_clip_boxes(
    boxes: Sequence[tuple[float, float, float, float]],
    image_width: int,
    image_height: int,
    coordinate_size: float | None,
) -> list[tuple[float, float, float, float]]:
    """Convert fixed-square bbox coordinates to image space and clip safely."""
    scale_x = float(image_width) / coordinate_size if coordinate_size else 1.0
    scale_y = float(image_height) / coordinate_size if coordinate_size else 1.0
    result = []
    for x_min, y_min, x_max, y_max in boxes:
        scaled = (
            max(0.0, min(float(image_width), x_min * scale_x)),
            max(0.0, min(float(image_height), y_min * scale_y)),
            max(0.0, min(float(image_width), x_max * scale_x)),
            max(0.0, min(float(image_height), y_max * scale_y)),
        )
        if scaled[2] > scaled[0] and scaled[3] > scaled[1]:
            result.append(scaled)
    return result


def summarize_pghit(rows: Sequence[dict[str, Any]], variant: str) -> dict[str, Any]:
    selected = [row for row in rows if row["variant"] == variant]
    per_concept: dict[str, dict[str, Any]] = {}
    for concept in sorted({row["class_name"] for row in selected}):
        concept_rows = [row for row in selected if row["class_name"] == concept]
        hits = sum(bool(row["pg_hit"]) for row in concept_rows)
        per_concept[concept] = {
            "samples": len(concept_rows),
            "hits": hits,
            "pg_hit_rate": hits / len(concept_rows),
            "empty_heatmaps": sum(bool(row["empty_heatmap"]) for row in concept_rows),
        }
    hits = sum(bool(row["pg_hit"]) for row in selected)
    rates = [item["pg_hit_rate"] for item in per_concept.values()]
    return {
        "samples": len(selected),
        "hits": hits,
        "pg_hit_rate": hits / len(selected) if selected else None,
        "macro_pg_hit_rate": float(np.mean(rates)) if rates else None,
        "empty_heatmaps": sum(bool(row["empty_heatmap"]) for row in selected),
        "per_concept": per_concept,
    }


def evaluate_vindr_pghit(
    bbox_csv: str | os.PathLike[str],
    image_ids: Sequence[str],
    image_path_by_id: dict[str, Path],
    concepts: Sequence[str],
    positive_text: torch.Tensor,
    conceptclip_model,
    processor,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate class-aware PGHit before and after within-image calibration."""
    grouped, skipped_classes, bbox_loader_stats = load_vindr_bboxes(bbox_csv, concepts)
    if bbox_loader_stats["blank_coordinate_rows"]:
        print(
            "[PGHit] skipped "
            f"{bbox_loader_stats['blank_coordinate_rows']} rows without bounding boxes: "
            f"{bbox_loader_stats['rows_without_bbox_by_class']}"
        )
    evaluated_ids = set(image_ids)
    eligible_ids = [image_id for image_id in image_ids if image_id in grouped]
    if args.pghit_max_images is not None:
        eligible_ids = eligible_ids[: args.pghit_max_images]
    if not eligible_ids:
        raise ValueError(
            "No bbox image_id matches the evaluated VinDr images and configured concept space"
        )

    independent_args = argparse.Namespace(**vars(args))
    independent_args.heatmap_calibration = "independent"
    calibrated_args = argparse.Namespace(**vars(args))
    calibrated_args.heatmap_calibration = "competitive"
    concept_to_index = {name: index for index, name in enumerate(concepts)}
    concept_text = positive_text.detach().float().cpu()
    rows: list[dict[str, Any]] = []

    for position, image_id in enumerate(eligible_ids, start=1):
        image_path = image_path_by_id[image_id]
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        width, height = image.size
        annotated = grouped[image_id]
        selected_concepts = [name for name in concepts if name in annotated]
        selected_indices = [concept_to_index[name] for name in selected_concepts]
        patch_tokens = extract_patch_tokens(
            conceptclip_model, processor, [image], device, args.amp
        )[0]

        variants = {
            "before_calibration": calibrated_concept_heatmaps(
                patch_tokens, concept_text, selected_indices, (height, width), independent_args
            ),
            "after_calibration": calibrated_concept_heatmaps(
                patch_tokens, concept_text, selected_indices, (height, width), calibrated_args
            ),
        }
        for variant, heatmaps in variants.items():
            for concept, heatmap in zip(selected_concepts, heatmaps):
                boxes = scale_and_clip_boxes(
                    annotated[concept], width, height, args.bbox_coord_size
                )
                if not boxes:
                    continue
                peak_x, peak_y, peak_map_x, peak_map_y, peak_value = pointing_game_peak(
                    heatmap, width, height
                )
                hit = any(
                    x_min <= peak_x <= x_max and y_min <= peak_y <= y_max
                    for x_min, y_min, x_max, y_max in boxes
                )
                rows.append(
                    {
                        "variant": variant,
                        "image_id": image_id,
                        "class_name": concept,
                        "pg_hit": bool(hit),
                        "peak_x": peak_x,
                        "peak_y": peak_y,
                        "peak_map_x": peak_map_x,
                        "peak_map_y": peak_map_y,
                        "peak_saliency": peak_value,
                        "empty_heatmap": bool(float(np.max(heatmap)) <= 1e-12),
                        "image_width": width,
                        "image_height": height,
                        "bbox_count": len(boxes),
                        "bboxes": json.dumps(boxes),
                    }
                )
        if position % args.pghit_print_freq == 0 or position == len(eligible_ids):
            print(f"[PGHit] {position}/{len(eligible_ids)} bbox images")

    summary = {
        "definition": "max-point PGHit; hit iff the class-specific heatmap peak is inside any same-class bbox",
        "sample_unit": "one image_id-class_name pair",
        "bbox_csv": str(Path(bbox_csv)),
        "bbox_coordinate_size": args.bbox_coord_size,
        "bbox_images_in_csv": len(grouped),
        "bbox_images_in_evaluation": len(eligible_ids),
        "bbox_images_not_in_evaluation": len(set(grouped) - evaluated_ids),
        "bbox_loader": bbox_loader_stats,
        "skipped_bbox_classes": skipped_classes,
        "before_calibration": summarize_pghit(rows, "before_calibration"),
        "after_calibration": summarize_pghit(rows, "after_calibration"),
    }
    before = summary["before_calibration"]["pg_hit_rate"]
    after = summary["after_calibration"]["pg_hit_rate"]
    summary["delta_pg_hit_rate"] = (
        after - before if before is not None and after is not None else None
    )
    return summary, rows


def resolve_query_index(
    image_ids: Sequence[str], query_id: str | None, query_index: int
) -> int:
    if query_id:
        normalized = Path(query_id).stem
        try:
            return list(image_ids).index(normalized)
        except ValueError as exc:
            raise ValueError(f"--visualize-query-id was not found: {query_id}") from exc
    if not 0 <= query_index < len(image_ids):
        raise IndexError(
            f"--visualize-query-index must be in [0, {len(image_ids) - 1}], got {query_index}"
        )
    return query_index


def visualize_ranking(
    *,
    output_path: Path,
    ranking_name: str,
    query_index: int,
    candidate_indices: torch.Tensor,
    candidate_scores: torch.Tensor,
    score_name: str,
    image_ids: Sequence[str],
    image_path_by_id: dict[str, Path],
    concept_confidences: torch.Tensor,
    concepts: Sequence[str],
    positive_text: torch.Tensor,
    conceptclip_model,
    processor,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """Create one baseline/CRAR explanation plot for a single query."""
    row_indices = [query_index] + candidate_indices.tolist()
    images: list[Image.Image] = []
    for index in row_indices:
        image_id = image_ids[index]
        with Image.open(image_path_by_id[image_id]) as handle:
            images.append(handle.convert("RGB"))
    patch_tokens = extract_patch_tokens(
        conceptclip_model,
        processor,
        images,
        device,
        use_amp=args.amp,
    )

    rows = len(row_indices)
    figure, axes = plt.subplots(rows, 4, figsize=(18, 4.4 * rows), squeeze=False)
    query_active = concept_confidences[query_index] >= args.concept_threshold
    prompt_embeddings = positive_text.detach().cpu()

    for row, (index, image, tokens) in enumerate(zip(row_indices, images, patch_tokens)):
        confidence = concept_confidences[index]
        top_count = min(args.visualize_top_concepts, len(concepts))
        top_scores, top_indices = torch.topk(confidence, k=top_count)

        image_axis = axes[row, 0]
        image_axis.imshow(image, cmap="gray")
        if row == 0:
            title = f"Query: {image_ids[index][:16]}..."
        else:
            title = (
                f"#{row}: {image_ids[index][:16]}...\n"
                f"{score_name}: {float(candidate_scores[row - 1]):.4f}"
            )
        image_axis.set_title(title, fontsize=11, fontweight="bold")
        image_axis.axis("off")

        text_axis = axes[row, 1]
        text_axis.axis("off")
        heading = "Top detected concepts" if row == 0 else "Concepts"
        lines = [heading + ":"]
        for order, (concept_index, score) in enumerate(zip(top_indices, top_scores), start=1):
            concept_index_int = int(concept_index)
            shared_marker = "* " if row > 0 and bool(query_active[concept_index_int]) else ""
            lines.append(
                f"{shared_marker}{order}. {concepts[concept_index_int]} "
                f"{{{float(score):.3f}}}"
            )
        face_color = "#FDE5C5" if row == 0 else "#DCE9FA"
        edge_color = "#D99100" if row == 0 else "#6B93C9"
        text_axis.text(
            0.5,
            0.5,
            "\n".join(lines),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=1.1",
                "facecolor": face_color,
                "edgecolor": edge_color,
                "linewidth": 1.3,
            },
        )

        # Only the two highest-confidence concepts receive spatial heatmaps.
        selected_heatmap_indices = [int(index) for index in top_indices[:2]]
        selected_heatmaps = calibrated_concept_heatmaps(
            tokens,
            prompt_embeddings,
            selected_heatmap_indices,
            output_size=(image.height, image.width),
            args=args,
        )
        if len(selected_heatmaps) == 2:
            overlap_numerator = np.minimum(
                selected_heatmaps[0], selected_heatmaps[1]
            ).sum()
            overlap_denominator = np.maximum(
                selected_heatmaps[0], selected_heatmaps[1]
            ).sum()
            soft_iou = float(overlap_numerator / max(overlap_denominator, 1e-8))
            print(
                f"[heatmap] {image_ids[index]} | "
                f"{concepts[selected_heatmap_indices[0]]} vs "
                f"{concepts[selected_heatmap_indices[1]]} | soft-IoU={soft_iou:.3f}"
            )
        for heatmap_column in range(2):
            axis = axes[row, heatmap_column + 2]
            if heatmap_column >= len(top_indices):
                axis.axis("off")
                continue
            concept_index = int(top_indices[heatmap_column])
            heatmap = selected_heatmaps[heatmap_column]
            axis.imshow(image, cmap="gray")
            # Dynamic alpha leaves low-score and out-of-ROI pixels untouched,
            # rather than painting the entire radiograph blue.
            axis.imshow(
                heatmap,
                cmap="jet",
                alpha=args.heatmap_alpha * np.power(heatmap, 0.7),
                vmin=0.0,
                vmax=1.0,
            )
            axis.set_title(
                f"{concepts[concept_index]} ({float(confidence[concept_index]):.3f})",
                fontsize=10,
                fontweight="bold",
            )
            axis.axis("off")

    figure.suptitle(
        f"{ranking_name} | Query {image_ids[query_index]}",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=args.visualize_dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved visualization: {output_path}")


def create_query_visualizations(
    *,
    base_ranks: torch.Tensor,
    crar_ranks: torch.Tensor,
    visual_embeddings: torch.Tensor,
    concept_confidences: torch.Tensor,
    concept_weights: torch.Tensor,
    concepts: Sequence[str],
    positive_text: torch.Tensor,
    image_ids: Sequence[str],
    image_path_by_id: dict[str, Path],
    conceptclip_model,
    processor,
    device: torch.device,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, str]:
    query_index = resolve_query_index(
        image_ids, args.visualize_query_id, args.visualize_query_index
    )
    top_k = min(args.visualize_top_k, len(image_ids) - 1)
    baseline_candidates = base_ranks[query_index, :top_k]
    crar_candidates = crar_ranks[query_index, :top_k]

    baseline_svis, _, _ = crar_pair_scores(
        query_index,
        baseline_candidates,
        visual_embeddings,
        concept_confidences,
        concept_weights,
        args.gamma,
        args.concept_threshold,
    )
    _, _, crar_total = crar_pair_scores(
        query_index,
        crar_candidates,
        visual_embeddings,
        concept_confidences,
        concept_weights,
        args.gamma,
        args.concept_threshold,
    )

    safe_query_id = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in image_ids[query_index]
    )
    baseline_path = output_dir / f"query_{safe_query_id}_baseline.png"
    crar_path = output_dir / f"query_{safe_query_id}_crar.png"
    common = {
        "query_index": query_index,
        "image_ids": image_ids,
        "image_path_by_id": image_path_by_id,
        "concept_confidences": concept_confidences,
        "concepts": concepts,
        "positive_text": positive_text,
        "conceptclip_model": conceptclip_model,
        "processor": processor,
        "device": device,
        "args": args,
    }
    visualize_ranking(
        output_path=baseline_path,
        ranking_name="Baseline ConvNeXtV2-SRA",
        candidate_indices=baseline_candidates,
        candidate_scores=baseline_svis,
        score_name="Svis",
        **common,
    )
    visualize_ranking(
        output_path=crar_path,
        ranking_name="After CRAR reranking",
        candidate_indices=crar_candidates,
        candidate_scores=crar_total,
        score_name="Stotal",
        **common,
    )
    return {"baseline": str(baseline_path), "crar": str(crar_path)}


def retrieval_metrics(
    ranks: torch.Tensor,
    labels: torch.Tensor,
    thresholds: Sequence[float],
    k_values: Sequence[int],
) -> dict[str, dict[str, float]]:
    labels = labels.float()
    result: dict[str, dict[str, float]] = {}
    for threshold in thresholds:
        aps: list[float] = []
        precision = {k: [] for k in k_values}
        recall = {k: [] for k in k_values}
        hit = {k: [] for k in k_values}
        skipped = 0
        for query_index in range(len(labels)):
            intersection = (labels[query_index] * labels).sum(dim=1)
            union = ((labels[query_index] + labels) > 0).float().sum(dim=1)
            relevance = (intersection / union.clamp_min(1e-8)) > threshold
            relevance[query_index] = False
            relevant_count = int(relevance.sum().item())
            if relevant_count == 0:
                skipped += 1
                continue
            ranked_relevance = relevance[ranks[query_index]].float()
            cumulative = torch.cumsum(ranked_relevance, dim=0)
            hit_positions = torch.nonzero(ranked_relevance, as_tuple=False).flatten()
            ap = (cumulative[hit_positions] / (hit_positions.float() + 1.0)).sum() / relevant_count
            aps.append(float(ap.item()))
            for k in k_values:
                actual_k = min(k, ranked_relevance.numel())
                found = float(ranked_relevance[:actual_k].sum().item())
                precision[k].append(found / max(1, actual_k))
                recall[k].append(found / relevant_count)
                hit[k].append(float(found > 0))

        key = f"jaccard>{threshold:g}"
        metrics = {
            "mAP": 100.0 * float(np.mean(aps)) if aps else 0.0,
            "evaluated_queries": len(aps),
            "skipped_queries": skipped,
        }
        for k in k_values:
            metrics[f"P@{k}"] = 100.0 * float(np.mean(precision[k])) if precision[k] else 0.0
            metrics[f"Recall@{k}"] = 100.0 * float(np.mean(recall[k])) if recall[k] else 0.0
            metrics[f"Hit@{k}"] = 100.0 * float(np.mean(hit[k])) if hit[k] else 0.0
        result[key] = metrics
    return result


def print_comparison(baseline: dict, reranked: dict) -> None:
    print("\n================ CRAR EVALUATION REPORT ================")
    for threshold, base_metrics in baseline.items():
        crar_metrics = reranked[threshold]
        delta = crar_metrics["mAP"] - base_metrics["mAP"]
        print(f"\nRelevance: {threshold}")
        print(f"  Baseline mAP : {base_metrics['mAP']:.3f}%")
        print(f"  CRAR mAP     : {crar_metrics['mAP']:.3f}%")
        print(f"  Delta mAP    : {delta:+.3f} percentage points")
        for key in ("P@1", "P@5", "P@10", "Recall@10", "Hit@10"):
            if key in base_metrics:
                print(f"  {key:<10}    : {base_metrics[key]:.3f}% -> {crar_metrics[key]:.3f}%")


def safe_divide(numerator: np.ndarray | float, denominator: np.ndarray | float):
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    return np.divide(
        numerator_array,
        denominator_array,
        out=np.zeros_like(numerator_array, dtype=np.float64),
        where=denominator_array != 0,
    )


def multilabel_metrics_at_threshold(
    probabilities: np.ndarray,
    ground_truth: np.ndarray,
    concepts: Sequence[str],
    threshold: float,
) -> dict[str, Any]:
    """Compute image-level multi-label metrics for ConceptCLIP predictions."""
    targets = np.asarray(ground_truth >= 0.5, dtype=bool)
    predictions = np.asarray(probabilities >= threshold, dtype=bool)
    true_positive = np.logical_and(predictions, targets).sum(axis=0)
    false_positive = np.logical_and(predictions, ~targets).sum(axis=0)
    false_negative = np.logical_and(~predictions, targets).sum(axis=0)
    true_negative = np.logical_and(~predictions, ~targets).sum(axis=0)

    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    support = targets.sum(axis=0)
    micro_tp = float(true_positive.sum())
    micro_fp = float(false_positive.sum())
    micro_fn = float(false_negative.sum())
    micro_precision = float(safe_divide(micro_tp, micro_tp + micro_fp))
    micro_recall = float(safe_divide(micro_tp, micro_tp + micro_fn))
    micro_f1 = float(
        safe_divide(
            2.0 * micro_precision * micro_recall,
            micro_precision + micro_recall,
        )
    )
    total_support = float(support.sum())

    return {
        "threshold": float(threshold),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(
            safe_divide(float((f1 * support).sum()), total_support)
        ),
        "exact_match_accuracy": float(np.all(predictions == targets, axis=1).mean()),
        "hamming_accuracy": float((predictions == targets).mean()),
        "predicted_positive_rate": float(predictions.mean()),
        "ground_truth_positive_rate": float(targets.mean()),
        "per_concept": {
            concept: {
                "support": int(support[index]),
                "predicted_positives": int(predictions[:, index].sum()),
                "true_positive": int(true_positive[index]),
                "false_positive": int(false_positive[index]),
                "false_negative": int(false_negative[index]),
                "true_negative": int(true_negative[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, concept in enumerate(concepts)
        },
    }


def threshold_free_classification_metrics(
    probabilities: np.ndarray,
    ground_truth: np.ndarray,
    concepts: Sequence[str],
) -> dict[str, Any]:
    targets = np.asarray(ground_truth >= 0.5, dtype=np.int32)
    per_concept: dict[str, dict[str, float | int | None]] = {}
    average_precisions: list[float] = []
    roc_aucs: list[float] = []
    for index, concept in enumerate(concepts):
        target = targets[:, index]
        probability = probabilities[:, index]
        positive_count = int(target.sum())
        negative_count = int(len(target) - positive_count)
        average_precision = (
            float(average_precision_score(target, probability))
            if positive_count > 0
            else None
        )
        roc_auc = (
            float(roc_auc_score(target, probability))
            if positive_count > 0 and negative_count > 0
            else None
        )
        if average_precision is not None:
            average_precisions.append(average_precision)
        if roc_auc is not None:
            roc_aucs.append(roc_auc)
        per_concept[concept] = {
            "support": positive_count,
            "average_precision": average_precision,
            "roc_auc": roc_auc,
        }

    flattened_targets = targets.reshape(-1)
    flattened_probabilities = probabilities.reshape(-1)
    return {
        "macro_average_precision": float(np.mean(average_precisions))
        if average_precisions
        else None,
        "micro_average_precision": float(
            average_precision_score(flattened_targets, flattened_probabilities)
        ) if flattened_targets.sum() > 0 else None,
        "macro_roc_auc": float(np.mean(roc_aucs)) if roc_aucs else None,
        "per_concept": per_concept,
    }


def evaluate_concept_classification(
    confidences: torch.Tensor,
    ground_truth: torch.Tensor,
    concepts: Sequence[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate ConceptCLIP labels and find a diagnostic global F1 threshold."""
    probabilities = confidences.float().numpy()
    targets = ground_truth.float().numpy()
    fixed = multilabel_metrics_at_threshold(
        probabilities, targets, concepts, args.concept_threshold
    )
    threshold_free = threshold_free_classification_metrics(
        probabilities, targets, concepts
    )
    thresholds = np.arange(
        args.classification_threshold_min,
        args.classification_threshold_max + args.classification_threshold_step * 0.5,
        args.classification_threshold_step,
    )
    thresholds = np.clip(thresholds, 0.0, 1.0)
    curve: list[dict[str, float]] = []
    candidates: list[dict[str, Any]] = []
    for threshold in thresholds:
        metrics = multilabel_metrics_at_threshold(
            probabilities, targets, concepts, float(threshold)
        )
        candidates.append(metrics)
        curve.append(
            {
                "threshold": float(threshold),
                "micro_f1": metrics["micro_f1"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )
    objective = args.classification_sweep_objective
    best = max(
        candidates,
        key=lambda metrics: (
            metrics[objective],
            -abs(metrics["threshold"] - args.concept_threshold),
        ),
    )
    return {
        "concepts": list(concepts),
        "num_images": len(targets),
        "threshold_free": threshold_free,
        "at_crar_threshold": fixed,
        "threshold_sweep": {
            "objective": objective,
            "minimum": args.classification_threshold_min,
            "maximum": args.classification_threshold_max,
            "step": args.classification_threshold_step,
            "best_threshold": best["threshold"],
            "best_metrics": best,
            "curve": curve,
            "selection_warning": (
                "This threshold was selected on the current evaluation labels. "
                "Treat it as post-hoc diagnostic only; select on validation data "
                "before reporting an unbiased test result."
            ),
        },
    }


def print_concept_classification(metrics: dict[str, Any]) -> None:
    fixed = metrics["at_crar_threshold"]
    sweep = metrics["threshold_sweep"]
    best = sweep["best_metrics"]
    threshold_free = metrics["threshold_free"]
    macro_ap = threshold_free["macro_average_precision"]
    micro_ap = threshold_free["micro_average_precision"]
    macro_ap_text = f"{macro_ap:.4f}" if macro_ap is not None else "N/A"
    micro_ap_text = f"{micro_ap:.4f}" if micro_ap is not None else "N/A"
    print("\n============= CONCEPT CLASSIFICATION REPORT =============")
    print(
        f"At CRAR threshold {fixed['threshold']:.3f}: "
        f"micro-F1={fixed['micro_f1']:.4f}, macro-F1={fixed['macro_f1']:.4f}, "
        f"P={fixed['micro_precision']:.4f}, R={fixed['micro_recall']:.4f}"
    )
    print(
        f"Best global threshold by {sweep['objective']}: {best['threshold']:.3f} | "
        f"micro-F1={best['micro_f1']:.4f}, macro-F1={best['macro_f1']:.4f}"
    )
    print(f"Threshold-free: macro-mAP={macro_ap_text}, micro-AP={micro_ap_text}")
    print("Note: the swept test threshold is diagnostic and is not applied to CRAR.")


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def select_concepts(space: str) -> list[str]:
    if space == "lesions":
        return list(LESION_CONCEPTS)
    if space == "diseases":
        return list(DISEASE_CONCEPTS)
    return list(ALL_CONCEPTS)


def main(args: argparse.Namespace) -> None:
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError("--gamma must be in [0, 1]")
    if not 0.0 < args.prompt_temperature:
        raise ValueError("--prompt-temperature must be greater than zero")
    if not 0.0 <= args.concept_threshold <= 1.0:
        raise ValueError("--concept-threshold must be in [0, 1]")
    if args.rerank_top_k < 1:
        raise ValueError("--rerank-top-k must be at least 1")
    if args.visualize_top_k < 1:
        raise ValueError("--visualize-top-k must be at least 1")
    if args.visualize_top_concepts < 2:
        raise ValueError("--visualize-top-concepts must be at least 2")
    if not 0.0 <= args.heatmap_alpha <= 1.0:
        raise ValueError("--heatmap-alpha must be in [0, 1]")
    if args.heatmap_calibration_temperature <= 0.0:
        raise ValueError("--heatmap-calibration-temperature must be greater than zero")
    if not 0.0 <= args.heatmap_activation_quantile < 1.0:
        raise ValueError("--heatmap-activation-quantile must be in [0, 1)")
    if args.heatmap_common_mode_strength < 0.0:
        raise ValueError("--heatmap-common-mode-strength must be non-negative")
    if not 0.0 <= args.classification_threshold_min <= 1.0:
        raise ValueError("--classification-threshold-min must be in [0, 1]")
    if not 0.0 <= args.classification_threshold_max <= 1.0:
        raise ValueError("--classification-threshold-max must be in [0, 1]")
    if args.classification_threshold_min > args.classification_threshold_max:
        raise ValueError("Classification threshold minimum cannot exceed maximum")
    if args.classification_threshold_step <= 0.0:
        raise ValueError("--classification-threshold-step must be greater than zero")
    if args.bbox_coord_size is not None and args.bbox_coord_size <= 0.0:
        raise ValueError("--bbox-coord-size must be greater than zero")
    if args.pghit_max_images is not None and args.pghit_max_images < 1:
        raise ValueError("--pghit-max-images must be at least 1")
    if args.pghit_print_freq < 1:
        raise ValueError("--pghit-print-freq must be at least 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    concepts = select_concepts(args.concept_space)
    evaluation_labels = select_concepts(args.evaluation_label_space)
    print(f"Device: {device}")
    print(f"CRAR concepts ({len(concepts)}): {', '.join(concepts)}")
    print(f"Evaluation labels ({len(evaluation_labels)}): {', '.join(evaluation_labels)}")

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # Load the union once, then select evaluation columns after extraction. This
    # keeps concept prompts and evaluation labels independently configurable.
    dataset_columns = list(dict.fromkeys(concepts + evaluation_labels))
    dataset = VinDrCRARDataset(args.test_dir, args.labels_csv, transform, dataset_columns)
    image_path_by_id = dict(zip(dataset.image_ids, dataset.image_paths))
    if args.max_samples is not None and 1 < args.max_samples < len(dataset):
        dataset = Subset(dataset, range(args.max_samples))
    if len(dataset) < 2:
        raise ValueError("Evaluation requires at least two images")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_vindr,
    )

    print("Loading ConvNeXtV2-SRA checkpoint...")
    base_model = load_base_model(args, device)
    print(f"Loading ConceptCLIP: {args.conceptclip_model}")
    conceptclip_model, processor = load_conceptclip(args, device)
    positive_text, negative_text = build_prompt_embeddings(
        conceptclip_model, processor, concepts, device
    )

    print("Extracting visual embeddings and ConceptCLIP confidences...")
    base_embeddings, _, confidences, dataset_labels, image_ids = extract_features(
        base_model,
        conceptclip_model,
        processor,
        loader,
        positive_text,
        negative_text,
        device,
        args,
    )
    column_to_index = {name: index for index, name in enumerate(dataset_columns)}
    eval_indices = [column_to_index[name] for name in evaluation_labels]
    concept_label_indices = [column_to_index[name] for name in concepts]
    labels = dataset_labels[:, eval_indices]
    concept_ground_truth = dataset_labels[:, concept_label_indices]

    concept_classification: dict[str, Any] | None = None
    if args.evaluate_concept_classification:
        concept_classification = evaluate_concept_classification(
            confidences, concept_ground_truth, concepts, args
        )
        print_concept_classification(concept_classification)

    weights = load_concept_weights(args, concepts, confidences)
    base_ranks, crar_ranks, crar_stats = crar_rerank(
        base_embeddings,
        confidences,
        weights,
        gamma=args.gamma,
        threshold=args.concept_threshold,
        rerank_top_k=args.rerank_top_k,
    )
    thresholds = parse_float_list(args.jaccard_thresholds)
    k_values = parse_int_list(args.k_values)
    baseline = retrieval_metrics(base_ranks, labels, thresholds, k_values)
    reranked = retrieval_metrics(crar_ranks, labels, thresholds, k_values)
    print_comparison(baseline, reranked)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    visualizations: dict[str, str] = {}
    visualization_error: str | None = None
    pending_visualization_error: Exception | None = None
    # The base model is no longer needed. Free it before requesting patch tokens
    # from the much larger ConceptCLIP model for visualization.
    del base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pghit: dict[str, Any] | None = None
    pghit_rows: list[dict[str, Any]] = []
    if args.bbox_csv:
        print("\nEvaluating class-aware VinDr PGHit before/after heatmap calibration...")
        pghit, pghit_rows = evaluate_vindr_pghit(
            bbox_csv=args.bbox_csv,
            image_ids=image_ids,
            image_path_by_id=image_path_by_id,
            concepts=concepts,
            positive_text=positive_text,
            conceptclip_model=conceptclip_model,
            processor=processor,
            device=device,
            args=args,
        )
        before_pg = pghit["before_calibration"]["pg_hit_rate"]
        after_pg = pghit["after_calibration"]["pg_hit_rate"]
        print(f"PGHit before calibration: {before_pg:.4f}")
        print(f"PGHit after calibration:  {after_pg:.4f}")
        print(f"PGHit delta:              {pghit['delta_pg_hit_rate']:+.4f}")

    if not args.no_visualize:
        try:
            visualizations = create_query_visualizations(
                base_ranks=base_ranks,
                crar_ranks=crar_ranks,
                visual_embeddings=base_embeddings,
                concept_confidences=confidences,
                concept_weights=weights,
                concepts=concepts,
                positive_text=positive_text,
                image_ids=image_ids,
                image_path_by_id=image_path_by_id,
                conceptclip_model=conceptclip_model,
                processor=processor,
                device=device,
                output_dir=output_dir,
                args=args,
            )
        except Exception as exc:  # Preserve the expensive evaluation outputs.
            visualization_error = f"{type(exc).__name__}: {exc}"
            pending_visualization_error = exc
            print(f"[visualization warning] {visualization_error}")
            print("Metrics and rankings will still be saved.")

    delta = {
        key: {"mAP": reranked[key]["mAP"] - baseline[key]["mAP"]}
        for key in baseline
    }
    report = {
        "algorithm": "CRAR Hybrid Concept-Visual Ranking",
        "dataset_size": len(labels),
        "base_model": "convnextv2_sra",
        "base_checkpoint": str(Path(args.base_checkpoint)),
        "conceptclip_model": args.conceptclip_model,
        "conceptclip_revision": args.conceptclip_revision,
        "conceptclip_checkpoint": args.conceptclip_checkpoint,
        "concept_space": args.concept_space,
        "evaluation_label_space": args.evaluation_label_space,
        "concepts": concepts,
        "concept_weights": {name: float(weight) for name, weight in zip(concepts, weights)},
        "gamma": args.gamma,
        "concept_threshold": args.concept_threshold,
        "prompt_temperature": args.prompt_temperature,
        "crar": crar_stats,
        "baseline": baseline,
        "after_crar": reranked,
        "delta": delta,
        "concept_classification": concept_classification,
        "pghit": pghit,
        "visualizations": visualizations,
        "visualization_error": visualization_error,
        "heatmap_configuration": {
            "calibration": args.heatmap_calibration,
            "common_mode_strength": args.heatmap_common_mode_strength,
            "calibration_temperature": args.heatmap_calibration_temperature,
            "activation_quantile": args.heatmap_activation_quantile,
            "anatomy_mask": not args.no_anatomy_mask,
        },
    }
    report_path = output_dir / "vindr_crar_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    if pghit is not None:
        pghit_summary_path = output_dir / "vindr_pghit_summary.json"
        with pghit_summary_path.open("w", encoding="utf-8") as handle:
            json.dump(pghit, handle, indent=2, ensure_ascii=False)
        pghit_csv_path = output_dir / "vindr_pghit_per_sample.csv"
        pd.DataFrame(pghit_rows).to_csv(pghit_csv_path, index=False)
        print(f"Saved PGHit summary: {pghit_summary_path}")
        print(f"Saved PGHit samples: {pghit_csv_path}")
    if concept_classification is not None:
        classification_path = output_dir / "vindr_concept_classification.json"
        with classification_path.open("w", encoding="utf-8") as handle:
            json.dump(concept_classification, handle, indent=2, ensure_ascii=False)
        per_concept_rows = []
        fixed_per_concept = concept_classification["at_crar_threshold"]["per_concept"]
        best_per_concept = concept_classification["threshold_sweep"]["best_metrics"]["per_concept"]
        threshold_free_per_concept = concept_classification["threshold_free"]["per_concept"]
        for concept in concepts:
            per_concept_rows.append(
                {
                    "concept": concept,
                    **fixed_per_concept[concept],
                    "average_precision": threshold_free_per_concept[concept]["average_precision"],
                    "roc_auc": threshold_free_per_concept[concept]["roc_auc"],
                    "best_global_threshold_f1": best_per_concept[concept]["f1"],
                }
            )
        classification_csv_path = output_dir / "vindr_concept_classification_per_class.csv"
        pd.DataFrame(per_concept_rows).to_csv(classification_csv_path, index=False)
        print(f"Saved concept classification: {classification_path}")
        print(f"Saved per-concept metrics: {classification_csv_path}")
    np.savez_compressed(
        output_dir / "vindr_crar_rankings.npz",
        image_ids=np.asarray(image_ids),
        labels=labels.numpy(),
        concept_labels=concept_ground_truth.numpy(),
        concept_confidences=confidences.numpy(),
        concept_weights=weights.numpy(),
        baseline_ranks=base_ranks.numpy(),
        crar_ranks=crar_ranks.numpy(),
        concepts=np.asarray(concepts),
        evaluation_labels=np.asarray(evaluation_labels),
    )
    print(f"\nSaved report: {report_path}")
    print(f"Saved rankings: {output_dir / 'vindr_crar_rankings.npz'}")
    if pending_visualization_error is not None and args.strict_visualize:
        raise RuntimeError("Visualization failed; evaluation outputs were saved") from pending_visualization_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate ConvNeXtV2-SRA before and after ConceptCLIP CRAR reranking on VinDr-CXR"
    )
    parser.add_argument("--test-dir", required=True, help="Directory containing VinDr <image_id>.png test images")
    parser.add_argument("--labels-csv", default="vindr/image_labels_test.csv", help="VinDr multi-label test CSV")
    parser.add_argument(
        "--bbox-csv",
        default=None,
        help=(
            "Optional VinDr bbox CSV with image_id,class_name,x_min,y_min,x_max,y_max; "
            "enables class-aware PGHit"
        ),
    )
    parser.add_argument(
        "--bbox-coord-size",
        default=None,
        type=float,
        help=(
            "Fixed square coordinate size used by the bbox CSV (for example 384). "
            "Omit when boxes already use each original image's pixel coordinates"
        ),
    )
    parser.add_argument(
        "--pghit-max-images",
        default=None,
        type=int,
        help="Optional development limit on bbox-annotated images used for PGHit",
    )
    parser.add_argument("--pghit-print-freq", default=25, type=int)
    parser.add_argument("--base-checkpoint", required=True, help="ConvNeXtV2-SRA checkpoint path")
    parser.add_argument("--model", default="convnextv2_sra", choices=["convnextv2_sra"], help="Base retrieval model")
    parser.add_argument("--sra-num-heads", default=8, type=int)
    parser.add_argument("--sra-lam", default=0.1, type=float)
    parser.add_argument("--conceptclip-model", default="JerrryNie/ConceptCLIP")
    parser.add_argument(
        "--conceptclip-revision",
        default="8120d7f1e07b590a7dce35bd2a01126b0e42b6c3",
        help="Pinned Hugging Face revision for reproducible remote code and weights",
    )
    parser.add_argument("--conceptclip-checkpoint", default=None, help="Optional fine-tuned ConceptCLIP checkpoint")
    parser.add_argument("--concept-space", choices=["all", "lesions", "diseases"], default="all")
    parser.add_argument("--evaluation-label-space", choices=["all", "lesions", "diseases"], default="all")
    parser.add_argument("--gamma", default=0.30, type=float, help="CRAR semantic balancing factor")
    parser.add_argument("--concept-threshold", default=0.50, type=float, help="Confidence threshold defining Cq and Cn")
    parser.add_argument(
        "--evaluate-concept-classification",
        action="store_true",
        help=(
            "Evaluate ConceptCLIP multi-label predictions against labels CSV and "
            "sweep a diagnostic global F1 threshold"
        ),
    )
    parser.add_argument("--classification-threshold-min", default=0.05, type=float)
    parser.add_argument("--classification-threshold-max", default=0.95, type=float)
    parser.add_argument("--classification-threshold-step", default=0.01, type=float)
    parser.add_argument(
        "--classification-sweep-objective",
        choices=["micro_f1", "macro_f1", "weighted_f1"],
        default="macro_f1",
        help="F1 statistic maximized by the diagnostic global threshold sweep",
    )
    parser.add_argument("--prompt-temperature", default=0.07, type=float, help="Positive/negative prompt softmax temperature")
    parser.add_argument("--concept-weighting", choices=["uniform", "idf"], default="uniform")
    parser.add_argument("--concept-weights-json", default=None, help="Optional JSON object mapping concept names to omega_j")
    parser.add_argument("--rerank-top-k", default=100, type=int, help="Rerank only the initial visual top-K")
    parser.add_argument("--jaccard-thresholds", default="0.25,0.4,0.5")
    parser.add_argument("--k-values", default="1,5,10")
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--max-samples", default=None, type=int)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision during feature extraction")
    parser.add_argument("--device", default=None, help="Example: cuda:0 or cpu")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--print-freq", default=10, type=int)
    parser.add_argument("--output-dir", default="./results/vindr_crar")
    parser.add_argument(
        "--visualize-query-id",
        default=None,
        help="VinDr image_id to visualize; overrides --visualize-query-index",
    )
    parser.add_argument(
        "--visualize-query-index",
        default=0,
        type=int,
        help="Zero-based query index used when --visualize-query-id is omitted",
    )
    parser.add_argument(
        "--visualize-top-k",
        default=2,
        type=int,
        help="Number of retrieved candidates shown in each baseline/CRAR plot",
    )
    parser.add_argument(
        "--visualize-top-concepts",
        default=5,
        type=int,
        help="Concept confidences listed per image; heatmaps are always limited to top 2",
    )
    parser.add_argument("--heatmap-alpha", default=0.50, type=float)
    parser.add_argument(
        "--heatmap-calibration",
        choices=["competitive", "independent"],
        default="competitive",
        help="Joint within-image concept calibration or raw independent cosine maps",
    )
    parser.add_argument("--heatmap-common-mode-strength", default=1.0, type=float)
    parser.add_argument("--heatmap-calibration-temperature", default=0.75, type=float)
    parser.add_argument(
        "--heatmap-activation-quantile",
        default=0.75,
        type=float,
        help="Suppress patch scores below this within-map quantile",
    )
    parser.add_argument(
        "--no-anatomy-mask",
        action="store_true",
        help="Disable the soft neck/shoulder, thorax and upper-abdomen ROI",
    )
    parser.add_argument("--visualize-dpi", default=150, type=int)
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument(
        "--strict-visualize",
        action="store_true",
        help="Return a non-zero exit status if plotting fails (metrics are still saved)",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
