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

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
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
    labels = dataset_labels[:, eval_indices]

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
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "vindr_crar_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    np.savez_compressed(
        output_dir / "vindr_crar_rankings.npz",
        image_ids=np.asarray(image_ids),
        labels=labels.numpy(),
        concept_confidences=confidences.numpy(),
        concept_weights=weights.numpy(),
        baseline_ranks=base_ranks.numpy(),
        crar_ranks=crar_ranks.numpy(),
        concepts=np.asarray(concepts),
        evaluation_labels=np.asarray(evaluation_labels),
    )
    print(f"\nSaved report: {report_path}")
    print(f"Saved rankings: {output_dir / 'vindr_crar_rankings.npz'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate ConvNeXtV2-SRA before and after ConceptCLIP CRAR reranking on VinDr-CXR"
    )
    parser.add_argument("--test-dir", required=True, help="Directory containing VinDr <image_id>.png test images")
    parser.add_argument("--labels-csv", default="vindr/image_labels_test.csv", help="VinDr multi-label test CSV")
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
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
