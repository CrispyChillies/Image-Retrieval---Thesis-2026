from __future__ import annotations

import argparse
import os

import torch

from nih_multilabel_retrieval import (
    build_nih_dataloader,
    build_nih_val_transform,
    get_backbone_image_config,
)
from nih_multilabel_training import BACKBONE_SPECS, evaluate_map, set_seed


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif "state-dict" in checkpoint:
        checkpoint = checkpoint["state-dict"]

    model.load_state_dict(checkpoint, strict=False)


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spec = BACKBONE_SPECS[args.model]
    image_config = get_backbone_image_config(args.model)
    transform = build_nih_val_transform(
        image_size=image_config["image_size"],
        resize_size=image_config["resize_size"],
    )

    model = spec.model_builder(
        args.num_labels,
        args.backbone_name or spec.default_backbone_name,
        False,
    ).to(device)
    load_checkpoint(model, args.resume, device)

    data_loader = build_nih_dataloader(
        data_dir=args.test_dataset_dir,
        image_list_file=args.test_image_list,
        batch_size=args.eval_batch_size,
        transform=transform,
        num_workers=args.workers,
        label_aware_sampling=False,
        shuffle=False,
        drop_last=False,
    )

    mean_ap = evaluate_map(
        model=model,
        data_loader=data_loader,
        device=device,
        jaccard_threshold=args.jaccard_threshold,
    )
    print(f"NIH {args.model} mAP: {mean_ap:.3f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NIH multi-label retrieval models")
    parser.add_argument(
        "--model",
        default="dinov2",
        choices=list(BACKBONE_SPECS),
        help="Backbone family to evaluate",
    )
    parser.add_argument("--backbone-name", default=None, help="Optional timm backbone override")
    parser.add_argument("--resume", required=True, help="Checkpoint path")
    parser.add_argument("--test-dataset-dir", required=True, help="NIH test/gallery directory")
    parser.add_argument("--test-image-list", default=None, help="Optional manifest file")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-labels", type=int, default=14)
    parser.add_argument("--jaccard-threshold", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
