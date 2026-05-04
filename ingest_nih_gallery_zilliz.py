from __future__ import annotations

import argparse

import torch

from nih_multilabel_training import set_seed
from nih_zilliz_utils import (
    build_collection_name,
    build_model_and_transform,
    connect_zilliz,
    create_nih_collection,
    disconnect_zilliz,
    encode_npy_paths,
    insert_rows,
    resolve_npy_paths,
)


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    connect_zilliz(args.uri, args.token)
    try:
        collection_name = args.collection_name or build_collection_name(args.model, "gallery")
        collection = create_nih_collection(
            collection_name=collection_name,
            drop_old=args.drop_old,
            metric_type=args.metric_type,
            index_type=args.index_type,
            nlist=args.nlist,
        )

        model, transform = build_model_and_transform(
            model_name=args.model,
            checkpoint_path=args.resume,
            backbone_name=args.backbone_name,
            device=device,
            num_labels=args.num_labels,
        )

        gallery_paths = resolve_npy_paths(
            data_dir=args.gallery_dir,
            image_list_file=args.gallery_image_list,
        )
        rows = encode_npy_paths(
            model=model,
            transform=transform,
            image_paths=gallery_paths,
            device=device,
            batch_size=args.batch_size,
            progress_desc="Encoding gallery",
        )
        insert_rows(collection, rows)
        print(f"collection={collection_name}")
        print(f"ingested_gallery_images={len(rows)}")
    finally:
        disconnect_zilliz()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest NIH gallery embeddings into Zilliz")
    parser.add_argument("--model", required=True, choices=["dinov2", "convnextv2"])
    parser.add_argument("--resume", required=True, help="Checkpoint path")
    parser.add_argument("--gallery-dir", required=True, help="Gallery directory")
    parser.add_argument("--gallery-image-list", default=None, help="Optional gallery manifest")
    parser.add_argument("--collection-name", default=None, help="Override collection name")
    parser.add_argument("--backbone-name", default=None, help="Optional timm backbone override")
    parser.add_argument("--uri", required=True, help="Zilliz cluster URI")
    parser.add_argument("--token", required=True, help="Zilliz token")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-labels", type=int, default=14)
    parser.add_argument("--metric-type", default="COSINE")
    parser.add_argument("--index-type", default="IVF_FLAT")
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--drop-old", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
