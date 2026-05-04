from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from nih_multilabel_training import set_seed
from nih_zilliz_utils import (
    build_collection_name,
    build_model_and_transform,
    connect_zilliz,
    disconnect_zilliz,
    encode_npy_paths,
    resolve_npy_paths,
    search_collection,
)
from pymilvus import Collection


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    connect_zilliz(args.uri, args.token)
    try:
        collection_name = args.collection_name or build_collection_name(args.model, "gallery")
        collection = Collection(collection_name)
        collection.load()

        model, transform = build_model_and_transform(
            model_name=args.model,
            checkpoint_path=args.resume,
            backbone_name=args.backbone_name,
            device=device,
            num_labels=args.num_labels,
        )

        query_paths = resolve_npy_paths(
            data_dir=args.query_dir,
            image_list_file=args.query_image_list,
        )
        query_rows = encode_npy_paths(
            model=model,
            transform=transform,
            image_paths=query_paths,
            device=device,
            batch_size=args.batch_size,
            progress_desc="Encoding queries",
        )
        effective_top_k = args.top_k if args.top_k and args.top_k > 0 else collection.num_entities

        all_results = []
        for row in tqdm(query_rows, desc="Searching queries", unit="query"):
            hits = search_collection(
                collection=collection,
                query_vector=row["embedding"].tolist(),
                top_k=effective_top_k,
                nprobe=args.nprobe,
            )
            all_results.append(
                {
                    "query_image_path": row["image_path"],
                    "query_image_name": row["image_name"],
                    "query_label_names": row["label_names"],
                    "query_label_vector": row["multi_hot"],
                    "results": hits,
                }
            )

        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(all_results, handle, indent=2)

        print(f"collection={collection_name}")
        print(f"queried_images={len(all_results)}")
        print(f"saved_results={output_path}")
        print(f"top_k={effective_top_k}")
    finally:
        disconnect_zilliz()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query NIH gallery embeddings from Zilliz")
    parser.add_argument("--model", required=True, choices=["dinov2", "convnextv2"])
    parser.add_argument("--resume", required=True, help="Checkpoint path")
    parser.add_argument("--query-dir", required=True, help="Query directory")
    parser.add_argument("--query-image-list", default=None, help="Optional query manifest")
    parser.add_argument("--collection-name", default=None, help="Override collection name")
    parser.add_argument("--backbone-name", default=None, help="Optional timm backbone override")
    parser.add_argument("--uri", required=True, help="Zilliz cluster URI")
    parser.add_argument("--token", required=True, help="Zilliz token")
    parser.add_argument("--output-json", default="nih_query_results.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=0, help="0 means retrieve the full gallery ranking")
    parser.add_argument("--nprobe", type=int, default=10)
    parser.add_argument("--num-labels", type=int, default=14)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
