from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ingest_embeddings import get_model_and_transform


def load_embeddings(npz_path: str) -> dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    required_keys = {"embeddings", "paths"}
    missing = required_keys.difference(data.files)
    if missing:
        raise ValueError(f"Missing keys in {npz_path}: {sorted(missing)}")

    return {
        "embeddings": data["embeddings"],
        "paths": data["paths"],
        "labels": data["labels"] if "labels" in data.files else None,
    }


def compute_query_embedding(model, transform, query_image: str, device: torch.device) -> torch.Tensor:
    image = Image.open(query_image).convert("RGB")
    query_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        query_embedding = model(query_tensor)
        query_embedding = F.normalize(query_embedding, p=2, dim=1)

    return query_embedding.squeeze(0).cpu()


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, transform = get_model_and_transform(
        args.model_type,
        args.model_weights,
        args.embedding_dim,
        device,
        args.dinov2_model_name,
    )

    store = load_embeddings(args.embeddings_npz)
    gallery_embeddings = torch.from_numpy(store["embeddings"]).float()
    gallery_embeddings = F.normalize(gallery_embeddings, p=2, dim=1)

    query_embedding = compute_query_embedding(model, transform, args.query_image, device)
    similarities = torch.matmul(gallery_embeddings, query_embedding)

    query_name = Path(args.query_image).name
    query_stem = Path(args.query_image).stem

    gallery_paths = [str(p) for p in store["paths"]]
    rankings = []
    for index, similarity in enumerate(similarities.tolist()):
        candidate_path = gallery_paths[index]
        candidate_name = Path(candidate_path).name
        if args.exclude_query and (candidate_name == query_name or Path(candidate_path).stem == query_stem):
            continue
        rankings.append((index, candidate_path, similarity))

    rankings.sort(key=lambda item: item[2], reverse=True)
    top_rankings = rankings[: args.top_k]

    results = []
    for rank, (_, candidate_path, similarity) in enumerate(top_rankings, start=1):
        results.append(
            {
                "rank": rank,
                "retrieved_image_path": candidate_path,
                "retrieved_image_name": Path(candidate_path).name,
                "similarity": float(similarity),
            }
        )

    output = {
        "query_image_path": args.query_image,
        "query_image_name": query_name,
        "embeddings_npz": args.embeddings_npz,
        "results": results,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    if args.copy_retrieved_dir:
        from shutil import copy2

        retrieved_dir = Path(args.copy_retrieved_dir) / query_stem
        retrieved_dir.mkdir(parents=True, exist_ok=True)
        for item in results:
            destination = retrieved_dir / f"rank_{item['rank']:02d}_{item['retrieved_image_name']}"
            copy2(item["retrieved_image_path"], destination)

    print(f"query_image={args.query_image}")
    print(f"embeddings_npz={args.embeddings_npz}")
    print(f"saved_results={output_path}")
    for item in results:
        print(f"{item['rank']}. {item['retrieved_image_name']} (similarity: {item['similarity']:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query saved local embeddings for top-k retrieval")
    parser.add_argument("--query_image", required=True, help="Path to the query image")
    parser.add_argument("--embeddings_npz", required=True, help="Local embeddings .npz saved by ingest_embeddings.py")
    parser.add_argument("--model_type", required=True, choices=["densenet121", "resnet50", "convnextv2", "convnextv2_sra", "dinov2", "medsiglip"])
    parser.add_argument("--model_weights", required=True, help="Checkpoint path used to build the query embedding")
    parser.add_argument("--embedding_dim", type=int, default=None, help="Embedding dimension used by the model")
    parser.add_argument("--dinov2_model_name", default="facebook/dinov2-base", help="DINOv2 backbone name when model_type=dinov2")
    parser.add_argument("--top_k", type=int, default=5, help="Number of retrieved images to return")
    parser.add_argument("--output_json", default="local_query_results.json", help="Where to save the local retrieval results")
    parser.add_argument("--copy_retrieved_dir", default=None, help="Optional directory to copy the top-k retrieved images into")
    parser.add_argument("--exclude_query", action="store_true", help="Skip the query image if it is present in the gallery")
    parser.add_argument("--device", default="cuda", help="Device to use for query embedding")
    main(parser.parse_args())