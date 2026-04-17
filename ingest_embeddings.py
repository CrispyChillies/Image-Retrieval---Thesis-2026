"""
Ingest image embeddings into Milvus collections
Supports batch processing for efficient insertion
"""

import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm
from timm.data import resolve_model_data_config
from model import DenseNet121, ResNet50, ConvNeXtV2, DinoV2, MedSigLIP
from milvus.milvus_setup import MilvusManager, MODEL_CONFIGS
from transformers import AutoProcessor
from tqdm import tqdm
import argparse

try:
    import boto3
except ImportError:
    boto3 = None

from dotenv import load_dotenv

load_dotenv()


def get_model_and_transform(
    model_type, model_weights, embedding_dim, device, dinov2_model_name
):
    """Load model and get appropriate transform"""

    # Load model
    if model_type == "densenet121":
        model = DenseNet121(embedding_dim=embedding_dim)
        img_size = 224
    elif model_type == "resnet50":
        model = ResNet50(embedding_dim=embedding_dim)
        img_size = 224
    elif model_type == "convnextv2":
        model = ConvNeXtV2(embedding_dim=embedding_dim)
        img_size = 384
    elif model_type == "dinov2":
        model = DinoV2(
            model_name=dinov2_model_name,
            embedding_dim=embedding_dim,
        )
    elif model_type == "medsiglip":
        embed_dim = embedding_dim if embedding_dim is not None else 512
        model = MedSigLIP(embed_dim=embed_dim)

        # Load weights
        checkpoint = torch.load(model_weights, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        model.to(device)

        # MedSigLIP uses AutoProcessor (448x448, normalisation baked in)
        processor = AutoProcessor.from_pretrained("google/medsiglip-448")

        def medsiglip_transform(img):
            inputs = processor(images=img.convert("RGB"), return_tensors="pt")
            return inputs["pixel_values"].squeeze(0)

        return model, medsiglip_transform
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights
    checkpoint = torch.load(model_weights, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    model.to(device)

    if model_type == "dinov2":
        temp_model = timm.create_model(
            dinov2_model_name,
            pretrained=False,
            num_classes=0,
        )
        data_config = resolve_model_data_config(temp_model)
        img_size = data_config["input_size"][-1]
        normalize = transforms.Normalize(data_config["mean"], data_config["std"])
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        # Setup transform
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if model_type == "convnextv2":
            # Match test.py exactly for ConvNeXtV2 so Milvus embeddings are
            # directly comparable to non-Milvus evaluation.
            transform = transforms.Compose(
                [
                    transforms.Lambda(lambda img: img.convert("RGB")),
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Lambda(lambda img: img.convert("RGB")),
                    transforms.Resize(256),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

    return model, transform


def load_image_list(image_list_file, data_dir):
    """Load image paths and labels from file

    Supports two formats:
    1. CSV format (ISIC dataset): image_id, melanoma, seborrheic_keratosis, ...
    2. Text format (ChestXray): idx filename label
    """
    images = []

    # Detect if file is CSV
    is_csv = image_list_file.endswith(".csv")

    if is_csv:
        # Parse ISIC CSV format
        import csv

        print(f"Parsing CSV file: {image_list_file}")

        with open(image_list_file, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)  # skip header
            print(f"CSV Header: {header}")

            line_count = 0
            for line in reader:
                line_count += 1
                if len(line) < 3:
                    print(
                        f"Warning: Line {line_count} has fewer than 3 columns: {line}"
                    )
                    continue

                image_name = line[0]

                try:
                    # Determine label based on columns
                    if float(line[1]) == 1:
                        label = "melanoma"
                    elif float(line[2]) == 1:
                        label = "seborrheic_keratosis"
                    else:
                        label = "nevus"

                    image_path = os.path.join(data_dir, image_name)
                    images.append((image_path, label))

                    # Show first few entries for debugging
                    if line_count <= 3:
                        print(f"  Sample {line_count}: {image_name} -> {label}")
                        print(f"    Full path: {image_path}")
                        print(f"    Exists: {os.path.exists(image_path)}")

                except (ValueError, IndexError) as e:
                    print(f"Error parsing line {line_count}: {line} - {e}")
                    continue

            print(f"Parsed {line_count} lines from CSV")
    else:
        # Parse text format (ChestXray)
        with open(image_list_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_path = os.path.join(data_dir, parts[1])
                    label = parts[2] if len(parts) > 2 else "unknown"
                    images.append((image_path, label))

    return images


def compute_embeddings_batch(
    model, image_paths, labels, transform, device, batch_size=32
):
    """Compute embeddings for a batch of images"""
    all_embeddings = []
    all_paths = []
    all_labels = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]
        batch_tensors = []
        valid_paths = []
        valid_labels = []

        for img_path, label in zip(batch_paths, batch_labels):
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img)
                batch_tensors.append(img_tensor)
                valid_paths.append(img_path)
                valid_labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

        if batch_tensors:
            batch = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                embeddings = model(batch)
                embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())
            all_paths.extend(valid_paths)
            all_labels.extend(valid_labels)

    import numpy as np

    return np.concatenate(all_embeddings, axis=0), all_paths, all_labels


def build_s3_client(args):
    """Build an S3 client from CLI args or environment variables."""
    if boto3 is None:
        raise ImportError(
            "boto3 is required for S3 uploads. Install it with `pip install boto3`."
        )

    aws_access_key_id = args.aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = args.aws_secret_access_key or os.getenv(
        "AWS_SECRET_ACCESS_KEY"
    )
    aws_region = args.aws_region or os.getenv("AWS_REGION")

    if not aws_access_key_id or not aws_secret_access_key or not aws_region:
        raise ValueError(
            "Missing AWS S3 credentials. Provide --aws_access_key_id, "
            "--aws_secret_access_key, --aws_region or set them in the environment."
        )

    return boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region,
    )


def upload_images_to_s3(image_paths, args):
    """Upload images to S3 and return the S3 URI stored in Milvus."""
    bucket_name = args.s3_bucket_name or os.getenv("S3_BUCKET_NAME")
    prefix = args.s3_prefix or os.getenv("S3_ORIGINAL_IMAGES_PREFIX", "original/")

    if not bucket_name:
        raise ValueError(
            "Missing S3 bucket name. Provide --s3_bucket_name or set S3_BUCKET_NAME."
        )

    prefix = prefix.strip("/")
    s3_client = build_s3_client(args)
    s3_paths = []

    print(f"\nUploading {len(image_paths)} images to s3://{bucket_name}/{prefix}/")
    for img_path in tqdm(image_paths, desc="Uploading to S3"):
        image_name = os.path.basename(img_path)
        s3_key = f"{prefix}/{image_name}" if prefix else image_name
        s3_client.upload_file(img_path, bucket_name, s3_key)
        s3_paths.append(f"s3://{bucket_name}/{s3_key}")

    return s3_paths


def resolve_stored_image_paths(image_paths, args):
    """Resolve which image paths should be stored in Milvus."""
    if args.store_local_paths:
        print("\nSkipping S3 upload. Storing local image paths in Milvus.")
        return image_paths

    print("\n" + "=" * 70)
    print("UPLOADING IMAGES TO S3")
    print("=" * 70)
    stored_image_paths = upload_images_to_s3(image_paths, args)
    print(f"Uploaded {len(stored_image_paths)} images to S3")
    if stored_image_paths:
        print(f"   Sample stored path: {stored_image_paths[0]}")
    return stored_image_paths


def ingest_embeddings(
    manager, model_type, embeddings, stored_image_paths, labels, batch_size=100
):
    """Insert embeddings into Milvus collection"""

    if model_type not in manager.collections:
        manager.load_collection(model_type)

    collection = manager.collections[model_type]

    total = len(embeddings)
    print(f"\nInserting {total} embeddings into {model_type} collection...")

    for i in tqdm(range(0, total, batch_size), desc="Ingesting"):
        batch_end = min(i + batch_size, total)

        batch_data = [
            stored_image_paths[i:batch_end],  # image_path
            labels[i:batch_end],  # label
            embeddings[i:batch_end].tolist(),  # embedding
        ]

        collection.insert(batch_data)

    # Flush to persist data
    collection.flush()
    print(f"✅ Inserted {total} embeddings")

    return total


def main():
    parser = argparse.ArgumentParser(description="Ingest image embeddings into Milvus")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["densenet121", "resnet50", "convnextv2", "dinov2", "medsiglip"],
        help="Model type to use",
    )
    parser.add_argument(
        "--model_weights", type=str, required=True, help="Path to model weights"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory containing images"
    )
    parser.add_argument(
        "--image_list",
        type=str,
        required=True,
        help="Text file with image list (format: idx filename label)",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=None,
        help="Custom embedding dimension if using projection",
    )
    parser.add_argument(
        "--dinov2-model-name",
        type=str,
        default="vit_base_patch14_dinov2.lvd142m",
        help="timm model name for DINOv2 ingestion",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding computation",
    )
    parser.add_argument(
        "--insert_batch_size",
        type=int,
        default=100,
        help="Batch size for Milvus insertion",
    )
    parser.add_argument("--uri", type=str, default=None, help="Zilliz Cloud URI")
    parser.add_argument("--token", type=str, default=None, help="Zilliz Cloud token")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--aws_access_key_id",
        type=str,
        default=None,
        help="AWS access key for S3 upload",
    )
    parser.add_argument(
        "--aws_secret_access_key",
        type=str,
        default=None,
        help="AWS secret key for S3 upload",
    )
    parser.add_argument(
        "--aws_region", type=str, default=None, help="AWS region for S3 upload"
    )
    parser.add_argument(
        "--s3_bucket_name",
        type=str,
        default=None,
        help="S3 bucket where source images will be uploaded",
    )
    parser.add_argument(
        "--s3_prefix",
        type=str,
        default=None,
        help="S3 key prefix to use for uploaded images",
    )
    parser.add_argument(
        "--store-local-paths",
        action="store_true",
        help="Store local image paths in Milvus instead of uploading images to S3",
    )

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Connect to Milvus
    print("\n" + "=" * 70)
    print("CONNECTING TO MILVUS")
    print("=" * 70)
    manager = MilvusManager(uri=args.uri, token=args.token)
    if not manager.connect():
        return

    try:
        # Ensure collection exists and is loaded
        print("\n" + "=" * 70)
        print("PREPARING COLLECTION")
        print("=" * 70)
        manager.create_collection(args.model_type, drop_old=False)
        manager.create_index(args.model_type)
        manager.load_collection(args.model_type)

        # Load model
        print("\n" + "=" * 70)
        print("LOADING MODEL")
        print("=" * 70)
        print(f"Model: {args.model_type}")
        print(f"Weights: {args.model_weights}")
        model, transform = get_model_and_transform(
            args.model_type,
            args.model_weights,
            args.embedding_dim,
            device,
            args.dinov2_model_name,
        )
        print("✅ Model loaded")

        # Load image list
        print("\n" + "=" * 70)
        print("LOADING IMAGE LIST")
        print("=" * 70)
        image_data = load_image_list(args.image_list, args.data_dir)
        image_paths = [path for path, _ in image_data]
        labels = [label for _, label in image_data]
        print(f"Found {len(image_paths)} images")

        # Compute embeddings
        print("\n" + "=" * 70)
        print("COMPUTING EMBEDDINGS")
        print("=" * 70)
        embeddings, valid_paths, valid_labels = compute_embeddings_batch(
            model, image_paths, labels, transform, device, args.batch_size
        )
        print(f"✅ Computed {len(embeddings)} embeddings")
        print(f"   Embedding shape: {embeddings.shape}")

        stored_image_paths = resolve_stored_image_paths(valid_paths, args)

        # Ingest into Milvus
        print("\n" + "=" * 70)
        print("INGESTING INTO MILVUS")
        print("=" * 70)
        total_inserted = ingest_embeddings(
            manager,
            args.model_type,
            embeddings,
            stored_image_paths,
            valid_labels,
            args.insert_batch_size,
        )

        # Show collection info
        info = manager.get_collection_info(args.model_type)
        print("\n" + "=" * 70)
        print("COLLECTION INFO")
        print("=" * 70)
        print(f"Collection: {info['name']}")
        print(f"Total entities: {info['num_entities']}")
        print(f"Embeddings ingested: {total_inserted}")

        print("\n✅ INGESTION COMPLETE!")

    finally:
        manager.disconnect()


if __name__ == "__main__":
    main()
