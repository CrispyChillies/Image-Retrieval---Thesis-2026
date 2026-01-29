"""
Extract and save embeddings from ConvNeXtV2 or ConceptCLIP for backbone comparison.

Usage:
    # Extract ConvNeXtV2 embeddings
    python extract_embeddings.py \
        --model convnextv2 \
        --resume model.pth \
        --dataset covid \
        --test-dataset-dir /path/to/test \
        --test-image-list test_COVIDx4.txt \
        --output convnext_embeddings.npy \
        --embedding-dim 1024

    # Extract ConceptCLIP embeddings
    python extract_embeddings.py \
        --model conceptclip \
        --dataset covid \
        --test-dataset-dir /path/to/test \
        --test-image-list test_COVIDx4.txt \
        --output conceptclip_embeddings.npy
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from read_data import ISICDataSet, ChestXrayDataSet, TBX11kDataSet
from model import ConvNeXtV2, ResNet50, DenseNet121, HybridConvNeXtViT


@torch.no_grad()
def extract_convnext_embeddings(model, loader, device):
    """Extract embeddings from ConvNeXtV2 or similar models."""
    model.eval()
    
    all_embeddings = []
    all_labels = []
    all_image_ids = []
    
    print("Extracting embeddings...")
    for data in tqdm(loader, desc="Processing batches"):
        images = data[0].to(device)
        labels = data[1]
        
        # Get image IDs if available (3rd element in data tuple)
        if len(data) > 2:
            image_ids = data[2]
        else:
            # Use batch indices as fallback
            image_ids = list(range(len(all_labels), len(all_labels) + len(labels)))
        
        # Extract embeddings
        embeddings = model(images)
        
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels)
        all_image_ids.extend(image_ids)
    
    # Concatenate all batches
    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    image_ids = np.array(all_image_ids)
    
    print(f"Extracted {len(embeddings)} embeddings with shape {embeddings.shape}")
    
    return embeddings, labels, image_ids


@torch.no_grad()
def extract_medclip_embeddings(loader, device, batch_size=8):
    """Extract embeddings from MedCLIP model."""
    from transformers import AutoModel, AutoTokenizer, CLIPVisionModel
    
    print("Loading MedCLIP model...")
    vision_model = CLIPVisionModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
    vision_model = vision_model.to(device)
    vision_model.eval()
    
    from torchvision import transforms as T
    preprocess = T.Compose([
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                   std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    
    all_embeddings = []
    all_labels = []
    all_image_ids = []
    
    print("Extracting MedCLIP embeddings...")
    for data in tqdm(loader, desc="Processing batches"):
        images = data[0].to(device)
        labels = data[1]
        
        if len(data) > 2:
            image_ids = data[2]
        else:
            image_ids = list(range(len(all_labels), len(all_labels) + len(labels)))
        
        # Denormalize and renormalize for MedCLIP
        images_denorm = images * std + mean
        images_clip = preprocess(images_denorm)
        
        batch_embeddings = []
        for i in range(0, len(images_clip), batch_size):
            mini_batch = images_clip[i:i + batch_size]
            outputs = vision_model(pixel_values=mini_batch)
            img_features = F.normalize(outputs.pooler_output, dim=-1)
            batch_embeddings.append(img_features.cpu())
            torch.cuda.empty_cache()
        
        batch_embeddings = torch.cat(batch_embeddings, dim=0)
        all_embeddings.append(batch_embeddings)
        all_labels.append(labels)
        all_image_ids.extend(image_ids)
    
    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    image_ids = np.array(all_image_ids)
    
    print(f"Extracted {len(embeddings)} MedCLIP embeddings with shape {embeddings.shape}")
    return embeddings, labels, image_ids


@torch.no_grad()
def extract_medsiglip_embeddings(loader, device, batch_size=8):
    """Extract embeddings from MedSigLIP model."""
    from transformers import AutoModel, AutoProcessor
    
    print("Loading MedSigLIP model...")
    model = AutoModel.from_pretrained("flaviagiammarino/medsiglip-vit-base-patch16-224").to(device)
    processor = AutoProcessor.from_pretrained("flaviagiammarino/medsiglip-vit-base-patch16-224")
    model.eval()
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    
    all_embeddings = []
    all_labels = []
    all_image_ids = []
    
    print("Extracting MedSigLIP embeddings...")
    for data in tqdm(loader, desc="Processing batches"):
        images = data[0].to(device)
        labels = data[1]
        
        if len(data) > 2:
            image_ids = data[2]
        else:
            image_ids = list(range(len(all_labels), len(all_labels) + len(labels)))
        
        images_denorm = images * std + mean
        images_denorm = torch.clamp(images_denorm, 0, 1)
        
        batch_embeddings = []
        for i in range(0, len(images_denorm), batch_size):
            mini_batch = images_denorm[i:i + batch_size]
            inputs = processor(images=[img for img in mini_batch], return_tensors='pt', padding=True).to(device)
            outputs = model.get_image_features(**inputs)
            img_features = F.normalize(outputs, dim=-1)
            batch_embeddings.append(img_features.cpu())
            del outputs, inputs
            torch.cuda.empty_cache()
        
        batch_embeddings = torch.cat(batch_embeddings, dim=0)
        all_embeddings.append(batch_embeddings)
        all_labels.append(labels)
        all_image_ids.extend(image_ids)
    
    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    image_ids = np.array(all_image_ids)
    
    print(f"Extracted {len(embeddings)} MedSigLIP embeddings with shape {embeddings.shape}")
    return embeddings, labels, image_ids


@torch.no_grad()
def extract_conceptclip_embeddings(loader, device, batch_size=8):
    """Extract embeddings from ConceptCLIP model."""
    from transformers import AutoModel, AutoProcessor
    
    print("Loading ConceptCLIP model...")
    model = AutoModel.from_pretrained(
        'JerrryNie/ConceptCLIP',
        trust_remote_code=True
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(
        'JerrryNie/ConceptCLIP',
        trust_remote_code=True
    )
    
    model.eval()
    
    # Define denormalization (ConceptCLIP expects unnormalized images)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    
    def denormalize(tensor):
        return tensor * std + mean
    
    all_embeddings = []
    all_labels = []
    all_image_ids = []
    
    print("Extracting ConceptCLIP embeddings...")
    for data in tqdm(loader, desc="Processing batches"):
        images = data[0].to(device)
        labels = data[1]
        
        # Get image IDs if available
        if len(data) > 2:
            image_ids = data[2]
        else:
            image_ids = list(range(len(all_labels), len(all_labels) + len(labels)))
        
        # Denormalize images for ConceptCLIP
        images_denorm = denormalize(images)
        
        # Process in smaller batches to avoid OOM
        batch_embeddings = []
        for i in range(0, len(images_denorm), batch_size):
            mini_batch = images_denorm[i:i + batch_size]
            
            # Convert to PIL-like format expected by processor
            # ConceptCLIP processor expects images in [0, 1] range
            mini_batch_list = [img for img in mini_batch]
            
            # Process with ConceptCLIP processor
            inputs = processor(
                images=mini_batch_list,
                return_tensors='pt',
                padding=True
            ).to(device)
            
            # Extract global image features
            outputs = model(**inputs)
            img_features = F.normalize(outputs["image_features"], dim=-1)
            
            batch_embeddings.append(img_features.cpu())
            
            # Free memory
            del outputs, inputs
            torch.cuda.empty_cache()
        
        batch_embeddings = torch.cat(batch_embeddings, dim=0)
        
        all_embeddings.append(batch_embeddings)
        all_labels.append(labels)
        all_image_ids.extend(image_ids)
    
    # Concatenate all batches
    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    image_ids = np.array(all_image_ids)
    
    print(f"Extracted {len(embeddings)} ConceptCLIP embeddings with shape {embeddings.shape}")
    
    return embeddings, labels, image_ids


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup transforms
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    
    # Use 384x384 for ConvNeXtV2 and Hybrid, 224x224 for others
    if args.model in ['convnextv2', 'hybrid_convnext_vit', 'conceptclip']:
        img_size = 384
    else:
        img_size = 224
    
    test_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Setup dataset
    if args.dataset == 'covid':
        test_dataset = ChestXrayDataSet(
            data_dir=args.test_dataset_dir,
            image_list_file=args.test_image_list,
            mask_dir=args.mask_dir,
            transform=test_transform
        )
    elif args.dataset == 'isic':
        test_dataset = ISICDataSet(
            data_dir=args.test_dataset_dir,
            image_list_file=args.test_image_list,
            mask_dir=args.mask_dir,
            transform=test_transform
        )
    elif args.dataset == 'tbx11k':
        test_dataset = TBX11kDataSet(
            data_dir=args.test_dataset_dir,
            csv_file=args.test_image_list,
            transform=test_transform
        )
    else:
        raise NotImplementedError(f'Dataset {args.dataset} not supported!')
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    print(f"Dataset: {args.dataset}")
    print(f"Number of samples: {len(test_dataset)}")
    
    # Extract embeddings based on model type
    if args.model == 'conceptclip':
        embeddings, labels, image_ids = extract_conceptclip_embeddings(
            test_loader, device, batch_size=args.conceptclip_batch_size
        )
    elif args.model == 'medclip':
        embeddings, labels, image_ids = extract_medclip_embeddings(
            test_loader, device, batch_size=args.conceptclip_batch_size
        )
    elif args.model == 'medsiglip':
        embeddings, labels, image_ids = extract_medsiglip_embeddings(
            test_loader, device, batch_size=args.conceptclip_batch_size
        )
    else:
        # Load model
        if args.model == 'densenet121':
            model = DenseNet121(embedding_dim=args.embedding_dim)
        elif args.model == 'resnet50':
            model = ResNet50(embedding_dim=args.embedding_dim)
        elif args.model == 'convnextv2':
            model = ConvNeXtV2(embedding_dim=args.embedding_dim)
        elif args.model == 'hybrid_convnext_vit':
            model = HybridConvNeXtViT(embedding_dim=args.embedding_dim)
        else:
            raise NotImplementedError(f'Model {args.model} not supported!')
        
        # Load checkpoint
        if args.resume and os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu')
            if 'state-dict' in checkpoint:
                checkpoint = checkpoint['state-dict']
            model.load_state_dict(checkpoint, strict=False)
            print("Checkpoint loaded successfully")
        else:
            print("WARNING: No checkpoint provided or file not found. Using randomly initialized model.")
        
        model.to(device)
        
        embeddings, labels, image_ids = extract_convnext_embeddings(
            model, test_loader, device
        )
    
    # Save embeddings
    output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else '.'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving embeddings to {args.output}")
    np.save(args.output, embeddings)
    
    # Save labels
    if args.labels_output:
        labels_path = args.labels_output
    else:
        labels_path = args.output.replace('.npy', '_labels.npy')
    print(f"Saving labels to {labels_path}")
    np.save(labels_path, labels)
    
    # Save image IDs
    if args.image_ids_output:
        ids_path = args.image_ids_output
    else:
        ids_path = args.output.replace('.npy', '_image_ids.npy')
    print(f"Saving image IDs to {ids_path}")
    np.save(ids_path, image_ids)
    
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE!")
    print("="*70)
    print(f"Embeddings: {embeddings.shape} -> {args.output}")
    print(f"Labels: {labels.shape} -> {labels_path}")
    print(f"Image IDs: {image_ids.shape} -> {ids_path}")
    print(f"Unique labels: {np.unique(labels)}")
    print("="*70)
    
    # Print example command for test_backbone.py
    print("\nTo run backbone comparison, use:")
    print(f"python test_backbone.py \\")
    print(f"    --convnext-embeddings <convnext_output>.npy \\")
    print(f"    --conceptclip-embeddings <conceptclip_output>.npy \\")
    print(f"    --labels {labels_path} \\")
    print(f"    --image-ids {ids_path} \\")
    print(f"    --image-dir {args.test_dataset_dir} \\")
    print(f"    --save-dir results/backbone_comparison")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract embeddings from ConvNeXtV2 or ConceptCLIP',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model arguments
    parser.add_argument('--model', required=True,
                       choices=['densenet121', 'resnet50', 'convnextv2', 
                               'hybrid_convnext_vit', 'conceptclip', 'medclip', 'medsiglip'],
                       help='Model to extract embeddings from')
    parser.add_argument('--resume', default='',
                       help='Path to model checkpoint (not needed for ConceptCLIP/MedCLIP/MedSigLIP)')
    parser.add_argument('--embedding-dim', default=None, type=int,
                       help='Embedding dimension (not needed for ConceptCLIP/MedCLIP/MedSigLIP)')
    
    # Dataset arguments
    parser.add_argument('--dataset', default='covid',
                       choices=['covid', 'isic', 'tbx11k'],
                       help='Dataset to use')
    parser.add_argument('--test-dataset-dir', required=True,
                       help='Test dataset directory path')
    parser.add_argument('--test-image-list', required=True,
                       help='Test image list file')
    parser.add_argument('--mask-dir', default=None,
                       help='Segmentation masks directory (optional)')
    
    # Output arguments
    parser.add_argument('--output', required=True,
                       help='Output path for embeddings (.npy)')
    parser.add_argument('--labels-output', default=None,
                       help='Output path for labels (default: <output>_labels.npy)')
    parser.add_argument('--image-ids-output', default=None,
                       help='Output path for image IDs (default: <output>_image_ids.npy)')
    
    # Processing arguments
    parser.add_argument('--batch-size', default=32, type=int,
                       help='Batch size for data loading')
    parser.add_argument('--conceptclip-batch-size', default=8, type=int,
                       help='Internal batch size for ConceptCLIP processing (to avoid OOM)')
    parser.add_argument('-j', '--workers', default=4, type=int,
                       help='Number of data loading workers')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
