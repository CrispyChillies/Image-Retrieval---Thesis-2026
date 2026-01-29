import os
import random
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoModel, AutoProcessor
from PIL import Image

from read_data import ChestXrayDataSet, ISICDataSet, TBX11kDataSet
from loss import TripletMarginLoss
from sampler import PKSampler


class ConceptCLIPEmbeddingModel(nn.Module):
    """Wrapper around ConceptCLIP to extract embeddings for retrieval.
    
    Supports two modes:
    - 'image_only': Uses only image features
    - 'image_text': Fuses image and text features for enhanced embeddings
    """
    def __init__(self, concept_clip_model, processor, mode='image_only', fusion_method='concat'):
        super(ConceptCLIPEmbeddingModel, self).__init__()
        self.concept_clip = concept_clip_model
        self.processor = processor
        self.mode = mode
        self.fusion_method = fusion_method
        
        # Get embedding dimension from model config
        self.embedding_dim = concept_clip_model.config.projection_dim
        
        # Fusion layer for image+text mode
        if mode == 'image_text':
            if fusion_method == 'concat':
                self.fusion_layer = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
            elif fusion_method == 'weighted':
                self.alpha = nn.Parameter(torch.tensor(0.5))
            # 'add' mode doesn't need extra parameters
        
    def forward(self, images, texts=None):
        """
        Args:
            images: Preprocessed image tensors
            texts: Optional text descriptions (required for image_text mode)
        
        Returns:
            Normalized embeddings for retrieval
        """
        if self.mode == 'image_only':
            # Extract only image features
            with torch.no_grad():
                image_features = self.concept_clip.get_image_features(pixel_values=images)
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features
            
        elif self.mode == 'image_text':
            if texts is None:
                raise ValueError("Text descriptions required for image_text mode")
            
            # Extract both image and text features
            with torch.no_grad():
                image_features = self.concept_clip.get_image_features(pixel_values=images)
                text_features = self.concept_clip.get_text_features(**texts)
            
            # Normalize individual features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Fuse features
            if self.fusion_method == 'concat':
                fused = torch.cat([image_features, text_features], dim=1)
                fused = self.fusion_layer(fused)
            elif self.fusion_method == 'add':
                fused = image_features + text_features
            elif self.fusion_method == 'weighted':
                fused = self.alpha * image_features + (1 - self.alpha) * text_features
            
            # Normalize fused features
            fused = fused / fused.norm(dim=-1, keepdim=True)
            return fused


def get_text_descriptions(labels, dataset_name):
    """Generate text descriptions for medical images based on labels.
    
    Args:
        labels: Tensor of class labels
        dataset_name: Name of the dataset ('covid', 'isic', or 'tbx11k')
    
    Returns:
        List of text descriptions
    """
    if dataset_name == 'covid':
        label_to_text = {
            0: "a chest X-ray of normal healthy lungs",
            1: "a chest X-ray showing bacterial or viral pneumonia",
            2: "a chest X-ray showing COVID-19 pneumonia with ground-glass opacity"
        }
    elif dataset_name == 'isic':
        label_to_text = {
            0: "a dermoscopic image of a benign melanocytic nevus skin lesion",
            1: "a dermoscopic image of seborrheic keratosis skin lesion",
            2: "a dermoscopic image of malignant melanoma skin cancer"
        }
    elif dataset_name == 'tbx11k':
        label_to_text = {
            0: "a chest X-ray showing active tuberculosis infection",
            1: "a chest X-ray of healthy normal lungs",
            2: "a chest X-ray showing abnormalities but not tuberculosis"
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return [label_to_text[label.item()] for label in labels]


def train_epoch(model, processor, optimizer, criterion, data_loader, device, epoch, print_freq, dataset_name, use_text):
    """Train for one epoch using triplet loss."""
    model.train()
    running_loss = 0
    running_frac_pos_triplets = 0

    pbar = tqdm(data_loader, desc=f'Epoch {epoch}')
    for i, data in enumerate(pbar):
        optimizer.zero_grad()
        
        images, labels = data[0].to(device), data[1].to(device)
        
        # Prepare inputs
        if use_text:
            # Generate text descriptions for this batch
            text_descriptions = get_text_descriptions(labels, dataset_name)
            text_inputs = processor(
                text=text_descriptions,
                return_tensors='pt',
                padding=True,
                truncation=True
            )
            # Move text inputs to device
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            embeddings = model(images, text_inputs)
        else:
            embeddings = model(images)
        
        # Compute triplet loss
        loss, frac_pos_triplets = criterion(embeddings, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_frac_pos_triplets += float(frac_pos_triplets)
        
        # Update progress bar
        if i % print_freq == 0:
            avg_loss = running_loss / (i + 1)
            avg_trip = 100.0 * running_frac_pos_triplets / (i + 1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'hard_triplets': f'{avg_trip:.2f}%'
            })


@torch.no_grad()
def evaluate(model, processor, loader, device, dataset_name, use_text):
    """Evaluate retrieval performance using R@1 accuracy."""
    model.eval()
    embeds, labels = [], []

    print("Extracting embeddings for evaluation...")
    for data in tqdm(loader, desc='Evaluating'):
        images = data[0].to(device)
        _labels = data[1].to(device)
        
        if use_text:
            text_descriptions = get_text_descriptions(_labels, dataset_name)
            text_inputs = processor(
                text=text_descriptions,
                return_tensors='pt',
                padding=True,
                truncation=True
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            out = model(images, text_inputs)
        else:
            out = model(images)
            
        embeds.append(out)
        labels.append(_labels)

    embeds = torch.cat(embeds, dim=0)
    labels = torch.cat(labels, dim=0)

    # Compute pairwise distances (negative for similarity)
    dists = -torch.cdist(embeds, embeds)
    dists.fill_diagonal_(float('-inf'))
    
    # Get top-1 predictions
    _, pred_indices = dists.max(dim=1)
    pred_labels = labels[pred_indices]
    
    # Compute R@1 accuracy
    correct = (pred_labels == labels).sum().item()
    accuracy = 100.0 * correct / len(labels)
    
    print(f'>> R@1 accuracy: {accuracy:.3f}%')
    return accuracy


def save_checkpoint(model, epoch, save_dir, args, is_best=False):
    """Save model checkpoint."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    mode_suffix = 'image_only' if args.mode == 'image_only' else f'image_text_{args.fusion_method}'
    file_name = f'{args.dataset}_conceptclip_{mode_suffix}'
    if args.anomaly:
        file_name += '_anomaly'
    file_name += f'_seed_{args.seed}'
    
    if is_best:
        file_name += '_best_ckpt.pth'
    else:
        file_name += f'_epoch_{epoch}_ckpt.pth'
    
    save_path = os.path.join(save_dir, file_name)
    
    # Save full model state (including fusion layer if present)
    torch.save({
        'model_state_dict': model.state_dict(),
        'concept_clip_state_dict': model.concept_clip.state_dict(),
        'mode': args.mode,
        'fusion_method': args.fusion_method if args.mode == 'image_text' else None,
        'epoch': epoch
    }, save_path)
    
    print(f'>> Checkpoint saved: {save_path}')
    return save_path


def main(args):
    # Set random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Mode: {args.mode}")
    if args.mode == 'image_text':
        print(f"Fusion method: {args.fusion_method}")
    
    # Batch size configuration for triplet sampling
    p = args.labels_per_batch if not args.anomaly else args.labels_per_batch - 1
    k = args.samples_per_label
    batch_size = p * k
    print(f"Batch configuration: {p} classes × {k} samples = {batch_size} total")
    
    # Load ConceptCLIP model and processor
    print("Loading ConceptCLIP model...")
    concept_clip_model = AutoModel.from_pretrained(
        'JerrryNie/ConceptCLIP',
        trust_remote_code=True
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(
        'JerrryNie/ConceptCLIP',
        trust_remote_code=True
    )
    
    print("ConceptCLIP model loaded successfully")
    
    # Create embedding model wrapper
    use_text = (args.mode == 'image_text')
    model = ConceptCLIPEmbeddingModel(
        concept_clip_model,
        processor,
        mode=args.mode,
        fusion_method=args.fusion_method if use_text else None
    )
    model.to(device)
    
    # Resume from checkpoint if provided
    if os.path.isfile(args.resume):
        print(f"=> Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'concept_clip_state_dict' in checkpoint:
            model.concept_clip.load_state_dict(checkpoint['concept_clip_state_dict'], strict=False)
        else:
            model.concept_clip.load_state_dict(checkpoint, strict=False)
        print("=> Checkpoint loaded successfully")
    
    # Setup loss and optimizer
    criterion = TripletMarginLoss(margin=args.margin, mining=args.mining)
    
    # Different learning rates for pretrained backbone vs fusion layer
    if use_text and hasattr(model, 'fusion_layer'):
        optimizer = torch.optim.AdamW([
            {'params': model.concept_clip.parameters(), 'lr': args.lr},
            {'params': model.fusion_layer.parameters(), 'lr': args.lr * 10}  # Higher LR for fusion
        ], weight_decay=0.01)
    elif use_text and hasattr(model, 'alpha'):
        optimizer = torch.optim.AdamW([
            {'params': model.concept_clip.parameters(), 'lr': args.lr},
            {'params': [model.alpha], 'lr': args.lr * 10}
        ], weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.01
        )
    
    # Define preprocessing transform using ConceptCLIP processor
    def transform_fn(image):
        """Transform function that wraps ConceptCLIP processor."""
        # Processor expects PIL images
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Process image - returns dict with pixel_values
        inputs = processor(images=image, return_tensors='pt')
        # Return just the pixel_values tensor, squeeze batch dimension
        return inputs['pixel_values'].squeeze(0)
    
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == 'covid':
        train_dataset = ChestXrayDataSet(
            data_dir=os.path.join(args.dataset_dir, 'train'),
            image_list_file=args.train_image_list,
            use_covid=not args.anomaly,
            mask_dir=os.path.join(args.mask_dir, 'train') if args.mask_dir else None,
            transform=transform_fn
        )
        test_dataset = ChestXrayDataSet(
            data_dir=os.path.join(args.dataset_dir, 'test'),
            image_list_file=args.test_image_list,
            mask_dir=os.path.join(args.mask_dir, 'test') if args.mask_dir else None,
            transform=transform_fn
        )
    elif args.dataset == 'isic':
        train_dataset = ISICDataSet(
            data_dir=os.path.join(args.dataset_dir, 'ISIC-2017_Training_Data'),
            image_list_file=args.train_image_list,
            use_melanoma=not args.anomaly,
            mask_dir=os.path.join(args.mask_dir, 'train') if args.mask_dir else None,
            transform=transform_fn
        )
        test_dataset = ISICDataSet(
            data_dir=os.path.join(args.dataset_dir, 'ISIC-2017_Test_v2_Data'),
            image_list_file=args.test_image_list,
            mask_dir=os.path.join(args.mask_dir, 'test') if args.mask_dir else None,
            transform=transform_fn
        )
    elif args.dataset == 'tbx11k':
        train_dataset = TBX11kDataSet(
            data_dir=os.path.join(args.dataset_dir, 'train'),
            csv_file=args.train_image_list,
            transform=transform_fn
        )
        test_dataset = TBX11kDataSet(
            data_dir=os.path.join(args.dataset_dir, 'test'),
            csv_file=args.test_image_list,
            transform=transform_fn
        )
    else:
        raise NotImplementedError(f'Dataset {args.dataset} not supported!')
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Setup data loaders with PKSampler for triplet mining
    targets = train_dataset.labels
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=PKSampler(targets, p, k),
        num_workers=args.workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    # Track best model
    best_accuracy = 0.0
    best_epoch = 0
    processor, optimizer, criterion,
    
    # Training loop
    print(f'\n{"="*60}')
    print(f'Starting training for {args.epochs} epochs')
    print(f'{"="*60}\n')
    
    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')
        print('-' * 60)
        
        # Train for one epoch
        train_epoch(
            model, processor, optimizer, criterion,
            train_loader, device, epoch, args.print_freq,
            args.dataset, use_text
        )
        
        # Evaluate every N epochs
        if epoch % args.eval_freq == 0:
            print(f'\nEvaluating at epoch {epoch}...')
            accuracy = evaluate(model, processor, test_loader, device, args.dataset, use_text)
            
            # Save if best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
                print(f'🎯 New best model! Accuracy: {accuracy:.3f}% (epoch {epoch})')
                save_checkpoint(model, epoch, args.save_dir, args, is_best=True)
            else:
                print(f'Current: {accuracy:.3f}%, Best: {best_accuracy:.3f}% (epoch {best_epoch})')
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                save_checkpoint(model, epoch, args.save_dir, args, is_best=False)
    
    # Final summary
    print(f'\n{"="*60}')
    print('Training completed!')
    print(f'{"="*60}')
    print(f'Best model: Epoch {best_epoch} with R@1 accuracy: {best_accuracy:.3f}%')
    print(f'Best model saved in: {args.save_dir}')
    print(f'{"="*60}\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Finetune ConceptCLIP for Medical Image Retrieval')
    
    # Dataset arguments
    parser.add_argument('--dataset', default='covid', choices=['covid', 'isic', 'tbx11k'],
                        help='Dataset to use')
    parser.add_argument('--dataset-dir', default='/data/brian.hu/COVID/data/',
                        help='Dataset directory path')
    parser.add_argument('--train-image-list', default='./train_split.txt',
                        help='Train image list')
    parser.add_argument('--test-image-list', default='./test_COVIDx4.txt',
                        help='Test image list')
    parser.add_argument('--mask-dir', default=None,
                        help='Segmentation masks path (if used)')
    parser.add_argument('--anomaly', action='store_true',
                        help='Train without anomaly class (COVID for covid dataset, melanoma for isic)')
    
    # Model arguments
    parser.add_argument('--mode', default='image_only', choices=['image_only', 'image_text'],
                        help='Embedding mode: image_only uses only image features, '
                             'image_text fuses image and text features')
    parser.add_argument('--fusion-method', default='concat', choices=['concat', 'add', 'weighted'],
                        help='Method to fuse image and text features (only for image_text mode). '
                             'concat: concatenate and project, add: element-wise addition, '
                             'weighted: learnable weighted sum')
    parser.add_argument('--resume', default='', type=str,
                        help='Path to checkpoint to resume from')
    
    # Training arguments
    parser.add_argument('-p', '--labels-per-batch', default=3, type=int,
                        help='Number of unique labels/classes per batch')
    parser.add_argument('-k', '--samples-per-label', default=16, type=int,
                        help='Number of samples per label in a batch')
    parser.add_argument('--eval-batch-size', default=64, type=int,
                        help='Batch size for evaluation')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Number of training epochs')
    parser.add_argument('--eval-freq', default=2, type=int,
                        help='Evaluate model every N epochs')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='Number of data loading workers')
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='Learning rate')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Triplet loss margin')
    parser.add_argument('--mining', default='batch_all', choices=['batch_all', 'batch_hard'],
                        help='Triplet mining strategy')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='Print frequency (batches)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--save-dir', default='./checkpoints',
                        help='Directory to save model checkpoints')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
