"""
Fine-tune ConceptCLIP on COVID CXR Dataset using Image-Text Contrastive Learning.

This script adapts ConceptCLIP to COVID-19 chest X-ray domain by training it
with class-specific text prompts and corresponding medical images.

Usage:
    python finetune_conceptclip.py \
        --dataset-dir /path/to/COVID/data \
        --train-image-list train_split.txt \
        --test-image-list test_COVIDx4.txt \
        --epochs 20 \
        --batch-size 32 \
        --lr 1e-5 \
        --save-dir ./checkpoints/conceptclip_finetuned
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from read_data import ChestXrayDataSet


class ConceptCLIPFineTuner(nn.Module):
    """Wrapper for ConceptCLIP with contrastive loss."""
    
    def __init__(self, model_name='JerrryNie/ConceptCLIP'):
        super(ConceptCLIPFineTuner, self).__init__()
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
    def forward(self, pixel_values, input_ids, attention_mask):
        """Forward pass returning image and text features."""
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs


def create_text_prompts(label_names, prompt_templates=None):
    """
    Create diverse text prompts for each medical condition.
    
    Args:
        label_names: List of class names
        prompt_templates: Optional custom templates
    
    Returns:
        Dictionary mapping label to list of text prompts
    """
    if prompt_templates is None:
        # Default medical imaging templates
        prompt_templates = [
            "a chest X-ray showing {}",
            "a chest radiograph of {}",
            "medical image of {} in lungs",
            "chest X-ray image with {}",
            "radiological finding of {}",
            "pulmonary imaging showing {}",
            "thoracic X-ray with {}",
            "a CXR demonstrating {}",
        ]
    
    # Class-specific descriptions
    class_descriptions = {
        "normal": [
            "normal lungs", "healthy chest", "no abnormalities",
            "clear lung fields", "normal pulmonary tissue"
        ],
        "pneumonia": [
            "pneumonia", "lung infection", "pulmonary consolidation",
            "bacterial pneumonia", "viral pneumonia", "lung infiltrates"
        ],
        "COVID-19": [
            "COVID-19", "coronavirus infection", "SARS-CoV-2",
            "COVID pneumonia", "viral COVID infection", "coronavirus disease"
        ]
    }
    
    prompts = {}
    for label_name in label_names:
        label_prompts = []
        descriptions = class_descriptions.get(label_name, [label_name])
        
        for desc in descriptions:
            for template in prompt_templates:
                label_prompts.append(template.format(desc))
        
        prompts[label_name] = label_prompts
    
    return prompts


def clip_contrastive_loss(image_features, text_features, logit_scale, temperature=0.07):
    """
    Compute CLIP-style contrastive loss.
    
    Args:
        image_features: [batch_size, feature_dim] normalized image features
        text_features: [batch_size, feature_dim] normalized text features
        logit_scale: learnable temperature parameter
        temperature: fixed temperature (if logit_scale is None)
    
    Returns:
        loss: contrastive loss
    """
    # Normalize features
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    
    # Compute similarity matrix
    if logit_scale is not None:
        logits = logit_scale * image_features @ text_features.t()
    else:
        logits = (image_features @ text_features.t()) / temperature
    
    # Create labels (diagonal = positive pairs)
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)
    
    # Symmetric loss (image-to-text and text-to-image)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    
    loss = (loss_i2t + loss_t2i) / 2
    
    return loss


def train_epoch(model, processor, train_loader, optimizer, device, epoch, 
                label_names, class_prompts, print_freq=10):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        # Move images to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Create text prompts for this batch
        batch_texts = []
        for label in labels:
            label_name = label_names[label.item()]
            # Randomly sample one prompt per image
            prompt = random.choice(class_prompts[label_name])
            batch_texts.append(prompt)
        
        # Process inputs
        inputs = processor(
            images=[img for img in images],  # Convert tensor to list
            text=batch_texts,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        
        # Move to device
        pixel_values = inputs['pixel_values'].to(device)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(pixel_values, input_ids, attention_mask)
        
        # Extract features
        image_features = outputs['image_features']
        text_features = outputs['text_features']
        logit_scale = outputs.get('logit_scale', None)
        
        # Compute contrastive loss
        loss = clip_contrastive_loss(image_features, text_features, logit_scale)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item()
        
        # Update progress bar
        if (batch_idx + 1) % print_freq == 0:
            avg_loss = running_loss / print_freq
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            running_loss = 0.0
    
    return running_loss / len(train_loader)


@torch.no_grad()
def evaluate(model, processor, test_loader, device, label_names, class_prompts):
    """
    Evaluate model using zero-shot classification with text prompts.
    """
    model.eval()
    
    # Encode all text prompts
    all_text_features = []
    prompt_to_label = []
    
    for label_idx, label_name in enumerate(label_names):
        prompts = class_prompts[label_name]
        
        # Use all prompts for this class
        for prompt in prompts[:5]:  # Use top 5 prompts per class to avoid OOM
            inputs = processor(
                text=prompt,
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(device)
            
            outputs = model.model(input_ids=inputs['input_ids'],
                                 attention_mask=inputs['attention_mask'])
            text_features = F.normalize(outputs['text_features'], dim=-1)
            all_text_features.append(text_features)
            prompt_to_label.append(label_idx)
    
    # Stack all text features
    all_text_features = torch.cat(all_text_features, dim=0)  # [num_prompts, dim]
    prompt_to_label = torch.tensor(prompt_to_label, device=device)
    
    # Evaluate on test set
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(test_loader, desc='Evaluating'):
        images = images.to(device)
        
        # Process images
        inputs = processor(
            images=[img for img in images],
            return_tensors='pt',
            padding=True
        )
        pixel_values = inputs['pixel_values'].to(device)
        
        # Extract image features
        outputs = model.model(pixel_values=pixel_values)
        image_features = F.normalize(outputs['image_features'], dim=-1)
        
        # Compute similarity with all text prompts
        similarity = image_features @ all_text_features.t()  # [batch, num_prompts]
        
        # For each image, find the most similar prompt and its label
        # Average similarities for prompts of the same class
        class_similarities = []
        for label_idx in range(len(label_names)):
            mask = prompt_to_label == label_idx
            class_sim = similarity[:, mask].mean(dim=1)
            class_similarities.append(class_sim)
        
        class_similarities = torch.stack(class_similarities, dim=1)  # [batch, num_classes]
        preds = class_similarities.argmax(dim=1)
        
        all_preds.append(preds.cpu())
        all_labels.append(labels)
    
    # Compute accuracy
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    accuracy = (all_preds == all_labels).float().mean().item() * 100
    
    # Per-class accuracy
    class_accuracies = {}
    for label_idx, label_name in enumerate(label_names):
        mask = all_labels == label_idx
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == all_labels[mask]).float().mean().item() * 100
            class_accuracies[label_name] = class_acc
    
    return accuracy, class_accuracies


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize normalized tensor."""
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean


def save_checkpoint(model, optimizer, epoch, accuracy, save_dir, is_best=False):
    """Save model checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
    }
    
    if is_best:
        path = os.path.join(save_dir, 'conceptclip_covid_best.pth')
    else:
        path = os.path.join(save_dir, f'conceptclip_covid_epoch_{epoch}.pth')
    
    torch.save(checkpoint, path)
    print(f'Checkpoint saved: {path}')


def main(args):
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Class names
    label_names = ['normal', 'pneumonia', 'COVID-19']
    
    # Create text prompts
    print('\nCreating text prompts...')
    class_prompts = create_text_prompts(label_names)
    for label, prompts in class_prompts.items():
        print(f'{label}: {len(prompts)} prompts')
        print(f'  Examples: {prompts[:3]}')
    
    # Load model and processor
    print('\nLoading ConceptCLIP model...')
    processor = AutoProcessor.from_pretrained('JerrryNie/ConceptCLIP', trust_remote_code=True)
    model = ConceptCLIPFineTuner('JerrryNie/ConceptCLIP')
    model = model.to(device)
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Denormalization will be needed since ConceptCLIP expects unnormalized images
    # But for consistency with the dataset, we'll normalize during data loading
    # and denormalize before passing to processor
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.Resize(384),
        transforms.RandomCrop(384),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    
    test_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        normalize
    ])
    
    # Setup datasets
    print('\nLoading datasets...')
    train_dataset = ChestXrayDataSet(
        data_dir=os.path.join(args.dataset_dir, 'train'),
        image_list_file=args.train_image_list,
        transform=train_transform
    )
    
    test_dataset = ChestXrayDataSet(
        data_dir=os.path.join(args.dataset_dir, 'test'),
        image_list_file=args.test_image_list,
        transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Test samples: {len(test_dataset)}')
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f'\n{"="*70}')
        print(f'Epoch {epoch}/{args.epochs}')
        print("="*70)
        
        # Train
        avg_loss = train_epoch(
            model, processor, train_loader, optimizer, device,
            epoch, label_names, class_prompts, args.print_freq
        )
        
        print(f'Average training loss: {avg_loss:.4f}')
        
        # Evaluate
        if epoch % args.eval_freq == 0:
            print('\nEvaluating...')
            accuracy, class_accuracies = evaluate(
                model, processor, test_loader, device,
                label_names, class_prompts
            )
            
            print(f'\nOverall Accuracy: {accuracy:.2f}%')
            print('Per-class Accuracy:')
            for label, acc in class_accuracies.items():
                print(f'  {label}: {acc:.2f}%')
            
            # Save checkpoint
            is_best = accuracy > best_accuracy
            if is_best:
                best_accuracy = accuracy
                print(f'New best accuracy: {best_accuracy:.2f}%')
            
            save_checkpoint(model, optimizer, epoch, accuracy, args.save_dir, is_best)
        
        # Save periodic checkpoint
        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, epoch, 0, args.save_dir, is_best=False)
    
    print(f'\n{"="*70}')
    print('Training completed!')
    print(f'Best accuracy: {best_accuracy:.2f}%')
    print(f'Model saved in: {args.save_dir}')
    print("="*70)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fine-tune ConceptCLIP on COVID CXR Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument('--dataset-dir', default='/data/brian.hu/COVID/data/',
                       help='Dataset directory path')
    parser.add_argument('--train-image-list', default='./train_split.txt',
                       help='Train image list file')
    parser.add_argument('--test-image-list', default='./test_COVIDx4.txt',
                       help='Test image list file')
    
    # Training arguments
    parser.add_argument('--epochs', default=20, type=int,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', default=32, type=int,
                       help='Training batch size')
    parser.add_argument('--eval-batch-size', default=16, type=int,
                       help='Evaluation batch size (smaller to avoid OOM)')
    parser.add_argument('--lr', default=1e-5, type=float,
                       help='Learning rate')
    parser.add_argument('--weight-decay', default=0.01, type=float,
                       help='Weight decay')
    parser.add_argument('--eval-freq', default=2, type=int,
                       help='Evaluate every N epochs')
    
    # Other arguments
    parser.add_argument('--workers', default=4, type=int,
                       help='Number of data loading workers')
    parser.add_argument('--print-freq', default=10, type=int,
                       help='Print frequency')
    parser.add_argument('--seed', default=42, type=int,
                       help='Random seed')
    parser.add_argument('--save-dir', default='./checkpoints/conceptclip_finetuned',
                       help='Directory to save checkpoints')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
