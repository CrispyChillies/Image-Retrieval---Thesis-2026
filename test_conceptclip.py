import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from transformers import AutoModel, AutoProcessor
from PIL import Image

from read_data import ChestXrayDataSet


def custom_collate(batch):
    """Custom collate function to keep PIL images as a list."""
    images = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    return images, labels


@torch.no_grad()
def evaluate_conceptclip(model, processor, loader, device, args):
    model.eval()
    
    # Define class labels and text prompts
    class_names = {
      'normal': 'a chest X-ray image of a healthy lung',
      'pneumonia': 'a chest X-ray image of pneumonia',
      'COVID-19': 'a chest X-ray image of COVID-19'
    }

    texts = list(class_names.values())

    
    print(f'Using text prompts: {texts}')
    
    all_predictions = []
    all_labels = []
    
    for batch_idx, data in enumerate(loader):
        images = data[0]  # List of PIL images
        labels = data[1].to(device)
        
        # Process images and texts through ConceptCLIP
        inputs = processor(
            images=images, 
            text=texts,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get model outputs
        outputs = model(**inputs)
        
        # Compute logits: logit_scale * image_features @ text_features.t()
        logits = (outputs['logit_scale'] * outputs['image_features'] @ outputs['text_features'].t())
        
        # Get predictions (argmax over classes)
        predictions = logits.argmax(dim=-1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if (batch_idx + 1) % 10 == 0:
            print(f'Processed {(batch_idx + 1) * args.eval_batch_size} images...')
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions) * 100.0
    precision_macro = precision_score(all_labels, all_predictions, average='macro', zero_division=0) * 100.0
    recall_macro = recall_score(all_labels, all_predictions, average='macro', zero_division=0) * 100.0
    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0) * 100.0
    
    precision_weighted = precision_score(all_labels, all_predictions, average='weighted', zero_division=0) * 100.0
    recall_weighted = recall_score(all_labels, all_predictions, average='weighted', zero_division=0) * 100.0
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0) * 100.0
    
    # Per-class metrics
    precision_per_class = precision_score(all_labels, all_predictions, average=None, zero_division=0) * 100.0
    recall_per_class = recall_score(all_labels, all_predictions, average=None, zero_division=0) * 100.0
    f1_per_class = f1_score(all_labels, all_predictions, average=None, zero_division=0) * 100.0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Print results
    print('\n' + '='*60)
    print('ConceptCLIP Zero-Shot Classification Results on COVIDx CXR')
    print('='*60)
    
    print(f'\nOverall Metrics:')
    print(f'  Accuracy: {accuracy:.2f}%')
    print(f'\nMacro-averaged Metrics:')
    print(f'  Precision: {precision_macro:.2f}%')
    print(f'  Recall: {recall_macro:.2f}%')
    print(f'  F1-Score: {f1_macro:.2f}%')
    print(f'\nWeighted-averaged Metrics:')
    print(f'  Precision: {precision_weighted:.2f}%')
    print(f'  Recall: {recall_weighted:.2f}%')
    print(f'  F1-Score: {f1_weighted:.2f}%')
    
    print(f'\nPer-Class Metrics:')
    for i, class_name in enumerate(class_names):
        print(f'\n  {class_name}:')
        print(f'    Precision: {precision_per_class[i]:.2f}%')
        print(f'    Recall: {recall_per_class[i]:.2f}%')
        print(f'    F1-Score: {f1_per_class[i]:.2f}%')
    
    print(f'\nConfusion Matrix:')
    print(f'  Rows: True labels, Columns: Predicted labels')
    print(f'  Classes: {list(class_names.keys())}')
    print(cm)
    
    # Save results
    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        
        save_path = os.path.join(args.save_dir, 'conceptclip_covidx_zeroshot_results')
        
        np.savez(
            save_path,
            predictions=all_predictions,
            labels=all_labels,
            accuracy=accuracy,
            precision_macro=precision_macro,
            recall_macro=recall_macro,
            f1_macro=f1_macro,
            precision_weighted=precision_weighted,
            recall_weighted=recall_weighted,
            f1_weighted=f1_weighted,
            precision_per_class=precision_per_class,
            recall_per_class=recall_per_class,
            f1_per_class=f1_per_class,
            confusion_matrix=cm,
            class_names=list(class_names.keys()),
            text_prompts=texts
        )
        print(f'\nResults saved to {save_path}.npz')


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load ConceptCLIP model
    print('\nLoading ConceptCLIP model...')
    model = AutoModel.from_pretrained('JerrryNie/ConceptCLIP', trust_remote_code=True)
    processor = AutoProcessor.from_pretrained('JerrryNie/ConceptCLIP', trust_remote_code=True)
    
    model.to(device)
    model.eval()
    print('Model loaded successfully')

    # Transform for test data - keep as PIL images for processor
    test_transform = transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

    # Set up dataset and dataloader
    print(f'\nLoading test dataset from: {args.test_dataset_dir}')
    test_dataset = ChestXrayDataSet(
        data_dir=args.test_dataset_dir,
        image_list_file=args.test_image_list,
        use_covid=True,
        mask_dir=args.mask_dir,
        transform=test_transform
    )
    
    print(f'Test dataset size: {len(test_dataset)}')

    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=custom_collate
    )

    print('\n' + '='*60)
    print('Starting Evaluation...')
    print('='*60)
    evaluate_conceptclip(model, processor, test_loader, device, args)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Test ConceptCLIP on COVIDx CXR (Zero-Shot Classification)')

    parser.add_argument('--test-dataset-dir', default='/data/brian.hu/COVID/data/test',
                        help='Test dataset directory path')
    parser.add_argument('--test-image-list', default='./test_COVIDx4.txt',
                        help='Test image list')
    parser.add_argument('--mask-dir', default=None,
                        help='Segmentation masks path (if used)')
    parser.add_argument('--eval-batch-size', default=32, type=int,
                        help='Batch size for evaluation')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='Number of data loading workers')
    parser.add_argument('--save-dir', default='./results',
                        help='Result save directory')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
