"""
Quick test script to verify ConceptCLIP evaluation works correctly.
Tests on a small subset (100 samples) of the validation set.

Usage:
    python test_eval_conceptclip.py
    python test_eval_conceptclip.py --checkpoint_path checkpoints/conceptclip_epoch1.pth
    python test_eval_conceptclip.py --csv_file vindr/image_labels_test.csv --data_dir path/to/test/images
"""

import torch
from torch.utils.data import DataLoader, Subset
import argparse
from model import conceptCLIP
from read_data import VINDRConceptCLIPDataSet
from train import evaluate_conceptclip, conceptclip_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description='Test ConceptCLIP evaluation')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint (optional, will test with random init if not provided)')
    parser.add_argument('--csv_file', type=str, 
                        default='vindr/image_labels_test.csv',
                        help='Path to test CSV')
    parser.add_argument('--data_dir', type=str,
                        default='D:/VinDR-CXR-dataset/vinbigdata-chest-xray-original-png/test/test',
                        help='Path to test images directory')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to test (default: 100 for quick test)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("Testing ConceptCLIP Evaluation")
    print("="*60)
    
    # 1. Load model
    print(f"\n1. Loading model...")
    model = conceptCLIP(
        model_name='JerrryNie/ConceptCLIP',
        unfreeze_vision_layers=4,
        unfreeze_text_layers=2
    )
    
    if args.checkpoint_path:
        print(f"   Loading checkpoint from: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("   ✓ Checkpoint loaded")
    else:
        print("   Using model with random initialization (no checkpoint provided)")
    
    model = model.to(device)
    model.eval()
    
    # 2. Create test dataset
    print(f"\n2. Creating test dataset...")
    print(f"   CSV: {args.csv_file}")
    print(f"   Images: {args.data_dir}")
    
    full_dataset = VINDRConceptCLIPDataSet(
        data_dir=args.data_dir,
        csv_file=args.csv_file,
        return_pil=True
    )
    
    print(f"   Total samples: {len(full_dataset)}")
    print(f"   Testing on: {args.num_samples} samples")
    
    # Create subset for quick testing
    indices = list(range(min(args.num_samples, len(full_dataset))))
    test_subset = Subset(full_dataset, indices)
    
    test_loader = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # 0 for debugging
        collate_fn=conceptclip_collate_fn
    )
    
    print(f"   ✓ DataLoader created: {len(test_loader)} batches")
    
    # 3. Test data loading
    print(f"\n3. Testing data loading...")
    batch = next(iter(test_loader))
    print(f"   Batch keys: {batch.keys()}")
    print(f"   Images: {len(batch['images'])} PIL Images")
    print(f"   First image size: {batch['images'][0].size}")
    print(f"   all_labels shape: {batch['all_labels'].shape}")
    print(f"   ✓ Data loading works")
    
    # 4. Test forward pass
    print(f"\n4. Testing forward pass...")
    with torch.no_grad():
        images = batch['images'][:2]  # Test with 2 images
        processor = model.processor
        img_inputs = processor(images=images, return_tensors="pt")
        pixel_values = img_inputs['pixel_values'].to(device)
        
        embeddings = model(pixel_values)
        print(f"   Input shape: {pixel_values.shape}")
        print(f"   Output shape: {embeddings.shape}")
        print(f"   Embedding range: [{embeddings.min().item():.4f}, {embeddings.max().item():.4f}]")
        print(f"   ✓ Forward pass works")
    
    # 5. Run full evaluation
    print(f"\n5. Running evaluation on {args.num_samples} samples...")
    print(f"   (This computes mAP with Jaccard>0.4 threshold)")
    
    mean_ap = evaluate_conceptclip(
        model=model,
        loader=test_loader,
        device=device,
        rank=0,
        world_size=1
    )
    
    print(f"\n{'='*60}")
    print(f"✓ Evaluation completed successfully!")
    print(f"  mAP on {args.num_samples} samples: {mean_ap:.3f}%")
    print(f"{'='*60}")
    
    if not args.checkpoint_path:
        print("\nNote: This is with random initialization.")
        print("      Expected mAP should be low (~5-15%).")
        print("      Run with --checkpoint_path to test a trained model.")
    
    return mean_ap


if __name__ == '__main__':
    main()
