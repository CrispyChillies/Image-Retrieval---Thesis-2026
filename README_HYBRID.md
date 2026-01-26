# 🔬 Hybrid ConvNeXtV2-ViT for Medical Image Retrieval

Implementation of the dual-backbone hybrid architecture combining ConvNeXtV2 and Vision Transformer for chest X-ray image retrieval.

## 🎯 Quick Start

### 1. Verify Installation
```bash
python test_hybrid_model.py
```

### 2. Train the Model
```bash
python train.py \
    --model hybrid_convnext_vit \
    --embedding-dim 1024 \
    --dataset covid \
    --dataset-dir ./data \
    --train-image-list ./train_split.txt \
    --test-image-list ./test_COVIDx4.txt \
    --epochs 50 \
    --lr 0.0001 \
    --batch-size 16
```

### 3. Evaluate the Model
```bash
python test.py \
    --model hybrid_convnext_vit \
    --embedding-dim 1024 \
    --dataset covid \
    --test-dataset-dir ./data/test \
    --test-image-list ./test_COVIDx4.txt \
    --resume ./checkpoints/your_checkpoint.pth
```

## 📊 Architecture Overview

```
Input (384×384×3)
    ↓
    ├─── ConvNeXtV2-Base ───→ MXA Block ───→ GAP ───→ [1024]
    │                         (CA + SA)
    │
    └─── ViT-B/16 ───→ CLS Token ───→ [768]
    
    [1024] + [768] = [1792]
         ↓
    Fusion Layer
         ↓
    Final Embedding [1024]
```

### Key Components

1. **ConvNeXt Branch**
   - ConvNeXtV2-Base pretrained on ImageNet-22k
   - Multi-scale Attention (MXA) Block
     - Channel Attention (CA)
     - Spatial Attention (SA)
   - Global Average Pooling
   - Output: 1024-dim features

2. **ViT Branch**
   - Vision Transformer Base (ViT-B/16)
   - Patch size: 16×16
   - CLS token extraction
   - Output: 768-dim features

3. **Feature Fusion**
   - Concatenation: 1792-dim
   - Linear → LayerNorm → ReLU
   - Final projection: 1024-dim
   - L2 normalization

## 🛠️ Installation

### Dependencies
```bash
pip install -r requirements_hybrid.txt
```

### Key Requirements
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- timm >= 0.9.0 (for model architectures)
- CUDA-capable GPU (12+ GB VRAM recommended)

## 📁 Files Modified

- **model.py**: Added `HybridConvNeXtViT`, `ChannelAttention`, `SpatialAttention`
- **train.py**: Updated to support hybrid model with 384×384 images
- **test.py**: Updated evaluation for hybrid model

## 📁 New Files

- **test_hybrid_model.py**: Quick validation script
- **HYBRID_MODEL_GUIDE.md**: Comprehensive documentation
- **QUICK_REFERENCE.py**: Command quick reference
- **IMPLEMENTATION_SUMMARY.txt**: Technical implementation details
- **requirements_hybrid.txt**: Updated dependencies

## 🚀 Training Tips

### Recommended Hyperparameters
```python
model = 'hybrid_convnext_vit'
embedding_dim = 1024
batch_size = 16  # Adjust based on GPU memory
epochs = 50-100
learning_rate = 1e-4
margin = 0.2  # Triplet loss margin
```

### Memory Optimization
- **Training**: ~12-16 GB GPU (batch_size=16)
- **Testing**: ~4-6 GB GPU (batch_size=64)
- Reduce batch size if OOM: try 8 or 4
- Use gradient accumulation for larger effective batch sizes

### Data Augmentation
- Random resized crop: 384×384
- Random horizontal flip
- Normalization: ImageNet mean/std

## 📈 Expected Performance

The hybrid architecture combines:
- **ConvNeXt**: Strong local feature extraction with hierarchical representations
- **ViT**: Global context modeling with self-attention
- **MXA Block**: Multi-scale attention for medical image specificity

Expected improvements over single backbone:
- Better R@K scores (especially R@1, R@5)
- Improved mAP (mean Average Precision)
- More robust embeddings for cross-domain retrieval

## 🔍 Model Validation

Run the test script to verify installation:
```bash
python test_hybrid_model.py
```

Expected output:
```
✓ Model created successfully
✓ Forward pass successful: (2, 1024)
✓ Output norm: ~1.0
✓ All tests passed!
```

## 💡 Usage Examples

### Example 1: Basic Training
```bash
python train.py --model hybrid_convnext_vit --embedding-dim 1024
```

### Example 2: Training with Custom Settings
```bash
python train.py \
    --model hybrid_convnext_vit \
    --embedding-dim 1024 \
    --labels-per-batch 3 \
    --samples-per-label 16 \
    --epochs 100 \
    --lr 0.0001 \
    --margin 0.2 \
    --rand-resize \
    --save-dir ./checkpoints/hybrid
```

### Example 3: Resume Training
```bash
python train.py \
    --model hybrid_convnext_vit \
    --embedding-dim 1024 \
    --resume ./checkpoints/hybrid/checkpoint_epoch_50.pth
```

### Example 4: Evaluation
```bash
python test.py \
    --model hybrid_convnext_vit \
    --embedding-dim 1024 \
    --dataset covid \
    --test-dataset-dir ./data/test \
    --test-image-list ./test_COVIDx4.txt \
    --resume ./checkpoints/best_model.pth \
    --eval-batch-size 64
```

## 🐛 Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch-size 8
--samples-per-label 8

# Or reduce number of workers
--workers 2
```

### Slow Training
```bash
# Increase data loading workers
--workers 8

# Enable data augmentation caching
# (requires code modification)
```

### Poor Convergence
- Check learning rate (try 5e-5 to 1e-4)
- Verify data normalization
- Check triplet loss margin (try 0.1 to 0.3)
- Ensure balanced sampling (PKSampler)

## 📚 Additional Documentation

- **HYBRID_MODEL_GUIDE.md**: Complete usage guide
- **IMPLEMENTATION_SUMMARY.txt**: Technical architecture details
- **QUICK_REFERENCE.py**: Quick command reference

## 🔗 Model Checkpoints

After training, checkpoints are saved as:
```
checkpoints/
└── {dataset}_{model}_embed_{dim}_seed_{seed}_epoch_{epoch}_ckpt.pth
```

Example:
```
covid_hybrid_convnext_vit_embed_1024_seed_0_epoch_50_ckpt.pth
```

## 🎓 Citation

If you use this implementation, please cite:
```
Hybrid ConvNeXtV2–ViT Architecture with Ontology-Driven Explainability 
for Transparent Chest X-Ray Diagnosis
```

## 📞 Support

For issues or questions:
1. Check **HYBRID_MODEL_GUIDE.md** for detailed documentation
2. Run `python test_hybrid_model.py` to verify installation
3. Check GPU memory with `nvidia-smi`
4. Verify data paths and file lists

## ✅ Implementation Status

- [x] Hybrid ConvNeXtV2-ViT architecture
- [x] Multi-scale Attention (MXA) Block
- [x] Channel Attention (CA)
- [x] Spatial Attention (SA)
- [x] Feature Fusion module
- [x] Training script integration
- [x] Testing script integration
- [x] Model validation
- [x] Documentation
- [ ] Grad-CAM visualization (TIER 3)
- [ ] OOD detection layer (TIER 3)
- [ ] Ontology-aware reasoning (TIER 3)


"""
Implementation Summary: Hybrid ConvNeXtV2-ViT Architecture
===========================================================

✅ COMPLETED IMPLEMENTATION

1. MODEL ARCHITECTURE (model.py)
   ├── ChannelAttention
   │   └── Squeeze & Excitation with avg/max pooling
   ├── SpatialAttention  
   │   └── Spatial focus using conv2d on channel-aggregated features
   └── HybridConvNeXtViT
       ├── ConvNeXt Branch
       │   ├── ConvNeXtV2-Base (pretrained ImageNet-22k)
       │   ├── MXA Block (CA + SA)
       │   └── Global Average Pooling → 1024-dim
       ├── ViT Branch
       │   ├── ViT-B/16 (pretrained ImageNet-21k)
       │   └── CLS Token Extraction → 768-dim
       └── Feature Fusion
           ├── Concatenate: [1024, 768] → 1792
           ├── Linear + LayerNorm + ReLU
           └── Final Linear → 1024-dim (normalized)

2. TRAINING SCRIPT (train.py)
   ✅ Added hybrid_convnext_vit model support
   ✅ Added 384×384 image processing for hybrid model
   ✅ Updated model selection logic
   ✅ Updated argument parser help text

3. TESTING SCRIPT (test.py)
   ✅ Added hybrid_convnext_vit model support
   ✅ Added 384×384 image processing for hybrid model
   ✅ Updated model selection logic
   ✅ Updated argument parser help text

4. DOCUMENTATION
   ✅ HYBRID_MODEL_GUIDE.md - Complete usage guide
   ✅ QUICK_REFERENCE.py - Quick command reference
   ✅ test_hybrid_model.py - Model validation script

5. VERIFICATION
   ✅ Syntax check: All files compile without errors
   ✅ Model instantiation: Successfully creates model
   ✅ Forward pass: Correctly processes 384×384 images
   ✅ Output validation: Proper shape and L2 normalization

ARCHITECTURE FLOW
=================

Input Image (384×384×3)
         |
         ├────────────────┬─────────────────┐
         │                │                 │
    ConvNeXtV2      ViT-B/16              │
         │                │                 │
   Feature Maps     Patch Tokens           │
   (B,1024,H,W)    (B,197,768)             │
         │                │                 │
   Channel Attn    Extract CLS             │
         │           Token                 │
   Spatial Attn    (B,768)                 │
         │                │                 │
   Global Avg            │                 │
    Pooling              │                 │
   (B,1024)              │                 │
         │                │                 │
         └────────┬───────┘                 │
                  │                         │
            Concatenate                     │
             (B,1792)                       │
                  │                         │
         Linear→LayerNorm→ReLU              │
             (B,1024)                       │
                  │                         │
           Final Linear                     │
             (B,1024)                       │
                  │                         │
           L2 Normalize                     │
                  │                         │
         Embedding Vector                   │
          (unit norm)                       │
                  └─────────────────────────┘

NEXT STEPS
==========
1. Train the model:
   python train.py --model hybrid_convnext_vit --embedding-dim 1024

2. Evaluate performance:
   python test.py --model hybrid_convnext_vit --embedding-dim 1024 --resume checkpoint.pth

3. Compare with baselines:
   - ConvNeXtV2 only
   - ResNet50
   - DenseNet121

4. Implement TIER 3 (Post-hoc modules from paper):
   - Grad-CAM visualization
   - OOD detection layer
   - Ontology-aware reasoning

TECHNICAL SPECIFICATIONS
=========================
Framework: PyTorch with timm
ConvNeXt: convnextv2_base.fcmae_ft_in22k_in1k_384
ViT: vit_base_patch16_384
Input: 384×384 RGB images
Loss: Triplet Margin Loss (batch-hard mining)
Optimization: Adam optimizer
"""

# Hybrid ConvNeXtV2-ViT Implementation Guide

## Overview
This implementation adds a hybrid dual-backbone architecture combining ConvNeXtV2 and Vision Transformer (ViT) as described in the paper "Hybrid ConvNeXtV2–ViT Architecture with Ontology-Driven Explainability for Transparent Chest X-Ray Diagnosis".

## Architecture Components

### 1. ConvNeXtV2 Branch
- **Base Model**: ConvNeXtV2-Base pretrained on ImageNet-22k
- **MXA Block**: Multi-scale Attention combining:
  - Channel Attention (CA): Focus on important feature channels
  - Spatial Attention (SA): Highlight important spatial regions
- **Output**: 1024-dimensional features from last stage

### 2. ViT Branch
- **Base Model**: ViT-B/16 (Vision Transformer Base, patch size 16)
- **Token Pooling**: Extracts CLS token representation
- **Output**: 768-dimensional features

### 3. Feature Fusion
- **Input**: Concatenated features (1024 + 768 = 1792)
- **Processing**: Linear projection → LayerNorm → ReLU
- **Output**: Configurable embedding dimension (default: 1024)

## Usage

### Training

#### Basic Training
```bash
python train.py \
    --model hybrid_convnext_vit \
    --dataset covid \
    --dataset-dir ./data \
    --train-image-list ./train_split.txt \
    --test-image-list ./test_COVIDx4.txt \
    --embedding-dim 1024 \
    --batch-size 16 \
    --epochs 50 \
    --lr 0.0001
```

#### Advanced Training with Custom Settings
```bash
python train.py \
    --model hybrid_convnext_vit \
    --dataset covid \
    --dataset-dir /path/to/data \
    --train-image-list ./train_split.txt \
    --test-image-list ./test_COVIDx4.txt \
    --embedding-dim 1024 \
    --labels-per-batch 3 \
    --samples-per-label 16 \
    --epochs 100 \
    --lr 0.0001 \
    --margin 0.2 \
    --rand-resize \
    --workers 8 \
    --save-dir ./checkpoints
```

#### Training Parameters
- `--model`: Set to `hybrid_convnext_vit`
- `--embedding-dim`: Output embedding dimension (recommended: 1024)
- `--labels-per-batch`: Number of unique classes per batch (for triplet loss)
- `--samples-per-label`: Number of samples per class in batch
- `--epochs`: Number of training epochs
- `--lr`: Learning rate (recommended: 1e-4)
- `--margin`: Triplet loss margin (default: 0.2)
- `--rand-resize`: Enable random resizing augmentation

### Testing

#### Basic Testing
```bash
python test.py \
    --model hybrid_convnext_vit \
    --dataset covid \
    --test-dataset-dir ./data/test \
    --test-image-list ./test_COVIDx4.txt \
    --embedding-dim 1024 \
    --resume ./checkpoints/covid_hybrid_convnext_vit_embed_1024_seed_0_epoch_50_ckpt.pth
```

#### Evaluation Metrics
The test script computes:
- **R@K**: Recall at K (K=1, 5, 10)
- **mAP**: Mean Average Precision
- **mP@K**: Mean Precision at K
- **Classification Metrics**: Accuracy, Precision, Recall, F1 (via majority voting)

### Model Comparison

Train and compare different backbones:

```bash
# ConvNeXtV2 only
python train.py --model convnextv2 --embedding-dim 1024 ...

# Hybrid model
python train.py --model hybrid_convnext_vit --embedding-dim 1024 ...
```

## Technical Details

### Image Preprocessing
- **Input Size**: 384×384 (required for both ConvNeXtV2 and ViT-B/16)
- **Normalization**: ImageNet mean/std
- **Augmentation**: 
  - Random resized crop (training)
  - Horizontal flip (training)
  - Center crop (testing)

### Model Architecture
```python
HybridConvNeXtViT(
    pretrained=True,  # Load ImageNet pretrained weights
    embedding_dim=1024  # Final embedding dimension
)
```

### Memory Requirements
- **Training**: ~12-16 GB GPU memory (batch_size=16)
- **Inference**: ~4-6 GB GPU memory (batch_size=64)

### Performance Expectations
The hybrid architecture should provide:
- Better feature representation than single backbone
- Improved retrieval accuracy (R@K)
- Better generalization across different chest X-ray conditions
- More robust embeddings for similarity search

## Files Modified

1. **model.py**: Added `HybridConvNeXtViT`, `ChannelAttention`, `SpatialAttention`
2. **train.py**: Updated to support hybrid model with 384×384 images
3. **test.py**: Updated to support hybrid model evaluation

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch-size` (try 8 or 4)
- Reduce `--samples-per-label`
- Use gradient checkpointing (requires model modification)

### Slow Training
- Increase `--workers` for faster data loading
- Use mixed precision training (requires code modification)
- Ensure data is on fast storage (SSD)

### Poor Performance
- Verify data augmentation is appropriate
- Check learning rate (try 1e-4 to 5e-5)
- Ensure proper normalization
- Try longer training (100+ epochs)
- Verify triplet loss hyperparameters

## Next Steps

1. **Train the model**: Start with baseline settings
2. **Evaluate**: Compare with ConvNeXtV2-only baseline
3. **Hyperparameter tuning**: Adjust lr, margin, embedding_dim
4. **Advanced features**: 
   - Add Grad-CAM visualization
   - Implement OOD detection
   - Add ontology-based reasoning (TIER 3 from paper)

## Citation

If you use this implementation, please cite the original paper:
```
Hybrid ConvNeXtV2–ViT Architecture with Ontology-Driven Explainability 
for Transparent Chest X-Ray Diagnosis
```
