# Backbone Comparison Workflow

This guide explains how to extract embeddings and compare ConvNeXtV2 vs ConceptCLIP retrieval behavior.

## Quick Start (Automated)

Edit `run_backbone_comparison.sh` to set your paths, then run:

```bash
bash run_backbone_comparison.sh
```

This will:
1. Extract ConvNeXtV2 embeddings
2. Extract ConceptCLIP embeddings
3. Run comparison analysis

## Manual Step-by-Step

### Step 1: Extract ConvNeXtV2 Embeddings

```bash
python extract_embeddings.py \
    --model convnextv2 \
    --resume model.pth \
    --embedding-dim 1024 \
    --dataset covid \
    --test-dataset-dir /path/to/test/data \
    --test-image-list test_COVIDx4.txt \
    --output embeddings/convnext_embeddings.npy \
    --batch-size 32
```

**Outputs:**
- `embeddings/convnext_embeddings.npy` - Embeddings [N, D]
- `embeddings/convnext_embeddings_labels.npy` - Labels [N]
- `embeddings/convnext_embeddings_image_ids.npy` - Image IDs [N]

### Step 2: Extract ConceptCLIP Embeddings

```bash
python extract_embeddings.py \
    --model conceptclip \
    --dataset covid \
    --test-dataset-dir /path/to/test/data \
    --test-image-list test_COVIDx4.txt \
    --output embeddings/conceptclip_embeddings.npy \
    --batch-size 16 \
    --conceptclip-batch-size 8
```

**Outputs:**
- `embeddings/conceptclip_embeddings.npy` - Embeddings [N, D]
- `embeddings/conceptclip_embeddings_labels.npy` - Labels [N]
- `embeddings/conceptclip_embeddings_image_ids.npy` - Image IDs [N]

**Note:** ConceptCLIP uses smaller batches (8) internally to avoid OOM on GPU.

### Step 3: Run Backbone Comparison

```bash
python test_backbone.py \
    --convnext-embeddings embeddings/convnext_embeddings.npy \
    --conceptclip-embeddings embeddings/conceptclip_embeddings.npy \
    --labels embeddings/convnext_embeddings_labels.npy \
    --image-ids embeddings/convnext_embeddings_image_ids.npy \
    --image-dir /path/to/test/data \
    --save-dir results/backbone_comparison \
    --k 5 \
    --visualize-samples 5 \
    --fusion-alphas 0.3 0.5 0.7
```

**Outputs:**
- `results/backbone_comparison/overlap_histogram.png` - Overlap distribution
- `results/backbone_comparison/query_*_comparison.png` - Visual comparisons
- `results/backbone_comparison/backbone_comparison_results.npz` - All metrics

## What You'll Learn

### 1. **Retrieval Performance**
- mAP@5 for each backbone
- Which model retrieves more accurately

### 2. **Retrieval Overlap**
- How many top-5 images are shared between models
- Low overlap (≤2) → complementary signals → fusion beneficial

### 3. **Rank Correlation**
- Spearman ρ between similarity rankings
- Low correlation → models rank differently → fusion beneficial

### 4. **Late Fusion**
- Test weighted combinations: `α * sim_convnext + (1-α) * sim_conceptclip`
- Compare fusion vs individual models

### 5. **Visual Analysis**
- Side-by-side retrieval comparisons
- See where models agree/disagree
- Understand failure modes

## Expected Outcomes

**If ConceptCLIP is beneficial for fusion:**
- Low overlap (mean < 3.0)
- Low Spearman ρ (< 0.6)
- Fusion mAP@5 > individual models

**If ConceptCLIP is NOT beneficial:**
- High overlap (mean > 4.0)
- High Spearman ρ (> 0.8)
- Fusion doesn't improve mAP@5

## Interpretation Guide

### Summary Table Example:
```
Model                     mAP@5        Mean Overlap@5     Spearman ρ
----------------------------------------------------------------------
ConvNeXtV2               0.7234       -                  -
ConceptCLIP              0.6512       2.3400             0.4821
----------------------------------------------------------------------
Fusion (α=0.3)           0.7156       -                  -
Fusion (α=0.5)           0.7389       -                  -
Fusion (α=0.7)           0.7401       -                  -
```

**This tells you:**
- ConvNeXtV2 is stronger (0.7234 > 0.6512)
- Low overlap (2.34/5) → models retrieve different images
- Low correlation (0.48) → models rank very differently
- Fusion improves over ConvNeXtV2 alone (0.7401 > 0.7234)
- → **ConceptCLIP provides complementary signals!**

## GPU Memory Considerations

- **ConvNeXtV2:** Can use batch_size=32-64 (depends on embedding_dim)
- **ConceptCLIP:** Use batch_size=16, conceptclip_batch_size=8 to avoid OOM
- If OOM occurs, reduce `--conceptclip-batch-size` to 4

## Troubleshooting

### Issue: "No module named transformers"
```bash
pip install transformers
```

### Issue: "Image not found" in visualizations
- Check `--image-dir` points to correct directory
- Images should be named as `{image_id}.png` or `{image_id}.jpg`

### Issue: Shape mismatch between embeddings
- Both models must process the SAME test set
- Use the same `--test-image-list` for both extractions

### Issue: CUDA OOM during ConceptCLIP extraction
- Reduce `--batch-size` to 8
- Reduce `--conceptclip-batch-size` to 4
- Use CPU: `CUDA_VISIBLE_DEVICES="" python extract_embeddings.py ...`

## Files Created

```
backbone_comparison_data/
├── convnext_embeddings.npy          # ConvNeXtV2 features
├── convnext_embeddings_labels.npy   # Class labels
├── convnext_embeddings_image_ids.npy # Image identifiers
├── conceptclip_embeddings.npy       # ConceptCLIP features
├── conceptclip_embeddings_labels.npy
└── conceptclip_embeddings_image_ids.npy

results/backbone_comparison/
├── overlap_histogram.png            # Overlap distribution
├── query_*_comparison.png           # Visual comparisons (5 samples)
└── backbone_comparison_results.npz  # All numerical results
```

## Next Steps After Analysis

**If fusion is beneficial:**
1. Implement early fusion (concatenate embeddings before final layer)
2. Train a fusion module on top of frozen backbones
3. Test on validation set before final evaluation

**If fusion is NOT beneficial:**
1. Stick with ConvNeXtV2 (simpler, likely better)
2. Focus on improving ConvNeXtV2 (data augmentation, better loss)
3. Consider domain-specific pretraining for ConceptCLIP
