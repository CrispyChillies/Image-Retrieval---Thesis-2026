# Ensemble Mode Usage Guide

## Overview
The ensemble mode allows you to combine predictions from two models (e.g., ConvNeXtV2 and SwinV2) to potentially improve retrieval performance.

## Files Modified
- **postprocess.py**: Rewritten to provide ensemble functionality including:
  - `ensemble_embeddings()`: Combines embeddings from multiple models
  - `get_ensemble_embeddings()`: Extracts and combines embeddings from models
  - Support for multiple ensemble methods: average, concatenate, weighted

- **test.py**: Enhanced with ensemble support:
  - New `evaluate_ensemble()` function
  - Support for loading two models with separate checkpoints
  - New command-line arguments for ensemble mode

## Usage

### Single Model Mode (Default)
```bash
python test.py \
    --dataset covid \
    --test-dataset-dir /path/to/test/data \
    --test-image-list ./test_COVIDx4.txt \
    --model convnextv2 \
    --embedding-dim 512 \
    --resume ./checkpoints/convnextv2_model.pth
```

### Ensemble Mode
```bash
python test.py \
    --ensemble \
    --dataset covid \
    --test-dataset-dir /path/to/test/data \
    --test-image-list ./test_COVIDx4.txt \
    --model convnextv2 \
    --embedding-dim 512 \
    --resume ./checkpoints/convnextv2_model.pth \
    --model2 swinv2 \
    --embedding-dim2 512 \
    --resume2 ./checkpoints/swinv2_model.pth \
    --ensemble-method average
```

## Command-Line Arguments

### Ensemble-Specific Arguments:
- `--ensemble`: Enable ensemble mode (flag)
- `--model2`: Second model architecture (densenet121, resnet50, convnextv2, or swinv2)
- `--embedding-dim2`: Embedding dimension of the second model
- `--resume2`: Checkpoint path for the second model (required when using --ensemble)
- `--ensemble-method`: Method to combine embeddings:
  - `average` (default): Average embeddings and re-normalize
  - `concatenate`: Concatenate embeddings and normalize
  - `weighted`: Weighted average of embeddings

## Ensemble Methods Explained

1. **Average**: Takes the mean of embeddings from both models and re-normalizes. Good for models with similar embedding dimensions.

2. **Concatenate**: Concatenates embeddings side-by-side, doubling the embedding dimension. Best when models capture different aspects.

3. **Weighted**: Weighted average (currently equal weights, but can be extended). Useful when one model is more reliable than the other.

## Example Scenarios

### ConvNeXtV2 + SwinV2 Ensemble
```bash
python test.py \
    --ensemble \
    --dataset covid \
    --test-dataset-dir ./data/test \
    --test-image-list ./test_COVIDx4.txt \
    --model convnextv2 \
    --embedding-dim 1024 \
    --resume ./checkpoints/convnextv2_epoch_20.pth \
    --model2 swinv2 \
    --embedding-dim2 1024 \
    --resume2 ./checkpoints/swinv2_epoch_20.pth \
    --ensemble-method average \
    --eval-batch-size 32
```

### ResNet50 + DenseNet121 Ensemble
```bash
python test.py \
    --ensemble \
    --dataset isic \
    --test-dataset-dir ./data/ISIC-2017_Test_v2_Data \
    --test-image-list ./ISIC-2017_Test_v2_Part3_GroundTruth_balanced.csv \
    --model resnet50 \
    --embedding-dim 512 \
    --resume ./checkpoints/resnet50_model.pth \
    --model2 densenet121 \
    --embedding-dim2 512 \
    --resume2 ./checkpoints/densenet121_model.pth \
    --ensemble-method concatenate
```

## Notes
- Both model checkpoints must exist when using `--ensemble` mode
- The ensemble method can significantly impact performance - experiment with different methods
- Concatenate method doubles the embedding dimension, which may increase memory usage
- Ensemble mode works with all supported datasets (covid, isic, tbx11k)
