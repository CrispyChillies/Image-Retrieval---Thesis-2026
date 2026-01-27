# Training Updates - Best Model Tracking

## Changes Made

### 1. **Periodic Evaluation**
- Model is now evaluated every 2 epochs (configurable with `--eval-freq`)
- Prevents overfitting by monitoring validation performance throughout training

### 2. **Best Model Saving**
- Tracks the best model based on R@1 accuracy
- Automatically saves the best checkpoint as `*_best_ckpt.pth`
- Also saves periodic checkpoints every 10 epochs

### 3. **Training Summary**
- Shows best epoch and accuracy at the end of training
- Clear visual feedback with formatted output

## Usage

### Basic Training (evaluates every 2 epochs)
```bash
python train.py \
    --model hybrid_convnext_vit \
    --embedding-dim 1024 \
    --epochs 50
```

### Custom Evaluation Frequency
```bash
python train.py \
    --model hybrid_convnext_vit \
    --embedding-dim 1024 \
    --epochs 50 \
    --eval-freq 5  # Evaluate every 5 epochs
```

### Evaluate Every Epoch
```bash
python train.py \
    --model hybrid_convnext_vit \
    --embedding-dim 1024 \
    --epochs 50 \
    --eval-freq 1
```

## Saved Checkpoints

### Best Model
```
checkpoints/covid_hybrid_convnext_vit_embed_1024_seed_0_best_ckpt.pth
```
- Saved automatically when validation accuracy improves
- Use this for final testing/deployment

### Periodic Checkpoints
```
checkpoints/covid_hybrid_convnext_vit_embed_1024_seed_0_epoch_10_ckpt.pth
checkpoints/covid_hybrid_convnext_vit_embed_1024_seed_0_epoch_20_ckpt.pth
checkpoints/covid_hybrid_convnext_vit_embed_1024_seed_0_epoch_30_ckpt.pth
...
```
- Saved every 10 epochs for backup

## Example Output

```
============================================================
Training epoch 10/50...
============================================================
[10, 5] | loss: 0.2345 | % avg hard triplets: 67.34%
...

============================================================
Evaluating epoch 10...
============================================================
>> R@1 accuracy: 85.234%

🎯 New best model! Accuracy: 85.234% (epoch 10)
>> Checkpoint saved: ./checkpoints/covid_hybrid_convnext_vit_embed_1024_seed_0_best_ckpt.pth
>> Checkpoint saved: ./checkpoints/covid_hybrid_convnext_vit_embed_1024_seed_0_epoch_10_ckpt.pth

============================================================
Training completed!
============================================================
Best model: Epoch 42 with R@1 accuracy: 89.567%
Best model saved in: ./checkpoints
============================================================
```

## Benefits

1. **Prevents Overfitting**: Best model is selected based on validation performance
2. **Training Insights**: Monitor performance throughout training
3. **Flexibility**: Configurable evaluation frequency
4. **Safety**: Periodic backups every 10 epochs
5. **Efficiency**: No need to train to completion if early stopping desired

## Testing the Best Model

```bash
python test.py \
    --model hybrid_convnext_vit \
    --embedding-dim 1024 \
    --dataset covid \
    --test-dataset-dir ./data/test \
    --test-image-list ./test_COVIDx4.txt \
    --resume ./checkpoints/covid_hybrid_convnext_vit_embed_1024_seed_0_best_ckpt.pth
```

## New Argument

- `--eval-freq`: Frequency of evaluation in epochs (default: 2)
  - Set to 1 for every epoch
  - Set to 5 for every 5 epochs
  - Adjust based on dataset size and training time
