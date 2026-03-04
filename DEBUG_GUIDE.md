# Debug Guide: Insertion/Deletion Metrics for ConvNeXtV2

This guide explains how to debug and validate the insertion/deletion metric calculation on a single image.

## Quick Start

### Option 1: With Pre-computed Saliency Map

If you have a pre-computed saliency map (.npy file):

```bash
python debug_insertion_deletion.py \
    --query_image /path/to/your/image.jpg \
    --model_weights /path/to/convnextv2_weights.pth \
    --saliency_map /path/to/saliency_map.npy \
    --output_dir ./debug_output
```

### Option 2: Without Saliency Map (Demo Mode)

To test the calculation pipeline with a random saliency map:

```bash
python debug_insertion_deletion.py \
    --query_image /path/to/your/image.jpg \
    --model_weights /path/to/convnextv2_weights.pth \
    --output_dir ./debug_output
```

### Option 3: Self-Similarity Test

To debug with the same image as query and retrieved:

```bash
python debug_insertion_deletion.py \
    --query_image /path/to/your/image.jpg \
    --retrieved_image /path/to/your/image.jpg \
    --model_weights /path/to/convnextv2_weights.pth \
    --saliency_map /path/to/saliency_map.npy \
    --output_dir ./debug_output
```

## Understanding the Output

The script will create a `debug_output` directory with two subdirectories:

### 1. Deletion Metrics (`debug_output/deletion/`)
- `del_start.png` - Original retrieved image
- `del_finish.png` - Fully masked (black) image  
- `del_step_XXX.png` - Intermediate images at various steps
- `del_curve.png` - Plot showing how similarity decreases as pixels are removed

### 2. Insertion Metrics (`debug_output/insertion/`)
- `ins_start.png` - Blurred retrieved image
- `ins_finish.png` - Original retrieved image
- `ins_step_XXX.png` - Intermediate images as pixels are inserted
- `ins_curve.png` - Plot showing how similarity increases as pixels are added

## What the Script Shows

The debug script provides detailed logging for:

1. **Initialization**: Model setup, image sizes, step sizes
2. **Feature Extraction**: Query and retrieved image embeddings
3. **Saliency Ranking**: Which pixels are most salient
4. **Iteration Progress**: Cosine similarity at each step
5. **Final Metrics**: AUC scores for both insertion and deletion

## Understanding the Metrics

### Deletion Metric
- **Start**: Original retrieved image (high similarity expected)
- **Process**: Progressively remove most salient pixels (set to black)
- **End**: All pixels removed (low similarity expected)
- **Good saliency**: Similarity should drop quickly (low AUC)

### Insertion Metric  
- **Start**: Blurred retrieved image (low similarity expected)
- **Process**: Progressively add back most salient pixels from original
- **End**: Full original image (high similarity expected)
- **Good saliency**: Similarity should increase quickly (high AUC)

## Parameters

- `--query_image`: Path to query image (required)
- `--retrieved_image`: Path to retrieved image (optional, defaults to query)
- `--saliency_map`: Path to .npy saliency map (optional, generates random if not provided)
- `--model_weights`: Path to ConvNeXtV2 checkpoint (required)
- `--embedding_dim`: Embedding dimension if model uses projection layer (optional)
- `--output_dir`: Where to save debug outputs (default: ./debug_output)
- `--step_size`: Pixels modified per iteration (default: 1000, smaller = more detailed)
- `--device`: cuda or cpu (default: cuda)

## Adjusting Detail Level

To see more detailed step-by-step progression, use a smaller `--step_size`:

```bash
# More detailed (slower, more images saved)
python debug_insertion_deletion.py \
    --query_image /path/to/image.jpg \
    --model_weights /path/to/weights.pth \
    --step_size 500 \
    --output_dir ./debug_detailed

# Less detailed (faster, fewer images saved)  
python debug_insertion_deletion.py \
    --query_image /path/to/image.jpg \
    --model_weights /path/to/weights.pth \
    --step_size 5000 \
    --output_dir ./debug_fast
```

## Generating Saliency Maps First

If you don't have saliency maps yet, first generate them using:

```bash
python compute_saliency.py \
    --model convnextv2 \
    --explainer simatt \
    --resume /path/to/weights.pth \
    --test-dataset-dir /path/to/images \
    --test-image-list test.txt \
    --save-dir ./saliency_maps
```

Then use the generated saliency maps with the debug script.

## Expected Console Output

The script prints detailed information:
- Model loading status
- Image dimensions
- Feature vector norms
- Per-iteration cosine similarities
- Final AUC scores

Look for the "SUMMARY" section at the end for final results.
