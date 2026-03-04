# Complete Kaggle Workflow: Debug Insertion/Deletion Metrics

This guide shows the complete workflow to debug insertion/deletion metrics on Kaggle.

## Step 1: Generate Saliency Map

First, generate a saliency map for your query image:

```python
# Cell 1: Generate saliency map
!python generate_single_saliency.py \
    --query_image /kaggle/input/rsna-png/data/test/47c78742-4998-4878-aec4-37b11b1354ac.png \
    --model_type convnextv2 \
    --model_weights /kaggle/input/convnextv2/pytorch/default/1/covid_convnextv2_seed_0_epoch_16_ckpt.pth \
    --explainer simatt \
    --output_path ./saliency_map.npy
```

This will create:
- `saliency_map.npy` - The saliency map file
- `saliency_map_visualization.png` - A visualization showing query image, retrieved image, and saliency map

## Step 2: Run Debug Script with Saliency Map

Now run the debug script with the generated saliency map:

```python
# Cell 2: Debug insertion/deletion metrics
!python debug_insertion_deletion.py \
    --query_image /kaggle/input/rsna-png/data/test/47c78742-4998-4878-aec4-37b11b1354ac.png \
    --model_weights /kaggle/input/convnextv2/pytorch/default/1/covid_convnextv2_seed_0_epoch_16_ckpt.pth \
    --saliency_map ./saliency_map.npy \
    --output_dir ./debug_output \
    --step_size 1000 \
    --device cuda
```

## Step 3: View Results

```python
# Cell 3: Display results
from IPython.display import Image, display
import matplotlib.pyplot as plt

print("=== Saliency Map Visualization ===")
display(Image('./saliency_map_visualization.png'))

print("\n=== Deletion Metric (Removing Salient Pixels) ===")
display(Image('./debug_output/deletion/del_curve.png'))

print("\n=== Insertion Metric (Adding Salient Pixels) ===")
display(Image('./debug_output/insertion/ins_curve.png'))

# Show some intermediate steps
print("\n=== Deletion Progress ===")
import os
del_steps = sorted([f for f in os.listdir('./debug_output/deletion') if f.startswith('del_step_')])
for step_file in del_steps[::15]:  # Show every 15th step
    print(f"\n{step_file}")
    display(Image(f'./debug_output/deletion/{step_file}'))
```

## Alternative: Test with Different Retrieved Image

To compute saliency between two different images:

```python
# Generate saliency for query-retrieved pair
!python generate_single_saliency.py \
    --query_image /path/to/query.png \
    --retrieved_image /path/to/retrieved.png \
    --model_type convnextv2 \
    --model_weights /path/to/weights.pth \
    --explainer simatt \
    --output_path ./saliency_query_retrieved.npy

# Then debug with the pair
!python debug_insertion_deletion.py \
    --query_image /path/to/query.png \
    --retrieved_image /path/to/retrieved.png \
    --model_weights /path/to/weights.pth \
    --saliency_map ./saliency_query_retrieved.npy \
    --output_dir ./debug_output
```

## Complete Kaggle Notebook Example

```python
# ============================================================
# COMPLETE KAGGLE NOTEBOOK SCRIPT
# ============================================================

# Cell 1: Setup
import os
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Define paths
QUERY_IMAGE = "/kaggle/input/rsna-png/data/test/47c78742-4998-4878-aec4-37b11b1354ac.png"
MODEL_WEIGHTS = "/kaggle/input/convnextv2/pytorch/default/1/covid_convnextv2_seed_0_epoch_16_ckpt.pth"

# Verify files exist
print(f"\nQuery image exists: {os.path.exists(QUERY_IMAGE)}")
print(f"Model weights exist: {os.path.exists(MODEL_WEIGHTS)}")

# Cell 2: Generate Saliency Map
print("\n" + "="*60)
print("STEP 1: Generating Saliency Map")
print("="*60)

!python generate_single_saliency.py \
    --query_image {QUERY_IMAGE} \
    --model_type convnextv2 \
    --model_weights {MODEL_WEIGHTS} \
    --explainer simatt \
    --output_path ./saliency_map.npy

# Cell 3: View Saliency Visualization
from IPython.display import Image, display
print("\n=== Saliency Map Visualization ===")
display(Image('./saliency_map_visualization.png'))

# Cell 4: Run Debug Script
print("\n" + "="*60)
print("STEP 2: Running Insertion/Deletion Debug")
print("="*60)

!python debug_insertion_deletion.py \
    --query_image {QUERY_IMAGE} \
    --model_weights {MODEL_WEIGHTS} \
    --saliency_map ./saliency_map.npy \
    --output_dir ./debug_output \
    --step_size 1000 \
    --device cuda

# Cell 5: Display Results
print("\n" + "="*60)
print("STEP 3: Viewing Results")
print("="*60)

print("\n=== DELETION METRIC ===")
print("(Similarity should DROP as salient pixels are removed)")
display(Image('./debug_output/deletion/del_curve.png'))

print("\n=== INSERTION METRIC ===")
print("(Similarity should RISE as salient pixels are added)")
display(Image('./debug_output/insertion/ins_curve.png'))

print("\n=== Deletion Start vs End ===")
from matplotlib import pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(plt.imread('./debug_output/deletion/del_start.png'))
axes[0].set_title('Start (Original)')
axes[0].axis('off')
axes[1].imshow(plt.imread('./debug_output/deletion/del_finish.png'))
axes[1].set_title('End (All pixels removed)')
axes[1].axis('off')
plt.tight_layout()
plt.show()

print("\n=== Insertion Start vs End ===")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(plt.imread('./debug_output/insertion/ins_start.png'))
axes[0].set_title('Start (Blurred)')
axes[0].axis('off')
axes[1].imshow(plt.imread('./debug_output/insertion/ins_finish.png'))
axes[1].set_title('End (Original)')
axes[1].axis('off')
plt.tight_layout()
plt.show()

# Cell 6: List All Generated Files
print("\n=== All Generated Files ===")
for root, dirs, files in os.walk('./debug_output'):
    level = root.replace('./debug_output', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in sorted(files):
        print(f'{subindent}{file}')
```

## Understanding the Output

### Good Saliency Map Indicators:

**Deletion Metric:**
- AUC should be **LOW** (closer to 0)
- Curve should drop **quickly**
- This means removing salient pixels destroys similarity fast

**Insertion Metric:**
- AUC should be **HIGH** (closer to 1)
- Curve should rise **quickly**
- This means adding salient pixels recovers similarity fast

### Files Generated:

**Saliency Generation:**
- `saliency_map.npy` - Raw saliency values
- `saliency_map_visualization.png` - Visual comparison

**Debug Output:**
- `deletion/del_curve.png` - Deletion metric plot
- `insertion/ins_curve.png` - Insertion metric plot
- `deletion/del_step_*.png` - Intermediate deletion steps
- `insertion/ins_step_*.png` - Intermediate insertion steps

## Troubleshooting

**If saliency generation fails:**
- Check that model weights match the model type
- Verify image paths are correct
- Ensure GPU is enabled in Kaggle settings

**If debug script is too slow:**
- Increase `--step_size` (e.g., 5000 for faster, less detailed)
- Reduce number of steps by using lower resolution saliency

**If out of memory:**
- Use CPU: `--device cpu` (slower but won't crash)
- Reduce batch size in explainer settings
