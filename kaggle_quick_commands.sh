#!/bin/bash
# Quick Kaggle Commands - Copy/Paste into Kaggle Notebook Cells

echo "================================================================"
echo "QUICK REFERENCE: Debug Insertion/Deletion for ConvNeXtV2"
echo "================================================================"
echo ""
echo "Run these commands in sequence in separate Kaggle notebook cells"
echo ""

cat << 'EOF'

# =================================================================
# CELL 1: Generate Saliency Map
# =================================================================
!python generate_single_saliency.py \
    --query_image /kaggle/input/rsna-png/data/test/47c78742-4998-4878-aec4-37b11b1354ac.png \
    --model_type convnextv2 \
    --model_weights /kaggle/input/convnextv2/pytorch/default/1/covid_convnextv2_seed_0_epoch_16_ckpt.pth \
    --explainer simatt \
    --output_path ./saliency_map.npy

# =================================================================
# CELL 2: View Saliency Visualization
# =================================================================
from IPython.display import Image, display
display(Image('./saliency_map_visualization.png'))

# =================================================================
# CELL 3: Run Debug Script (with the generated saliency map)
# =================================================================
!python debug_insertion_deletion.py \
    --query_image /kaggle/input/rsna-png/data/test/47c78742-4998-4878-aec4-37b11b1354ac.png \
    --model_weights /kaggle/input/convnextv2/pytorch/default/1/covid_convnextv2_seed_0_epoch_16_ckpt.pth \
    --saliency_map ./saliency_map.npy \
    --output_dir ./debug_output \
    --step_size 1000 \
    --device cuda

# =================================================================
# CELL 4: Display Results
# =================================================================
from IPython.display import Image, display

print("=== DELETION METRIC ===")
display(Image('./debug_output/deletion/del_curve.png'))

print("\n=== INSERTION METRIC ===")
display(Image('./debug_output/insertion/ins_curve.png'))

# =================================================================
# OPTIONAL: Show intermediate steps
# =================================================================
import os
import matplotlib.pyplot as plt

# Show deletion progress
del_images = sorted([f for f in os.listdir('./debug_output/deletion') if 'step' in f])
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Deletion Progress (Removing Salient Pixels)', fontsize=16)
for i, img_name in enumerate(del_images[::len(del_images)//10][:10]):
    ax = axes[i//5, i%5]
    img = plt.imread(f'./debug_output/deletion/{img_name}')
    ax.imshow(img)
    ax.set_title(img_name.replace('del_step_', 'Step ').replace('.png', ''))
    ax.axis('off')
plt.tight_layout()
plt.show()

# Show insertion progress
ins_images = sorted([f for f in os.listdir('./debug_output/insertion') if 'step' in f])
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Insertion Progress (Adding Salient Pixels)', fontsize=16)
for i, img_name in enumerate(ins_images[::len(ins_images)//10][:10]):
    ax = axes[i//5, i%5]
    img = plt.imread(f'./debug_output/insertion/{img_name}')
    ax.imshow(img)
    ax.set_title(img_name.replace('ins_step_', 'Step ').replace('.png', ''))
    ax.axis('off')
plt.tight_layout()
plt.show()

EOF
