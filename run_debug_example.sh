#!/bin/bash

# Example script to debug insertion/deletion metrics for ConvNeXtV2
# This uses your actual COVID dataset and model

echo "==================================================================="
echo "Debug Script: Insertion/Deletion Metrics for ConvNeXtV2"
echo "==================================================================="
echo ""

# Configuration - UPDATE THESE PATHS TO MATCH YOUR SETUP
QUERY_IMAGE="/data/brian.hu/COVID/data/test/47c78742-4998-4878-aec4-37b11b1354ac.png"
MODEL_WEIGHTS="./model.pth"  # or path to your ConvNeXtV2 checkpoint
OUTPUT_DIR="./debug_output"
STEP_SIZE=1000  # Number of pixels to modify per iteration (lower = more detailed)

# Optional: if you want to test with a different retrieved image
# RETRIEVED_IMAGE="/path/to/retrieved/image.png"

# Optional: if you have a pre-computed saliency map
# SALIENCY_MAP="/path/to/saliency_map.npy"

echo "Configuration:"
echo "  Query Image: $QUERY_IMAGE"
echo "  Model Weights: $MODEL_WEIGHTS"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Step Size: $STEP_SIZE pixels"
echo ""

# Check if query image exists
if [ ! -f "$QUERY_IMAGE" ]; then
    echo "ERROR: Query image not found at: $QUERY_IMAGE"
    echo ""
    echo "Please update the QUERY_IMAGE path in this script."
    echo "You can find valid image names in test_COVIDx4.txt"
    echo ""
    echo "Example valid paths (if using standard COVID dataset structure):"
    head -5 test_COVIDx4.txt | while read line; do
        filename=$(echo $line | awk '{print $2}')
        echo "  /data/brian.hu/COVID/data/test/$filename"
    done
    exit 1
fi

# Check if model weights exist
if [ ! -f "$MODEL_WEIGHTS" ]; then
    echo "WARNING: Model weights not found at: $MODEL_WEIGHTS"
    echo "The script will run with random initialization (for testing only)"
    echo ""
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting debug script..."
echo ""
echo "This will:"
echo "  1. Load the ConvNeXtV2 model"
echo "  2. Process the query image"
echo "  3. Calculate DELETION metrics (removing salient pixels)"
echo "  4. Calculate INSERTION metrics (adding salient pixels)"
echo "  5. Save detailed outputs and visualizations"
echo ""
echo "==================================================================="
echo ""

# Run the debug script
python debug_insertion_deletion.py \
    --query_image "$QUERY_IMAGE" \
    --model_weights "$MODEL_WEIGHTS" \
    --output_dir "$OUTPUT_DIR" \
    --step_size "$STEP_SIZE" \
    --device cuda

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "==================================================================="
    echo "Debug completed successfully!"
    echo "==================================================================="
    echo ""
    echo "Check the output directory for results:"
    echo "  $OUTPUT_DIR/deletion/   - Deletion metric visualizations"
    echo "  $OUTPUT_DIR/insertion/  - Insertion metric visualizations"
    echo ""
    echo "Key files to examine:"
    echo "  - del_curve.png and ins_curve.png: Shows the metric curves"
    echo "  - *_step_*.png: Intermediate images at different stages"
    echo ""
    echo "To run with a saliency map, add:"
    echo "  --saliency_map /path/to/saliency.npy"
    echo ""
else
    echo ""
    echo "==================================================================="
    echo "Debug script failed. Please check the error messages above."
    echo "==================================================================="
    exit 1
fi
