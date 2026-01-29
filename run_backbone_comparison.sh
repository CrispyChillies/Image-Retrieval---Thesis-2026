#!/bin/bash

# Full pipeline for backbone comparison
# Usage: bash run_backbone_comparison.sh

# Configuration
DATASET="covid"
TEST_DIR="/data/brian.hu/COVID/data/test"
TEST_LIST="./test_COVIDx4.txt"
CONVNEXT_CHECKPOINT="model.pth"
EMBEDDING_DIM=1024
BATCH_SIZE=32
OUTPUT_DIR="./backbone_comparison_data"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "STEP 1: Extracting ConvNeXtV2 Embeddings"
echo "=============================================="

python extract_embeddings.py \
    --model convnextv2 \
    --resume $CONVNEXT_CHECKPOINT \
    --embedding-dim $EMBEDDING_DIM \
    --dataset $DATASET \
    --test-dataset-dir $TEST_DIR \
    --test-image-list $TEST_LIST \
    --output $OUTPUT_DIR/convnext_embeddings.npy \
    --batch-size $BATCH_SIZE

if [ $? -ne 0 ]; then
    echo "ERROR: ConvNeXtV2 embedding extraction failed!"
    exit 1
fi

echo ""
echo "=============================================="
echo "STEP 2: Extracting ConceptCLIP Embeddings"
echo "=============================================="

python extract_embeddings.py \
    --model conceptclip \
    --dataset $DATASET \
    --test-dataset-dir $TEST_DIR \
    --test-image-list $TEST_LIST \
    --output $OUTPUT_DIR/conceptclip_embeddings.npy \
    --batch-size 16 \
    --conceptclip-batch-size 8

if [ $? -ne 0 ]; then
    echo "ERROR: ConceptCLIP embedding extraction failed!"
    exit 1
fi

echo ""
echo "=============================================="
echo "STEP 3: Running Backbone Comparison"
echo "=============================================="

python test_backbone.py \
    --convnext-embeddings $OUTPUT_DIR/convnext_embeddings.npy \
    --conceptclip-embeddings $OUTPUT_DIR/conceptclip_embeddings.npy \
    --labels $OUTPUT_DIR/convnext_embeddings_labels.npy \
    --image-ids $OUTPUT_DIR/convnext_embeddings_image_ids.npy \
    --image-dir $TEST_DIR \
    --save-dir ./results/backbone_comparison \
    --k 5 \
    --visualize-samples 5 \
    --fusion-alphas 0.3 0.5 0.7

if [ $? -ne 0 ]; then
    echo "ERROR: Backbone comparison failed!"
    exit 1
fi

echo ""
echo "=============================================="
echo "PIPELINE COMPLETE!"
echo "=============================================="
echo "Results saved to: ./results/backbone_comparison"
echo "Embeddings saved to: $OUTPUT_DIR"
