# Fixing Low Similarity Scores in Milvus

## Problem
You're seeing very low similarity scores (0.03-0.04) instead of high scores (0.96+).

**Root Cause**: Your Milvus index was created with **L2 distance metric** instead of **COSINE similarity**.

## Quick Diagnosis

Run this command to check your index configuration:
```bash
python check_milvus_index.py --model_type densenet121 --uri $ZILLIZ_URI --token $ZILLIZ_TOKEN
```

## Solutions

### Option 1: Specify the Correct Metric (Quick Fix)

If your index uses L2, just tell the script to use L2:

```bash
python debug_pipeline_with_milvus.py \
    --query_image ./test.png \
    --model_type densenet121 \
    --model_weights ./weights.pth \
    --metric_type L2 \
    --uri $ZILLIZ_URI --token $ZILLIZ_TOKEN
```

The code will automatically convert L2 distances to cosine similarities using:
- `similarity = 1.0 - (L2_distance² / 2)`

### Option 2: Recreate Index with COSINE (Recommended)

For better performance and intuitive results, recreate the index with COSINE metric:

```bash
# 1. Drop and recreate collections
python milvus_setup.py \
    --model densenet121 \
    --drop_old \
    --metric_type COSINE \
    --uri $ZILLIZ_URI --token $ZILLIZ_TOKEN

# 2. Re-ingest embeddings (make sure they're normalized!)
python ingest_embeddings.py \
    --model_type densenet121 \
    --model_weights ./weights.pth \
    --data_dir ./data/ \
    --image_list ./test.txt \
    --uri $ZILLIZ_URI --token $ZILLIZ_TOKEN

# 3. Run pipeline with COSINE (default)
python debug_pipeline_with_milvus.py \
    --query_image ./test.png \
    --model_type densenet121 \
    --model_weights ./weights.pth \
    --metric_type COSINE \
    --uri $ZILLIZ_URI --token $ZILLIZ_TOKEN
```

## Understanding the Metrics

### COSINE Similarity (Recommended)
- **Range**: Similarity 0 to 1 (higher = more similar)
- **Distance**: 0 to 2 (lower = more similar)
- **Conversion**: `similarity = 1 - distance`
- **Best for**: Normalized embeddings (most common)

### L2 Distance (Euclidean)
- **Range**: 0 to ∞ (lower = more similar)
- **Conversion**: `similarity = 1 - (distance² / 2)` for normalized vectors
- **Issue**: Low scores like 0.03 instead of 0.96
- **Best for**: When absolute distance matters

### IP (Inner Product)
- **Range**: -1 to 1 (higher = more similar)
- **Conversion**: `similarity = distance` for normalized vectors
- **Best for**: When vectors are already normalized

## Verification

After fixing, you should see:
```
Query: CR.1.2.840.113564.1722810170.20200320071047906500.1003000225002.png

Rank   Retrieved Image                          Sim      DEL      INS     
----------------------------------------------------------------------
1      covid-19-caso-86-RX-Torace-2.jpg         0.9632   0.4626   0.6722  
2      SARS-10.1148rg.242035193-g04mr34g0-Fig8b-day5.jpeg 0.9631   0.6304   0.5692  
3      1-s2.0-S1341321X20301124-gr3_lrg-d.png   0.9592   0.5511   0.7081  
4      COVID-93.png                             0.9559   0.5377   0.6841  
5      f44373474437c99b2740062c914438_jumbo.jpeg 0.9531   0.5631   0.7252  
```

✅ High similarity scores (0.95+)
✅ DEL and INS metrics unchanged (they only depend on saliency, not retrieval)

## What Changed

**Updated Files:**
1. **milvus_retrieval.py**: Now properly converts L2/IP distances to similarities
2. **debug_pipeline_with_milvus.py**: Added `--metric_type` parameter
3. **check_milvus_index.py**: New diagnostic tool

**Key Fixes:**
- L2 distance → similarity conversion: `1 - (distance²/2)`
- COSINE distance → similarity: `1 - distance`
- IP distance → similarity: `distance` (for normalized vectors)
