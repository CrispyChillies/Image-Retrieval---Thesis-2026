#!/bin/bash
#SBATCH --job-name=nih_dinov2_weighted
#SBATCH --output=/media/vhviet03/Image-Retrieval---Thesis-2026/logs/nih_dinov2_weighted_%j.out
#SBATCH --error=/media/vhviet03/Image-Retrieval---Thesis-2026/logs/nih_dinov2_weighted_%j.out
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00

set -euo pipefail

REPO_DIR="/media/vhviet03/Image-Retrieval---Thesis-2026"
DATA_DIR="/media/vhviet03/NIH"
SAVE_DIR="${REPO_DIR}/checkpoints/nih_dinov2_weighted"

cd "${REPO_DIR}"
mkdir -p logs "${SAVE_DIR}"

source ~/.bashrc
conda activate thesis

nvidia-smi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python train.py \
  --dataset nih \
  --model dinov2 \
  --loss-name weighted_multilabel_triplet \
  --dataset-dir "${DATA_DIR}" \
  --val-dataset-dir "${DATA_DIR}" \
  --train-image-list "${REPO_DIR}/nih/train_val_list.txt" \
  --val-image-list "${REPO_DIR}/nih/test_list.txt" \
  --nih-labels-csv "${DATA_DIR}/Data_Entry_2017.csv" \
  --embedding-dim 256 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --epochs 20 \
  --lr 0.0001 \
  --save-dir "${SAVE_DIR}"
