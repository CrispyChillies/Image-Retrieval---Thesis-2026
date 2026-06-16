#!/bin/bash
#SBATCH --job-name=nih_dinov2_weighted_triplet
#SBATCH --output=/media/vhviet03/Image-Retrieval---Thesis-2026/slurm-%j.out
#SBATCH --error=/media/vhviet03/Image-Retrieval---Thesis-2026/slurm-%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -euo pipefail

REPO_DIR="/media/vhviet03/Image-Retrieval---Thesis-2026"

cd "${REPO_DIR}"

source ~/.bashrc
conda activate thesis

nvidia-smi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python train.py \
  --dataset nih \
  --dataset-dir "/media/vhviet03/Dataset/train" \
  --val-dataset-dir "/media/vhviet03/Dataset/gallery" \
  --model dinov2 \
  --loss-name weighted_triplet_loss \
  --epochs 30 \
  --eval-max-samples 10296 \
  --save-dir "results/nih_dinov2_weighted_triplet" \
  --eval-freq 2 \
  --batch-size 24 \
  --lr 1e-4 \
  --workers 4
