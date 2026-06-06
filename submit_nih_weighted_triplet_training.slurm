#!/bin/bash
#SBATCH --job-name=nih_densenet_weighted
#SBATCH --output=/media/vhviet03/Image-Retrieval---Thesis-2026/nih_densenet_weighted_%j.out
#SBATCH --error=/media/vhviet03/Image-Retrieval---Thesis-2026/nih_densenet_weighted_%j.out
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00

set -euo pipefail

REPO_DIR="/media/vhviet03/Image-Retrieval---Thesis-2026"
DATA_DIR="/media/vhviet03/Dataset"
SAVE_DIR="${REPO_DIR}/checkpoints/nih_densenet121_weighted_multilabel_triplet"

cd "${REPO_DIR}"
mkdir -p "${SAVE_DIR}"

source ~/.bashrc
conda activate thesis

nvidia-smi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python train.py \
  --dataset nih \
  --dataset-dir "${DATA_DIR}" \
  --model densenet121 \
  --loss-name weighted_multilabel_triplet \
  --workers "${SLURM_CPUS_PER_TASK:-8}" \
  --save-dir "${SAVE_DIR}"
