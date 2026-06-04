#!/bin/bash
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/am525/3d-brain-ssl/slurm_logs/supervised/nnunet_preprocess.%N.%j.log
#SBATCH --time=0-12:00:00
#SBATCH --job-name=nnunet_preprocess

set -euo pipefail

REPO=/vol/biomedic2/bglocker_studproj/am525/3d-brain-ssl

CONDA_ROOT=/vol/bitbucket/am525/miniconda3
ENV_NAME=3d-brain-ssl-env

export MKL_INTERFACE_LAYER=GNU
export PYTHONPATH="$REPO/src:$REPO:$PYTHONPATH"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export nnUNet_raw="$REPO/data/nnUNet_raw"
export nnUNet_preprocessed="$REPO/data/nnUNet_preprocessed"
export nnUNet_results="$REPO/data/nnUNet_results"

mkdir -p "$nnUNet_preprocessed"
mkdir -p "$nnUNet_results"
mkdir -p "$REPO/slurm_logs/supervised"

export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1

echo "=== Environment ==="
hostname
date

echo "=== CUDA ==="
nvidia-smi || true

echo "=== nnUNet paths ==="
echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"

echo "=== Starting preprocessing ==="

nnUNetv2_plan_and_preprocess \
    -d 1 \
    --verify_dataset_integrity

echo "=== Done ==="
date