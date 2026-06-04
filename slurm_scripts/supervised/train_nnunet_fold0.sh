#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/am525/3d-brain-ssl/slurm_logs/supervised/train_nnunet_fold0.%N.%j.log
#SBATCH --time=20-00:00:00
#SBATCH --job-name=train_nnunet_fold0

set -eo pipefail

REPO=/vol/biomedic2/bglocker_studproj/am525/3d-brain-ssl

CONDA_ROOT=/vol/bitbucket/am525/miniconda3
ENV_NAME=3d-brain-ssl-env

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export nnUNet_compile=false

export PYTHONPATH="$REPO/src:$REPO:$PYTHONPATH"
export nnUNet_raw="$REPO/data/nnUNet_raw"
export nnUNet_preprocessed="$REPO/data/nnUNet_preprocessed"
export nnUNet_results="$REPO/data/nnUNet_results"

if [[ -f "$HOME/.wandb_env" ]]; then
    source "$HOME/.wandb_env"
fi

export nnUNet_wandb_enabled=1
export nnUNet_wandb_project=3d-brain-ssl
export nnUNet_wandb_mode=online
export WANDB_NAME=nnunet_bratsmen_fold0
export WANDB_DIR="$REPO/wandb"

mkdir -p "$WANDB_DIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONFAULTHANDLER=1

echo "=== Environment ==="
hostname
date

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi

echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"

echo "=== CUDA test ==="

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

echo "=== Starting nnUNet training ==="

nnUNetv2_train \
    1 \
    3d_fullres \
    0 \
    --npz

echo "=== Training finished ==="
date