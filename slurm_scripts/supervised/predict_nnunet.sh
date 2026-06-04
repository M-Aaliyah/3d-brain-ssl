#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/am525/3d-brain-ssl/slurm_logs/supervised/nnunet_inference.%N.%j.log
#SBATCH --time=10-00:00:00
#SBATCH --job-name=nnunet_inference

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

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONFAULTHANDLER=1

INPUT_FOLDER="$nnUNet_raw/Dataset001_BraTSMEN/imagesTs"

OUTPUT_FOLDER="$REPO/data/nnunet_predictions"
OUTPUT_FOLDER_PP="$REPO/data/nnunet_predictions_postprocessed"

POSTPROCESSING_PKL="$nnUNet_results/Dataset001_BraTSMEN/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl"

PLANS_JSON="$nnUNet_results/Dataset001_BraTSMEN/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json"

mkdir -p "$OUTPUT_FOLDER"
mkdir -p "$OUTPUT_FOLDER_PP"

echo "=== Environment ==="
hostname
date

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi

echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"

echo "=== Starting inference ==="

nnUNetv2_predict \
    -d Dataset001_BraTSMEN \
    -i "$INPUT_FOLDER" \
    -o "$OUTPUT_FOLDER" \
    -f 0 1 2 3 4 \
    -tr nnUNetTrainer \
    -c 3d_fullres \
    -p nnUNetPlans

echo "=== Inference completed ==="

echo "=== Applying postprocessing ==="

nnUNetv2_apply_postprocessing \
    -i "$OUTPUT_FOLDER" \
    -o "$OUTPUT_FOLDER_PP" \
    -pp_pkl_file "$POSTPROCESSING_PKL" \
    -np 8 \
    -plans_json "$PLANS_JSON"

echo "=== Postprocessing completed ==="
date