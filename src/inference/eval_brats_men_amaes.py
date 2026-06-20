#!/usr/bin/env python
"""
Inference and evaluation for the AMAES finetuned model on BraTS-MEN.

Loads from brats_men_yucca_preprocessed (.npy + .pkl), the same preprocessed
files used during finetuning. This ensures train/eval preprocessing
consistency.

Pipeline:
  1. Load preprocessed .npy (shape 5,D,H,W): channels 0-3 are modalities,
     channel 4 is the segmentation label.
  2. Run sliding window via model.model.predict().
  3. Apply keep_largest_component postprocessing per class (matching
     FOMO2JOMO).
  4. Unpad prediction AND GT from cropped space to original 240×240×155 image space using the
     .pkl key 'crop_to_nonzero' = [[x0,x1], [y0,y1], [z0,z1]] records where
     preprocess_bratsmen_yucca.py cropped the background. Mirrors crop_to_box:
       out[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1],
           bbox[2][0]:bbox[2][1]]
     Enables direct metric comparison with nnUNet (original image space).
  5. Save as .nii.gz (affine=np.eye(4) 1mm isotropic) for both pred and GT.
  6. Call evaluate_brats_men.py (--pred_dir, --gt_dir, --output_dir), which
     runs get_LesionWiseResults(challenge_name='BraTS-MEN'), the same metric
     as nnUNet.

All BraTS-MEN cases are 240x240x155.
"""

import os
import sys
import json
import argparse
import cc3d
import subprocess
from pathlib import Path

import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import load_pickle

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for p in [SRC_ROOT, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.supervised_seg import SupervisedSegModel


BRATS_MEN_ORIGINAL_SIZE = (240, 240, 155)


def load_split_cases(splits_json: str, split: str) -> list:
    with open(splits_json) as f:
        splits = json.load(f)
    if "train" in splits and "val" in splits:
        return splits[split]
    elif "splits" in splits:
        if split == "test":
            return splits.get("test", [])
        return splits["splits"][0][split]
    raise ValueError(f"Unrecognised splits JSON format in {splits_json}")


def keep_largest_component(pred: np.ndarray) -> np.ndarray:
    """Retain only the largest connected component per class."""
    out = np.zeros_like(pred)
    for cls in [1, 2, 3]:
        mask = (pred == cls).astype(np.uint8)
        if mask.sum() == 0:
            continue
        labels = cc3d.connected_components(mask, connectivity=26)
        if labels.max() == 0:
            continue
        largest = np.argmax(np.bincount(labels.flat)[1:]) + 1
        out[labels == largest] = cls
    return out


def unpad_to_original(pred: np.ndarray, pkl_path: str) -> np.ndarray:
    """
    Place cropped prediction back into original 240×240×155 image space.

    Mirrors preprocess_bratsmen_yucca.py crop_to_box exactly:
      crop_to_box: vol[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1],
                       bbox[2][0]:bbox[2][1]]
    bbox format: [[x0,x1], [y0,y1], [z0,z1]]
    Direct exclusive-end slice indices.
    """
    props = load_pickle(pkl_path)
    bbox = props["crop_to_nonzero"] # [[x0,x1], [y0,y1], [z0,z1]]

    out = np.zeros(BRATS_MEN_ORIGINAL_SIZE, dtype=pred.dtype)
    out[
        bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1],
    ] = pred
    return out


def run_case(model, case_id: str, data_dir: str, pred_dir: str, gt_dir: str,
             patch_size: tuple, overlap: float, device: torch.device):
    npy_path = os.path.join(data_dir, case_id + ".npy")
    pkl_path = os.path.join(data_dir, case_id + ".pkl")

    data = np.load(npy_path).astype(np.float32) # (5, D, H, W)
    image = data[:4] 
    label = data[4] # GT segmentation in cropped space

    data_tensor = torch.from_numpy(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model.model.predict(
            data=data_tensor,
            mode="3D",
            mirror=False,
            overlap=overlap,
            patch_size=patch_size,
            sliding_window_prediction=True,
            device=device,
        )

    pred = torch.argmax(logits[0], dim=0).cpu().numpy().astype(np.uint8)
    pred = keep_largest_component(pred)

    # Unpad both pred and GT from cropped space to original 240×240×155
    pred = unpad_to_original(pred, pkl_path)
    label = unpad_to_original(label.astype(np.uint8), pkl_path)

    # BraTS is 1mm isotropic
    affine = np.eye(4)
    nib.save(
        nib.Nifti1Image(pred, affine),
        os.path.join(pred_dir, case_id + ".nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(label, affine),
        os.path.join(gt_dir, case_id + ".nii.gz"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="AMAES inference + BraTS-MEN evaluation from preprocessed .npy"
    )
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--splits_json", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )

    args = parser.parse_args()

    pred_dir = args.output_dir
    gt_dir = args.output_dir.rstrip("/") + "_gt"
    for d in [pred_dir, gt_dir, args.results_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size = (args.patch_size,) * 3

    print(f"Loading model from: {args.ckpt_path}")
    model = SupervisedSegModel.load_from_checkpoint(
        args.ckpt_path,
        map_location=device,
    )
    model.eval()
    model.to(device)

    case_ids = load_split_cases(args.splits_json, args.split)
    print(f"Running inference on {len(case_ids)} '{args.split}' cases")

    for case_id in tqdm(case_ids, desc="Inference"):
        out_path = os.path.join(pred_dir, case_id + ".nii.gz")
        if os.path.exists(out_path):
            continue
        run_case(
            model=model,
            case_id=case_id,
            data_dir=args.data_dir,
            pred_dir=pred_dir,
            gt_dir=gt_dir,
            patch_size=patch_size,
            overlap=args.overlap,
            device=device,
        )

    print(f"\nPredictions saved to: {pred_dir}")
    print("Running BraTS-MEN evaluation ...")

    eval_script = os.path.join(SRC_ROOT, "eval", "evaluate_brats_men.py")
    subprocess.run(
        [
            sys.executable,
            eval_script,
            "--pred_dir",
            pred_dir,
            "--gt_dir",
            gt_dir,
            "--output_dir",
            args.results_dir,
        ],
        check=True,
    )

    print(f"\nResults: {args.results_dir}/brats_men_summary.csv")


if __name__ == "__main__":
    main()
