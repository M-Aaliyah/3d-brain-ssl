#!/usr/bin/env python
"""
Inference and evaluation of FOMO2JOMO finetuned on the BraTS-MEN test set.

Data is already preprocessed (.npy + .pkl from preprocess_brats_men_yucca.py).

  .npy shape: (5, D, H, W)
  Channels 0-3 are the four BraTS modalities.
  Channel 4 is the segmentation label.

  .pkl key 'crop_to_nonzero' = [[x0,x1], [y0,y1], [z0,z1]]
  This is the bounding box saved by preprocess_bratsmen_yucca.py after
  cropping the background. x1, y1, and z1 are exclusive-end indices.

Pipeline:
  1. Load preprocessed .npy (shape 5,D,H,W): channels 0-3 are modalities,
     and channel 4 is the segmentation label.
  2. Run mmunetvae sliding window via model.model.predict().
  3. Apply keep_largest_component postprocessing per class.
  4. Unpad prediction and GT from cropped space to the original
     240×240×155 space. This mirrors preprocess_bratsmen_yucca.py
     crop_to_box exactly:
       out[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1],
           bbox[2][0]:bbox[2][1]]
     This enables direct metric comparison with nnUNet in the original
     image space.
  5. Save as .nii.gz using affine=np.eye(4), giving 1mm isotropic spacing
     for BraTS HD95.
  6. Call evaluate_brats_men.py with --pred_dir, --gt_dir, and --output_dir.
     This runs get_LesionWiseResults(challenge_name='BraTS-MEN'), the same
     metric used for nnUNet.

All BraTS-MEN cases are 240×240×155.
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

from models.fomo2jomo_supervised_seg import FOMO2JOMOSegModel


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


def unpad_to_original(pred: np.ndarray, pkl_path: str) -> np.ndarray:
    """
    Place cropped prediction back into the original 240×240×155 image space.

    Mirrors preprocess_bratsmen_yucca.py crop_to_box exactly:
      crop_to_box: vol[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1],
                       bbox[2][0]:bbox[2][1]]

    bbox format: [[x0,x1], [y0,y1], [z0,z1]]
    These are direct exclusive-end slice indices.
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


def run_inference(model, case_id: str, data_dir: str, pred_dir: str,
                  gt_dir: str, patch_size: tuple, overlap: float,
                  device: torch.device):
    npy_path = os.path.join(data_dir, case_id + ".npy")
    pkl_path = os.path.join(data_dir, case_id + ".pkl")

    data = np.load(npy_path).astype(np.float32)
    image = data[:4] # Four BraTS modalities
    label = data[4] # GT segmentation in cropped space

    data_tensor = torch.from_numpy(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model.model.predict(
            mode="3D",
            data=data_tensor,
            patch_size=patch_size,
            overlap=overlap,
            sliding_window_prediction=True,
            mirror=False,
            device=device,
        )

    pred = torch.argmax(logits[0], dim=0).cpu().numpy().astype(np.uint8)
    pred = keep_largest_component(pred)

    # Unpad pred and GT to the original 240×240×155 space for evaluation
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
        description="Inference + evaluation of FOMO2JOMO on BraTS-MEN test set"
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--splits_json", type=str, required=True)
    parser.add_argument("--finetune_ckpt", type=str, required=True)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    pred_dir = args.output_dir
    gt_dir = args.output_dir.rstrip("/") + "_gt"
    for d in [pred_dir, gt_dir, args.results_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size = (args.patch_size,) * 3

    print(f"Loading finetuned model from: {args.finetune_ckpt}")
    model = FOMO2JOMOSegModel.load_from_checkpoint(
        args.finetune_ckpt,
        map_location=device,
    )
    model.eval()
    model.to(device)

    case_ids = load_split_cases(args.splits_json, args.split)
    if not case_ids:
        raise ValueError(f"No '{args.split}' cases found in {args.splits_json}")
    print(f"Running inference on {len(case_ids)} '{args.split}' cases ...")

    for case_id in tqdm(case_ids, desc="Inference"):
        out_path = os.path.join(pred_dir, case_id + ".nii.gz")
        if os.path.exists(out_path):
            continue # Resume-safe
        run_inference(
            model=model,
            case_id=case_id,
            data_dir=args.data_dir,
            pred_dir=pred_dir,
            gt_dir=gt_dir,
            patch_size=patch_size,
            overlap=args.overlap,
            device=device,
        )

    print("\nRunning BraTS-MEN evaluation ...")
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

    print("\nEvaluation complete.")
    print(f"Predictions: {pred_dir}")
    print(f"Results: {args.results_dir}/brats_men_metrics.csv")
    print(f"Results: {args.results_dir}/brats_men_summary.csv")


if __name__ == "__main__":
    main()