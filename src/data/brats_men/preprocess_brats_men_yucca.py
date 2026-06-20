#!/usr/bin/env python
"""
Preprocess BraTS-MEN data into yucca format for SSL finetuning.

Yucca format per case:
  <case_id>.npy  — shape (C+1, D, H, W), dtype float32
                   C modality channels stacked, label appended last:
                   [T1, T1ce, T2, T2FLAIR, seg_label]
  <case_id>.pkl  — dict with foreground_locations, crop_to_nonzero,
                   size_before_transpose, size_after_transpose etc.

BraTS-MEN label convention:
  0=background, 1=NCR, 2=ED, 3=ET

Preprocessing steps:
  1. Load 4 modalities + seg as NIfTI
  2. Reorient to RAS
  3. Crop to nonzero bounding box
  4. Resample to 1mm isotropic spacing
  5. Z-normalise each modality independently (volume_wise_znorm on brain mask)
  6. Stack as (5, D, H, W), save as .npy
  7. Compute foreground locations, save metadata as .pkl

Output is used by both AMAES and FOMO2JOMO finetuning. Run once only.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import nibabel as nib
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import (
    save_pickle,
    maybe_mkdir_p as ensure_dir_exists,
)
from skimage.transform import resize

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_ROOT  = os.path.join(REPO_ROOT, "src")
for p in [SRC_ROOT, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data.brats_men.brats_men_config import bratsmen_config

# BraTS-MEN modality suffixes, order matches bratsmen_config["modalities"]
MODALITY_SUFFIXES = ["-t1n.nii.gz", "-t1c.nii.gz", "-t2w.nii.gz", "-t2f.nii.gz"]
SEG_SUFFIX        = "-seg.nii.gz"


def get_nib_spacing(nib_img):
    return np.abs(np.diag(nib_img.affine)[:3]).tolist()


def reorient_to_ras(nib_img):
    ras_ornt     = nib.orientations.axcodes2ornt(("R", "A", "S"))
    current_ornt = nib.orientations.io_orientation(nib_img.affine)
    transform    = nib.orientations.ornt_transform(current_ornt, ras_ornt)
    return nib_img.as_reoriented(transform)


def resample_volume(vol, original_spacing, target_spacing=(1., 1., 1.), order=3):
    original_spacing = np.array(original_spacing)
    target_spacing   = np.array(target_spacing)
    original_size    = np.array(vol.shape)
    target_size      = np.round(original_size * original_spacing / target_spacing).astype(int)
    target_size      = np.maximum(target_size, 1)
    if np.all(original_size == target_size):
        return vol.astype(np.float32)
    return resize(
        vol.astype(np.float32), target_size,
        order=order, mode="edge", cval=0,
        anti_aliasing=False, preserve_range=True,
    ).astype(np.float32)


def volume_wise_znorm(vol, mask=None):
    if mask is not None and mask.sum() > 0:
        mean, std = vol[mask].mean(), vol[mask].std()
    else:
        mean, std = vol.mean(), vol.std()
    return (vol - mean) / std if std > 1e-8 else vol - mean


def get_bbox_for_foreground(vol, background_val=0):
    nz = np.where(vol != background_val)
    if len(nz[0]) == 0:
        return [[0, vol.shape[0]], [0, vol.shape[1]], [0, vol.shape[2]]]
    return [[nz[i].min(), nz[i].max() + 1] for i in range(3)]


def crop_to_box(vol, bbox):
    return vol[bbox[0][0]:bbox[0][1],
               bbox[1][0]:bbox[1][1],
               bbox[2][0]:bbox[2][1]]


def get_foreground_locations(label, max_locs=100000):
    locs = np.array(np.nonzero(label)).T[::10].tolist()
    if len(locs) > max_locs:
        locs = locs[::round(len(locs) / max_locs)]
    return {"1": locs} if locs else {}


def preprocess_case(case_dir: Path, output_dir: Path,
                    target_spacing=(1., 1., 1.)):
    case_id = case_dir.name

    # Load modalities
    modality_paths = [case_dir / (case_id + s) for s in MODALITY_SUFFIXES]
    for p in modality_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    seg_path  = case_dir / (case_id + SEG_SUFFIX)
    has_label = seg_path.exists()

    nib_imgs = [reorient_to_ras(nib.load(str(p))) for p in modality_paths]
    spacing  = get_nib_spacing(nib_imgs[0])
    vols     = [img.get_fdata(dtype=np.float32) for img in nib_imgs]

    if has_label:
        seg = reorient_to_ras(nib.load(str(seg_path))).get_fdata(dtype=np.float32)
    else:
        seg = np.zeros_like(vols[0])

    # Brain mask is the union of nonzero across modalities
    brain_mask = np.zeros_like(vols[0], dtype=bool)
    for v in vols:
        brain_mask |= (v != 0)

    # Crop to nonzero
    bbox = get_bbox_for_foreground(brain_mask.astype(np.float32))
    vols = [crop_to_box(v, bbox) for v in vols]
    seg  = crop_to_box(seg,  bbox)
    size_before = list(vols[0].shape)

    # Resample
    vols = [resample_volume(v, spacing, target_spacing, order=3) for v in vols]
    seg  = resample_volume(seg, spacing, target_spacing, order=1)  # order=1 matches nnUNet
    seg  = np.clip(np.round(seg), 0, bratsmen_config["num_classes"] - 1).astype(np.float32)
    size_after = list(vols[0].shape)

    # Recompute brain mask from resampled volumes which avoids shape mismatch that occurs when resampling the mask separately (different rounding)
    brain_mask_rs = np.zeros_like(vols[0], dtype=bool)
    for v in vols:
        brain_mask_rs |= (v != 0)

    # Z-normalise each modality independently
    vols = [volume_wise_znorm(v, mask=brain_mask_rs) for v in vols]

    # Stack (C+1, D, H, W)
    stacked = np.stack(vols + [seg], axis=0).astype(np.float32)

    metadata = {
        "foreground_locations":  get_foreground_locations(seg),
        "crop_to_nonzero":       bbox,
        "size_before_transpose": size_before,
        "size_after_transpose":  size_after,
        "original_spacing":      spacing,
        "target_spacing":        list(target_spacing),
        "case_id":               case_id,
        "has_label":             has_label,
    }

    np.save(str(output_dir / (case_id + ".npy")), stacked)
    save_pickle(metadata, str(output_dir / (case_id + ".pkl")))
    return case_id


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess BraTS-MEN into yucca .npy+.pkl format"
    )
    parser.add_argument("--data_dir",    type=str, required=True,
                        help="BraTS-MEN raw data root (one folder per case)")
    parser.add_argument("--output_dir",  type=str, required=True,
                        help="Output directory for .npy + .pkl files")
    parser.add_argument("--splits_json", type=str, default=None,
                        help="Optional: only process cases in this splits JSON")
    parser.add_argument("--target_spacing", type=float, nargs=3,
                        default=[1.0, 1.0, 1.0])
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Parallel workers (default 8, use 1 for sequential)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ensure_dir_exists(str(output_dir))
    target_spacing = tuple(args.target_spacing)

    if args.splits_json is not None:
        with open(args.splits_json) as f:
            splits = json.load(f)
        all_ids = set()
        for key in ("train", "val", "test"):
            all_ids.update(splits.get(key, []))
        if "splits" in splits:
            for fold in splits["splits"]:
                all_ids.update(fold.get("train", []))
                all_ids.update(fold.get("val", []))
            all_ids.update(splits.get("test", []))
        case_dirs = sorted([data_dir / cid for cid in all_ids
                            if (data_dir / cid).is_dir()])
    else:
        case_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    # Skip already-done cases
    pending = [d for d in case_dirs
               if not (output_dir / (d.name + ".npy")).exists()]
    print(f"Found {len(case_dirs)} cases | {len(pending)} to process | Output: {output_dir}")

    def _worker(case_dir):
        try:
            preprocess_case(case_dir, output_dir, target_spacing)
            return (case_dir.name, None)
        except Exception as e:
            return (case_dir.name, str(e))

    errors = []
    if args.num_workers <= 1:
        for case_dir in tqdm(pending, desc="Preprocessing"):
            _, err = _worker(case_dir)
            if err:
                logging.error(f"Failed {case_dir.name}: {err}")
                errors.append((case_dir.name, err))
    else:
        from multiprocessing import Pool
        with Pool(processes=args.num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(_worker, pending),
                total=len(pending),
                desc=f"Preprocessing ({args.num_workers} workers)",
            ))
        errors = [(cid, err) for cid, err in results if err is not None]
        for cid, err in errors:
            logging.error(f"Failed {cid}: {err}")

    print(f"\nDone. {len(case_dirs) - len(errors)}/{len(case_dirs)} cases preprocessed.")
    if errors:
        print(f"Errors ({len(errors)}):")
        for case_id, err in errors[:20]:
            print(f"  {case_id}: {err}")


if __name__ == "__main__":
    main()