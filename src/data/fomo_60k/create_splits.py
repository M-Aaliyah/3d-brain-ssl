"""
Generates and saves FOMO60K train/val splits to a JSON file so all future SSL pretraining experiments
use the exact same split.

Usage (run once):
    python src/data/fomo_60k/create_splits.py \
        --data_dir ../data/fomo-60k-preprocessed/FOMO60k \
        --output_path artifacts/splits/fomo60k_splits.json \
        --val_ratio 0.01 \
        --seed 42
"""

import argparse
import json
import os
import random
from pathlib import Path


def get_case_ids(data_dir: str) -> list[str]:
    """
    Collect all case IDs from the FOMO60K preprocessed directory.
    FOMO60K preprocessed files are .npy arrays; each case is one file
    (single-channel, already cropped/normalised by the fomo25 pipeline).
    We strip the extension to get the case ID.
    """
    data_path = Path(data_dir)
    # Collect .npy files (preprocessed volumes)
    files = sorted(data_path.glob("*.npy"))
    if not files:
        # Some preprocessed datasets store files in sub-folders, try one level deep
        files = sorted(data_path.rglob("*.npy"))
    if not files:
        raise FileNotFoundError(
            f"No .npy files found under {data_dir}. "
            "Check that the FOMO60K preprocessed directory is correct."
        )
    # Case ID = filename without extension
    case_ids = [f.stem for f in files]
    return case_ids


def create_splits(
    data_dir: str,
    output_path: str,
    val_ratio: float = 0.01,
    seed: int = 42,
) -> dict:
    """
    Split FOMO60K cases into train and val.

    Parameters:
    data_dir    : path to /data/fomo-60k-preprocessed/FOMO60K
    output_path : where to save the JSON  (e.g. artifacts/splits/fomo60k_splits.json)
    val_ratio   : fraction of data for validation (fomo25 baseline uses 0.01 = 1%)
    seed        : random seed for reproducibility

    Returns:
    splits dict saved to JSON and returned
    """
    case_ids = get_case_ids(data_dir)
    print(f"Found {len(case_ids)} cases in {data_dir}")

    random.seed(seed)
    shuffled = case_ids.copy()
    random.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_ratio))
    val_cases = shuffled[:n_val]
    train_cases = shuffled[n_val:]

    splits = {
        "metadata": {
            "data_dir": str(data_dir),
            "total_cases": len(case_ids),
            "num_train": len(train_cases),
            "num_val": len(val_cases),
            "val_ratio": val_ratio,
            "seed": seed,
            "description": (
                "FOMO60K pretrain/val split. "
                "Train is used for SSL pretraining. "
                "Val (1%) is used for reconstruction loss monitoring. "
                "Fixed split – reuse for all SSL methods for fair comparison."
            ),
        },
        "train": sorted(train_cases),
        "val": sorted(val_cases),
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\nSplit summary:")
    print(f"  Train : {len(train_cases)} cases")
    print(f"  Val   : {len(val_cases)} cases")
    print(f"  Saved : {out_path}")
    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create and save FOMO60K train/val splits."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/fomo-60k-preprocessed/FOMO60K",
        help="Path to the preprocessed FOMO60K directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="artifacts/splits/fomo60k_splits.json",
        help="Where to save the splits JSON",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.01,
        help="Fraction of data for validation (default 0.01 = 1%%, matching fomo25 baseline)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()
    create_splits(
        data_dir=args.data_dir,
        output_path=args.output_path,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
