"""
Loads pre-saved FOMO60K splits from JSON and returns a splits config object
compatible with PretrainDataModule (same interface as fomo25's pretrain_split).

This is so that all SSL experiments use the exact same FOMO60K train/val split.
"""

import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class FOMO60KSplitsConfig:
    """
    Minimal splits config compatible with PretrainDataModule.
    The train/val splits are pre-saved to a JSON file (artifacts/splits/fomo60k_splits.json).
    """
    _train: list[str]
    _val: list[str]

    def train(self, idx: int = 0) -> list[str]:
        return self._train

    def val(self, idx: int = 0) -> list[str]:
        return self._val


def load_fomo60k_splits(splits_json_path: str) -> FOMO60KSplitsConfig:
    """
    Load FOMO60K splits from the pre-saved JSON file.

    Parameters:
    splits_json_path : path to artifacts/splits/fomo60k_splits.json

    Returns:
    FOMO60KSplitsConfig (has .train(0) and .val(0) methods)
    """
    path = Path(splits_json_path)
    if not path.exists():
        raise FileNotFoundError(
            f"FOMO60K splits file not found at {splits_json_path}.\n"
            f"Run:  python src/data/fomo_60k/create_splits.py  first."
        )
    with open(path, "r") as f:
        splits = json.load(f)

    train_cases = splits["train"]
    val_cases = splits["val"]
    meta = splits.get("metadata", {})

    print(
        f"Loaded FOMO60K splits from {splits_json_path}\n"
        f"  Train: {len(train_cases)} | Val: {len(val_cases)} "
        f"(seed={meta.get('seed', '?')}, val_ratio={meta.get('val_ratio', '?')})"
    )
    return FOMO60KSplitsConfig(_train=train_cases, _val=val_cases)
