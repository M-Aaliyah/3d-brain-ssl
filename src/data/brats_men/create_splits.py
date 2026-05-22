"""
Create reproducible BraTS-MEN patient-level train/val/test splits.

This script performs patient-level splitting to prevent data leakage of scans from the same patient across splits.

Example:
    BraTS-MEN-00074-000
    BraTS-MEN-00074-001
    BraTS-MEN-00074-002

All belong to patient:
    BraTS-MEN-00074

These should stay in the same split.

This script:
1. Scans BraTS-MEN TrainData
2. Verifies required modality + segmentation files exist
3. Groups scans by patient
4. Creates reproducible patient-level splits
5. Saves:
    - JSON split file
    - CSV spreadsheets
    - XLSX spreadsheets

Example:
python create_splits.py \
    --data_dir /vol/biodata/data/BraTS_2023_MEN/TrainData \
    --output_dir artifacts/splits \
    --seed 42
"""

import os
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd


REQUIRED_MODALITIES = {
    "t1n": "t1n",
    "t1c": "t1c",
    "t2w": "t2w",
    "t2f": "t2f",
}

SEGMENTATION_KEYWORDS = [
    "seg",
]


def find_file(files, keyword):
    """
    Find file containing keyword.
    """
    for f in files:
        if keyword.lower() in f.lower():
            return f
    return None


def extract_patient_id(case_id):
    """
    Extract patient ID.

    Example:
        BraTS-MEN-00074-003
        -> BraTS-MEN-00074
    """
    return "-".join(case_id.split("-")[:3])


def extract_session_id(case_id):
    """
    Extract session/scan ID.

    Example:
        BraTS-MEN-00074-003
        -> 003
    """
    return case_id.split("-")[-1]


def verify_case(case_dir):
    """
    Verify a BraTS-MEN case contains:
    - required modalities
    - segmentation

    Returns:
        dict with paths if valid
        None if invalid
    """
    files = os.listdir(case_dir)
    modality_paths = {}

    for modality_name, keyword in REQUIRED_MODALITIES.items():
        file_name = find_file(files, keyword)

        if file_name is None:
            return None

        modality_paths[modality_name] = os.path.join(case_dir, file_name)

    seg_path = None

    for keyword in SEGMENTATION_KEYWORDS:
        seg_file = find_file(files, keyword)

        if seg_file is not None:
            seg_path = os.path.join(case_dir, seg_file)
            break

    if seg_path is None:
        return None

    case_id = os.path.basename(case_dir)

    return {
        "case_id": case_id,
        "patient_id": extract_patient_id(case_id),
        "session_id": extract_session_id(case_id),
        "segmentation": seg_path,
        **modality_paths,
    }


def create_splits(
    data_dir,
    output_dir,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    seed=42,
):
    """
    Create reproducible patient-level dataset splits.
    """
    random.seed(seed)

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Scanning BraTS-MEN dataset...")
    print("=" * 80)

    valid_cases = []
    invalid_cases = []

    for case_name in sorted(os.listdir(data_dir)):
        case_dir = data_dir / case_name

        if not case_dir.is_dir():
            continue

        case_info = verify_case(case_dir)

        if case_info is None:
            invalid_cases.append(case_name)
            continue

        valid_cases.append(case_info)

    print(f"Valid cases   : {len(valid_cases)}")
    print(f"Invalid cases : {len(invalid_cases)}")

    if len(invalid_cases) > 0:
        invalid_df = pd.DataFrame({
            "invalid_case_id": invalid_cases
        })
        invalid_df.to_csv(
            output_dir / "invalid_cases.csv",
            index=False,
        )

    patient_to_cases = defaultdict(list)

    for case in valid_cases:
        patient_to_cases[case["patient_id"]].append(case)

    patient_ids = sorted(patient_to_cases.keys())

    print(f"\nUnique patients : {len(patient_ids)}")

    random.Random(seed).shuffle(patient_ids)

    n_patients = len(patient_ids)

    n_train_patients = int(train_ratio * n_patients)
    n_val_patients = int(val_ratio * n_patients)

    train_patient_ids = patient_ids[:n_train_patients]

    val_patient_ids = patient_ids[
        n_train_patients:n_train_patients + n_val_patients
    ]

    test_patient_ids = patient_ids[
        n_train_patients + n_val_patients:
    ]

    train_cases = []
    val_cases = []
    test_cases = []

    for patient_id in train_patient_ids:
        train_cases.extend(patient_to_cases[patient_id])

    for patient_id in val_patient_ids:
        val_cases.extend(patient_to_cases[patient_id])

    for patient_id in test_patient_ids:
        test_cases.extend(patient_to_cases[patient_id])

    train_ids = sorted([x["case_id"] for x in train_cases])
    val_ids = sorted([x["case_id"] for x in val_cases])
    test_ids = sorted([x["case_id"] for x in test_cases])

    print("\n")
    print("=" * 80)
    print("Split Summary")
    print("=" * 80)

    print(f"Train patients : {len(train_patient_ids)}")
    print(f"Val patients   : {len(val_patient_ids)}")
    print(f"Test patients  : {len(test_patient_ids)}")

    print()

    print(f"Train scans : {len(train_ids)}")
    print(f"Val scans   : {len(val_ids)}")
    print(f"Test scans  : {len(test_ids)}")

    train_patients = set(train_patient_ids)
    val_patients = set(val_patient_ids)
    test_patients = set(test_patient_ids)

    assert len(train_patients & val_patients) == 0
    assert len(train_patients & test_patients) == 0
    assert len(val_patients & test_patients) == 0

    print("\nNo patient leakage detected.")

    split_json = {
        "seed": seed,
        "train_patients": sorted(train_patient_ids),
        "val_patients": sorted(val_patient_ids),
        "test_patients": sorted(test_patient_ids),
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    with open(output_dir / "brats_men_splits.json", "w") as f:
        json.dump(split_json, f, indent=4)

    print("\nSaved split JSON.")

    save_split_dataframe(train_cases, output_dir, "train")
    save_split_dataframe(val_cases, output_dir, "val")
    save_split_dataframe(test_cases, output_dir, "test")

    print("\nSaved spreadsheets.")
    print("\nDone.")


def save_split_dataframe(cases, output_dir, split_name):
    df = pd.DataFrame(cases)

    df = df.sort_values(
        by=["patient_id", "session_id"]
    )

    csv_path = output_dir / f"{split_name}.csv"
    xlsx_path = output_dir / f"{split_name}.xlsx"

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    print(f"{split_name:<5} -> {len(df)} scans")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to BraTS-MEN TrainData",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save splits",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
    )

    args = parser.parse_args()

    total = (
        args.train_ratio
        + args.val_ratio
        + args.test_ratio
    )

    assert abs(total - 1.0) < 1e-6, \
        "train/val/test ratios must sum to 1."

    create_splits(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
