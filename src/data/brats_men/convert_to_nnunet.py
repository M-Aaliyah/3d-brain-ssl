"""
Convert BraTS-MEN dataset into nnUNetv2 format.

Creates:
    nnUNet_raw/
        Dataset001_BraTSMEN/
            imagesTr/
            labelsTr/
            dataset.json

Uses predefined patient-level splits.

Modalities:
    0000 -> t1n
    0001 -> t1c
    0002 -> t2w
    0003 -> t2f

Segmentation labels:
    0 -> background
    1 -> NETC
    2 -> SNFH
    3 -> ET
"""

import json
import shutil
import argparse
from pathlib import Path

from tqdm import tqdm


MODALITY_MAP = {
    "t1n": "0000",
    "t1c": "0001",
    "t2w": "0002",
    "t2f": "0003",
}


def load_splits(split_json_path):
    with open(split_json_path, "r") as f:
        return json.load(f)


def find_modality_file(case_dir, keyword):
    matches = sorted(case_dir.glob(f"*{keyword}*.nii.gz"))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Missing modality '{keyword}' in {case_dir}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple files found for '{keyword}' in {case_dir}"
        )

    return matches[0]


def find_segmentation_file(case_dir):
    matches = sorted(case_dir.glob("*seg*.nii.gz"))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Missing segmentation in {case_dir}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple segmentation files found in {case_dir}"
        )

    return matches[0]


def copy_case(case_id, source_dir, imagesTr, labelsTr):
    case_dir = source_dir / case_id

    for modality_name, modality_idx in MODALITY_MAP.items():
        src = find_modality_file(case_dir, modality_name)

        dst = imagesTr / (
            f"{case_id}_{modality_idx}.nii.gz"
        )

        shutil.copy2(src, dst)

    seg_src = find_segmentation_file(case_dir)

    seg_dst = labelsTr / f"{case_id}.nii.gz"

    shutil.copy2(seg_src, seg_dst)


def create_dataset_json(dataset_dir, num_training_cases):
    dataset_json = {
        "channel_names": {
            "0": "t1n",
            "1": "t1c",
            "2": "t2w",
            "3": "t2f",
        },
        "labels": {
            "background": 0,
            "NETC": 1,
            "SNFH": 2,
            "ET": 3,
        },
        "numTraining": num_training_cases,
        "file_ending": ".nii.gz",
    }

    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)


def main(args):
    source_dir = Path(args.data_dir)

    splits = load_splits(args.split_json)

    train_cases = splits["train"]
    val_cases = splits["val"]

    all_training_cases = (
        train_cases + val_cases
    )

    dataset_dir = (
        Path(args.nnunet_raw)
        / "Dataset001_BraTSMEN"
    )

    imagesTr = dataset_dir / "imagesTr"
    labelsTr = dataset_dir / "labelsTr"

    imagesTr.mkdir(
        parents=True,
        exist_ok=True,
    )

    labelsTr.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("Converting BraTS-MEN to nnUNet format")
    print("=" * 80)

    for case_id in tqdm(all_training_cases):
        copy_case(
            case_id=case_id,
            source_dir=source_dir,
            imagesTr=imagesTr,
            labelsTr=labelsTr,
        )

    create_dataset_json(
        dataset_dir=dataset_dir,
        num_training_cases=len(
            all_training_cases
        ),
    )

    print("\nDone.")
    print(
        f"\nDataset written to:\n{dataset_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to BraTS-MEN TrainData",
    )

    parser.add_argument(
        "--split_json",
        type=str,
        required=True,
        help="Path to split JSON",
    )

    parser.add_argument(
        "--nnunet_raw",
        type=str,
        required=True,
        help="Path to nnUNet_raw",
    )

    args = parser.parse_args()

    main(args)
