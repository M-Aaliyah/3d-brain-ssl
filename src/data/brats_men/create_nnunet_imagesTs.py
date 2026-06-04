"""
Create nnUNetv2 imagesTs folder from predefined test split.

Creates:
    nnUNet_raw/
        Dataset001_BraTSMEN/
            imagesTs/

Modalities:
    0000 -> t1n
    0001 -> t1c
    0002 -> t2w
    0003 -> t2f

Example:
python create_imagesTs.py \
    --data_dir /vol/biodata/data/BraTS_2023_MEN/TrainData \
    --split_json artifacts/splits/brats_men_splits.json \
    --nnunet_raw /vol/biomedic2/bglocker_studproj/am525/3d-brain-ssl/data/nnUNet_raw
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
    """
    Find modality file in case directory.
    """
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


def copy_test_case(case_id, source_dir, imagesTs):
    """
    Copy modalities into imagesTs with nnUNet naming.
    """
    case_dir = source_dir / case_id

    for modality_name, modality_idx in MODALITY_MAP.items():
        src = find_modality_file(case_dir, modality_name)

        dst = imagesTs / (
            f"{case_id}_{modality_idx}.nii.gz"
        )

        shutil.copy2(src, dst)


def main(args):
    source_dir = Path(args.data_dir)

    splits = load_splits(args.split_json)

    test_cases = splits["test"]

    dataset_dir = (
        Path(args.nnunet_raw)
        / "Dataset001_BraTSMEN"
    )

    imagesTs = dataset_dir / "imagesTs"

    imagesTs.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("Creating nnUNet imagesTs")
    print("=" * 80)

    print(f"Number of test cases: {len(test_cases)}")

    for case_id in tqdm(test_cases):
        copy_test_case(
            case_id=case_id,
            source_dir=source_dir,
            imagesTs=imagesTs,
        )

    print("\nDone.")
    print(f"\nimagesTs written to:\n{imagesTs}")


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