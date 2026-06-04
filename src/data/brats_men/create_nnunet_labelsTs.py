import json
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


def load_splits(split_json_path):
    with open(split_json_path, "r") as f:
        return json.load(f)


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


def main(args):
    source_dir = Path(args.data_dir)

    splits = load_splits(args.split_json)

    test_cases = splits["test"]

    dataset_dir = (
        Path(args.nnunet_raw)
        / "Dataset001_BraTSMEN"
    )

    labelsTs = dataset_dir / "labelsTs"

    labelsTs.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("Creating nnUNet labelsTs")
    print("=" * 80)

    print(f"Number of test cases: {len(test_cases)}")

    for case_id in tqdm(test_cases):
        case_dir = source_dir / case_id

        seg_src = find_segmentation_file(case_dir)

        seg_dst = labelsTs / f"{case_id}.nii.gz"

        shutil.copy2(seg_src, seg_dst)

    print("\nDone.")
    print(f"\nlabelsTs written to:\n{labelsTs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--split_json",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--nnunet_raw",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    main(args)