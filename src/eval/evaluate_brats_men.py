"""
Evaluate BraTS-MEN predictions using brats_men_metrics. This script computes lesion-wise metrics for
each case and saves both per-case and summary results to CSV files.

Usage:
    python evaluate_brats_men.py --pred_dir /path/to/predictions --gt_dir /path/to/ground_truths --output_dir /path/to/save/results
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from brats_men_metrics import get_LesionWiseResults


def main(args):
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_dir.glob("*.nii.gz"))

    all_results = []

    print("=" * 80)
    print("BraTS-MEN Evaluation")
    print("=" * 80)

    for pred_file in tqdm(pred_files):
        case_id = pred_file.name

        gt_file = gt_dir / case_id

        if not gt_file.exists():
            print(f"Missing GT for {case_id}")
            continue

        case_df = get_LesionWiseResults(
            pred_file=str(pred_file),
            gt_file=str(gt_file),
            challenge_name="BraTS-MEN",
        )

        case_df.insert(0, "Case", case_id)

        all_results.append(case_df)

    results_df = pd.concat(all_results, ignore_index=True)

    results_csv = output_dir / "brats_men_metrics.csv"
    results_df.to_csv(results_csv, index=False)

    print("\nSaved per-case metrics:")
    print(results_csv)

    summary_df = (
        results_df
        .groupby("Labels")
        .mean(numeric_only=True)
        .reset_index()
    )

    summary_csv = output_dir / "brats_men_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\nSaved summary metrics:")
    print(summary_csv)

    print("\n=== Mean Metrics ===")
    print(summary_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--gt_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    main(args)