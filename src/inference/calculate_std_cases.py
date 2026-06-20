#!/usr/bin/env python
"""
Compute mean +- std for LesionWise and Legacy Dice/HD95 from a brats_men_metrics.csv produced by
evaluate_brats_men.py.

Usage:
    python src/inference/calculate_std_cases.py \
        --csv        artifacts/results/fomo2jomo/mmunetvae/fomo2jomo_eval_bratsmen/brats_men_metrics.csv \
        --model      fomo2jomo \
        --output     artifacts/results/fomo2jomo/fomo2jomo_bratsmen_std_cases.log

    python src/inference/calculate_std_cases.py \
        --csv        /vol/biomedic2/bglocker_studproj/am525/3d-brain-ssl/artifacts/results/amaes/brats_men_metrics.csv \
        --model      amaes \
        --output     artifacts/results/amaes/amaes_bratsmen_std_cases.log

    python src/inference/calculate_std_cases.py \
        --csv        /vol/biomedic2/bglocker_studproj/am525/3d-brain-ssl/artifacts/results/nnunet/nnunet_150epochs/brats_men_metrics.csv \
        --model      nnunet \
        --output     artifacts/results/nnunet/nnunet_150epochs/nnunet_bratsmen_std_cases.log
"""

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

METRICS = {
    "LesionWise Dice": "LesionWise_Score_Dice",
    "LesionWise HD95": "LesionWise_Score_HD95",
    "Legacy Dice":     "Legacy_Dice",
    "Legacy HD95":     "Legacy_HD95",
}
LABELS = ["ET", "TC", "WT"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    type=str, required=True,
                        help="Path to brats_men_metrics.csv")
    parser.add_argument("--model",  type=str, required=True,
                        help="Model name for the log header")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to write the summary log")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    lines = []
    lines.append(f"Model : {args.model}")
    lines.append(f"CSV   : {args.csv}")
    lines.append(f"Cases : {df['Case'].nunique()}")
    lines.append(f"Date  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)

    for label in LABELS:
        sub = df[df["Labels"] == label]
        lines.append(f"\n{label}")
        for metric_name, col in METRICS.items():
            mean = sub[col].mean()
            std  = sub[col].std()
            lines.append(f"  {metric_name:20s}: {mean:.3f} ± {std:.3f}")

    output = "\n".join(lines)
    print(output)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output + "\n")
    print(f"\nLog written to: {out_path}")


if __name__ == "__main__":
    main()