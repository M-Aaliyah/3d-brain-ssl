#!/usr/bin/env python
"""
Generate hyperparameter search config files for AMAES and FOMO2JOMO.

Search space:
  LR: {2e-4, 1e-4} — NVAUTO rank-1 and CNMC/AMAES/FOMO2JOMO
  WD: {1e-5, 3e-5} — NVAUTO rank-1 and CNMC/AMAES/FOMO2JOMO
  loss_fn: {dicece, ce, dice, tversky_ce, dice_focal}
  lr_scheduler: {cosine, warmup_cosine, cyclic}
  Total: 2 × 2 × 5 × 3 = 60 configs

Both methods use the same search space for a fair comparison.
Writes a single shared artifacts/hp_search/hp_configs.json.

Usage:
  python src/finetuning/generate_hp_configs.py \
      --output_dir artifacts/hp_search
"""

import os
import json
import itertools
import argparse


def generate_configs(lr_values, wd_values, loss_fns, schedulers):
    configs = []
    for lr, wd, loss_fn, sched in itertools.product(
        lr_values, wd_values, loss_fns, schedulers
    ):
        lr_tag = f"lr_{lr:.0e}".replace("-", "m")
        wd_tag = f"wd_{wd:.0e}".replace("-", "m")
        run_tag = f"{lr_tag}_{wd_tag}_{loss_fn}_{sched}"
        configs.append({
            "idx": len(configs),
            "lr": lr,
            "wd": wd,
            "loss_fn": loss_fn,
            "lr_scheduler": sched,
            "run_tag": run_tag,
        })
    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Generate HP search config JSON for AMAES and FOMO2JOMO"
    )
    parser.add_argument(
        "--output_dir", type=str, default="artifacts/hp_search",
    )
    parser.add_argument(
        "--lr_values", type=float, nargs="+", default=[2e-4, 1e-4],
        help="LR values. Default from NVAUTO (2e-4) and CNMC/AMAES/FOMO2JOMO (1e-4)."
    )
    parser.add_argument(
        "--wd_values", type=float, nargs="+", default=[1e-5, 3e-5],
        help="WD values. Default from NVAUTO (1e-5) and CNMC/AMAES/FOMO2JOMO (3e-5)."
    )
    parser.add_argument(
        "--loss_fns", type=str, nargs="+",
        default=["dicece", "ce", "dice", "tversky_ce", "dice_focal"],
    )
    parser.add_argument(
        "--schedulers", type=str, nargs="+",
        default=["cosine", "warmup_cosine", "cyclic"],
        help="LR schedulers to search."
    )
    args = parser.parse_args()

    configs = generate_configs(
        args.lr_values, args.wd_values, args.loss_fns, args.schedulers
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "hp_configs.json")
    with open(output_path, "w") as f:
        json.dump(configs, f, indent=2)

    print(f"Generated {len(configs)} configs → {output_path}")
    print(f"  LR: {args.lr_values}")
    print(f"  WD: {args.wd_values}")
    print(f"  loss_fns: {args.loss_fns}")
    print(f"  schedulers: {args.schedulers}")
    print(f"  SLURM array: --array=0-{len(configs)-1}%12")
    print()
    print("Configs (first 6):")
    for c in configs[:6]:
        print(f"  [{c['idx']:2d}] lr={c['lr']:.0e} wd={c['wd']:.0e}"
              f" loss={c['loss_fn']:<12} sched={c['lr_scheduler']}")
    print(f"  ... ({len(configs)} total)")


if __name__ == "__main__":
    main()