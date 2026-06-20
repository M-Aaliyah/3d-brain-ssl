#!/usr/bin/env python
"""
src/finetuning/collect_hp_results.py

Aggregate per-config result.json files written by SLURM array tasks
into a single best_config.json.

Example usage:

Run after all array tasks complete:
  python src/finetuning/collect_hp_results.py \
      --search_dir artifacts/hp_search/amaes \
      --output     artifacts/hp_search/amaes/best_config.json

Or for fomo2jomo:
  python src/finetuning/collect_hp_results.py \
      --search_dir artifacts/hp_search/fomo2jomo \
      --output     artifacts/hp_search/fomo2jomo/best_config.json
"""
import os
import json
import argparse
import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_dir", required=True,
                        help="artifacts/hp_search/amaes or fomo2jomo")
    parser.add_argument("--output",     required=True,
                        help="Where to write best_config.json")
    parser.add_argument("--method",     default=None)
    args = parser.parse_args()

    method = args.method or os.path.basename(args.search_dir.rstrip("/"))

    # Collect all result.json files written by array tasks
    result_files = sorted(glob.glob(
        os.path.join(args.search_dir, "*", "result.json")
    ))

    if not result_files:
        print(f"ERROR: No result.json files found under {args.search_dir}")
        print("Make sure all SLURM array tasks have completed.")
        return

    all_results = []
    for rf in result_files:
        with open(rf) as f:
            all_results.append(json.load(f))

    # Check for expected number of configs
    expected = 60  # 2 LR x 2 WD x 5 loss x 3 scheduler
    if len(all_results) < expected:
        missing_n = expected - len(all_results)
        print(f"WARNING: Only {len(all_results)}/{expected} configs have results.")
        print(f"  {missing_n} array tasks may still be running or failed.")
        print(f"  Re-run collect_hp_results.py when all tasks finish.")

    # Select best by val/dice
    best = max(all_results, key=lambda r: r["best_val_dice"])

    # Write consolidated results (all configs sorted by val/dice)
    results_path = os.path.join(args.search_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "method":      method,
            "n_configs":   expected,
            "n_done":      len(all_results),
            "all_results": sorted(all_results,
                                  key=lambda r: r["best_val_dice"],
                                  reverse=True),
            "best":        best,
        }, f, indent=2)

    # Write best_config, includes lr_scheduler since it is searched
    best_config = {
        "method":       method,
        "lr":            best["lr"],
        "weight_decay":  best["weight_decay"],
        "loss_fn":       best["loss_fn"],
        "lr_scheduler":  best["lr_scheduler"],
        "best_val_dice": best["best_val_dice"],
        "checkpoint":    best["checkpoint"],
        "next_step": (
            f"Run finetune_brats_men_{method}.py with "
            f"--learning_rate {best['lr']:.0e} "
            f"--weight_decay {best['weight_decay']:.0e} "
            f"--loss_fn {best['loss_fn']} "
            f"--lr_scheduler {best['lr_scheduler']} "
        ),
    }
    with open(args.output, "w") as f:
        json.dump(best_config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"{method.upper()} HP SEARCH RESULTS ({len(all_results)}/{expected} configs)")
    print(f"\nTop 10 configs:")
    for r in sorted(all_results, key=lambda x: x["best_val_dice"],
                    reverse=True)[:10]:
        print(f"  LR={r['lr']:.0e}  WD={r['weight_decay']:.0e}  "
              f"loss={r['loss_fn']:<12}  sched={r['lr_scheduler']:<15}  "
              f"dice={r['best_val_dice']:.4f}")

    print(f"\nBest: LR={best['lr']:.0e}  WD={best['weight_decay']:.0e}  "
          f"loss={best['loss_fn']}  sched={best['lr_scheduler']}")
    print(f"val_dice={best['best_val_dice']:.4f}")
    print(f"\nAll results -> {results_path}")
    print(f"Best config -> {args.output}")
    print(f"\nNext: {best_config['next_step']}")


if __name__ == "__main__":
    main()