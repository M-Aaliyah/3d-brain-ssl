#!/usr/bin/env python
"""
Full joint HP search for FOMO2JOMO finetuning on BraTS-MEN.

Search space:
  learning_rate in {2e-4, 1e-4} (NVAUTO=2e-4, CNMC/AMAES/FOMO2JOMO=1e-4)
  weight_decay in {1e-5, 3e-5} (NVAUTO=1e-5, CNMC/AMAES/FOMO2JOMO=3e-5)
  loss_fn in {dicece, ce, dice, tversky_ce, dice_focal}
  lr_scheduler in {cosine, warmup_cosine, cyclic}
  Total: 2 × 2 × 5 × 3 = 60 configs

Modes:
  Sequential (1 GPU)
  Array (4 GPUs)

Writes:
  - artifacts/hp_search/fomo2jomo/results.json
  - artifacts/hp_search/fomo2jomo/best_config.json
"""

import os
import sys
import json
import itertools
import argparse
import warnings
import logging

import torch
import lightning as L
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from batchgenerators.utilities.file_and_folder_operations import (
    maybe_mkdir_p as ensure_dir_exists,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for p in [SRC_ROOT, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.supervised_base import BaseSupervisedModel
from augmentations.finetune_augmentation_presets import get_finetune_augmentation_params
from utils.fomo2jomo_utils import setup_seed, load_pretrained_weights
from yucca.modules.data.augmentation.YuccaAugmentationComposer import YuccaAugmentationComposer
from data.brats_men.brats_men_config import bratsmen_config
from data.yucca_datamodule import ModYuccaDataModule
from data.yucca_dataset import ModYuccaTrainDataset


class BraTSMENSplitsConfig:
    def __init__(self, splits_json_path):
        with open(splits_json_path) as f:
            splits = json.load(f)
        if "train" in splits and "val" in splits:
            self._train = splits["train"]
            self._val = splits["val"]
            self._test = splits.get("test", [])
        elif "splits" in splits:
            self._train = splits["splits"][0]["train"]
            self._val = splits["splits"][0]["val"]
            self._test = splits.get("test", [])
        else:
            raise ValueError("Unrecognised brats_men_splits.json format")
        print(f"BraTS-MEN: train={len(self._train)} | "
              f"val={len(self._val)} | test={len(self._test)} (untouched)")

    def train(self, idx=0): return self._train
    def val(self, idx=0): return self._val
    def test(self): return self._test


class ValDiceTracker(L.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_dices = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if "val/dice" in trainer.callback_metrics:
            self.epoch_dices.append(
                trainer.callback_metrics["val/dice"].item()
            )

    def best_val_dice(self) -> float:
        return max(self.epoch_dices) if self.epoch_dices else 0.0


def run_config(run_tag, lr, wd, loss_fn, lr_scheduler,
               args, splits_config, task_type,
               num_classes, num_modalities, search_dir):
    run_dir = os.path.join(search_dir, run_tag)
    ensure_dir_exists(run_dir)

    config = {
        "experiment": f"fomo2jomo_hp_{run_tag}",
        "run_type": "finetune",
        "task_type": task_type,
        "model_name": args.model_name,
        "model_dimensions": "3D",
        "version": 0,
        "save_dir": run_dir,
        "train_data_dir": args.data_dir,
        "version_dir": run_dir,
        "pretrained_ckpt": args.pretrained_ckpt,
        "seed": setup_seed(False),
        "num_classes": num_classes,
        "num_modalities": num_modalities,
        "image_extension": ".npy",
        "allow_missing_modalities": False,
        "labels": list(bratsmen_config["labels"].keys()),
        "batch_size": args.batch_size,
        "learning_rate": lr,
        "weight_decay": wd,
        "patch_size": (args.patch_size,) * 3,
        "precision": args.precision,
        "augmentation_preset": args.augmentation_preset,
        "epochs": args.search_epochs,
        "train_batches_per_epoch": args.train_batches_per_epoch,
        "effective_batch_size": args.num_devices * args.batch_size,
        "train_dataset_size": len(splits_config.train(0)),
        "val_dataset_size": len(splits_config.val(0)),
        "num_devices": args.num_devices,
        "num_workers": args.num_workers,
        "compile": False,
        "compile_mode": None,
        "fast_dev_run": False,
        "gradient_clip_val": 1.0,
        "loss_fn": loss_fn,
        "lr_scheduler": lr_scheduler,
    }

    aug_params = get_finetune_augmentation_params(args.augmentation_preset)
    aug_params["mask_ratio"] = 0
    augmenter = YuccaAugmentationComposer(
        patch_size=config["patch_size"],
        task_type_preset="segmentation",
        parameter_dict=aug_params,
        deep_supervision=False,
    )
    data_module = ModYuccaDataModule(
        train_dataset_class=ModYuccaTrainDataset,
        composed_train_transforms=augmenter.train_transforms,
        composed_val_transforms=augmenter.val_transforms,
        patch_size=config["patch_size"],
        batch_size=config["batch_size"],
        train_data_dir=config["train_data_dir"],
        image_extension=config["image_extension"],
        task_type=config["task_type"],
        splits_config=splits_config,
        split_idx=0,
        num_workers=args.num_workers,
        val_sampler=None,
    )

    model = BaseSupervisedModel.create(
        task_type=task_type,
        config=config,
        learning_rate=lr,
        weight_decay=wd,
        loss_fn=loss_fn,
        lr_scheduler=lr_scheduler,
        do_compile=False,
        compile_mode=None,
    )
    state_dict = load_pretrained_weights(args.pretrained_ckpt, False)
    n_transferred = model.load_state_dict(state_dict=state_dict, strict=False)
    assert n_transferred > 0, "No weights transferred!"

    wandb_logger = L.pytorch.loggers.WandbLogger(
        project=args.wandb_project,
        name=f"fomo2jomo_hp_{run_tag}",
        save_dir=os.path.join(REPO_ROOT, "wandb"),
        log_model=False,
        tags=["hp_search", "fomo2jomo", "brats-men"],
        config={
            "lr": lr, "weight_decay": wd,
            "loss_fn": loss_fn, "lr_scheduler": lr_scheduler,
        },
    )

    dice_tracker = ValDiceTracker()
    checkpoint_cb = ModelCheckpoint(
        dirpath=run_dir,
        every_n_epochs=10,
        save_top_k=1,
        filename="best",
        enable_version_counter=False,
        monitor="val/dice",
        mode="max",
    )

    trainer = L.Trainer(
        callbacks=[checkpoint_cb, LearningRateMonitor("step"), dice_tracker],
        logger=wandb_logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.num_devices,
        max_epochs=args.search_epochs,
        limit_train_batches=args.train_batches_per_epoch,
        precision=args.precision,
        log_every_n_steps=1,
        gradient_clip_val=config["gradient_clip_val"],
    )
    trainer.fit(model=model, datamodule=data_module)

    best_dice = dice_tracker.best_val_dice()
    wandb.summary["best_val_dice"] = best_dice
    wandb.finish()

    result = {
        "run_tag": run_tag,
        "lr": lr,
        "weight_decay": wd,
        "loss_fn": loss_fn,
        "lr_scheduler": lr_scheduler,
        "best_val_dice": best_dice,
        "checkpoint": os.path.join(run_dir, "best.ckpt"),
    }
    print(f"  [{run_tag}] val_dice={best_dice:.4f}")
    return result


def main():
    logging.getLogger().setLevel(logging.INFO)
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--splits_json", type=str, required=True)
    parser.add_argument("--pretrained_ckpt", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="mmunetvae")
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--search_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_batches_per_epoch", type=int, default=100)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--augmentation_preset", type=str, default="none")
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--wandb_project", type=str, default="3d-brain-ssl")
    parser.add_argument("--lr_values", type=float, nargs="+",
                        default=[2e-4, 1e-4])
    parser.add_argument("--wd_values", type=float, nargs="+",
                        default=[1e-5, 3e-5])
    parser.add_argument("--loss_fns", type=str, nargs="+",
                        default=["dicece", "ce", "dice",
                                 "tversky_ce", "dice_focal"])
    parser.add_argument("--schedulers", type=str, nargs="+",
                        default=["cosine", "warmup_cosine", "cyclic"])
    parser.add_argument("--resume_from", type=str, default=None)
    # SLURM array mode
    parser.add_argument("--configs_json", type=str, default=None,
                        help="Pre-generated config list for array mode")
    parser.add_argument("--config_idx", type=int, default=None,
                        help="Index into configs_json (= SLURM_ARRAY_TASK_ID)")

    args = parser.parse_args()

    search_dir = os.path.join(args.save_dir, "hp_search", "fomo2jomo")
    results_path = os.path.join(search_dir, "results.json")
    ensure_dir_exists(search_dir)

    splits_config = BraTSMENSplitsConfig(args.splits_json)
    task_type = bratsmen_config["task_type"]
    num_classes = bratsmen_config["num_classes"]
    num_modalities = len(bratsmen_config["modalities"])

    # SLURM array mode: run exactly one config and exit
    if args.config_idx is not None:
        assert args.configs_json is not None, \
            "--configs_json required when --config_idx is set"
        with open(args.configs_json) as f:
            all_configs = json.load(f)
        cfg = all_configs[args.config_idx]
        lr = cfg["lr"]
        wd = cfg["wd"]
        loss_fn = cfg["loss_fn"]
        lr_scheduler = cfg.get("lr_scheduler", "cosine")
        run_tag = cfg["run_tag"]
        print(f"\nArray task {args.config_idx}/{len(all_configs)-1}: "
              f"LR={lr:.0e} WD={wd:.0e} loss={loss_fn}")
        result = run_config(
            run_tag, lr, wd, loss_fn, lr_scheduler,
            args, splits_config, task_type,
            num_classes, num_modalities, search_dir,
        )
        result_path = os.path.join(search_dir, run_tag, "result.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result saved: {result_path}")
        return

    # Sequential mode
    configs = list(itertools.product(
        args.lr_values, args.wd_values, args.loss_fns, args.schedulers,
    ))
    n = len(configs)

    print(f"\n{'='*65}")
    print(f"FOMO2JOMO HP search — {n} configs (sequential mode)")
    print(f"  LR: {args.lr_values}")
    print(f"  WD: {args.wd_values}")
    print(f"  loss_fn: {args.loss_fns}")
    print(f"  scheduler: cosine annealing (fixed)")
    print(f"  epochs: {args.search_epochs}")
    print(f"  selection: val/dice | test split untouched")
    print(f"  After search: re-run best config for 100 epochs")
    print(f"{'='*65}\n")

    done_tags = set()
    all_results = []
    if args.resume_from and os.path.exists(args.resume_from):
        with open(args.resume_from) as f:
            all_results = json.load(f)["all_results"]
        done_tags = {r["run_tag"] for r in all_results}
        print(f"Resuming: {len(done_tags)} done, {n - len(done_tags)} remaining\n")

    for i, (lr, wd, loss_fn, lr_scheduler) in enumerate(configs):
        lr_tag = f"lr_{lr:.0e}".replace("-", "m")
        wd_tag = f"wd_{wd:.0e}".replace("-", "m")
        run_tag = f"{lr_tag}_{wd_tag}_{loss_fn}_{lr_scheduler}"

        if run_tag in done_tags:
            continue

        print(f"\n[{i+1}/{n}] LR={lr:.0e} WD={wd:.0e} "
              f"loss={loss_fn} scheduler={lr_scheduler}")
        result = run_config(
            run_tag, lr, wd, loss_fn, lr_scheduler,
            args, splits_config, task_type,
            num_classes, num_modalities, search_dir,
        )
        all_results.append(result)

        best_so_far = max(all_results, key=lambda r: r["best_val_dice"])
        with open(results_path, "w") as f:
            json.dump({
                "method": "fomo2jomo",
                "search_epochs": args.search_epochs,
                "n_configs": n,
                "n_done": len(all_results),
                "all_results": all_results,
                "best_so_far": best_so_far,
            }, f, indent=2)

    best = max(all_results, key=lambda r: r["best_val_dice"])
    best_config = {
        "method": "fomo2jomo",
        "lr": best["lr"],
        "weight_decay": best["weight_decay"],
        "loss_fn": best["loss_fn"],
        "lr_scheduler": best["lr_scheduler"],
        "best_val_dice": best["best_val_dice"],
        "checkpoint": best["checkpoint"],
        "next_step": (
            f"Run finetune_brats_men_fomo2jomo.py with "
            f"--learning_rate {best['lr']:.0e} "
            f"--weight_decay {best['weight_decay']:.0e} "
            f"--loss_fn {best['loss_fn']} "
            f"--lr_scheduler {best['lr_scheduler']} "
            f"--epochs 100"
        ),
    }
    best_path = os.path.join(search_dir, "best_config.json")
    with open(best_path, "w") as f:
        json.dump(best_config, f, indent=2)

    print(f"\n{'='*65}")
    print("FOMO2JOMO HP SEARCH COMPLETE")
    print(f"\nTop configs:")
    for r in sorted(all_results, key=lambda x: x["best_val_dice"],
                    reverse=True):
        print(f"  LR={r['lr']:.0e} WD={r['weight_decay']:.0e} "
              f"loss={r['loss_fn']:<12} dice={r['best_val_dice']:.4f}")
    print(f"\nBest config -> {best_path}")
    print(f"\nNext: {best_config['next_step']}")


if __name__ == "__main__":
    main()
