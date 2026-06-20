#!/usr/bin/env python
"""
Finetune FOMO2JOMO-pretrained mmunetvae on BraTS-MEN segmentation.
All HP (lr, wd, loss_fn, lr_scheduler) set from HP search results.
"""

import os, sys, json, argparse, warnings, logging
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
from utils.fomo2jomo_utils import setup_seed, find_checkpoint, load_pretrained_weights
from yucca.modules.data.augmentation.YuccaAugmentationComposer import YuccaAugmentationComposer
from yucca.modules.callbacks.loggers import YuccaLogger
from yucca.pipeline.configuration.configure_paths import detect_version
from data.brats_men.brats_men_config import bratsmen_config
from data.yucca_datamodule import ModYuccaDataModule
from data.yucca_dataset import ModYuccaTrainDataset


class BraTSMENSplitsConfig:
    def __init__(self, path):
        with open(path) as f: s = json.load(f)
        if "train" in s:
            self._train, self._val, self._test = (
                s["train"], s["val"], s.get("test", []))
        else:
            self._train = s["splits"][0]["train"]
            self._val = s["splits"][0]["val"]
            self._test = s.get("test", [])
        print(f"splits: train={len(self._train)} val={len(self._val)} "
              f"test={len(self._test)}")
    def train(self, idx=0): return self._train
    def val(self, idx=0): return self._val
    def test(self): return self._test


def main():
    logging.getLogger().setLevel(logging.INFO)
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision("high")

    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--splits_json", required=True)
    p.add_argument("--pretrained_ckpt", default=None)
    p.add_argument("--model_name", default="mmunetvae")
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--train_batches_per_epoch", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=3e-5)
    p.add_argument("--loss_fn", default="tversky_ce",
                   choices=["dicece","ce","dice","tversky_ce","dice_focal"])
    p.add_argument("--lr_scheduler", default="cosine",
                   choices=["cosine","warmup_cosine","cyclic"])
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--augmentation_preset", default="none")
    p.add_argument("--num_devices", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile_mode", default=None)
    p.add_argument("--experiment", default="fomo2jomo_finetune_bratsmen")
    p.add_argument("--wandb_project", default="3d-brain-ssl")
    p.add_argument("--new_version", action="store_true")
    p.add_argument("--fast_dev_run", action="store_true")
    args = p.parse_args()

    run_type = "finetune" if args.pretrained_ckpt else "scratch"
    experiment_name = f"fomo2jomo_{run_type}_{args.model_name}"

    save_dir = os.path.join(args.save_dir, "models", "finetuning",
                            "fomo2jomo", args.model_name)
    continue_from = not args.new_version
    version = detect_version(save_dir, continue_from)
    version_dir = os.path.join(save_dir, f"version_{version}")
    ensure_dir_exists(version_dir)
    ckpt_path = find_checkpoint(version_dir, continue_from)

    splits = BraTSMENSplitsConfig(args.splits_json)
    task_type = bratsmen_config["task_type"]
    num_classes = bratsmen_config["num_classes"]
    num_mod = len(bratsmen_config["modalities"])

    config = {
        "experiment": experiment_name, "run_type": run_type,
        "task_type": task_type, "model_name": args.model_name,
        "model_dimensions": "3D", "version": version,
        "save_dir": save_dir, "train_data_dir": args.data_dir,
        "version_dir": version_dir,
        "pretrained_ckpt": args.pretrained_ckpt,
        "seed": setup_seed(continue_from),
        "num_classes": num_classes, "num_modalities": num_mod,
        "image_extension": ".npy", "allow_missing_modalities": False,
        "labels": list(bratsmen_config["labels"].keys()),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "loss_fn": args.loss_fn,
        "lr_scheduler": args.lr_scheduler,
        "patch_size": (args.patch_size,) * 3,
        "precision": args.precision,
        "augmentation_preset": args.augmentation_preset,
        "epochs": args.epochs,
        "train_batches_per_epoch": args.train_batches_per_epoch,
        "effective_batch_size": args.num_devices * args.batch_size,
        "train_dataset_size": len(splits.train(0)),
        "val_dataset_size": len(splits.val(0)),
        "num_devices": args.num_devices, "num_workers": args.num_workers,
        "gradient_clip_val": 1.0,
    }

    aug_params = get_finetune_augmentation_params(args.augmentation_preset)
    aug_params["mask_ratio"] = 0
    augmenter = YuccaAugmentationComposer(
        patch_size=config["patch_size"],
        task_type_preset="segmentation",
        parameter_dict=aug_params, deep_supervision=False,
    )
    data_module = ModYuccaDataModule(
        train_dataset_class=ModYuccaTrainDataset,
        composed_train_transforms=augmenter.train_transforms,
        composed_val_transforms=augmenter.val_transforms,
        patch_size=config["patch_size"], batch_size=config["batch_size"],
        train_data_dir=config["train_data_dir"],
        image_extension=config["image_extension"],
        task_type=config["task_type"], splits_config=splits,
        split_idx=0, num_workers=args.num_workers, val_sampler=None,
    )

    model = BaseSupervisedModel.create(
        task_type=task_type, config=config,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        loss_fn=args.loss_fn, lr_scheduler=args.lr_scheduler,
        do_compile=args.compile,
        compile_mode="default" if not args.compile_mode else args.compile_mode,
    )

    if run_type == "finetune":
        assert ckpt_path is None
        sd = load_pretrained_weights(args.pretrained_ckpt, args.compile)
        n = model.load_state_dict(state_dict=sd, strict=False)
        assert n > 0
        print(f"Transferred {n} tensors from {args.pretrained_ckpt}")

    wandb_logger = L.pytorch.loggers.WandbLogger(
        project=args.wandb_project,
        name=f"{experiment_name}_v{version}",
        save_dir=os.path.join(REPO_ROOT, "wandb"), log_model=False,
        tags=["finetuning", "fomo2jomo", "brats-men", run_type],
    )
    yucca_logger = YuccaLogger(
        save_dir=save_dir, version=version,
        steps_per_epoch=args.train_batches_per_epoch,
    )
    ckpt_cb = ModelCheckpoint(
        dirpath=version_dir, every_n_epochs=10, save_top_k=1,
        filename="last", enable_version_counter=False,
        monitor="val/full_case_dice", mode="max",
    )

    trainer = L.Trainer(
        callbacks=[ckpt_cb, LearningRateMonitor("step")],
        logger=[yucca_logger, wandb_logger],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="auto", num_nodes=1, devices=args.num_devices,
        default_root_dir=save_dir, max_epochs=args.epochs,
        limit_train_batches=args.train_batches_per_epoch,
        precision=args.precision, fast_dev_run=args.fast_dev_run,
        log_every_n_steps=1,
        gradient_clip_val=config["gradient_clip_val"],
    )
    trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt_path)

    if torch.cuda.is_available():
        wandb.summary["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    if ckpt_cb.best_model_score is not None:
        wandb.summary["best_val_dice"] = ckpt_cb.best_model_score.item()
    wandb.finish()
    print(f"Done. Checkpoint: {version_dir}/last.ckpt")


if __name__ == "__main__":
    main()