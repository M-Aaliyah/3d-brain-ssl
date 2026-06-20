"""
src/models/fomo2jomo_supervised_seg.py

FOMO2JOMOSegModel for mmunetvae finetuning.

mmunetvae.forward(x_list=inputs) returns {'task_output': tensor}.
Overrides training_step and validation_step to extract 'task_output'
before computing loss and metrics.

Configurable loss_fn. LR scheduler is fixed to cosine annealing.

supervised_base.py is NOT modified.
"""

from typing import Optional, Literal

import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import Dice
from torch.optim import AdamW

from yucca.modules.optimization.loss_functions.deep_supervision import DeepSupervisionLoss
from yucca.modules.metrics.training_metrics import F1

from models.supervised_base import BaseSupervisedModel
from models.seg_losses import build_loss
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, CyclicLR, SequentialLR, LinearLR
)

class FOMO2JOMOSegModel(BaseSupervisedModel):
    """
    Supervised segmentation model for mmunetvae (FOMO2JOMO).

    The only structural differences from SupervisedSegModel:
      1. forward() calls self.model(x_list=inputs) instead of self.model(inputs)
         because mmunetvae expects keyword argument x_list.
      2. training_step and validation_step extract out_dict["task_output"]
         before computing loss and metrics, because mmunetvae.forward()
         returns a dict rather than a tensor.

    Everything else (loss, scheduler, metrics, weight transfer) is
    identical to SupervisedSegModel.
    """

    def __init__(
        self,
        config: dict = {},
        learning_rate: float = 1e-3,
        do_compile: Optional[bool] = False,
        compile_mode: Optional[str] = "default",
        weight_decay: float = 3e-5,
        amsgrad: bool = False,
        eps: float = 1e-8,
        betas: tuple = (0.9, 0.999),
        deep_supervision: bool = False,
        loss_fn: Literal[
            "dicece", "ce", "dice", "tversky_ce", "dice_focal"
        ] = "tversky_ce", # FOMO2JOMO paper default; overridden by HP search
        lr_scheduler: Literal[
            "cosine", "warmup_cosine", "cyclic"
        ] = "cosine",
    ):
        self._loss_fn_name      = loss_fn
        self._lr_scheduler_name = lr_scheduler
        super().__init__(
            config=config,
            learning_rate=learning_rate,
            do_compile=do_compile,
            compile_mode=compile_mode,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            eps=eps,
            betas=betas,
            deep_supervision=deep_supervision,
        )
        self.val_case_dices = []
        self.case_dice_metric = Dice(
            num_classes=self.num_classes,
            ignore_index=0 if self.num_classes > 1 else None,
        )

    def load_model(self):
        from yucca.functional.utils.kwargs import filter_kwargs
        from models import networks
        print(f"Loading Model: 3D {self.model_name}")
        model_class = getattr(networks, self.model_name)
        print("Found model class: ", model_class)
        print("MODALITIES", self.num_modalities)
        model_kwargs = {
            "input_channels":       self.num_modalities,
            "output_channels":      self.num_classes,
            "deep_supervision":     self.deep_supervision,
            "conv_op":              torch.nn.Conv3d,
            "norm_op":              torch.nn.InstanceNorm3d,
            "checkpoint_style":     None,
            "mode":                 self.task_type,
            "use_vae":              False,
            "use_skip_connections": True,
        }
        model_kwargs = filter_kwargs(model_class, model_kwargs)
        self.model = model_class(**model_kwargs)
    
    def on_fit_start(self):
        """Freeze reconstruction decoder to match original FOMO2JOMO finetuning.
        model.decoder = pretraining reconstruction decoder (not used for segmentation)
        model.decoder_task = segmentation decoder (stays trainable)
        """
        if hasattr(self.model, "decoder"):
            print("Freezing reconstruction decoder (model.decoder)...")
            for param in self.model.decoder.parameters():
                param.requires_grad = False
            frozen = sum(p.numel() for p in self.model.decoder.parameters())
            print(f"Frozen {frozen:,} parameters in model.decoder")
        else:
            print("Warning: model.decoder not found, skipping freeze")

    def forward(self, inputs: torch.Tensor) -> dict:
        """inputs: (B, num_modalities, D, H, W)
        Returns:
        {'task_output': (B, num_classes, D, H, W)}
        """
        return self.model(x_list=inputs)

    def training_step(self, batch, _batch_idx):
        assert batch["image"].shape[1] == self.num_modalities, (
            f"Expected {self.num_modalities} channels, "
            f"got {batch['image'].shape[1]}"
        )
        inputs, target, _ = self._process_batch(batch)
        output = self(inputs)["task_output"]
        loss   = self.loss_fn_train(output, target)
        metrics = self.compute_metrics(self.train_metrics, output, target)
        self.log_dict(
            {"train/loss": loss} | metrics,
            prog_bar=self.progress_bar, logger=True,
            sync_dist=True, on_step=True, on_epoch=True,
        )
        return loss

    def run_predict(self, inputs: torch.Tensor) -> dict:
        with torch.autocast("cuda", enabled=False): # Uses sliding window inference
            preds = self.model.predict(
                data=inputs.float(),
                mode="3D",
                mirror=False,
                overlap=0.5,
                patch_size=self.patch_size,
                sliding_window_prediction=True,
                device=inputs.device,
            )
        return {"task_output": preds}
    
    def validation_step(self, batch, _batch_idx):
        assert batch["image"].shape[1] == self.num_modalities, (
            f"Expected {self.num_modalities} channels, "
            f"got {batch['image'].shape[1]}"
        )
        inputs, target, _ = self._process_batch(batch)
        output = self.run_predict(inputs.float())["task_output"]

        # Per-case dice accumulated, averaged at epoch end
        self.val_case_dices.append(
            self.case_dice_metric(output, target).item()
        )

        loss    = self.loss_fn_val(output, target)
        metrics = self.compute_metrics(self.val_metrics, output, target)
        self.log_dict(
            {"val/loss": loss} | metrics,
            prog_bar=self.progress_bar, logger=True,
            sync_dist=True, on_step=False, on_epoch=True,
        )

    def on_validation_epoch_end(self):
        if self.val_case_dices:
            mean_dice = sum(self.val_case_dices) / len(self.val_case_dices)
            self.log("val/full_case_dice", mean_dice,
                    prog_bar=True, sync_dist=True,
                    on_step=False, on_epoch=True)
            self.val_case_dices.clear()

    def _configure_losses(self):
        loss_fn_train = build_loss(self._loss_fn_name)
        loss_fn_val   = build_loss(self._loss_fn_name)
        if self.deep_supervision:
            loss_fn_train = DeepSupervisionLoss(loss_fn_train, weights=None)
        return loss_fn_train, loss_fn_val

    def configure_optimizers(self):
        self.loss_fn_train, self.loss_fn_val = self._configure_losses()
        self.optim = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
            eps=self.eps,
            betas=self.betas,
        )
        max_epochs    = self.trainer.max_epochs
        steps_per_ep  = self.trainer.limit_train_batches or 100

        if self._lr_scheduler_name == "cosine":
            sched = CosineAnnealingLR(
                self.optim,
                T_max=int(max_epochs * 1.15),
                eta_min=1e-9,
            )
            sched_cfg = {"scheduler": sched, "interval": "epoch", "frequency": 1}

        elif self._lr_scheduler_name == "warmup_cosine":
            warmup_epochs = max(1, int(max_epochs * 0.1))
            warmup = LinearLR(self.optim, start_factor=1e-3, end_factor=1.0,
                              total_iters=warmup_epochs)
            cosine = CosineAnnealingLR(
                self.optim, T_max=max(1, max_epochs - warmup_epochs), eta_min=1e-9
            )
            sched = SequentialLR(self.optim, schedulers=[warmup, cosine],
                                 milestones=[warmup_epochs])
            sched_cfg = {"scheduler": sched, "interval": "epoch", "frequency": 1}

        elif self._lr_scheduler_name == "cyclic":
            step_size_up = max(1, int(steps_per_ep * max_epochs * 0.1))
            sched = CyclicLR(
                self.optim,
                base_lr=self.learning_rate / 100,
                max_lr=self.learning_rate,
                step_size_up=step_size_up,
                mode="triangular2",
                cycle_momentum=False,
            )
            sched_cfg = {"scheduler": sched, "interval": "step", "frequency": 1}

        else:
            raise ValueError(
                f"Unknown lr_scheduler '{self._lr_scheduler_name}'. "
                f"Choose from: cosine, warmup_cosine, cyclic"
            )

        self.lr_scheduler = sched
        return {"optimizer": self.optim, "lr_scheduler": sched_cfg}

    def _configure_metrics(self, prefix: str):
        return MetricCollection({
            f"{prefix}/dice": Dice(
                num_classes=self.num_classes,
                ignore_index=0 if self.num_classes > 1 else None,
            ),
            f"{prefix}/F1": F1(
                num_classes=self.num_classes,
                ignore_index=0 if self.num_classes > 1 else None,
                average=None,
            ),
        })

    def compute_metrics(self, metrics, output, target, ignore_index: int = 0):
        metrics = metrics(output, target)
        tmp = {}
        to_drop = []
        for key in metrics.keys():
            if metrics[key].numel() > 1:
                to_drop.append(key)
                for i, val in enumerate(metrics[key]):
                    if i != ignore_index:
                        tmp[key + "_" + str(i)] = val
        for k in to_drop:
            metrics.pop(k)
        metrics.update(tmp)
        return metrics