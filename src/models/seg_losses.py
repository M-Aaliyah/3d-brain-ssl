"""
Segmentation loss functions for BraTS-MEN finetuning.

All losses expect:
  logits: (B, C, D, H, W) raw network output, C=4 for BraTS-MEN
  targets: (B, 1, D, H, W) or (B, D, H, W) long class indices

Available via build_loss(name):
  "dicece" DiceCE is Dice + CrossEntropy. nnUNet standard.
  "ce" CrossEntropy only. Baseline; does not handle imbalance.
  "dice" Soft Dice, background ignored. Most directly motivated
  by the downstream BraTS Dice evaluation metric.
  "tversky_ce" Tversky(α=0.7,β=0.3) + CrossEntropy. From FOMO2JOMO paper.
  "dice_focal" Dice + Focal(γ=2). Used by NVAUTO, rank-1 team in
  BraTS 2023 Meningioma challenge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.losses import DiceCELoss, DiceFocalLoss as MonaiDiceFocalLoss
from yucca.modules.optimization.loss_functions.nnUNet_losses import DiceCE


class CrossEntropyOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets.squeeze(1).long())


class SoftDiceLoss(nn.Module):
    """
    Macro soft Dice loss, background class (index 0) ignored.

    Computed as: 1 - mean(Dice_c) for c in {NCR=1, ED=2, ET=3}
    """

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.squeeze(1).long()
        probs = F.softmax(logits, dim=1)
        num_classes = probs.shape[1]
        targets_one_hot = (
            F.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float()
        )

        dice_sum = torch.tensor(0.0, device=probs.device)
        n_fg = 0
        for c in range(1, num_classes): # skip background (class 0)
            p = probs[:, c]
            t = targets_one_hot[:, c]
            intersection = (p * t).sum()
            denom = p.sum() + t.sum()
            dice_c = (2.0 * intersection + self.smooth) / (
                denom + self.smooth
            )
            dice_sum = dice_sum + dice_c
            n_fg += 1

        return 1.0 - dice_sum / max(n_fg, 1)


class TverskyCrossEntropyLoss(nn.Module):
    """
    Tversky(a=0.7, B=0.3) + CrossEntropy.

    Only computed on patches that contain foreground to avoid numerical
    issues on all-background patches (which dominate by volume).
    Used by the FOMO2JOMO paper for segmentation finetuning.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        smooth: float = 1e-6,
        weight_ce: float = 0.5,
        weight_tversky: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.weight_ce = weight_ce
        self.weight_tversky = weight_tversky
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.squeeze(1).long()
        ce_loss = self.ce(logits, targets)

        probs = F.softmax(logits, dim=1)
        num_classes = probs.shape[1]
        targets_one_hot = (
            F.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float()
        )

        # Only compute Tversky on patches containing foreground
        fg_mask = targets_one_hot[:, 1:].sum(dim=(1, 2, 3, 4)) > 0
        if fg_mask.any():
            p = probs[fg_mask]
            t = targets_one_hot[fg_mask]
            dims = (0, 2, 3, 4)
            TP = torch.sum(p * t, dim=dims)
            FP = torch.sum(p * (1 - t), dim=dims)
            FN = torch.sum((1 - p) * t, dim=dims)
            tversky = (TP + self.smooth) / (
                TP + self.alpha * FN + self.beta * FP + self.smooth
            )
            tversky_loss = 1.0 - tversky.mean()
        else:
            tversky_loss = torch.tensor(0.0, device=logits.device)

        return self.weight_ce * ce_loss + self.weight_tversky * tversky_loss


def build_loss(name: str) -> nn.Module:
    """
    Return the loss function for the given name string.

    Args:
        name: one of "dicece", "ce", "dice", "tversky_ce", "dice_focal"

    Loss selection rationale for BraTS-MEN (LaBella et al. 2025,
    BraTS 2023 Meningioma challenge):
      dice_focal NVAUTO rank-1; Focal γ=2 focuses on hard small lesions
      dicece nnUNet standard; ranks 2+3 in BraTS-MEN challenge
      tversky_ce FOMO2JOMO paper; α=0.7 upweights FN (missed lesion = 0)
      dice most direct proxy for downstream BraTS Dice metric
      ce baseline only; no top BraTS team used plain CE
    """
    if name == "dicece":
        # MONAI DiceCELoss: softmax + Dice(include_background=False) + CE
        # include_background=False matches the val/dice metric (bg ignored)
        return MonaiDiceCEWrapper()

    elif name == "ce":
        return CrossEntropyOnly()

    elif name == "dice":
        return SoftDiceLoss(smooth=1e-5)

    elif name == "tversky_ce":
        return TverskyCrossEntropyLoss(
            alpha=0.7,
            beta=0.3,
            weight_ce=0.5,
            weight_tversky=1.0,
        )

    elif name == "dice_focal":
        # MONAI DiceFocalLoss: Dice(include_background=False) + Focal(y=2)
        return MonaiDiceFocalWrapper()

    else:
        raise ValueError(
            f"Unknown loss_fn '{name}'. "
            f"Choose from: dicece, ce, dice, tversky_ce, dice_focal"
        )


class MonaiDiceCEWrapper(nn.Module):
    """
    Wraps MONAI DiceCELoss to accept (B,C,D,H,W) logits and
    (B,1,D,H,W) or (B,D,H,W) long targets, matching our pipeline's
    convention. Background excluded from Dice (include_background=False),
    consistent with val/dice metric (ignore_index=0).
    """

    def __init__(self):
        super().__init__()
        self.loss = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # MONAI expects targets as (B, 1, D, H, W)
        if targets.dim() == 4:
            targets = targets.unsqueeze(1)
        return self.loss(logits, targets)


class MonaiDiceFocalWrapper(nn.Module):
    """
    Wraps MONAI DiceFocalLoss. y=2 is standard focal parameter.
    Background excluded from both Dice and Focal terms.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.loss = MonaiDiceFocalLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            gamma=gamma,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 4:
            targets = targets.unsqueeze(1)
        return self.loss(logits, targets)