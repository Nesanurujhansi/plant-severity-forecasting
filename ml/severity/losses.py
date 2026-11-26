"""Loss Functions for Lesion and Leaf Mask Segmentation.

Provides combined Binary Cross-Entropy (with positive-class weighting for rare lesion pixels)
and soft Dice Loss, supporting selective mask loss computation based on ground truth availability.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation maps."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute soft Dice loss.

        Args:
            logits: Logit predictions [B, 1, H, W] or [B, H, W].
            targets: Binary ground truth targets [B, 1, H, W] or [B, H, W] (0.0 or 1.0).
            mask: Optional sample validity mask [B] indicating which samples to include.

        Returns:
            Scalar Dice loss tensor.
        """
        probs = torch.sigmoid(logits)

        if logits.ndim == 4:
            probs = probs.squeeze(1)
            targets = targets.squeeze(1)

        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            probs = probs[mask]
            targets = targets[mask]

        probs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1)

        intersection = (probs_flat * targets_flat).sum()
        cardinality = probs_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - dice_score


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss targeting small, highly class-imbalanced lesion segmentation."""

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 1.33, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha  # Focus higher penalty on false negatives (missed lesions)
        self.beta = beta
        self.gamma = gamma  # Focal exponent focusing on hard samples
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        if logits.ndim == 4:
            probs = probs.squeeze(1)
            targets = targets.squeeze(1)

        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            probs = probs[mask]
            targets = targets[mask]

        p_flat = probs.reshape(-1)
        t_flat = targets.float().reshape(-1)

        tp = (p_flat * t_flat).sum()
        fp = (p_flat * (1.0 - t_flat)).sum()
        fn = ((1.0 - p_flat) * t_flat).sum()

        tversky_index = (tp + self.eps) / (tp + self.alpha * fn + self.beta * fp + self.eps)
        focal_tversky = torch.pow(1.0 - tversky_index, self.gamma)
        return focal_tversky


class LogicalConstraintLoss(nn.Module):
    """Loss penalizing violation of domain rule: lesion_prob <= leaf_prob."""

    def forward(self, lesion_logits: torch.Tensor, leaf_logits: torch.Tensor) -> torch.Tensor:
        lesion_p = torch.sigmoid(lesion_logits)
        leaf_p = torch.sigmoid(leaf_logits)
        violation = F.relu(lesion_p - leaf_p)
        return violation.mean()


class SoftSeverityConsistencyLoss(nn.Module):
    """Loss penalizing discrepancy between predicted soft severity fraction and ground truth severity."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        lesion_logits: torch.Tensor,
        leaf_logits: torch.Tensor,
        lesion_targets: torch.Tensor,
        leaf_targets: torch.Tensor
    ) -> torch.Tensor:
        lesion_p = torch.sigmoid(lesion_logits)
        leaf_p = torch.sigmoid(leaf_logits)
        leaf_p_constrained = torch.maximum(leaf_p, lesion_p)

        sum_lesion_pred = lesion_p.sum(dim=(-2, -1))
        sum_leaf_pred = leaf_p_constrained.sum(dim=(-2, -1))
        pred_sev = (sum_lesion_pred + self.eps) / (sum_leaf_pred + self.eps)

        sum_lesion_true = lesion_targets.float().sum(dim=(-2, -1))
        sum_leaf_true = torch.maximum(leaf_targets.float(), lesion_targets.float()).sum(dim=(-2, -1))
        true_sev = (sum_lesion_true + self.eps) / (sum_leaf_true + self.eps)

        return F.smooth_l1_loss(pred_sev, true_sev)


class CombinedBCEDiceLoss(nn.Module):
    """Combined weighted BCE and Dice Loss for binary segmentation."""

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: float = 5.0,
        eps: float = 1e-6
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight
        self.dice_fn = DiceLoss(eps=eps)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute combined BCE + Dice loss.

        Args:
            logits: Predicted logits [B, 1, H, W].
            targets: Binary targets [B, 1, H, W].
            valid_mask: Optional boolean tensor [B] indicating valid ground truth samples.

        Returns:
            Combined scalar loss tensor.
        """
        if valid_mask is not None and valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        if valid_mask is not None:
            sub_logits = logits[valid_mask]
            sub_targets = targets[valid_mask]
        else:
            sub_logits = logits
            sub_targets = targets

        pos_wt_tensor = torch.tensor([self.pos_weight], device=logits.device)
        bce_loss = F.binary_cross_entropy_with_logits(
            sub_logits, sub_targets.float(), pos_weight=pos_wt_tensor
        )

        dice_loss = self.dice_fn(sub_logits, sub_targets.float())
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def compute_severity_losses(
    lesion_logits: torch.Tensor,
    leaf_logits: torch.Tensor,
    lesion_targets: torch.Tensor,
    leaf_targets: torch.Tensor,
    has_lesion_mask: Optional[torch.Tensor] = None,
    has_leaf_mask: Optional[torch.Tensor] = None,
    lesion_pos_weight: float = 5.0,
    leaf_pos_weight: float = 1.0,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    lesion_aux_logits: Optional[torch.Tensor] = None,
    leaf_aux_logits: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """Compute combined multi-task lesion and leaf segmentation losses.

    Args:
        lesion_logits: Predicted lesion logits [B, 1, H, W].
        leaf_logits: Predicted leaf logits [B, 1, H, W].
        lesion_targets: Binary lesion ground truth [B, 1, H, W].
        leaf_targets: Binary leaf ground truth [B, 1, H, W].
        has_lesion_mask: Boolean tensor [B] indicating which samples have lesion ground truth.
        has_leaf_mask: Boolean tensor [B] indicating which samples have leaf ground truth.
        lesion_pos_weight: BCE positive class weight for rare lesion pixels.
        leaf_pos_weight: BCE positive class weight for leaf pixels.
        bce_weight: Weight for BCE term.
        dice_weight: Weight for Dice term.
        lesion_aux_logits: Optional auxiliary lesion logits for deep supervision.
        leaf_aux_logits: Optional auxiliary leaf logits for deep supervision.

    Returns:
        Dict containing 'total_loss', 'lesion_loss', 'leaf_loss', 'focal_tversky_loss',
        'constraint_loss', and 'severity_consistency_loss'.
    """
    lesion_loss_fn = CombinedBCEDiceLoss(
        bce_weight=bce_weight, dice_weight=dice_weight, pos_weight=lesion_pos_weight
    )
    leaf_loss_fn = CombinedBCEDiceLoss(
        bce_weight=bce_weight, dice_weight=dice_weight, pos_weight=leaf_pos_weight
    )

    focal_tversky_fn = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.33)
    constraint_fn = LogicalConstraintLoss()
    sev_consistency_fn = SoftSeverityConsistencyLoss()

    lesion_loss = lesion_loss_fn(lesion_logits, lesion_targets, valid_mask=has_lesion_mask)
    leaf_loss = leaf_loss_fn(leaf_logits, leaf_targets, valid_mask=has_leaf_mask)

    focal_tversky_loss = focal_tversky_fn(lesion_logits, lesion_targets, mask=has_lesion_mask)
    constraint_loss = constraint_fn(lesion_logits, leaf_logits)
    sev_consistency_loss = sev_consistency_fn(lesion_logits, leaf_logits, lesion_targets, leaf_targets)

    total_loss = (
        lesion_loss + leaf_loss +
        0.5 * focal_tversky_loss +
        1.0 * constraint_loss +
        2.0 * sev_consistency_loss
    )

    # Auxiliary loss if deep supervision is enabled
    if lesion_aux_logits is not None and leaf_aux_logits is not None:
        les_aux_loss = lesion_loss_fn(lesion_aux_logits, lesion_targets, valid_mask=has_lesion_mask)
        lf_aux_loss = leaf_loss_fn(leaf_aux_logits, leaf_targets, valid_mask=has_leaf_mask)
        total_loss = total_loss + 0.4 * (les_aux_loss + lf_aux_loss)

    return {
        "total_loss": total_loss,
        "lesion_loss": lesion_loss,
        "leaf_loss": leaf_loss,
        "focal_tversky_loss": focal_tversky_loss,
        "constraint_loss": constraint_loss,
        "severity_consistency_loss": sev_consistency_loss
    }
