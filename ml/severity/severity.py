"""Severity Fraction, Ordinal Binning, and Connected-Component Lesion Analysis.

Calculates disease severity percentage from predicted lesion and leaf mask logits/probabilities,
enforces structural leaf-union-lesion constraints, assigns ordinal severity classes, and computes
connected-component lesion count, mean/max lesion sizes, and lesion boundary lengths.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import cv2
from scipy import ndimage
import torch
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_BINS = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
DEFAULT_BIN_LABELS = ["<1%", "1-5%", "5-10%", "10-25%", "25-50%", ">50%"]


class SeverityMetricsResult(BaseModel):
    """Result object containing calculated severity metrics and lesion statistics."""
    severity_fraction_hard: float = Field(..., description="Hard binarized severity fraction (0.0 to 1.0).")
    severity_fraction_soft: float = Field(..., description="Soft probability sum severity fraction (0.0 to 1.0).")
    severity_percentage_hard: float = Field(..., description="Hard severity as a percentage (0.0 to 100.0%).")
    severity_percentage_soft: float = Field(..., description="Soft severity as a percentage (0.0 to 100.0%).")
    ordinal_class_idx: int = Field(..., description="Ordinal severity bin index (0 to 5).")
    ordinal_class_label: str = Field(..., description="Ordinal severity bin label (e.g. '1-5%').")
    leaf_pixel_count: int = Field(..., description="Total leaf area in pixels.")
    lesion_pixel_count: int = Field(..., description="Total lesion area in pixels.")
    senescence_pixel_count: int = Field(default=0, description="Total senescent/chlorotic leaf area in pixels.")
    senescence_fraction: float = Field(default=0.0, description="Senescent area as fraction of total leaf area.")
    senescence_percentage: float = Field(default=0.0, description="Senescent area as percentage of total leaf area.")
    n_lesions: int = Field(..., description="Count of distinct connected lesion components.")
    mean_lesion_area: float = Field(..., description="Average lesion area in pixels.")
    max_lesion_area: int = Field(..., description="Maximum lesion component area in pixels.")
    lesion_boundary_length: float = Field(..., description="Total perimeter length of lesion boundaries in pixels.")
    leaf_coverage_ratio: float = Field(..., description="Leaf coverage ratio relative to total image area.")


def map_severity_to_ordinal_class(
    severity_fraction: float,
    bins: Optional[List[float]] = None,
    bin_labels: Optional[List[str]] = None
) -> Tuple[int, str]:
    """Map a continuous severity fraction [0, 1] to an ordinal class index and human-readable label.

    Args:
        severity_fraction: Continuous disease severity fraction in [0.0, 1.0].
        bins: Bin edges list (default [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0]).
        bin_labels: List of label strings.

    Returns:
        Tuple of (ordinal_class_idx, ordinal_class_label).
    """
    if bins is None:
        bins = DEFAULT_BINS
    if bin_labels is None:
        bin_labels = DEFAULT_BIN_LABELS

    s_val = float(np.clip(severity_fraction, 0.0, 1.0))
    n_bins = len(bins) - 1

    for idx in range(n_bins):
        low = bins[idx]
        high = bins[idx + 1]
        if idx == n_bins - 1:
            if low <= s_val <= high:
                return idx, bin_labels[idx]
        else:
            if low <= s_val < high:
                return idx, bin_labels[idx]

    return n_bins - 1, bin_labels[-1]


def compute_connected_lesion_stats(binary_lesion_mask: np.ndarray) -> Dict[str, Any]:
    """Compute individual lesion statistics from a binary lesion mask.
    Employs distance-transform peak decomposition to resolve touching or overlapping
    foliar lesion spots, providing accurate count and mean spot dimensions.

    Args:
        binary_lesion_mask: 2D boolean or uint8 numpy array (H, W) where 1 indicates lesion.

    Returns:
        Dict containing n_lesions, mean_lesion_area, max_lesion_area, and lesion_boundary_length.
    """
    mask = (binary_lesion_mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return {
            "n_lesions": 0,
            "mean_lesion_area": 0.0,
            "max_lesion_area": 0,
            "lesion_boundary_length": 0.0
        }

    # 1. Connected components analysis
    num_conn, labels_conn, stats_conn, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_conn <= 1:
        return {
            "n_lesions": 0,
            "mean_lesion_area": 0.0,
            "max_lesion_area": 0,
            "lesion_boundary_length": 0.0
        }

    from scipy.ndimage import maximum_filter

    total_spots = 0
    spot_areas: List[float] = []
    max_area = 0

    for c_idx in range(1, num_conn):
        c_area = stats_conn[c_idx, cv2.CC_STAT_AREA]
        if c_area < 20:
            continue
        c_mask = (labels_conn == c_idx).astype(np.uint8)
        c_dist = cv2.distanceTransform(c_mask, cv2.DIST_L2, 5)
        max_d = c_dist.max()

        if max_d < 4.0 or c_area < 80:
            # Simple single small spot
            total_spots += 1
            spot_areas.append(float(c_area))
            max_area = max(max_area, int(c_area))
        else:
            # Decompose overlapping or touching spots via distance transform peaks
            peaks_in_c = (maximum_filter(c_dist, size=15) == c_dist) & (c_dist >= max(3.0, 0.3 * max_d))
            peaks_dil = cv2.morphologyEx(
                peaks_in_c.astype(np.uint8),
                cv2.MORPH_DILATE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            )
            n_p, _, _, _ = cv2.connectedComponentsWithStats(peaks_dil.astype(np.uint8))
            n_sub = max(1, n_p - 1)
            total_spots += n_sub
            sub_area = float(c_area / n_sub)
            for _ in range(n_sub):
                spot_areas.append(sub_area)
            max_area = max(max_area, int(sub_area))

    mean_area = float(np.mean(spot_areas)) if spot_areas else 0.0

    # Boundary length computation via erosion difference
    eroded = ndimage.binary_erosion(mask)
    boundary = mask.astype(bool) & ~eroded
    boundary_length = float(np.sum(boundary))

    return {
        "n_lesions": int(total_spots),
        "mean_lesion_area": round(mean_area, 2),
        "max_lesion_area": int(max_area),
        "lesion_boundary_length": round(boundary_length, 2)
    }


def calculate_severity_metrics(
    lesion_pred: Union[torch.Tensor, np.ndarray],
    leaf_pred: Union[torch.Tensor, np.ndarray],
    prob_threshold: float = 0.5,
    senescence_pred: Optional[Union[torch.Tensor, np.ndarray]] = None,
    days_since_heading: Optional[float] = None,
    bins: Optional[List[float]] = None,
    bin_labels: Optional[List[str]] = None
) -> SeverityMetricsResult:
    """Calculate severity fractions, ordinal classes, and connected-component lesion statistics.

    Enforces the logical structural constraint: leaf := leaf UNION lesion (leaf mask must contain lesion).
    Optionally partitions physiological senescence / chlorosis so natural aging does not inflate pathogen necrosis.

    Args:
        lesion_pred: Logit map or probability map for lesion prediction.
        leaf_pred: Logit map or probability map for leaf prediction.
        prob_threshold: Probability threshold for hard binarization.
        senescence_pred: Optional logit/probability map for physiological senescence.
        days_since_heading: Growth stage in days since heading.
        bins: Ordinal severity bin edges.
        bin_labels: Ordinal severity bin labels.

    Returns:
        SeverityMetricsResult instance.
    """
    # Convert PyTorch tensors to numpy arrays
    if isinstance(lesion_pred, torch.Tensor):
        lesion_pred = lesion_pred.detach().cpu().numpy()
    if isinstance(leaf_pred, torch.Tensor):
        leaf_pred = leaf_pred.detach().cpu().numpy()

    # Squeeze extra dimensions if present
    lesion_pred = np.squeeze(lesion_pred)
    leaf_pred = np.squeeze(leaf_pred)

    # Convert logits to probabilities if input range includes negatives or values outside [0, 1]
    if lesion_pred.min() < 0.0 or lesion_pred.max() > 1.0:
        lesion_prob = 1.0 / (1.0 + np.exp(-np.clip(lesion_pred, -20.0, 20.0)))
    else:
        lesion_prob = lesion_pred

    if leaf_pred.min() < 0.0 or leaf_pred.max() > 1.0:
        leaf_prob = 1.0 / (1.0 + np.exp(-np.clip(leaf_pred, -20.0, 20.0)))
    else:
        leaf_prob = leaf_pred

    # Parse optional senescence
    senescence_binary = np.zeros_like(leaf_prob, dtype=np.uint8)
    if senescence_pred is not None:
        if isinstance(senescence_pred, torch.Tensor):
            senescence_pred = senescence_pred.detach().cpu().numpy()
        senescence_pred = np.squeeze(senescence_pred)
        if senescence_pred.min() < 0.0 or senescence_pred.max() > 1.0:
            sen_prob = 1.0 / (1.0 + np.exp(-np.clip(senescence_pred, -20.0, 20.0)))
        else:
            sen_prob = senescence_pred

        # Penalize premature senescence if crop is before physiological maturity (e.g. heading stage < 30 days)
        if days_since_heading is not None and days_since_heading < 30.0:
            sen_prob = sen_prob * 0.4

        senescence_binary = (sen_prob >= prob_threshold).astype(np.uint8)

    # Soft constraint enforcement: leaf_prob = max(leaf_prob, lesion_prob)
    leaf_prob_constrained = np.maximum(leaf_prob, lesion_prob)

    # Hard binarization
    lesion_binary = (lesion_prob >= prob_threshold).astype(np.uint8)
    # Lesion takes precedence over senescence
    senescence_binary = (senescence_binary & (lesion_binary == 0)).astype(np.uint8)

    leaf_binary = (leaf_prob_constrained >= prob_threshold).astype(np.uint8)
    # Hard constraint enforcement: leaf_binary contains leaf, lesion, and senescence
    leaf_binary_constrained = (leaf_binary | lesion_binary | senescence_binary).astype(np.uint8)

    # Pixel counts
    lesion_pixel_count = int(np.sum(lesion_binary))
    leaf_pixel_count = int(np.sum(leaf_binary_constrained))
    senescence_pixel_count = int(np.sum(senescence_binary))
    total_image_pixels = float(lesion_binary.size)

    # Severity calculations
    if leaf_pixel_count > 0:
        severity_hard = float(np.clip(lesion_pixel_count / float(leaf_pixel_count), 0.0, 1.0))
        senescence_frac = float(np.clip(senescence_pixel_count / float(leaf_pixel_count), 0.0, 1.0))
    else:
        severity_hard = 0.0
        senescence_frac = 0.0

    sum_lesion_prob = float(np.sum(lesion_prob))
    sum_leaf_prob = float(np.sum(leaf_prob_constrained))
    if sum_leaf_prob > 0.0:
        severity_soft = float(np.clip(sum_lesion_prob / sum_leaf_prob, 0.0, 1.0))
    else:
        severity_soft = 0.0

    # Ordinal class assignment
    ordinal_idx, ordinal_label = map_severity_to_ordinal_class(severity_hard, bins=bins, bin_labels=bin_labels)

    # Connected component lesion stats
    lesion_stats = compute_connected_lesion_stats(lesion_binary)

    leaf_coverage = float(leaf_pixel_count / total_image_pixels) if total_image_pixels > 0 else 0.0

    return SeverityMetricsResult(
        severity_fraction_hard=round(severity_hard, 6),
        severity_fraction_soft=round(severity_soft, 6),
        severity_percentage_hard=round(severity_hard * 100.0, 4),
        severity_percentage_soft=round(severity_soft * 100.0, 4),
        ordinal_class_idx=ordinal_idx,
        ordinal_class_label=ordinal_label,
        leaf_pixel_count=leaf_pixel_count,
        lesion_pixel_count=lesion_pixel_count,
        senescence_pixel_count=senescence_pixel_count,
        senescence_fraction=round(senescence_frac, 6),
        senescence_percentage=round(senescence_frac * 100.0, 4),
        n_lesions=lesion_stats["n_lesions"],
        mean_lesion_area=lesion_stats["mean_lesion_area"],
        max_lesion_area=lesion_stats["max_lesion_area"],
        lesion_boundary_length=lesion_stats["lesion_boundary_length"],
        leaf_coverage_ratio=round(leaf_coverage, 4)
    )


def calculate_multiclass_severity_metrics(
    multiclass_pred: Union[torch.Tensor, np.ndarray],
    days_since_heading: Optional[float] = None,
    bins: Optional[List[float]] = None,
    bin_labels: Optional[List[str]] = None
) -> SeverityMetricsResult:
    """Calculate severity from a 4-channel pixel logits tensor:
    Channel 0: Background
    Channel 1: Healthy Green Photosynthetic
    Channel 2: Physiological Senescence / Chlorosis
    Channel 3: Active Pathogen Lesion / Necrosis
    """
    if isinstance(multiclass_pred, torch.Tensor):
        logits = multiclass_pred.detach().cpu().numpy()
    else:
        logits = np.array(multiclass_pred)

    if logits.ndim == 4:
        logits = logits.squeeze(0)

    # Apply softmax across channels if needed
    if logits.shape[0] == 4:
        exp_l = np.exp(logits - np.max(logits, axis=0, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=0, keepdims=True)

        healthy_prob = probs[1]
        senescence_prob = probs[2]
        lesion_prob = probs[3]

        leaf_prob = healthy_prob + senescence_prob + lesion_prob
        return calculate_severity_metrics(
            lesion_pred=lesion_prob,
            leaf_pred=leaf_prob,
            senescence_pred=senescence_prob,
            days_since_heading=days_since_heading,
            bins=bins,
            bin_labels=bin_labels
        )
    else:
        raise ValueError(f"Expected 4-channel multiclass input, got shape {logits.shape}")
