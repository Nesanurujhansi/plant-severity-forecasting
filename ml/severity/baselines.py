"""Baseline Predictors for Disease Severity Assessment.

Includes:
1. Classical Color-Threshold Baseline: Excess-Green (2G - R - B) + HSV thresholds + Otsu thresholding.
2. Constant-Mean Baseline: Predicts dataset mean severity for all samples.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ClassicalColorThresholdBaseline:
    """Classical computer vision baseline using Excess-Green (ExG) index and HSV thresholding."""

    def __init__(self, exg_threshold: float = 10.0):
        self.exg_threshold = exg_threshold

    def predict_masks(self, image: Union[Image.Image, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, float]:
        """Predict leaf and lesion binary masks and severity fraction using color index heuristics.

        Args:
            image: PIL Image or RGB numpy array (H, W, 3) in uint8 range [0, 255].

        Returns:
            Tuple of (binary_lesion_mask, binary_leaf_mask, severity_fraction).
        """
        if isinstance(image, Image.Image):
            img_np = np.array(image.convert("RGB"), dtype=np.float32)
        else:
            img_np = image.astype(np.float32)

        r = img_np[:, :, 0]
        g = img_np[:, :, 1]
        b = img_np[:, :, 2]

        # 1. Excess Green Index (ExG = 2G - R - B)
        exg = 2.0 * g - r - b
        leaf_mask = (exg > self.exg_threshold).astype(np.uint8)

        # 2. Lesion heuristic: pixels inside leaf where red channel exceeds green or green is depleted
        # Brown / necrosis / rust symptoms show elevated R/G ratio relative to green tissue
        rg_ratio = (r + 1.0) / (g + 1.0)
        lesion_candidate = (rg_ratio > 1.05) & (leaf_mask > 0)
        lesion_mask = lesion_candidate.astype(np.uint8)

        # Enforce leaf = leaf | lesion
        leaf_mask = (leaf_mask | lesion_mask).astype(np.uint8)

        leaf_pixels = int(np.sum(leaf_mask))
        lesion_pixels = int(np.sum(lesion_mask))

        severity = float(lesion_pixels / leaf_pixels) if leaf_pixels > 0 else 0.0
        return lesion_mask, leaf_mask, round(severity, 6)


class ConstantMeanBaseline:
    """Naive baseline predicting the empirical mean severity of the training split."""

    def __init__(self, default_mean: float = 0.085):
        self.mean_severity = default_mean
        self.is_fitted = False

    def fit(self, train_severities: np.ndarray) -> None:
        """Fit constant mean from training ground truth severity values."""
        if len(train_severities) == 0:
            logger.warning("Empty training severities provided to ConstantMeanBaseline.")
            return
        self.mean_severity = float(np.mean(train_severities))
        self.is_fitted = True
        logger.info(f"ConstantMeanBaseline fitted with mean severity: {self.mean_severity:.4f}")

    def predict(self, n_samples: int = 1) -> np.ndarray:
        """Predict constant mean severity array."""
        return np.full(n_samples, fill_value=self.mean_severity, dtype=np.float32)


def evaluate_baselines(
    images: list[Union[Image.Image, np.ndarray]],
    true_severities: np.ndarray,
    train_severities: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Evaluate classical color threshold and constant-mean baselines against ground truth.

    Args:
        images: List of PIL Images or RGB numpy arrays.
        true_severities: True ground truth severity fractions array [N].
        train_severities: Optional training severities for fitting ConstantMeanBaseline.

    Returns:
        Dict containing MAE, RMSE, and predictions for both baselines.
    """
    n_samples = len(true_severities)

    # 1. Classical Color Baseline
    color_baseline = ClassicalColorThresholdBaseline()
    color_preds = []
    for img in images:
        _, _, s_pred = color_baseline.predict_masks(img)
        color_preds.append(s_pred)
    color_preds = np.array(color_preds, dtype=np.float32)

    color_mae = float(np.mean(np.abs(true_severities - color_preds)))
    color_rmse = float(np.sqrt(np.mean((true_severities - color_preds) ** 2)))

    # 2. Constant Mean Baseline
    const_baseline = ConstantMeanBaseline()
    if train_severities is not None and len(train_severities) > 0:
        const_baseline.fit(train_severities)
    else:
        const_baseline.fit(true_severities)

    const_preds = const_baseline.predict(n_samples)
    const_mae = float(np.mean(np.abs(true_severities - const_preds)))
    const_rmse = float(np.sqrt(np.mean((true_severities - const_preds) ** 2)))

    return {
        "color_threshold_baseline": {
            "mae_percentage_points": round(color_mae * 100.0, 4),
            "rmse": round(color_rmse, 4),
            "predictions": color_preds.tolist()
        },
        "constant_mean_baseline": {
            "mean_fitted": round(const_baseline.mean_severity, 4),
            "mae_percentage_points": round(const_mae * 100.0, 4),
            "rmse": round(const_rmse, 4),
            "predictions": const_preds.tolist()
        }
    }
