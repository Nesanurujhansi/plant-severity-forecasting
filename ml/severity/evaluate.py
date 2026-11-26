"""Evaluation Metrics and Visualizations for Severity Module.

Computes segmentation Dice and IoU scores, severity MAE (percentage points), RMSE,
Lin's Concordance Correlation Coefficient (CCC), Bland-Altman bias and limits of agreement,
ordinal classification accuracy, and Quadratic Weighted Kappa (QWK), generating tables and plots.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger
from ml.severity.severity import calculate_severity_metrics, map_severity_to_ordinal_class

logger = get_logger(__name__)


def compute_dice_iou(pred_binary: np.ndarray, target_binary: np.ndarray, eps: float = 1e-6) -> Tuple[float, float]:
    """Compute Dice coefficient and Intersection over Union (IoU) for binary masks.

    Args:
        pred_binary: Binary predicted mask (H, W) or (N, H, W) (0 or 1).
        target_binary: Binary ground truth mask (H, W) or (N, H, W) (0 or 1).
        eps: Small epsilon to prevent zero division.

    Returns:
        Tuple of (dice_score, iou_score).
    """
    p = (pred_binary > 0).astype(np.float32).reshape(-1)
    t = (target_binary > 0).astype(np.float32).reshape(-1)

    intersection = np.sum(p * t)
    cardinality = np.sum(p) + np.sum(t)
    union = cardinality - intersection

    dice = float((2.0 * intersection + eps) / (cardinality + eps))
    iou = float((intersection + eps) / (union + eps))
    return round(dice, 4), round(iou, 4)


def compute_lins_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Lin's Concordance Correlation Coefficient (CCC).

    CCC = (2 * s_xy) / (s_x^2 + s_y^2 + (mean_x - mean_y)^2)
    """
    x = np.array(y_true, dtype=np.float64)
    y = np.array(y_pred, dtype=np.float64)

    if len(x) < 2 or np.var(x) == 0 and np.var(y) == 0:
        return 1.0

    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    cov_xy = np.cov(x, y)[0, 1]

    numerator = 2.0 * cov_xy
    denominator = var_x + var_y + (mean_x - mean_y) ** 2

    if denominator == 0:
        return 1.0

    ccc = numerator / denominator
    return round(float(ccc), 4)


def compute_bland_altman_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute Bland-Altman mean bias and 95% limits of agreement (1.96 * SD)."""
    diffs = np.array(y_pred, dtype=np.float64) - np.array(y_true, dtype=np.float64)
    mean_bias = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

    lower_loa = mean_bias - 1.96 * std_diff
    upper_loa = mean_bias + 1.96 * std_diff

    return {
        "mean_bias": round(mean_bias, 6),
        "mean_bias_pct": round(mean_bias * 100.0, 4),
        "std_diff": round(std_diff, 6),
        "lower_limit_of_agreement": round(lower_loa, 6),
        "upper_limit_of_agreement": round(upper_loa, 6),
        "lower_loa_pct": round(lower_loa * 100.0, 4),
        "upper_loa_pct": round(upper_loa * 100.0, 4)
    }


def compute_quadratic_weighted_kappa(y_true_ordinal: np.ndarray, y_pred_ordinal: np.ndarray, n_classes: int = 6) -> float:
    """Compute Quadratic Weighted Kappa (QWK) for ordinal severity classes."""
    from sklearn.metrics import cohen_kappa_score
    try:
        qwk = cohen_kappa_score(y_true_ordinal, y_pred_ordinal, weights="quadratic")
        return round(float(qwk), 4)
    except Exception:
        return 0.0


def evaluate_severity_predictions(
    y_true_fraction: np.ndarray,
    y_pred_fraction: np.ndarray,
    lesion_preds: Optional[np.ndarray] = None,
    lesion_targets: Optional[np.ndarray] = None,
    leaf_preds: Optional[np.ndarray] = None,
    leaf_targets: Optional[np.ndarray] = None,
    sources: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics for severity estimation.

    Args:
        y_true_fraction: Ground truth severity fractions [N] in [0, 1].
        y_pred_fraction: Predicted severity fractions [N] in [0, 1].
        lesion_preds: Optional binary lesion prediction masks [N, H, W].
        lesion_targets: Optional binary lesion target masks [N, H, W].
        leaf_preds: Optional binary leaf prediction masks [N, H, W].
        leaf_targets: Optional binary leaf target masks [N, H, W].
        sources: Optional dataset source labels per sample.

    Returns:
        Dict containing MAE, RMSE, CCC, Bland-Altman stats, Ordinal Accuracy, QWK, and Dice/IoU scores.
    """
    y_true = np.clip(np.array(y_true_fraction, dtype=np.float32), 0.0, 1.0)
    y_pred = np.clip(np.array(y_pred_fraction, dtype=np.float32), 0.0, 1.0)
    n_samples = len(y_true)

    mae_pct = float(np.mean(np.abs(y_true - y_pred)) * 100.0)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ccc = compute_lins_ccc(y_true, y_pred)
    ba_stats = compute_bland_altman_stats(y_true, y_pred)

    # Ordinal class evaluation
    y_true_ord = np.array([map_severity_to_ordinal_class(val)[0] for val in y_true])
    y_pred_ord = np.array([map_severity_to_ordinal_class(val)[0] for val in y_pred])
    ordinal_acc = float(np.mean(y_true_ord == y_pred_ord))
    qwk = compute_quadratic_weighted_kappa(y_true_ord, y_pred_ord)

    # Segmentation metrics if masks provided
    segmentation_metrics = {}
    if lesion_preds is not None and lesion_targets is not None:
        l_dice, l_iou = compute_dice_iou(lesion_preds, lesion_targets)
        segmentation_metrics["lesion_dice"] = l_dice
        segmentation_metrics["lesion_iou"] = l_iou

    if leaf_preds is not None and leaf_targets is not None:
        lf_dice, lf_iou = compute_dice_iou(leaf_preds, leaf_targets)
        segmentation_metrics["leaf_dice"] = lf_dice
        segmentation_metrics["leaf_iou"] = lf_iou

    # Per-source breakdown if sources provided
    source_breakdown = {}
    if sources is not None and len(sources) == n_samples:
        for s in set(sources):
            s_mask = np.array([src == s for src in sources])
            if np.sum(s_mask) > 0:
                s_mae = float(np.mean(np.abs(y_true[s_mask] - y_pred[s_mask])) * 100.0)
                s_ccc = compute_lins_ccc(y_true[s_mask], y_pred[s_mask])
                source_breakdown[s] = {
                    "count": int(np.sum(s_mask)),
                    "mae_percentage_points": round(s_mae, 4),
                    "ccc": s_ccc
                }

    return {
        "n_samples": n_samples,
        "severity_mae_percentage_points": round(mae_pct, 4),
        "severity_rmse": round(rmse, 4),
        "lins_ccc": ccc,
        "bland_altman": ba_stats,
        "ordinal_accuracy": round(ordinal_acc, 4),
        "quadratic_weighted_kappa": qwk,
        "segmentation": segmentation_metrics,
        "source_breakdown": source_breakdown
    }


def generate_evaluation_reports(
    eval_results: Dict[str, Any],
    reports_dir: Optional[Path] = None
) -> Tuple[Path, Path]:
    """Export evaluation summary tables to reports/tables and text summary to reports/."""
    paths = get_paths()
    if reports_dir is None:
        reports_dir = paths.reports_dir
    tables_dir = reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Main summary metrics table
    metrics_summary = [
        {"metric": "Severity MAE (% points)", "value": eval_results.get("severity_mae_percentage_points")},
        {"metric": "Severity RMSE", "value": eval_results.get("severity_rmse")},
        {"metric": "Lin's CCC", "value": eval_results.get("lins_ccc")},
        {"metric": "Bland-Altman Bias (%)", "value": eval_results.get("bland_altman", {}).get("mean_bias_pct")},
        {"metric": "Ordinal Accuracy", "value": eval_results.get("ordinal_accuracy")},
        {"metric": "Quadratic Weighted Kappa", "value": eval_results.get("quadratic_weighted_kappa")},
        {"metric": "Lesion Dice", "value": eval_results.get("segmentation", {}).get("lesion_dice", "N/A")},
        {"metric": "Leaf Dice", "value": eval_results.get("segmentation", {}).get("leaf_dice", "N/A")},
    ]

    df_metrics = pd.DataFrame(metrics_summary)
    csv_path = tables_dir / "phase05_severity_metrics.csv"
    df_metrics.to_csv(csv_path, index=False)

    # 2. Text summary
    txt_path = reports_dir / "phase05_evaluation_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("PHASE 05 SEVERITY MODULE EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n")
        for row in metrics_summary:
            f.write(f"{row['metric']:<30}: {row['value']}\n")
        f.write("=" * 60 + "\n")

    logger.info(f"Exported evaluation report to {csv_path} and {txt_path}")
    return csv_path, txt_path
