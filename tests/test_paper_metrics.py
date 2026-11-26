"""Unit and Integration Tests for IEEE Research Paper Evaluation Metrics.

Verifies mathematical correctness of all segmentation, regression, conformal,
ordinal classification, ROC-AUC, and forecasting evaluation metrics, as well as
the exported reports/tables/paper_evaluation_metrics.json artifact.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_paper_evaluation import (
    compute_pixel_metrics,
    evaluate_regression_series,
    evaluate_binary_roc,
)
from ml.severity.evaluate import (
    compute_dice_iou,
    compute_lins_ccc,
    compute_bland_altman_stats,
    compute_quadratic_weighted_kappa,
)
from ml.forecast.evaluate import compute_pinball_loss, compute_forecast_metrics


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_pixel_segmentation_metrics_perfect():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:16, 8:16] = 1
    m = compute_pixel_metrics(mask, mask)
    assert m["dice"] == pytest.approx(1.0, abs=1e-4)
    assert m["iou"] == pytest.approx(1.0, abs=1e-4)
    assert m["pixel_accuracy"] == pytest.approx(1.0, abs=1e-4)
    assert m["precision"] == pytest.approx(1.0, abs=1e-4)
    assert m["recall"] == pytest.approx(1.0, abs=1e-4)
    assert m["specificity"] == pytest.approx(1.0, abs=1e-4)


def test_pixel_segmentation_metrics_partial_overlap():
    gt = np.zeros((10, 10), dtype=np.uint8)
    pred = np.zeros((10, 10), dtype=np.uint8)
    gt[0:4, 0:4] = 1      # 16 pixels
    pred[0:4, 2:6] = 1    # 16 pixels; overlap = 8 pixels
    m = compute_pixel_metrics(pred, gt)
    # Dice = 2*8 / (16+16) = 0.5; IoU = 8 / 24 = 0.3333
    assert m["dice"] == pytest.approx(0.5, abs=1e-3)
    assert m["iou"] == pytest.approx(1.0 / 3.0, abs=1e-3)
    assert m["precision"] == pytest.approx(0.5, abs=1e-3)
    assert m["recall"] == pytest.approx(0.5, abs=1e-3)


def test_regression_and_agreement_metrics():
    y_true = np.array([2.0, 5.0, 10.0, 20.0, 35.0])
    y_pred = np.array([2.5, 4.5, 11.0, 19.0, 34.0])
    reg = evaluate_regression_series(y_true, y_pred)
    assert reg["mae_pct"] == pytest.approx(0.8, abs=1e-3)
    assert reg["r2"] > 0.99
    assert reg["lins_ccc"] > 0.99
    assert reg["pearson_r"] > 0.99
    assert reg["spearman_rho"] == pytest.approx(1.0, abs=1e-3)


def test_binary_roc_auc_separation():
    y_true = np.array([1.0, 2.0, 3.0, 8.0, 12.0, 20.0])
    y_pred = np.array([1.2, 2.5, 3.1, 7.8, 11.5, 19.0])
    roc = evaluate_binary_roc(y_true, y_pred, threshold=5.0)
    assert roc["roc_auc"] == pytest.approx(1.0, abs=1e-4)
    assert roc["pr_auc"] == pytest.approx(1.0, abs=1e-4)
    assert roc["accuracy"] == pytest.approx(1.0, abs=1e-4)
    assert roc["f1_score"] == pytest.approx(1.0, abs=1e-4)


def test_ordinal_qwk_and_forecast_pinball():
    y_ord = np.array([0, 1, 2, 3, 4, 5])
    assert compute_quadratic_weighted_kappa(y_ord, y_ord) == pytest.approx(1.0, abs=1e-4)

    yt = np.array([0.10, 0.20, 0.30])
    yp = np.array([0.10, 0.20, 0.30])
    assert compute_pinball_loss(yt, yp, 0.5) == pytest.approx(0.0, abs=1e-6)

    fm = compute_forecast_metrics(
        y_true=yt,
        y_pred_med=yp,
        y_pred_low=yt - 0.02,
        y_pred_high=yt + 0.02,
        y_persistence=np.array([0.05, 0.10, 0.15]),
    )
    assert fm["coverage_90_pct"] == pytest.approx(100.0, abs=1e-2)
    assert fm["mae_pct_points"] == pytest.approx(0.0, abs=1e-4)
    assert fm["skill_score_vs_persistence"] == pytest.approx(1.0, abs=1e-4)


def test_exported_paper_evaluation_artifacts():
    metrics_json = PROJECT_ROOT / "reports" / "tables" / "paper_evaluation_metrics.json"
    if not metrics_json.exists():
        pytest.skip("paper_evaluation_metrics.json not yet generated.")

    data = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert data["segmentation"]["n_images"] == 100
    assert 0.0 < data["segmentation"]["macro_dice"] <= 1.0
    assert 0.0 < data["segmentation"]["macro_iou"] <= 1.0
    assert 0.0 <= data["binary_detection_roc"]["threshold_5pct"]["roc_auc"] <= 1.0
    assert 0.0 <= data["binary_detection_roc"]["threshold_10pct"]["roc_auc"] <= 1.0
    assert data["forecast_7day_real_nwp_eval"]["n_real_nwp_locations"] >= 1
    assert data["forecast_7day_real_nwp_eval"]["interval_bound_compliance_pct"] == 100.0
