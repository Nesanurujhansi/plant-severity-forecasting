"""Unit Tests for Severity Module (CPU Execution Only).

Tests decoder heads, losses, severity metric calculations, baselines, Mondrian conformal prediction,
SeverityPredictor CPU inference, and artifact export tooling.
"""

import sys
import json
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.severity.heads import SeveritySegmentationHeads, build_severity_heads
from ml.severity.losses import CombinedBCEDiceLoss, DiceLoss, compute_severity_losses
from ml.severity.severity import (
    calculate_severity_metrics,
    map_severity_to_ordinal_class,
    compute_connected_lesion_stats
)
from ml.severity.baselines import (
    ClassicalColorThresholdBaseline,
    ConstantMeanBaseline,
    evaluate_baselines
)
from ml.severity.conformal import MondrianConformalCalibrator
from ml.severity.predict import SeverityPredictor, benchmark_cpu_latency
from ml.severity.export import export_severity_artifacts


def test_severity_segmentation_heads_shape_and_params():
    """Test SeveritySegmentationHeads output logit tensor shapes and parameter budget."""
    model = build_severity_heads(config_dict={"input_dim": 384, "reduced_dim": 64})
    model.eval()

    # Input patch grid [B, 24, 32, 384]
    x = torch.randn(2, 24, 32, 384)
    with torch.no_grad():
        outputs = model(x, target_size=(112, 112), return_aux=True)

    assert "lesion_logits" in outputs
    assert "leaf_logits" in outputs
    assert "lesion_aux_logits" in outputs
    assert "leaf_aux_logits" in outputs
    assert outputs["lesion_logits"].shape == (2, 1, 112, 112)
    assert outputs["leaf_logits"].shape == (2, 1, 112, 112)
    assert outputs["lesion_aux_logits"].shape == (2, 1, 112, 112)

    params = model.count_parameters()
    assert params["total_parameters"] < 200000, f"Parameter count {params['total_parameters']} exceeds 200k limit."


def test_combined_bce_dice_and_multi_task_losses():
    """Test multi-task losses including FocalTverskyLoss, LogicalConstraintLoss, and SoftSeverityConsistencyLoss."""
    logits = torch.randn(4, 1, 28, 28)
    targets = torch.zeros(4, 1, 28, 28)
    targets[0, 0, 5:10, 5:10] = 1.0

    valid_mask = torch.tensor([True, True, False, False])
    loss_dict = compute_severity_losses(
        lesion_logits=logits,
        leaf_logits=logits,
        lesion_targets=targets,
        leaf_targets=targets,
        has_lesion_mask=valid_mask,
        has_leaf_mask=valid_mask
    )

    assert "total_loss" in loss_dict
    assert "focal_tversky_loss" in loss_dict
    assert "constraint_loss" in loss_dict
    assert "severity_consistency_loss" in loss_dict
    assert loss_dict["total_loss"].item() > 0.0


def test_severity_metrics_calculation_and_constraints():
    """Test hard/soft severity calculation, constraint enforcement, and connected component stats."""
    lesion_pred = torch.full((1, 100, 100), -5.0)
    leaf_pred = torch.full((1, 100, 100), -5.0)

    # Leaf area 20x20 = 400 pixels
    leaf_pred[0, 10:30, 10:30] = 5.0  # High positive logit
    # Lesion area 10x10 = 100 pixels inside leaf
    lesion_pred[0, 15:25, 15:25] = 5.0

    metrics = calculate_severity_metrics(lesion_pred, leaf_pred, prob_threshold=0.5)

    assert metrics.leaf_pixel_count == 400
    assert metrics.lesion_pixel_count == 100
    assert abs(metrics.severity_fraction_hard - 0.25) < 1e-3
    assert abs(metrics.severity_percentage_hard - 25.0) < 1e-2
    assert metrics.n_lesions == 1
    assert metrics.max_lesion_area == 100
    assert metrics.ordinal_class_label == "25-50%"


def test_baselines_evaluation():
    """Test ClassicalColorThresholdBaseline and ConstantMeanBaseline."""
    img_np = np.zeros((100, 100, 3), dtype=np.uint8)
    img_np[10:50, 10:50, 1] = 200  # Green leaf
    img_np[20:30, 20:30, 0] = 220  # Reddish rust lesion

    img_pil = Image.fromarray(img_np)

    baseline = ClassicalColorThresholdBaseline()
    lesion_m, leaf_m, sev = baseline.predict_masks(img_pil)

    assert leaf_m.sum() > 0
    assert 0.0 <= sev <= 1.0

    eval_out = evaluate_baselines([img_pil], np.array([0.15]), train_severities=np.array([0.10, 0.20]))
    assert "color_threshold_baseline" in eval_out
    assert "constant_mean_baseline" in eval_out


def test_mondrian_conformal_calibrator():
    """Test Mondrian split conformal calibration, interval prediction, and JSON serialization."""
    np.random.seed(42)
    y_true = np.random.uniform(0.0, 0.5, size=50)
    y_pred = y_true + np.random.normal(0.0, 0.02, size=50)

    calibrator = MondrianConformalCalibrator(alpha=0.10, min_bin_samples=5)
    fit_summary = calibrator.fit(y_true, y_pred)

    assert calibrator.is_calibrated
    assert fit_summary["confidence_level"] == 0.90

    # Predict interval for new prediction 0.20
    lower, upper, width = calibrator.predict_interval(0.20)
    assert 0.0 <= lower <= 0.20 <= upper <= 1.0
    assert abs(width - (upper - lower)) < 1e-3

    # Evaluate coverage
    cov_eval = calibrator.evaluate_coverage(y_true, y_pred)
    assert cov_eval["empirical_coverage"] >= 0.80

    # Test serialization
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_p = Path(tmp_dir) / "conformal.json"
        calibrator.save_json(json_p)
        assert json_p.exists()

        loaded_calibrator = MondrianConformalCalibrator.load_json(json_p)
        assert loaded_calibrator.is_calibrated
        assert loaded_calibrator.global_quantile == calibrator.global_quantile


def test_severity_predictor_cpu_inference():
    """Test SeverityPredictor end-to-end CPU prediction with stub weights."""
    predictor = SeverityPredictor(is_stub=True, device="cpu", resolution=448)

    test_img = Image.fromarray(np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8))
    result = predictor.predict(test_img)

    assert 0.0 <= result.severity_fraction <= 1.0
    assert 0.0 <= result.prediction_interval_lower <= result.prediction_interval_upper <= 1.0
    assert len(result.cls_feature_vector) == 384
    assert result.device == "cpu"
    assert result.inference_time_sec > 0.0


def test_cpu_latency_benchmark():
    """Test benchmark_cpu_latency utility function."""
    res = benchmark_cpu_latency(n_iterations=2, resolution=448)
    assert "mean_latency_sec" in res
    assert "throughput_fps" in res
    assert res["mean_latency_sec"] > 0.0


def test_export_severity_artifacts():
    """Test artifact bundling and metadata embedding inside safetensors header."""
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as export_dir:
        src_p, exp_p = Path(src_dir), Path(export_dir)

        # Create dummy weights and configs
        model = build_severity_heads()
        torch.save(model.state_dict(), src_p / "dummy.pt")

        calibrator = MondrianConformalCalibrator()
        calibrator.save_json(src_p / "conformal.json")

        exported = export_severity_artifacts(source_dir=src_p, export_dir=exp_p)

        assert (exp_p / "severity_heads.safetensors").exists()
        assert (exp_p / "config.json").exists()
        assert (exp_p / "conformal.json").exists()
