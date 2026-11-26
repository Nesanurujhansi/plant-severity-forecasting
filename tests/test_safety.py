"""Unit tests for Phase 08 Safety Layer and OOD Detection.

Tests image quality checks, Free Energy/Mahalanobis OOD detection, input range validation,
policy combination logic, per-module gating, stress testing, and zero-pickle export.
"""

import os
import json
import pytest
import numpy as np
from pathlib import Path
from safetensors.torch import load_file

from ml.safety.quality import (
    compute_laplacian_variance,
    check_image_blur,
    check_image_dimensions,
    check_exposure_and_brightness,
    check_leaf_coverage,
    evaluate_image_quality
)
from ml.safety.ood import (
    MahalanobisDistanceClassifier,
    KNNDistanceClassifier,
    FreeEnergyClassifier,
    DeepOODSafetyEvaluator
)
from ml.safety.ranges import (
    validate_weather_inputs,
    validate_time_horizon,
    validate_crop_type,
    evaluate_input_ranges
)
from ml.safety.policy import evaluate_safety_policy, evaluate_full_safety
from ml.safety.stress_test import run_synthetic_stress_tests
from ml.safety.export import export_safety_artifacts


def test_quality_check_dimensions():
    img_small = np.zeros((100, 100, 3), dtype=np.uint8)
    res_small = check_image_dimensions(img_small, min_width=224, min_height=224)
    assert res_small["passed"] is False
    assert "too low" in res_small["message"]

    img_ok = np.zeros((300, 400, 3), dtype=np.uint8)
    res_ok = check_image_dimensions(img_ok, min_width=224, min_height=224)
    assert res_ok["passed"] is True


def test_quality_check_blur():
    # Crisp image: sharp checkerboard pattern
    crisp = np.zeros((200, 200), dtype=np.uint8)
    crisp[::2, ::2] = 255
    res_crisp = check_image_blur(crisp, blur_threshold=50.0)
    assert res_crisp["passed"] is True

    # Blurred image: flat smooth surface
    blurred = np.ones((200, 200), dtype=np.uint8) * 128
    res_blurred = check_image_blur(blurred, blur_threshold=50.0)
    assert res_blurred["passed"] is False
    assert "blurry" in res_blurred["message"]


def test_quality_check_exposure_and_brightness():
    # Extremely dark image
    black_img = np.zeros((200, 200), dtype=np.uint8)
    res_dark = check_exposure_and_brightness(black_img, min_brightness=40.0)
    assert res_dark["passed"] is False
    assert "too dark" in res_dark["message"]

    # Extremely bright overexposed image
    white_img = np.ones((200, 200), dtype=np.uint8) * 250
    res_bright = check_exposure_and_brightness(white_img, max_overexposure_frac=0.20)
    assert res_bright["passed"] is False

    # Normal lighting image
    normal_img = (np.random.rand(200, 200) * 150 + 50).astype(np.uint8)
    res_norm = check_exposure_and_brightness(normal_img)
    assert res_norm["passed"] is True


def test_quality_check_leaf_coverage():
    # High leaf coverage
    mask_full = np.ones((200, 200), dtype=np.uint8)
    res_full = check_leaf_coverage(mask_full, min_coverage_frac=0.10)
    assert res_full["passed"] is True

    # Low leaf coverage
    mask_tiny = np.zeros((200, 200), dtype=np.uint8)
    mask_tiny[:5, :5] = 1
    res_tiny = check_leaf_coverage(mask_tiny, min_coverage_frac=0.10)
    assert res_tiny["passed"] is False


def test_ood_mahalanobis_classifier():
    clf = MahalanobisDistanceClassifier(feature_dim=384, p95_threshold=10.0, p99_threshold=20.0)

    # In-distribution sample near zero mean
    feat_in = np.zeros(384, dtype=np.float32)
    res_in = clf.predict(feat_in)
    assert res_in["status"] == "IN_DISTRIBUTION"

    # Out-of-distribution sample far from mean
    feat_ood = np.ones(384, dtype=np.float32) * 5.0
    res_ood = clf.predict(feat_ood)
    assert res_ood["status"] == "OUT_OF_DISTRIBUTION"


def test_ood_knn_classifier():
    refs = np.random.randn(20, 384).astype(np.float32) * 0.01
    clf = KNNDistanceClassifier(reference_embeddings=refs, feature_dim=384, p95_threshold=1.0, p99_threshold=2.0)
    feat_in = refs[0] + np.random.randn(384).astype(np.float32) * 0.001
    res_in = clf.predict(feat_in)
    assert res_in["status"] in ["IN_DISTRIBUTION", "BORDERLINE"]

    feat_ood = np.ones(384, dtype=np.float32) * 50.0
    res_ood = clf.predict(feat_ood)
    assert res_ood["status"] == "OUT_OF_DISTRIBUTION"


def test_ood_free_energy_classifier():
    clf = FreeEnergyClassifier(energy_threshold=8.5)
    # High logits produce low free energy (in-distribution)
    logits_in = np.array([10.0, 5.0, 2.0])
    res_in = clf.predict(np.zeros(384), logits=logits_in)
    assert res_in["status"] == "IN_DISTRIBUTION"

    # Negative logits produce high free energy (OOD)
    logits_ood = np.array([-15.0, -20.0, -18.0])
    res_ood = clf.predict(np.zeros(384), logits=logits_ood)
    assert res_ood["status"] == "OUT_OF_DISTRIBUTION"


def test_range_validation():
    # Valid weather
    res_valid = validate_weather_inputs(temp_c=22.0, rh_pct=65.0, vpd_kpa=1.2)
    assert res_valid["passed"] is True

    # Extreme temperature out of bounds
    res_temp = validate_weather_inputs(temp_c=55.0, rh_pct=65.0)
    assert res_temp["passed"] is False
    assert len(res_temp["warnings"]) > 0

    # Crop type validation
    crop_wheat = validate_crop_type("Wheat")
    assert crop_wheat["status"] == "FULLY_VALIDATED"
    assert crop_wheat["severity_allowed"] is True
    assert crop_wheat["progression_allowed"] is True

    crop_rice = validate_crop_type("Rice")
    assert crop_rice["is_supported"] is True

    crop_unknown = validate_crop_type("Dragonfruit")
    assert crop_unknown["status"] == "UNSUPPORTED_CROP"
    assert crop_unknown["progression_allowed"] is False


def test_safety_policy_aggregation():
    quality_ok = {"passed_all_quality_checks": True, "quality_score": 1.0, "failed_reasons": [], "recommended_actions": []}
    ood_in = {"overall_ood_status": "IN_DISTRIBUTION", "is_ood": False, "is_borderline": False}
    ranges_ok = {"all_ranges_valid": True, "warnings": [], "details": {"crop": {"severity_allowed": True, "progression_allowed": True, "treatment_allowed": True}}}

    pol_high = evaluate_safety_policy(quality_ok, ood_in, ranges_ok)
    assert pol_high["confidence_tier"] == "HIGH"
    assert pol_high["should_abstain"] is False
    assert pol_high["module_gating"]["allow_severity"] is True
    assert pol_high["module_gating"]["allow_progression"] is True

    # Rejection due to OOD
    ood_out = {"overall_ood_status": "OUT_OF_DISTRIBUTION", "is_ood": True, "is_borderline": False}
    pol_abstain = evaluate_safety_policy(quality_ok, ood_out, ranges_ok)
    assert pol_abstain["confidence_tier"] == "LOW"
    assert pol_abstain["should_abstain"] is True
    assert pol_abstain["module_gating"]["allow_severity"] is False

    # Conformal interval too wide
    pol_wide = evaluate_safety_policy(quality_ok, ood_in, ranges_ok, conformal_interval_width=45.0, max_interval_width=35.0)
    assert pol_wide["confidence_tier"] == "MEDIUM"
    assert pol_wide["module_gating"]["allow_progression"] is False


def test_synthetic_stress_testing(tmp_path):
    out_file = tmp_path / "stress_results.json"
    summary = run_synthetic_stress_tests(output_path=out_file)
    assert summary["total_stress_cases_tested"] >= 6
    assert summary["safety_detection_rate_pct"] > 0.0
    assert out_file.exists()


def test_export_safety_artifacts(tmp_path):
    exported = export_safety_artifacts(output_dir=tmp_path)
    assert exported["config"].exists()
    assert exported["safetensors"].exists()
    assert exported["meta"].exists()

    # Verify zero-pickle safety
    tensors = load_file(str(exported["safetensors"]))
    assert "mean_vector" in tensors
    assert "precision_matrix" in tensors

    with open(exported["config"], "r") as f:
        cfg = json.load(f)
    assert cfg["version"] == "1.0.0"
    assert "quality_thresholds" in cfg
