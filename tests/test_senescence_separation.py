"""Unit tests for Phase 02: Pathological Differentiation (Senescence vs. Fungal Necrosis)."""

import pytest
import numpy as np
import torch

from ml.severity.severity import (
    calculate_severity_metrics,
    calculate_multiclass_severity_metrics,
    SeverityMetricsResult
)


def test_senescence_partitioning():
    # Synthetic leaf of 1650 total pixels:
    # 1000 healthy green, 500 senescent yellowing, 150 fungal necrotic spots
    h, w = 50, 50
    leaf_pred = np.zeros((h, w), dtype=np.float32)
    leaf_pred[:33, :] = 1.0  # 33 * 50 = 1650 total leaf pixels

    senescence_pred = np.zeros((h, w), dtype=np.float32)
    senescence_pred[20:30, :] = 1.0  # 10 * 50 = 500 senescent pixels

    lesion_pred = np.zeros((h, w), dtype=np.float32)
    lesion_pred[5:8, :50] = 1.0  # 3 * 50 = 150 lesion pixels

    # Run severity metrics with explicit senescence partitioning
    res = calculate_severity_metrics(
        lesion_pred=lesion_pred,
        leaf_pred=leaf_pred,
        senescence_pred=senescence_pred
    )

    assert isinstance(res, SeverityMetricsResult)
    assert res.leaf_pixel_count == 1650
    assert res.lesion_pixel_count == 150
    assert res.senescence_pixel_count == 500

    # True pathogen severity must be ~9.09%, NOT inflated to ~39.4%
    expected_severity_pct = round((150.0 / 1650.0) * 100.0, 4)
    expected_senescence_pct = round((500.0 / 1650.0) * 100.0, 4)

    assert abs(res.severity_percentage_hard - expected_severity_pct) < 0.01
    assert abs(res.senescence_percentage - expected_senescence_pct) < 0.01
    assert res.ordinal_class_label == "5-10%"  # 9.09% falls in 5-10% bin, not >25%


def test_phenological_growth_stage_regularization():
    h, w = 40, 40
    leaf_pred = np.ones((h, w), dtype=np.float32)
    lesion_pred = np.zeros((h, w), dtype=np.float32)

    # Moderate chlorosis signal (prob = 0.60)
    senescence_pred = np.full((h, w), 0.60, dtype=np.float32)

    # 1. Early stage: days_since_heading = 15 (flowering stage)
    # Premature senescence should be suppressed by the 0.4x factor (0.60 * 0.4 = 0.24 < 0.5 threshold)
    res_early = calculate_severity_metrics(
        lesion_pred=lesion_pred,
        leaf_pred=leaf_pred,
        senescence_pred=senescence_pred,
        days_since_heading=15.0
    )
    assert res_early.senescence_pixel_count == 0
    assert res_early.senescence_percentage == 0.0

    # 2. Mature stage: days_since_heading = 45 (grain ripening stage)
    # Natural senescence is fully accepted
    res_late = calculate_severity_metrics(
        lesion_pred=lesion_pred,
        leaf_pred=leaf_pred,
        senescence_pred=senescence_pred,
        days_since_heading=45.0
    )
    assert res_late.senescence_pixel_count == 1600
    assert res_late.senescence_percentage == 100.0


def test_multiclass_severity_metrics_4channel():
    # 4 channels: [0: background, 1: healthy, 2: senescence, 3: lesion]
    logits = np.zeros((4, 30, 30), dtype=np.float32)

    # Set background as default
    logits[0, :, :] = 2.0

    # Set center region as leaf (healthy green)
    logits[1, 5:25, 5:25] = 5.0

    # Add a senescent patch
    logits[2, 10:15, 10:25] = 6.0

    # Add active lesion spots
    logits[3, 20:23, 10:15] = 7.0

    res = calculate_multiclass_severity_metrics(logits, days_since_heading=40.0)

    assert isinstance(res, SeverityMetricsResult)
    assert res.leaf_pixel_count > 0
    assert res.lesion_pixel_count > 0
    assert res.senescence_pixel_count > 0
    assert 0.0 < res.severity_percentage_hard < 100.0
