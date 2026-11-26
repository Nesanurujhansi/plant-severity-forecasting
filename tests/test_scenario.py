"""Unit tests for Treatment Scenario Module (Phase 07)."""

import pytest
from pathlib import Path
import numpy as np
import torch

from ml.scenario.identifiability import analyze_treatment_identifiability
from ml.scenario.mixed_effects import prepare_mixed_effects_dataset, fit_mixed_effects_treatment_model
from ml.scenario.bootstrap import run_cluster_bootstrap
from ml.scenario.treatment_net import DeepTreatmentTarNet
from ml.scenario.simulate import ScenarioSimulator, TreatmentScenarioResult
from ml.scenario.validate import validate_treatment_scenarios
from ml.scenario.export import export_scenario_artifacts


def test_identifiability_analysis() -> None:
    """Test treatment identifiability report structure."""
    report = analyze_treatment_identifiability()
    assert isinstance(report, dict)
    assert "arm_breakdown" in report
    assert "FI2" in report["arm_breakdown"]
    assert "0F0I" in report["arm_breakdown"]
    assert report["recommended_fungicide_multiplier_contrast"] == "FI2_vs_0F0I"


def test_mixed_effects_fit() -> None:
    """Test linear mixed-effects model estimation."""
    df = prepare_mixed_effects_dataset()
    res = fit_mixed_effects_treatment_model(df)
    assert isinstance(res, dict)
    assert "treatment_multiplier" in res
    assert 0.05 <= res["treatment_multiplier"] <= 1.0


def test_cluster_bootstrap() -> None:
    """Test plot-clustered bootstrap resampling."""
    df = prepare_mixed_effects_dataset()
    boot = run_cluster_bootstrap(df, n_resamples=10, seed=42)
    assert isinstance(boot, dict)
    assert boot["n_resamples_successful"] > 0
    assert len(boot["treatment_multiplier_ci"]) == 2
    assert boot["treatment_multiplier_ci"][0] <= boot["treatment_multiplier_ci"][1]


def test_deep_treatment_tarnet() -> None:
    """Test Deep Treatment TarNet forward pass and Monte Carlo Dropout."""
    model = DeepTreatmentTarNet(input_dim=16, hidden_dim=32)
    x = torch.randn(8, 16)
    u_pred, p_pred = model(x)
    assert u_pred.shape == (8, 1)
    assert p_pred.shape == (8, 1)

    mc_out = model.predict_counterfactuals_mc_dropout(x, n_mc_samples=10)
    assert "unprotected_mean_delta_r" in mc_out
    assert "protected_mean_delta_r" in mc_out
    assert mc_out["unprotected_mean_delta_r"].shape == (8,)


def test_scenario_simulator_monotonicity_and_width() -> None:
    """Test scenario simulator monotonicity and combined interval non-narrowness."""
    simulator = ScenarioSimulator(treatment_multiplier=0.50, treatment_ci=[0.35, 0.65])
    res = simulator.simulate_scenario(initial_severity_pct=10.0, horizon_days=7.0)

    assert isinstance(res, TreatmentScenarioResult)
    # Monotonicity check: fungicide treatment should never INCREASE disease severity
    assert res.protected_median_percentage <= res.untreated_median_percentage
    assert res.disease_reduction_pct_points >= 0.0

    # Mandatory disclaimer check
    assert "not a spray recommendation" in res.disclaimer.lower()
    assert res.combined_interval_wider is True


def test_validate_treatment_scenarios() -> None:
    """Test Leave-Cultivar-Out validation logic."""
    df = prepare_mixed_effects_dataset()
    val_report = validate_treatment_scenarios(df)
    assert isinstance(val_report, dict)
    assert "sign_accuracy_percentage" in val_report
    assert val_report["sign_accuracy_percentage"] >= 0.0


def test_export_scenario_artifacts(tmp_path: Path) -> None:
    """Test zero-pickle artifact export for scenario module."""
    exported = export_scenario_artifacts(output_dir=tmp_path)
    assert isinstance(exported, dict)
    assert (tmp_path / "mixed_effects.json").exists()
    assert (tmp_path / "effect_bootstrap.json").exists()
    assert (tmp_path / "scenario_config.json").exists()
    assert (tmp_path / "treatment_net.safetensors").exists()
