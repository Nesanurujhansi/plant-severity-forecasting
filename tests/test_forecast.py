"""Unit tests for Phase 06 Epidemiological Progression Forecasting Module.

Tests feature parity, growth curve ODE integration, SciPy parameter fitting, LightGBM residual learning,
CQR conformal prediction intervals, benchmark baselines, multi-horizon evaluation, noise robustness,
Forecaster inference engine, and zero-pickle artifact export on CPU without external network/GPU dependencies.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ml.forecast.features import (
    CORE_FEATURES,
    EXTENDED_FEATURES,
    build_feature_vector,
    get_feature_matrix,
    verify_feature_parity,
    export_feature_schema
)
from ml.forecast.growth import (
    thermal_response,
    humidity_response,
    compute_growth_rate,
    predict_growth_curve,
    analyze_temperature_sensitivity,
    fit_growth_curve,
    export_growth_params,
    DEFAULT_GROWTH_PARAMS
)
from ml.forecast.residual import (
    logit,
    inv_logit,
    compute_residuals,
    train_residual_model,
    predict_residual_and_combine,
    export_residual_model
)
from ml.forecast.cqr import (
    train_cqr_quantile_models,
    calibrate_cqr_intervals,
    predict_cqr_interval,
    export_cqr_artifacts
)
from ml.forecast.baselines import (
    PersistenceBaseline,
    CultivarMeanGrowthBaseline,
    GrowthCurveOnlyBaseline,
    LightGBMOnlyBaseline,
    NaiveLateFusionBaseline
)
from ml.forecast.evaluate import (
    compute_pinball_loss,
    compute_forecast_metrics,
    evaluate_forecast_pipeline
)
from ml.forecast.noise_robustness import (
    inject_severity_noise,
    evaluate_noise_robustness
)
from ml.forecast.predict import Forecaster, ForecastResult
from ml.forecast.export import export_progression_artifacts
from ml.scenario.treatment_net import DeepTreatmentTarNet
from ml.safety.deep_ood import DeepOODSafetyEvaluator, compute_free_energy_score
from ml.forecast.pinn_residual import HybridPINNODEResidualModel, train_pinn_residual_model


@pytest.fixture
def synthetic_pairs_df():
    """Create synthetic sequence pairs DataFrame for unit testing."""
    rng = np.random.RandomState(42)
    n = 60

    dates = pd.date_range("2024-01-01", periods=15, freq="D")
    dummy_w = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "t_mean": np.full(15, 18.5),
        "t_max": np.full(15, 23.0),
        "t_min": np.full(15, 14.0),
        "rh_mean": np.full(15, 76.0),
        "precip": np.full(15, 1.2)
    })

    records = []
    for i in range(n):
        s0 = float(rng.uniform(0.01, 0.20))
        dt = float(rng.uniform(1.0, 8.0))
        nl = int(rng.randint(1, 10))
        la = float(rng.uniform(10.0, 200.0))
        cs = 55.0
        cg = float(rng.choice([1.0, 2.0, 3.0]))

        feat = build_feature_vector(s0, dt, dummy_w, n_lesions_i=nl, mean_lesion_area_i=la, crop_stage=cs, cultivar_group=cg)
        sg = predict_growth_curve(s0=s0, dt_days=dt, t_mean=18.5, rh_mean=76.0)
        noise = float(rng.normal(0.0, 0.005))
        feat.update({
            "sample_id": f"sample_{i % 10}",
            "cultivar": str(rng.choice(["ArinaLR", "CH_Claro", "Forno"])),
            "sample_weight": 1.0,
            "severity_j": float(np.clip(sg + noise, 0.001, 0.95))
        })
        records.append(feat)

    return pd.DataFrame(records)


@pytest.fixture
def synthetic_weather_df():
    """Create synthetic weather DataFrame for unit testing."""
    dates = pd.date_range("2024-01-01", periods=14, freq="D")
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "t_mean": np.random.uniform(15.0, 22.0, size=14),
        "t_max": np.random.uniform(22.0, 28.0, size=14),
        "t_min": np.random.uniform(10.0, 15.0, size=14),
        "rh_mean": np.random.uniform(65.0, 85.0, size=14),
        "precip": np.random.uniform(0.0, 5.0, size=14)
    })


def test_feature_parity_offline_online(synthetic_weather_df):
    """Test offline training vs online serving feature parity."""
    sample_row = {
        "severity_i": 0.08,
        "dt_days": 3.0,
        "n_lesions_i": 5,
        "mean_lesion_area_i": 45.2,
        "crop_stage": 55.0,
        "cultivar_group": 2.0
    }
    assert verify_feature_parity(sample_row, synthetic_weather_df) is True


def test_growth_curve_integration_and_bounds():
    """Test growth curve numerical integration and bounds."""
    # Test positive growth
    s_next = predict_growth_curve(s0=0.05, dt_days=3.0, t_mean=21.0, rh_mean=80.0)
    assert s_next > 0.05
    assert s_next <= 0.95

    # Test zero initial infection stays zero
    s_zero = predict_growth_curve(s0=0.0, dt_days=7.0, t_mean=21.0, rh_mean=80.0)
    assert s_zero == 0.0


def test_temperature_sensitivity_analysis():
    """Test temperature sensitivity calculation."""
    sens = analyze_temperature_sensitivity(T_base=18.0, temp_delta=5.0)
    assert "r_base" in sens
    assert "r_warmed" in sens
    assert isinstance(sens["description"], str)


def test_growth_curve_fitting(synthetic_pairs_df):
    """Test SciPy least-squares fitting of growth curve parameters."""
    fit_res = fit_growth_curve(synthetic_pairs_df, scale="logit")
    assert fit_res["status"] == "success"
    assert "r0" in fit_res["params"]
    assert 0.001 <= fit_res["params"]["r0"] <= 0.50


def test_residual_model_training(synthetic_pairs_df):
    """Test LightGBM residual model training."""
    res = train_residual_model(synthetic_pairs_df, feature_set="core")
    assert res["status"] == "success"
    assert res["booster"] is not None


def test_cqr_quantile_models_and_calibration(synthetic_pairs_df):
    """Test CQR quantile models training and split-conformal calibration."""
    boosters = train_cqr_quantile_models(synthetic_pairs_df, feature_set="core")
    assert "low" in boosters and "med" in boosters and "high" in boosters

    calib = calibrate_cqr_intervals(boosters, synthetic_pairs_df, feature_set="core")
    assert calib["is_calibrated"] is True
    assert calib["conformal_quantile"] >= 0.0

    feat = build_feature_vector(0.05, 3.0, pd.DataFrame())
    s_med, s_low, s_high, width = predict_cqr_interval(
        boosters, calib, s0=0.05, dt_days=3.0, t_mean=18.0, rh_mean=75.0, feat_dict=feat
    )
    assert s_low <= s_med <= s_high
    assert width >= 0.0


def test_baselines(synthetic_pairs_df):
    """Test benchmark baselines."""
    pers = PersistenceBaseline()
    assert pers.predict(0.12) == 0.12

    cult = CultivarMeanGrowthBaseline().fit(synthetic_pairs_df)
    pred_c = cult.predict(0.10, dt_days=3.0, cultivar="ArinaLR")
    assert pred_c >= 0.10

    growth = GrowthCurveOnlyBaseline()
    assert growth.predict(0.05, 3.0) > 0.05


def test_evaluate_forecast_pipeline(synthetic_pairs_df):
    """Test forecast pipeline evaluation."""
    n = len(synthetic_pairs_df)
    train_idx = np.arange(int(n * 0.7))
    test_idx = np.arange(int(n * 0.7), n)

    eval_res = evaluate_forecast_pipeline(synthetic_pairs_df, train_idx, test_idx, feature_set="core")
    assert "overall" in eval_res
    assert "cqr_model" in eval_res["overall"]
    assert eval_res["overall"]["cqr_model"]["mae_pct_points"] >= 0.0


def test_severity_noise_robustness(synthetic_pairs_df):
    """Test injection of Phase 5 measurement noise into initial severity."""
    s_noisy = inject_severity_noise(0.10, bias=-0.001, std_dev=0.0133)
    assert 0.0 <= s_noisy <= 1.0

    boosters = train_cqr_quantile_models(synthetic_pairs_df)
    calib = calibrate_cqr_intervals(boosters, synthetic_pairs_df)
    robust_res = evaluate_noise_robustness(synthetic_pairs_df, boosters, calib)
    assert robust_res["status"] == "success"
    assert "gt_mae_pct" in robust_res


def test_forecaster_predict_and_missing_weather(synthetic_weather_df):
    """Test Forecaster serving engine with real and missing weather series."""
    forecaster = Forecaster(is_stub=True)

    # 1. Normal inference with weather
    res_w = forecaster.predict(current_severity=0.05, dt_days=3.0, weather_df=synthetic_weather_df)
    assert isinstance(res_w, ForecastResult)
    assert res_w.weather_missing is False
    assert res_w.interval_lower_percentage <= res_w.median_severity_percentage <= res_w.interval_upper_percentage

    # 2. Missing weather fallback
    res_noweather = forecaster.predict(current_severity=0.05, dt_days=3.0, weather_df=None)
    assert res_noweather.weather_missing is True
    assert res_noweather.warning_flag is not None


def test_export_progression_artifacts(tmp_path, synthetic_pairs_df):
    """Test zero-pickle artifact exporter."""
    exported = export_progression_artifacts(synthetic_pairs_df, output_dir=tmp_path)
    assert "feature_schema" in exported
    assert "growth_params" in exported
    assert "forecast_config" in exported
    assert (tmp_path / "growth_params.json").exists()
    assert (tmp_path / "forecast_config.json").exists()


def test_deep_treatment_tarnet():
    """Test Deep Counterfactual Treatment Network forward pass and MC Dropout uncertainty."""
    import torch
    net = DeepTreatmentTarNet(input_dim=16, hidden_dim=32)
    x = torch.randn(4, 16)
    u_pred, p_pred = net(x)
    assert u_pred.shape == (4, 1)
    assert p_pred.shape == (4, 1)

    mc_res = net.predict_counterfactuals_mc_dropout(x, n_mc_samples=10)
    assert "unprotected_mean_delta_r" in mc_res
    assert "protected_mean_delta_r" in mc_res
    assert "net_treatment_reduction" in mc_res
    assert len(mc_res["unprotected_mean_delta_r"]) == 4


def test_deep_ood_safety_evaluator():
    """Test Deep Free Energy OOD detector and quality evaluator."""
    evaluator = DeepOODSafetyEvaluator(energy_threshold=8.5)
    dummy_feat = np.random.randn(2, 384).astype(np.float32)

    # Free Energy calculation test
    score = compute_free_energy_score(np.array([2.0, 5.0, 1.0]))
    assert isinstance(score, float)

    # Safety evaluation test
    res = evaluator.evaluate_safety(dummy_feat)
    assert "free_energy_score" in res
    assert "is_ood" in res
    assert "safety_status" in res
    assert res["safety_status"] in ["ACCEPT_IN_DISTRIBUTION", "REJECT_OOD"]


def test_hybrid_pinn_residual_model(synthetic_pairs_df):
    """Test PyTorch Hybrid PINN-ODE derivative vector field and Deep Residual MLP."""
    import torch
    model = HybridPINNODEResidualModel(input_dim=19, hidden_dim=32)
    s0 = torch.tensor([[0.05], [0.10]])
    dt = torch.tensor([[3.0], [7.0]])
    temp = torch.tensor([[18.5], [21.0]])
    vpd = torch.tensor([[0.5], [0.8]])
    feat = torch.randn(2, 19)

    pred, z_ode, y_res = model(s0, dt, temp, vpd, feat)
    assert pred.shape == (2, 1)
    assert z_ode.shape == (2, 1)
    assert y_res.shape == (2, 1)
    assert 0.0 <= pred[0].item() <= 1.0

    train_res = train_pinn_residual_model(synthetic_pairs_df, epochs=5, batch_size=16)
    assert train_res["status"] == "success"
    assert "train_mae_pct" in train_res


def test_foliar_seir_rk4_solver():
    """Test 4-compartment Foliar-SEIR numerical ODE solver with Magarey microclimate forcing."""
    from ml.forecast.ode_model import (
        solve_foliar_seir_rk4,
        compute_magarey_infection_risk,
        forecast_foliar_seir_trajectory
    )

    # 1. Test Magarey infection risk
    risk_dry = compute_magarey_infection_risk(temp_c=20.0, rh_pct=45.0, precip_mm=0.0, lwd_hours=0.0)
    risk_wet = compute_magarey_infection_risk(temp_c=20.0, rh_pct=92.0, precip_mm=10.0, lwd_hours=8.0)
    assert 0.0 <= risk_dry < risk_wet <= 1.0
    assert risk_wet > 0.60

    # 2. Test RK4 solver mass conservation & carrying capacity bounds
    weather_series = [
        {"date": "Day 1", "t_mean": 20.0, "rh_mean": 70.0, "precip": 0.0, "lwd_hours": 2.0},
        {"date": "Day 2", "t_mean": 18.5, "rh_mean": 94.0, "precip": 12.0, "lwd_hours": 9.0}, # Storm day
        {"date": "Day 3", "t_mean": 21.0, "rh_mean": 80.0, "precip": 0.5, "lwd_hours": 4.0},
        {"date": "Day 4", "t_mean": 22.0, "rh_mean": 75.0, "precip": 0.0, "lwd_hours": 3.0},
        {"date": "Day 5", "t_mean": 21.5, "rh_mean": 74.0, "precip": 0.0, "lwd_hours": 2.5},
        {"date": "Day 6", "t_mean": 20.0, "rh_mean": 76.0, "precip": 0.0, "lwd_hours": 3.0},
        {"date": "Day 7", "t_mean": 19.5, "rh_mean": 78.0, "precip": 0.0, "lwd_hours": 3.0},
    ]

    t_vals, S, L, I, R, Reff, Risk = solve_foliar_seir_rk4(
        initial_necrotic_fraction=0.08,
        initial_chlorotic_fraction=0.03,
        daily_weather_series=weather_series,
        carrying_capacity=0.95
    )

    # Verify mass conservation at all steps: S + L + I + R == 1.0
    total_mass = S + L + I + R
    np.testing.assert_allclose(total_mass, 1.0, atol=1e-5)

    # Verify non-negativity
    assert np.all(S >= 0.0)
    assert np.all(L >= 0.0)
    assert np.all(I >= 0.0)
    assert np.all(R >= 0.0)

    # Visible severity must not exceed carrying capacity
    vis = I + R
    assert np.all(vis <= 0.95 + 1e-5)

    # Storm on Day 2 must cause latent tissue L to surge
    assert np.max(L) > L[0]

    # 3. Test High-Level FoliarSEIR Trajectory Entrypoint
    traj_res = forecast_foliar_seir_trajectory(
        necrotic_pct=8.0,
        chlorotic_pct=3.0,
        daily_weather_series=weather_series
    )
    assert len(traj_res.visible_severity_pct_trajectory) == 8 # Days 0 through 7
    assert traj_res.visible_severity_pct_trajectory[-1] >= traj_res.visible_severity_pct_trajectory[0]
    assert len(traj_res.latent_pct_trajectory) == 8
    assert len(traj_res.daily_magarey_risks) == 8
    assert traj_res.trajectory_summary != ""


def test_forecaster_realtime_predict():
    """Test Forecaster real-time empirical CQR inference with live weather DataFrame."""
    forecaster = Forecaster(is_stub=True)
    w_df = pd.DataFrame([
        {"date": f"2026-05-{i+1:02d}", "t_mean": 19.5, "rh_mean": 78.0, "precip": 1.0, "lwd_hours": 4.0, "vpd": 0.50}
        for i in range(7)
    ])

    res = forecaster.predict(
        current_severity=0.10,
        dt_days=7.0,
        weather_df=w_df
    )

    assert isinstance(res, ForecastResult)
    assert res.horizon_days == 7.0
    assert res.median_severity_percentage >= 10.0
    assert res.interval_lower_percentage <= res.median_severity_percentage <= res.interval_upper_percentage
    assert res.confidence_tier in ["HIGH", "MEDIUM", "LOW"]
    assert res.partial_effect_explanation != ""
