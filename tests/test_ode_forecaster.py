"""Unit tests for Phase 04: Non-Linear Epidemiological ODE Forecaster."""

import pytest
import numpy as np

from ml.forecast.ode_model import (
    solve_logistic_epidemic_ode,
    scale_flag_leaf_to_canopy,
    forecast_continuous_trajectory,
    ContinuousTrajectoryResult
)


def test_rk4_logistic_ode_stability_and_bounds():
    # Simulate high disease risk for 21 days: S(0) = 15%
    initial_sev = 0.15
    daily_risks = [0.9] * 21
    carrying_capacity = 0.90  # 90% theoretical maximum

    t_vals, s_vals = solve_logistic_epidemic_ode(
        initial_severity_fraction=initial_sev,
        daily_risk_indices=daily_risks,
        base_growth_rate=0.15,
        carrying_capacity=carrying_capacity
    )

    # Invariants:
    # 1. Monotonically non-decreasing under positive disease pressure
    diffs = np.diff(s_vals)
    assert np.all(diffs >= -1e-6)

    # 2. Strict bound: Must NEVER exceed carrying capacity K
    assert np.all(s_vals <= carrying_capacity + 1e-6)

    # 3. Starts at initial condition
    assert abs(s_vals[0] - initial_sev) < 1e-5

    # 4. Long-range asymptote check: should plateau towards K at day 21
    assert s_vals[-1] > 0.60
    assert s_vals[-1] <= carrying_capacity


def test_zero_disease_remains_zero():
    # If initial severity is 0, spore-free leaf stays healthy
    t_vals, s_vals = solve_logistic_epidemic_ode(
        initial_severity_fraction=0.0,
        daily_risk_indices=[1.0] * 7,
        base_growth_rate=0.20
    )
    assert np.all(s_vals == 0.0)


def test_canopy_vertical_scaling():
    flag_sev = 0.08  # 8% on flag leaf
    canopy_sev = scale_flag_leaf_to_canopy(flag_sev)

    # Canopy severity must be higher than top flag leaf due to lower-canopy inoculum
    assert canopy_sev > flag_sev
    assert 0.12 <= canopy_sev <= 0.20

    # Upper boundary test: 100% flag leaf cannot scale beyond 100% canopy
    assert scale_flag_leaf_to_canopy(1.0) == 1.0


def test_forecast_continuous_trajectory_pipeline():
    res = forecast_continuous_trajectory(
        initial_severity_pct=10.0,
        daily_risk_indices=[0.75] * 14,
        horizon_days=14.0,
        carrying_capacity_pct=92.0
    )
    assert isinstance(res, ContinuousTrajectoryResult)
    assert len(res.horizons_days) == 15  # Day 0 to Day 14
    assert res.severity_pct_trajectory[0] == 10.0
    assert res.canopy_severity_pct_trajectory[0] > 10.0
    assert res.severity_pct_trajectory[-1] <= res.carrying_capacity_pct
