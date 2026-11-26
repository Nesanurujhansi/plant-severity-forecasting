"""Unit tests for Phase 03: Dynamic NWP Assimilation and Canopy Leaf Wetness Duration."""

import pytest
import pandas as pd
import numpy as np

from ml.data.weather_nwp import _generate_fallback_nwp, _parse_open_meteo_payload, fetch_open_meteo_forecast
from ml.data.microclimate import (
    estimate_leaf_wetness_duration_hourly,
    aggregate_daily_microclimate_risk,
    DailyMicroclimateRisk
)
from app.inference.weather import resolve_weather


def test_nwp_fallback_generation():
    df = _generate_fallback_nwp(latitude=47.45, longitude=8.68, days=7)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 7 * 24
    expected_cols = ["datetime", "temp_2m", "rh_2m", "dew_point_2m", "precip_mm", "pressure_hpa", "cloud_cover_pct"]
    for col in expected_cols:
        assert col in df.columns

    # Verify diurnal range
    assert df["temp_2m"].min() >= 10.0
    assert df["temp_2m"].max() <= 30.0
    assert df["rh_2m"].min() >= 40.0
    assert df["rh_2m"].max() <= 100.0


def test_cartier_leaf_wetness_duration():
    # Construct 24-hour day
    # Hours 00:00 to 06:00: RH = 94%, T = 15°C, Dew point depression = 1.0°C (DEW CONDITION)
    # Hours 07:00 to 11:00: RH = 70%, T = 20°C, Dew point depression = 6.0°C (DRY)
    # Hours 12:00 to 14:00: RH = 80%, T = 18°C, Rain = 1.5 mm/hr (RAIN WET CONDITION)
    # Hours 15:00 to 23:00: RH = 65%, T = 22°C (DRY)
    hrs = 24
    temps = [15.0]*7 + [20.0]*5 + [18.0]*3 + [22.0]*9
    rhs = [94.0]*7 + [70.0]*5 + [80.0]*3 + [65.0]*9
    dew_pts = [14.0]*7 + [14.0]*5 + [15.0]*3 + [15.0]*9
    precips = [0.0]*12 + [1.5]*3 + [0.0]*9

    df = pd.DataFrame({
        "temp_2m": temps,
        "rh_2m": rhs,
        "dew_point_2m": dew_pts,
        "precip_mm": precips
    })

    df_wet = estimate_leaf_wetness_duration_hourly(df)
    assert "is_wet" in df_wet.columns
    assert "is_dew" in df_wet.columns
    assert "is_rain" in df_wet.columns

    # Morning dew should be detected (first 7 hours)
    assert df_wet["is_dew"].iloc[:7].sum() == 7
    # Afternoon rain wetness should be detected (3 hours)
    assert df_wet["is_rain"].iloc[12:15].sum() == 3
    # Total wet hours = 7 + 3 = 10 hours
    assert df_wet["is_wet"].sum() == 10


def test_daily_microclimate_risk_aggregation():
    nwp_df = _generate_fallback_nwp(latitude=47.45, longitude=8.68, days=3)
    daily_risks = aggregate_daily_microclimate_risk(nwp_df)

    assert isinstance(daily_risks, list)
    assert 3 <= len(daily_risks) <= 4

    for risk in daily_risks:
        assert isinstance(risk, DailyMicroclimateRisk)
        assert 0.0 <= risk.infection_risk_index <= 1.0
        assert 0.0 <= risk.leaf_wetness_hours <= 24.0
        assert risk.vpd_kpa >= 0.0


def test_resolve_weather_live_nwp():
    res, warnings = resolve_weather(
        latitude=47.45,
        longitude=8.68,
        weather_scenario="live_nwp"
    )
    assert "source" in res
    assert "NWP" in res["source"] or "Open-Meteo" in res["source"]
    assert "summary_table" in res
    assert len(res["summary_table"]) >= 3
