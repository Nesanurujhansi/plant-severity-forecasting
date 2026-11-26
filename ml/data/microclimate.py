"""In-Canopy Boundary Layer Microclimate and Leaf Wetness Duration (LWD) Engine.

Foliar pathogens (e.g. Zymoseptoria tritici, Puccinia striiformis) require a microscopic
liquid water film on the leaf surface for spore germination and stomatal penetration.
This module translates ambient 2m meteorological forecasts into physical canopy microclimate
estimates and calculates Leaf Wetness Duration (LWD) and disease risk indices.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class DailyMicroclimateRisk(BaseModel):
    """Structured daily in-canopy microclimate and disease risk summary."""
    date: str
    mean_temp_c: float
    mean_rh_pct: float
    leaf_wetness_hours: float = Field(..., description="Estimated hours per day with liquid film on leaf surface.")
    dew_hours: float = Field(..., description="Hours of dew condensation.")
    rain_hours: float = Field(..., description="Hours of active rainfall interception.")
    vpd_kpa: float = Field(..., description="Mean vapor pressure deficit in kPa.")
    conducive_hours: float = Field(..., description="Hours with simultaneous wetness and favorable temperature (14-24°C).")
    infection_risk_index: float = Field(..., description="Normalized daily pathogen germination risk (0.0 to 1.0).")
    max_wind_kmh: float = Field(default=8.0, description="Daily peak sustained wind speed in km/h.")
    spray_feasibility: str = Field(default="FAVORABLE", description="Spray window rating: FAVORABLE, MARGINAL, UNFAVORABLE.")
    spray_advisory: str = Field(default="", description="Agronomic operational spray justification.")


def estimate_leaf_wetness_duration_hourly(
    hourly_df: pd.DataFrame,
    rh_dew_threshold: float = 88.0,
    dew_point_depression_threshold: float = 1.8,
    precip_threshold_mm: float = 0.1
) -> pd.DataFrame:
    """Calculate hourly Leaf Wetness Duration (LWD) from meteorological records.

    Uses the extended Cartier-Gillespie empirical dew deposition criteria:
    LWD = 1 if:
      - Relative humidity >= 88% OR
      - Dew point depression (T - T_dew) <= 1.8°C OR
      - Precipitation >= 0.1 mm/hr

    Args:
        hourly_df: DataFrame with columns: ['temp_2m', 'rh_2m', 'dew_point_2m', 'precip_mm'].

    Returns:
        DataFrame with added columns: ['is_wet', 'is_dew', 'is_rain', 'is_conducive'].
    """
    df = hourly_df.copy()

    # Calculate dew point depression
    if "dew_point_2m" in df.columns:
        dp_depression = df["temp_2m"] - df["dew_point_2m"]
    else:
        # Magnus approximation for dew point
        dp_depression = (100.0 - df["rh_2m"]) / 5.0

    is_rain = df["precip_mm"] >= precip_threshold_mm
    is_high_rh = df["rh_2m"] >= rh_dew_threshold
    is_dew_dep = dp_depression <= dew_point_depression_threshold
    is_dew = (is_high_rh | is_dew_dep) & ~is_rain

    is_wet = is_rain | is_dew

    # Conducive temperature window for cereal rust / septoria: 12.0°C to 24.0°C
    temp_conducive = (df["temp_2m"] >= 12.0) & (df["temp_2m"] <= 24.0)
    is_conducive = is_wet & temp_conducive

    df["is_wet"] = is_wet.astype(int)
    df["is_dew"] = is_dew.astype(int)
    df["is_rain"] = is_rain.astype(int)
    df["is_conducive"] = is_conducive.astype(int)

    return df


def aggregate_daily_microclimate_risk(
    hourly_df: pd.DataFrame,
    opt_temp_c: float = 18.0,
    temp_sigma: float = 4.5
) -> List[DailyMicroclimateRisk]:
    """Aggregate hourly predictions into daily epidemiological risk profiles.

    Args:
        hourly_df: Hourly DataFrame containing datetime, temp_2m, rh_2m, etc.
        opt_temp_c: Optimal temperature for fungal spore germination (default 18.0°C).
        temp_sigma: Temperature tolerance curve standard deviation.

    Returns:
        List of DailyMicroclimateRisk instances.
    """
    df = estimate_leaf_wetness_duration_hourly(hourly_df)

    if "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d")
    else:
        # Synthesize daily index assuming 24 records per day
        df["date"] = [f"Day_{i // 24 + 1}" for i in range(len(df))]

    daily_results = []
    for date_str, group in df.groupby("date", sort=False):
        wet_hours = float(group["is_wet"].sum())
        dew_hours = float(group["is_dew"].sum())
        rain_hours = float(group["is_rain"].sum())
        conducive_hours = float(group["is_conducive"].sum())

        mean_temp = float(group["temp_2m"].mean())
        mean_rh = float(group["rh_2m"].mean())
        max_wind = float(group["wind_speed_10m"].max()) if "wind_speed_10m" in group.columns else 8.0

        # Vapor pressure deficit
        e_s = 0.61078 * np.exp((17.27 * mean_temp) / (mean_temp + 237.3))
        vpd = float(max(0.0, e_s * (1.0 - (mean_rh / 100.0))))

        # Normalized risk index: non-linear thermal suitability during wet periods
        wet_temps = group.loc[group["is_wet"] == 1, "temp_2m"]
        if len(wet_temps) > 0:
            thermal_score = np.mean(np.exp(-((wet_temps - opt_temp_c) ** 2) / (2.0 * temp_sigma ** 2)))
            raw_risk = (wet_hours / 14.0) * float(thermal_score)
            risk_index = float(np.clip(raw_risk, 0.0, 1.0))
        else:
            risk_index = 0.0

        # Operational Spray Feasibility Evaluation
        if rain_hours >= 1.0:
            spray_feas = "UNFAVORABLE"
            spray_adv = "Wash-off hazard: precipitation forecasted (>1 hr rainfall interception)."
        elif max_wind >= 18.0:
            spray_feas = "UNFAVORABLE"
            spray_adv = f"High drift hazard: peak wind reaches {round(max_wind, 1)} km/h (safety limit 18 km/h)."
        elif mean_temp >= 34.0:
            spray_feas = "UNFAVORABLE"
            spray_adv = f"Rapid evaporation: high temperature ({round(mean_temp, 1)}°C) degrades spray droplet coverage."
        elif max_wind >= 12.0:
            spray_feas = "MARGINAL"
            spray_adv = f"Moderate breeze ({round(max_wind, 1)} km/h). Use drift-reduction nozzles and lower pressure."
        elif rain_hours >= 0.1 or mean_rh >= 88.0:
            spray_feas = "MARGINAL"
            spray_adv = "High canopy humidity / dew risk. Extended foliar drying time expected."
        else:
            spray_feas = "FAVORABLE"
            spray_adv = f"Calm conditions (peak wind {round(max_wind, 1)} km/h), zero rain. Ideal foliar uptake."

        daily_results.append(DailyMicroclimateRisk(
            date=date_str,
            mean_temp_c=round(mean_temp, 1),
            mean_rh_pct=round(mean_rh, 1),
            leaf_wetness_hours=round(wet_hours, 1),
            dew_hours=round(dew_hours, 1),
            rain_hours=round(rain_hours, 1),
            vpd_kpa=round(vpd, 2),
            conducive_hours=round(conducive_hours, 1),
            infection_risk_index=round(risk_index, 3),
            max_wind_kmh=round(max_wind, 1),
            spray_feasibility=spray_feas,
            spray_advisory=spray_adv
        ))

    return daily_results
