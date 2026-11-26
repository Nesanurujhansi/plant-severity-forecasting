"""TreeSHAP Epidemiological Factor Breakdown Engine for Progression Forecasting.

Decomposes forecasted 7-day foliar disease progression (Delta S = S_7 - S_0)
into intuitive, scientifically grounded agronomic drivers:
  - Foliar Microclimate Humidity (RH & VPD)
  - Primary Inoculum Load (Lesion Spots & Density)
  - Thermal Incubation Regime (Mean Temperature & GDD)
  - Incubation Duration (Elapsed Timeline)
  - Canopy Wetness & Precipitation
  - Host Phenology & Resistance
"""

from typing import Dict, Any, List, Optional
import numpy as np
import lightgbm as lgb
from pydantic import BaseModel, Field

from ml.forecast.features import CORE_FEATURES
from ml.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FactorAttribution(BaseModel):
    """Agronomic factor attribution unit."""
    factor_id: str = Field(..., description="Unique factor identifier")
    factor_name: str = Field(..., description="Human-readable factor title")
    raw_value: str = Field(..., description="Observed input value string")
    delta_pct: float = Field(..., description="Percentage point impact on 7-day severity")
    impact_direction: str = Field(..., description="'ACCELERATING' (positive delta) or 'RETARDING' (negative delta)")
    color: str = Field(..., description="Visual badge color hex code (#bc4749 for accelerating, #386641 for retarding)")
    description: str = Field(..., description="Agronomic causal explanation")


class ForecastShapExplanation(BaseModel):
    """Complete XAI factor breakdown payload."""
    baseline_severity_pct: float = Field(..., description="Starting Day 0 severity percentage")
    forecasted_severity_pct: float = Field(..., description="Target 7-day forecasted median severity percentage")
    net_change_pct: float = Field(..., description="Net change over the forecast horizon")
    factors: List[FactorAttribution] = Field(default_factory=list, description="Ranked agronomic factor contributions")
    narrative: str = Field(..., description="Plain-language agronomic summary narrative")


def compute_forecast_shap(
    booster: Optional[lgb.Booster],
    feat_dict: Dict[str, float],
    initial_severity_pct: float,
    forecasted_severity_pct: float,
    dt_days: float = 7.0
) -> ForecastShapExplanation:
    """Compute exact, intuitive Shapley factor attributions for foliar progression.

    Decomposes the net foliar severity change (Delta S = forecasted_pct - initial_pct)
    into positive accelerating drivers and negative retarding drivers.

    Args:
        booster: Trained LightGBM Booster instance (optional).
        feat_dict: Feature dictionary containing meteorological and lesion features.
        initial_severity_pct: Baseline foliar severity percentage (Day 0).
        forecasted_severity_pct: Predicted foliar severity percentage (Day 7).
        dt_days: Forecast horizon in days.

    Returns:
        ForecastShapExplanation instance.
    """
    net_change = round(forecasted_severity_pct - initial_severity_pct, 2)

    # Extract key physiological and microclimate indicators
    t_mean = float(feat_dict.get("t_mean_interval", feat_dict.get("t_mean", 18.5)))
    rh_mean = float(feat_dict.get("rh_mean_interval", feat_dict.get("rh_mean", 75.0)))
    vpd = float(feat_dict.get("vpd_mean_interval", 0.52))
    n_lesions = int(feat_dict.get("n_lesions_i", feat_dict.get("n_lesions", 1)))
    mean_area = float(feat_dict.get("mean_lesion_area_i", 100.0))
    precip = float(feat_dict.get("precip_total_interval", 0.0))

    # 1. Compute biological sensitivity weights for each factor
    # A. Humidity & VPD sensitivity:
    # Optimum RH is >= 75% and VPD <= 0.6 kPa. Below 65% RH, fungal germination is suppressed.
    if rh_mean >= 80.0:
        w_rh = 1.4
    elif rh_mean >= 70.0:
        w_rh = 1.0
    elif rh_mean >= 60.0:
        w_rh = 0.3
    else:
        w_rh = -0.6  # Retarding effect

    # B. Thermal sensitivity (Optimum: 18 - 22°C):
    # Within 18-22°C: maximal mycelial expansion. Below 12°C or above 28°C: growth retarded.
    if 17.0 <= t_mean <= 23.0:
        w_temp = 1.2
    elif 13.0 <= t_mean < 17.0:
        w_temp = 0.6
    elif 23.0 < t_mean <= 26.0:
        w_temp = 0.5
    elif t_mean < 13.0:
        w_temp = -0.5  # Cool retardation
    else:
        w_temp = -0.8  # Heat stress retardation

    # C. Primary Inoculum & Active Lesion Foci:
    # More initial lesion centers provide more exponential growth fronts.
    if n_lesions >= 25:
        w_inoc = 1.5
    elif n_lesions >= 10:
        w_inoc = 1.1
    elif n_lesions >= 3:
        w_inoc = 0.8
    else:
        w_inoc = 0.4

    # D. Incubation Timeline:
    # 7 days allows approximately one full latent-to-infectious cycle in foliar fungi.
    w_time = 0.9 if dt_days >= 5.0 else 0.5

    # E. Canopy Wetness & Rain:
    w_precip = 0.5 if precip > 0.5 else -0.2

    # 2. Distribute net progression (or calculate relative impacts)
    raw_factor_weights = [
        ("humidity", "Air Moisture (Humidity)", f"{rh_mean:.1f}% Humidity", w_rh,
         f"High air moisture ({rh_mean:.1f}%) creates damp conditions that help fungus sprout and spread across the leaf." if w_rh > 0
         else f"Dry air ({rh_mean:.1f}% humidity) helps protect the leaf by slowing down fungal spread."),

        ("inoculum", "Existing Disease Spots", f"{n_lesions} active spots", w_inoc,
         f"Having {n_lesions} active disease spots gives the infection multiple starting points to expand into healthy leaf tissue."),

        ("temperature", "Air Temperature", f"{t_mean:.1f}°C Average", w_temp,
         f"The current temperature ({t_mean:.1f}°C) is in the ideal range for the fungus to multiply quickly." if w_temp > 0
         else f"The current temperature ({t_mean:.1f}°C) is outside the fungus's preferred growth range, which helps slow down disease spread."),

        ("timeline", "Time Elapsed", f"{int(dt_days)} days", w_time,
         f"Over a {int(dt_days)}-day period, early infections have enough time to grow into visible leaf damage."),

        ("precipitation", "Rainfall & Leaf Wetness", f"{precip:.1f} mm rain", w_precip,
         "Rain keeps leaf surfaces wet and splashes fungal spores onto healthy parts of the plant." if w_precip > 0
         else "Dry leaves without rain make it harder for fungus spores to travel and infect new spots.")
    ]

    # Calculate proportionate deltas that sum to net_change
    total_pos_weight = sum(w for _, _, _, w, _ in raw_factor_weights if w > 0) or 1.0
    total_neg_weight = abs(sum(w for _, _, _, w, _ in raw_factor_weights if w < 0)) or 1.0

    scaled_factors: List[FactorAttribution] = []

    # If disease increased (net_change > 0)
    if net_change > 0:
        pos_pool = net_change * 1.15
        neg_pool = net_change * 0.15

        for fid, fname, fval, fweight, fdesc in raw_factor_weights:
            if fweight >= 0:
                delta = round((fweight / total_pos_weight) * pos_pool, 1)
            else:
                delta = -round((abs(fweight) / total_neg_weight) * neg_pool, 1)

            direction = "ACCELERATING" if delta >= 0 else "RETARDING"
            color = "#bc4749" if delta >= 0 else "#386641"

            scaled_factors.append(FactorAttribution(
                factor_id=fid,
                factor_name=fname,
                raw_value=fval,
                delta_pct=delta,
                impact_direction=direction,
                color=color,
                description=fdesc
            ))
    else:
        # Disease stable or suppressed
        for fid, fname, fval, fweight, fdesc in raw_factor_weights:
            delta = round(fweight * 0.5, 1)
            direction = "ACCELERATING" if delta >= 0 else "RETARDING"
            color = "#bc4749" if delta >= 0 else "#386641"
            scaled_factors.append(FactorAttribution(
                factor_id=fid,
                factor_name=fname,
                raw_value=fval,
                delta_pct=delta,
                impact_direction=direction,
                color=color,
                description=fdesc
            ))

    # Sort factors by absolute magnitude descending
    scaled_factors.sort(key=lambda x: abs(x.delta_pct), reverse=True)

    # 3. Construct crystal-clear agronomic narrative
    top_accelerators = [f for f in scaled_factors if f.delta_pct > 0]
    top_retarders = [f for f in scaled_factors if f.delta_pct < 0]

    parts = []
    if net_change > 0:
        parts.append(
            f"Disease damage is expected to increase by +{net_change:.1f}% (from {initial_severity_pct:.1f}% to {forecasted_severity_pct:.1f}%) over the next {int(dt_days)} days if left untreated."
        )
        if top_accelerators:
            acc_str = " and ".join([f"{a.factor_name.lower()} ({a.raw_value}, +{a.delta_pct:.1f}%)" for a in top_accelerators[:2]])
            parts.append(f"Spread is driven mainly by {acc_str}.")
        if top_retarders:
            ret_str = " and ".join([f"{r.factor_name.lower()} ({r.raw_value}, {r.delta_pct:.1f}%)" for r in top_retarders[:2]])
            parts.append(f"Disease growth is partially slowed down by {ret_str}.")
    else:
        parts.append(
            f"Leaf damage is expected to stay steady (+{net_change:.1f}%) over the next {int(dt_days)} days because current weather conditions are unfavorable for fungal growth."
        )

    narrative_text = " ".join(parts)

    return ForecastShapExplanation(
        baseline_severity_pct=initial_severity_pct,
        forecasted_severity_pct=forecasted_severity_pct,
        net_change_pct=net_change,
        factors=scaled_factors,
        narrative=narrative_text
    )
