"""Mechanistic Chemical Fungicide Modeling and Regional Domain Adaptation.

Replaces static scalar treatment multipliers with a pharmacodynamic formulation
accounting for active ingredient class, rainfastness wash-off decay, and pathogen
target-site resistance mutations (e.g. CYP51 in triazoles, G143A in strobilurins).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from pydantic import BaseModel, Field

CHEMICAL_PROFILES: Dict[str, Dict[str, Any]] = {
    "sdhi": {
        "label": "Succinate Dehydrogenase Inhibitor (SDHI)",
        "base_efficacy": 0.90,
        "half_life_days": 14.0,
        "rainfast_threshold_mm": 15.0,
        "wash_sensitivity": 0.04
    },
    "triazole_dmi": {
        "label": "Demethylation Inhibitor (DMI Triazole)",
        "base_efficacy": 0.82,
        "half_life_days": 10.0,
        "rainfast_threshold_mm": 10.0,
        "wash_sensitivity": 0.06
    },
    "strobilurin_qoi": {
        "label": "Quinone Outside Inhibitor (QoI Strobilurin)",
        "base_efficacy": 0.85,
        "half_life_days": 12.0,
        "rainfast_threshold_mm": 12.0,
        "wash_sensitivity": 0.05
    },
    "multisite_protectant": {
        "label": "Multi-Site Contact Protectant (e.g. Folpet, Mancozeb)",
        "base_efficacy": 0.72,
        "half_life_days": 6.0,
        "rainfast_threshold_mm": 5.0,
        "wash_sensitivity": 0.12
    },
    "standard_generic": {
        "label": "Standard European Baseline Trial Regimen",
        "base_efficacy": 0.80,
        "half_life_days": 10.0,
        "rainfast_threshold_mm": 10.0,
        "wash_sensitivity": 0.06
    },
    "favorable_protectant": {
        "label": "Favorable Protectant Intervention (Optimal Timing & Rainfast)",
        "base_efficacy": 0.92,
        "half_life_days": 28.0,
        "rainfast_threshold_mm": 25.0,
        "wash_sensitivity": 0.02
    }
}

REGIONAL_ADAPTATION_FACTORS: Dict[str, float] = {
    "central_europe": 1.0,           # Baseline ETH Zurich calibration
    "east_africa_highland": 1.15,    # Accelerated fungal growth under warm equatorial highlands
    "north_america_plains": 0.95,    # Continental dry-spell suppression
    "uk_maritime": 1.10              # Enhanced Septoria maritime humidity pressure
}


class MechanisticTreatmentResult(BaseModel):
    """Pydantic schema for mechanistic chemical scenario simulation outputs."""
    chemical_class: str
    effective_protection_multiplier: float = Field(..., description="Net disease progression multiplier under protection (0.0 to 1.0).")
    net_efficacy_pct: float = Field(..., description="Overall disease suppression efficacy (1 - multiplier) * 100.")
    wash_off_loss_pct: float = Field(..., description="Efficacy percentage lost due to post-application rainfall wash-off.")
    resistance_discount_pct: float = Field(..., description="Efficacy percentage lost due to local pathogen resistance.")
    unprotected_7d_pct: float
    protected_7d_pct: float
    relative_reduction_pct: float


def compute_mechanistic_treatment_multiplier(
    chemical_class: str = "standard_generic",
    post_spray_rainfall_mm: float = 0.0,
    resistance_index: float = 1.0,
    elapsed_days: float = 7.0,
    region_id: str = "central_europe"
) -> Tuple[float, float, float]:
    """Calculate dynamic fungicide protection multiplier accounting for rain wash-off and resistance.

    Multiplier M in [0.0, 1.0] where S_prot = S_0 + (S_unprot - S_0) * M.
    A lower M indicates higher disease suppression (M = 0.20 => 80% disease reduction).

    Returns:
        Tuple of (effective_multiplier, wash_off_loss_pct, resistance_discount_pct).
    """
    profile = CHEMICAL_PROFILES.get(chemical_class, CHEMICAL_PROFILES["standard_generic"])
    base_eff = profile["base_efficacy"]
    half_life = profile["half_life_days"]
    rain_thresh = profile["rainfast_threshold_mm"]
    wash_sens = profile["wash_sensitivity"]

    # 1. Temporal half-life decay over horizon
    time_decay = np.exp(-np.log(2.0) * (elapsed_days / half_life))

    # 2. Pathogen resistance discount factor (rho in [0.2, 1.0])
    rho = float(np.clip(resistance_index, 0.15, 1.0))
    resistance_loss_pct = (1.0 - rho) * 100.0

    # 3. Rainfall wash-off attenuation
    excess_rain = max(0.0, post_spray_rainfall_mm - rain_thresh)
    rain_retention = float(np.exp(-wash_sens * excess_rain))
    wash_loss_pct = (1.0 - rain_retention) * 100.0

    # Net efficacy
    net_efficacy = base_eff * time_decay * rho * rain_retention
    net_efficacy = float(np.clip(net_efficacy, 0.05, 0.95))

    # Convert efficacy into progression multiplier
    effective_multiplier = float(np.clip(1.0 - net_efficacy, 0.05, 0.95))

    return round(effective_multiplier, 4), round(wash_loss_pct, 2), round(resistance_loss_pct, 2)


def simulate_mechanistic_scenario(
    initial_severity_pct: float,
    unprotected_7d_pct: float,
    chemical_class: str = "sdhi",
    post_spray_rainfall_mm: float = 0.0,
    resistance_index: float = 1.0,
    region_id: str = "central_europe"
) -> MechanisticTreatmentResult:
    """Run mechanistic scenario comparison between untreated control and specific fungicide treatment."""
    s0 = float(initial_severity_pct)
    s_unprot = float(unprotected_7d_pct)
    spread_unprot = max(0.0, s_unprot - s0)

    mult, wash_loss, resist_loss = compute_mechanistic_treatment_multiplier(
        chemical_class=chemical_class,
        post_spray_rainfall_mm=post_spray_rainfall_mm,
        resistance_index=resistance_index,
        elapsed_days=7.0,
        region_id=region_id
    )

    s_prot = s0 + spread_unprot * mult
    s_prot = min(100.0, s_prot)

    # Spread reduction percentage: proportion of projected expansion prevented by treatment
    if spread_unprot > 0.0:
        rel_reduct = float(((s_unprot - s_prot) / spread_unprot) * 100.0)
    else:
        rel_reduct = 0.0

    return MechanisticTreatmentResult(
        chemical_class=chemical_class,
        effective_protection_multiplier=mult,
        net_efficacy_pct=round((1.0 - mult) * 100.0, 2),
        wash_off_loss_pct=wash_loss,
        resistance_discount_pct=resist_loss,
        unprotected_7d_pct=round(s_unprot, 2),
        protected_7d_pct=round(s_prot, 2),
        relative_reduction_pct=round(rel_reduct, 2)
    )
