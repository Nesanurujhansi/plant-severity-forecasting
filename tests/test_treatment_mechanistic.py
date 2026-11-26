"""Unit tests for Phase 05: Mechanistic Chemical Modeling & Regional Domain Adaptation."""

import pytest
import numpy as np

from ml.scenario.mechanistic import (
    compute_mechanistic_treatment_multiplier,
    simulate_mechanistic_scenario,
    MechanisticTreatmentResult,
    CHEMICAL_PROFILES
)


def test_fungicide_chemical_profiles():
    # 1. SDHI (systemic, long half-life) vs Multi-site protectant (contact, short half-life)
    mult_sdhi, _, _ = compute_mechanistic_treatment_multiplier(chemical_class="sdhi", post_spray_rainfall_mm=0.0)
    mult_contact, _, _ = compute_mechanistic_treatment_multiplier(chemical_class="multisite_protectant", post_spray_rainfall_mm=0.0)

    # Lower multiplier means higher disease suppression
    assert mult_sdhi < mult_contact
    assert mult_sdhi < 0.40  # SDHI suppression > 60%


def test_rainfall_wash_off_decay():
    # Dry conditions vs 45 mm downpour
    mult_dry, wash_dry, _ = compute_mechanistic_treatment_multiplier(
        chemical_class="multisite_protectant",
        post_spray_rainfall_mm=0.0
    )
    mult_rain, wash_rain, _ = compute_mechanistic_treatment_multiplier(
        chemical_class="multisite_protectant",
        post_spray_rainfall_mm=45.0
    )

    assert wash_dry == 0.0
    assert wash_rain > 30.0  # Significant wash-off loss (>30%)
    # Multiplier increases (efficacy drops) after heavy rain
    assert mult_rain > mult_dry


def test_pathogen_target_site_resistance():
    # Wild-type sensitive pathogen (index = 1.0) vs Resistant strain (index = 0.30)
    mult_sensitive, _, loss_sens = compute_mechanistic_treatment_multiplier(
        chemical_class="strobilurin_qoi",
        resistance_index=1.0
    )
    mult_resistant, _, loss_res = compute_mechanistic_treatment_multiplier(
        chemical_class="strobilurin_qoi",
        resistance_index=0.30
    )

    assert loss_sens == 0.0
    assert loss_res == 70.0  # 70% loss due to resistance
    assert mult_resistant > mult_sensitive


def test_simulate_mechanistic_scenario():
    res = simulate_mechanistic_scenario(
        initial_severity_pct=12.0,
        unprotected_7d_pct=32.0,
        chemical_class="sdhi",
        post_spray_rainfall_mm=5.0,
        resistance_index=1.0
    )

    assert isinstance(res, MechanisticTreatmentResult)
    assert res.chemical_class == "sdhi"
    assert res.protected_7d_pct < res.unprotected_7d_pct
    assert res.protected_7d_pct >= 12.0  # Cannot drop below initial severity
    assert res.relative_reduction_pct > 30.0
