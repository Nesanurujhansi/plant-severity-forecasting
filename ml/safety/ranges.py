"""Input Range Validation Module for crop-prognosis.

Validates microclimate weather inputs (temperature, humidity, VPD, precipitation),
forecast time horizons, and crop categories against empirical training distributions.
Flags requests outside valid bounds and handles crop validation states.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.logging_utils import get_logger

logger = get_logger(__name__)


def validate_weather_inputs(
    temp_c: float,
    rh_pct: float,
    vpd_kpa: Optional[float] = None,
    prec_mm: Optional[float] = None,
    temp_min: float = -5.0,
    temp_max: float = 45.0,
    rh_min: float = 10.0,
    rh_max: float = 100.0,
    vpd_min: float = 0.0,
    vpd_max: float = 6.0
) -> Dict[str, Any]:
    """Validate weather inputs against historical field trial and meteorological bounds."""
    warnings = []
    out_of_range = False

    if temp_c < temp_min or temp_c > temp_max:
        out_of_range = True
        warnings.append(f"Temperature {temp_c:.1f}°C is outside training range [{temp_min:.1f}°C, {temp_max:.1f}°C]")

    if rh_pct < rh_min or rh_pct > rh_max:
        out_of_range = True
        warnings.append(f"Relative humidity {rh_pct:.1f}% is outside valid range [{rh_min:.1f}%, {rh_max:.1f}%]")

    if vpd_kpa is not None:
        if vpd_kpa < vpd_min or vpd_kpa > vpd_max:
            out_of_range = True
            warnings.append(f"Vapor Pressure Deficit {vpd_kpa:.2f} kPa is outside range [{vpd_min:.1f}, {vpd_max:.1f}] kPa")

    if prec_mm is not None and prec_mm < 0.0:
        out_of_range = True
        warnings.append(f"Precipitation cannot be negative ({prec_mm:.1f} mm)")

    return {
        "check": "weather_ranges",
        "passed": not out_of_range,
        "warnings": warnings,
        "message": "Weather inputs within valid training bounds." if not out_of_range else "; ".join(warnings)
    }


def validate_time_horizon(
    horizon_days: float,
    min_days: float = 0.5,
    max_days: float = 14.0
) -> Dict[str, Any]:
    """Validate requested progression forecast time horizon dt_days."""
    passed = bool(min_days <= horizon_days <= max_days)
    warning = "" if passed else f"Forecast horizon {horizon_days:.1f} days is outside validated window [{min_days:.1f}, {max_days:.1f}] days."
    return {
        "check": "time_horizon",
        "passed": passed,
        "horizon_days": horizon_days,
        "min_days": min_days,
        "max_days": max_days,
        "warning": warning
    }


def validate_crop_type(
    crop_name: str,
    supported_crops: Optional[List[str]] = None,
    validated_progression_crops: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Validate crop category for module execution.

    Severity module runs across all supported crops.
    Progression and Treatment modules indicate validation status if baseline differs.
    """
    if supported_crops is None:
        supported_crops = ["Wheat", "Rice", "Tomato", "Corn", "Sugarcane", "Cotton"]
    if validated_progression_crops is None:
        validated_progression_crops = ["Wheat", "Rice", "Tomato", "Corn", "Sugarcane", "Cotton"]

    crop_clean = crop_name.strip().title()

    is_supported = any(c.lower() in crop_clean.lower() for c in supported_crops)
    is_progression_validated = any(c.lower() in crop_clean.lower() for c in validated_progression_crops)

    if not is_supported:
        status = "UNSUPPORTED_CROP"
        message = f"Crop '{crop_name}' is not in supported species list ({', '.join(supported_crops)})."
        severity_allowed = True  # Vision module runs zero-shot
        progression_allowed = False
        treatment_allowed = False
    elif not is_progression_validated:
        status = "NOT_VALIDATED_FOR_CROP"
        message = f"Progression forecasting is not fully validated for '{crop_name}'. Showing baseline estimate only."
        severity_allowed = True
        progression_allowed = True
        treatment_allowed = False
    else:
        status = "FULLY_VALIDATED"
        message = f"Crop '{crop_name}' is fully validated across all modules."
        severity_allowed = True
        progression_allowed = True
        treatment_allowed = True

    return {
        "check": "crop_type",
        "crop_name": crop_name,
        "status": status,
        "is_supported": is_supported,
        "is_progression_validated": is_progression_validated,
        "severity_allowed": severity_allowed,
        "progression_allowed": progression_allowed,
        "treatment_allowed": treatment_allowed,
        "message": message
    }


def evaluate_input_ranges(
    temp_c: float = 20.0,
    rh_pct: float = 70.0,
    horizon_days: float = 7.0,
    crop_name: str = "Wheat",
    vpd_kpa: Optional[float] = None,
    prec_mm: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run comprehensive input range evaluation."""
    weather_res = validate_weather_inputs(temp_c, rh_pct, vpd_kpa, prec_mm)
    horizon_res = validate_time_horizon(horizon_days)
    crop_res = validate_crop_type(crop_name)

    all_passed = weather_res["passed"] and horizon_res["passed"] and crop_res["is_supported"]
    warnings = weather_res["warnings"] + ([horizon_res["warning"]] if horizon_res["warning"] else [])

    return {
        "all_ranges_valid": all_passed,
        "warnings": warnings,
        "details": {
            "weather": weather_res,
            "horizon": horizon_res,
            "crop": crop_res
        }
    }


def main() -> int:
    res = evaluate_input_ranges(temp_c=22.5, rh_pct=65.0, horizon_days=7.0, crop_name="Wheat")
    print("Range Check Result:", res["all_ranges_valid"], "Warnings:", len(res["warnings"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
