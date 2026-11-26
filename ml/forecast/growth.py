"""Enhanced Epidemiological Logistic Growth Engine.

Implements biological differential growth dynamics with:
1. Beta-type asymmetric thermal response curve g_T(T) (Yan & Hunt model)
2. Relative humidity response curve g_H(RH) & Vapor Pressure Deficit (VPD)
3. Microclimate vapor pressure deficit (VPD in kPa) physics
4. Temperature sensitivity analysis (+5°C shift)
5. SciPy least-squares optimization & zero-pickle serialization to growth_params.json
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_GROWTH_PARAMS: Dict[str, float] = {
    "r0": 0.15,          # Base growth rate (1/day)
    "t_opt": 20.5,       # Optimum temperature (°C)
    "t_min": 4.0,        # Minimum cardinal temperature (°C)
    "t_max": 34.0,       # Maximum cardinal temperature (°C)
    "t_sigma": 6.5,      # Thermal spread (°C)
    "k_h": 2.2,          # Humidity steepness
    "rh_50": 65.0,       # Humidity 50% inflection point (%)
    "K": 0.95,           # Maximum carrying capacity fraction
    "seed_floor": 1e-5   # Tiny seed floor for non-zero severity
}

PARAM_BOUNDS = (
    [0.001, 12.0, 2.0, 30.0, 3.0, 0.1, 40.0, 0.50],  # Lower bounds
    [0.500, 30.0, 8.0, 38.0, 15.0, 5.0, 95.0, 1.00]   # Upper bounds
)


def compute_vpd(T: Union[float, np.ndarray], RH: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Compute Vapor Pressure Deficit (VPD in kPa) using Tetens equation.

    Args:
        T: Temperature in °C.
        RH: Relative humidity in %.

    Returns:
        VPD in kPa (>= 0.0).
    """
    T = np.asarray(T, dtype=np.float64)
    RH = np.clip(np.asarray(RH, dtype=np.float64), 0.0, 100.0)

    svp = 0.61078 * np.exp((17.27 * T) / (T + 237.3))
    vpd = svp * (1.0 - RH / 100.0)
    return np.maximum(vpd, 0.0)


def thermal_response_beta(
    T: Union[float, np.ndarray],
    t_opt: float = 20.5,
    t_min: float = 4.0,
    t_max: float = 34.0
) -> Union[float, np.ndarray]:
    """Asymmetric Beta thermal response curve g_T(T) (Yan & Hunt model).

    g_T(T) = ((t_max - T)/(t_max - t_opt)) * ((T - t_min)/(t_opt - t_min))^((t_opt - t_min)/(t_max - t_opt))
    Bounded in [0.0, 1.0].
    """
    T_arr = np.asarray(T, dtype=np.float64)
    g_t = np.zeros_like(T_arr)

    mask = (T_arr > t_min) & (T_arr < t_max)
    if np.any(mask):
        T_m = T_arr[mask]
        alpha = (t_opt - t_min) / max(t_max - t_opt, 1e-4)
        num1 = (t_max - T_m) / max(t_max - t_opt, 1e-4)
        num2 = (T_m - t_min) / max(t_opt - t_min, 1e-4)
        val = num1 * (num2 ** alpha)
        g_t[mask] = np.clip(val, 0.0, 1.0)

    return g_t if isinstance(T, np.ndarray) else float(g_t)


def thermal_response(T: Union[float, np.ndarray], t_opt: float = 20.5, t_sigma: float = 6.5) -> Union[float, np.ndarray]:
    """Gaussian thermal response curve fallback g_T(T)."""
    g_t = np.exp(-0.5 * ((T - t_opt) / max(t_sigma, 1e-4)) ** 2)
    return np.clip(g_t, 0.0, 1.0)


def humidity_response(RH: Union[float, np.ndarray], k_h: float = 2.2, rh_50: float = 65.0) -> Union[float, np.ndarray]:
    """Logistic relative humidity response curve g_H(RH)."""
    g_h = 1.0 / (1.0 + np.exp(-k_h * (RH - rh_50) / 10.0))
    return np.clip(g_h, 0.0, 1.0)


def compute_growth_rate(
    T: float,
    RH: float,
    r0: float = 0.15,
    t_opt: float = 20.5,
    t_sigma: float = 6.5,
    k_h: float = 2.2,
    rh_50: float = 65.0,
    use_beta: bool = True
) -> float:
    """Compute environmental growth rate r(T, RH)."""
    if use_beta:
        g_t = float(thermal_response_beta(T, t_opt=t_opt, t_min=4.0, t_max=34.0))
    else:
        g_t = float(thermal_response(T, t_opt=t_opt, t_sigma=t_sigma))

    g_h = float(humidity_response(RH, k_h=k_h, rh_50=rh_50))
    return float(r0 * g_t * g_h)


def predict_growth_curve(
    s0: float,
    dt_days: float,
    t_mean: float = 18.0,
    rh_mean: float = 75.0,
    params: Optional[Dict[str, float]] = None,
    n_steps_per_day: int = 4
) -> float:
    """Integrate logistic differential equation ds/dt = r(T, RH) * s * (1 - s/K) over dt_days."""
    p = DEFAULT_GROWTH_PARAMS if params is None else {**DEFAULT_GROWTH_PARAMS, **params}

    s0 = float(np.clip(s0, 0.0, 1.0))
    if s0 == 0.0:
        return 0.0  # Zero initial infection stays zero

    seed_floor = p.get("seed_floor", 1e-5)
    s = max(s0, seed_floor)
    K = p.get("K", 0.95)

    if s >= K:
        return float(K)

    dt_step = 1.0 / float(max(n_steps_per_day, 1))
    total_steps = max(int(round(dt_days * n_steps_per_day)), 1)
    actual_step = float(dt_days) / float(total_steps)

    r_val = compute_growth_rate(
        T=t_mean,
        RH=rh_mean,
        r0=p.get("r0", 0.15),
        t_opt=p.get("t_opt", 20.5),
        t_sigma=p.get("t_sigma", 6.5),
        k_h=p.get("k_h", 2.2),
        rh_50=p.get("rh_50", 65.0)
    )

    for _ in range(total_steps):
        ds = r_val * s * (1.0 - s / K)
        s += ds * actual_step
        s = min(s, K)

    return float(np.clip(s, 0.0, K))


def analyze_temperature_sensitivity(
    T_base: float = 18.0,
    RH_base: float = 75.0,
    temp_delta: float = 5.0,
    params: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Calculate growth rate change Δr for a +5°C temperature shift."""
    p = DEFAULT_GROWTH_PARAMS if params is None else {**DEFAULT_GROWTH_PARAMS, **params}

    r_base = compute_growth_rate(T_base, RH_base, p["r0"], p["t_opt"], p["t_sigma"], p["k_h"], p["rh_50"])
    r_warmed = compute_growth_rate(T_base + temp_delta, RH_base, p["r0"], p["t_opt"], p["t_sigma"], p["k_h"], p["rh_50"])

    diff_r = r_warmed - r_base
    pct_change = (diff_r / max(r_base, 1e-6)) * 100.0

    if diff_r > 0:
        effect_desc = f"A +{temp_delta}°C temperature shift increases growth rate from {r_base:.4f} to {r_warmed:.4f} (+{pct_change:.1f}%) closer to optimum ({p['t_opt']}°C)."
    else:
        effect_desc = f"A +{temp_delta}°C temperature shift decreases growth rate from {r_base:.4f} to {r_warmed:.4f} ({pct_change:.1f}%) exceeding optimum ({p['t_opt']}°C)."

    return {
        "t_base": T_base,
        "t_warmed": T_base + temp_delta,
        "r_base": round(r_base, 5),
        "r_warmed": round(r_warmed, 5),
        "r_delta": round(diff_r, 5),
        "pct_change": round(pct_change, 2),
        "description": effect_desc
    }


def fit_growth_curve(
    pairs_df: pd.DataFrame,
    scale: str = "logit",
    use_sample_weights: bool = True
) -> Dict[str, Any]:
    """Fit growth curve parameters using SciPy least-squares optimization."""
    if pairs_df.empty or len(pairs_df) < 10:
        logger.warning("Fewer than 10 training pairs provided. Returning default growth parameters.")
        return {
            "params": DEFAULT_GROWTH_PARAMS,
            "fitted_scale": scale,
            "cost": 0.0,
            "n_samples": len(pairs_df),
            "status": "default_fallback"
        }

    s0_arr = pairs_df["severity_i"].values.astype(np.float64)
    s1_arr = pairs_df["severity_j"].values.astype(np.float64)
    dt_arr = pairs_df["dt_days"].values.astype(np.float64)

    t_arr = pairs_df["t_mean_interval"].values.astype(np.float64) if "t_mean_interval" in pairs_df.columns else np.full(len(s0_arr), 18.0)
    rh_arr = pairs_df["rh_mean_interval"].values.astype(np.float64) if "rh_mean_interval" in pairs_df.columns else np.full(len(s0_arr), 75.0)
    weights = pairs_df["sample_weight"].values.astype(np.float64) if (use_sample_weights and "sample_weight" in pairs_df.columns) else np.ones(len(s0_arr))

    eps = 1e-5

    def residual_func(param_vec: np.ndarray) -> np.ndarray:
        r0, t_opt, t_sigma, k_h, rh_50, K = param_vec[:6]
        p_dict = {"r0": r0, "t_opt": t_opt, "t_sigma": t_sigma, "k_h": k_h, "rh_50": rh_50, "K": K, "seed_floor": eps}

        preds = np.zeros_like(s1_arr)
        for idx in range(len(s0_arr)):
            preds[idx] = predict_growth_curve(
                s0=s0_arr[idx],
                dt_days=dt_arr[idx],
                t_mean=t_arr[idx],
                rh_mean=rh_arr[idx],
                params=p_dict
            )

        if scale == "logit":
            y_true_logit = np.log((np.clip(s1_arr, eps, 1.0 - eps)) / (1.0 - np.clip(s1_arr, eps, 1.0 - eps)))
            y_pred_logit = np.log((np.clip(preds, eps, 1.0 - eps)) / (1.0 - np.clip(preds, eps, 1.0 - eps)))
            res = (y_pred_logit - y_true_logit) * np.sqrt(weights)
        else:
            res = (preds - s1_arr) * np.sqrt(weights)

        return np.nan_to_num(res, nan=0.0)

    x0 = [DEFAULT_GROWTH_PARAMS[k] for k in ["r0", "t_opt", "t_sigma", "k_h", "rh_50", "K"]]
    b_low = [0.001, 12.0, 2.0, 0.1, 40.0, 0.50]
    b_high = [0.500, 30.0, 15.0, 5.0, 95.0, 1.00]

    try:
        opt_res = least_squares(residual_func, x0=x0, bounds=(b_low, b_high), max_nfev=300)
        fitted_vec = opt_res.x

        fitted_params = {
            "r0": round(float(fitted_vec[0]), 5),
            "t_opt": round(float(fitted_vec[1]), 2),
            "t_sigma": round(float(fitted_vec[2]), 2),
            "k_h": round(float(fitted_vec[3]), 2),
            "rh_50": round(float(fitted_vec[4]), 2),
            "K": round(float(fitted_vec[5]), 4),
            "seed_floor": eps
        }

        logger.info(f"Fitted growth curve on '{scale}' scale: {fitted_params}")
        return {
            "params": fitted_params,
            "fitted_scale": scale,
            "cost": float(opt_res.cost),
            "n_samples": len(pairs_df),
            "status": "success"
        }
    except Exception as e:
        logger.warning(f"Optimization failed ({e}). Returning default parameters.")
        return {
            "params": DEFAULT_GROWTH_PARAMS,
            "fitted_scale": scale,
            "cost": 0.0,
            "n_samples": len(pairs_df),
            "status": f"error: {e}"
        }


def export_growth_params(
    params_dict: Dict[str, Any],
    output_path: Optional[Path] = None
) -> Path:
    """Export growth parameters JSON artifact."""
    paths = get_paths()
    if output_path is None:
        output_path = paths.project_root / "models" / "forecast" / "growth_params.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(params_dict, indent=2), encoding="utf-8")
    logger.info(f"Exported growth_params.json to {output_path}")
    return output_path
