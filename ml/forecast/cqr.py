"""Conformalized Quantile Regression (CQR) for Epidemiological Forecasting.

Trains LightGBM quantile regression models (q_low=0.05, q_med=0.50, q_high=0.95),
calibrates non-parametric conformal prediction intervals on a dedicated calibration split,
and exports zero-pickle text models and cqr_calibration.json.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import lightgbm as lgb

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger
from ml.forecast.features import get_feature_matrix, CORE_FEATURES
from ml.forecast.residual import logit, inv_logit, compute_residuals

logger = get_logger(__name__)


def train_cqr_quantile_models(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    growth_params: Optional[Dict[str, float]] = None,
    quantiles: Optional[Dict[str, float]] = None,
    feature_set: str = "core"
) -> Dict[str, lgb.Booster]:
    """Train LightGBM quantile regression models for low (0.05), med (0.50), high (0.95)."""
    if quantiles is None:
        quantiles = {"low": 0.05, "med": 0.50, "high": 0.95}

    if train_df.empty:
        logger.warning("Empty train_df passed to train_cqr_quantile_models.")
        return {}

    y_train, _ = compute_residuals(train_df, growth_params)
    X_train, feature_names = get_feature_matrix(train_df, feature_set=feature_set)
    weights_train = train_df["sample_weight"].values if "sample_weight" in train_df.columns else None

    lgb_train = lgb.Dataset(X_train, label=y_train, weight=weights_train, feature_name=feature_names)

    boosters: Dict[str, lgb.Booster] = {}

    for q_name, alpha in quantiles.items():
        params = {
            "objective": "quantile",
            "alpha": float(alpha),
            "metric": "quantile",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbosity": -1,
            "random_state": 42
        }

        booster = lgb.train(
            params=params,
            train_set=lgb_train,
            num_boost_round=250
        )
        boosters[q_name] = booster
        logger.info(f"Trained CQR quantile model '{q_name}' (alpha={alpha}).")

    return boosters


def calibrate_cqr_intervals(
    boosters: Dict[str, lgb.Booster],
    cal_df: pd.DataFrame,
    growth_params: Optional[Dict[str, float]] = None,
    alpha: float = 0.10,
    feature_set: str = "core"
) -> Dict[str, Any]:
    """Calibrate CQR conformal error adjustment factor Q_1_alpha on calibration split."""
    if cal_df.empty or "low" not in boosters or "high" not in boosters:
        logger.warning("Empty calibration DataFrame or missing boosters. Returning fallback zero margin.")
        return {
            "alpha": alpha,
            "confidence_level": 1.0 - alpha,
            "conformal_quantile": 0.0,
            "n_calib_samples": len(cal_df),
            "is_calibrated": False
        }

    y_cal, _ = compute_residuals(cal_df, growth_params)
    X_cal, _ = get_feature_matrix(cal_df, feature_set=feature_set)

    q_low_preds = boosters["low"].predict(X_cal)
    q_high_preds = boosters["high"].predict(X_cal)

    # Compute non-conformity scores E_i = max(q_low - y, y - q_high)
    scores = np.maximum(q_low_preds - y_cal, y_cal - q_high_preds)

    n = len(scores)
    q_level = np.ceil((n + 1) * (1.0 - alpha)) / float(n)
    q_level = min(max(q_level, 0.0), 1.0)

    q_conformal = float(np.quantile(scores, q_level))

    # Mondrian binning by initial severity s0
    s0_arr = cal_df["severity_i"].values if "severity_i" in cal_df.columns else np.zeros(n)
    bin_edges = [0.0, 0.05, 0.15, 0.30, 1.0]
    mondrian_quantiles: Dict[int, float] = {}

    for b_idx in range(len(bin_edges) - 1):
        low_e, high_e = bin_edges[b_idx], bin_edges[b_idx + 1]
        bin_mask = (s0_arr >= low_e) & (s0_arr < high_e) if b_idx < len(bin_edges) - 2 else (s0_arr >= low_e) & (s0_arr <= high_e)
        if np.sum(bin_mask) >= 10:
            sub_scores = scores[bin_mask]
            sub_q = float(np.quantile(sub_scores, q_level))
            mondrian_quantiles[b_idx] = round(sub_q, 5)
        else:
            mondrian_quantiles[b_idx] = round(q_conformal, 5)

    result = {
        "alpha": alpha,
        "confidence_level": 1.0 - alpha,
        "conformal_quantile": round(q_conformal, 5),
        "mondrian_quantiles": mondrian_quantiles,
        "bin_edges": bin_edges,
        "n_calib_samples": n,
        "is_calibrated": True
    }
    logger.info(f"Calibrated CQR conformal prediction interval: Q_{1.0-alpha:.2f} = {q_conformal:.5f} over {n} samples.")
    return result


def predict_cqr_interval(
    boosters: Dict[str, lgb.Booster],
    cqr_calib: Dict[str, Any],
    s0: float,
    dt_days: float,
    t_mean: float,
    rh_mean: float,
    feat_dict: Dict[str, float],
    growth_params: Optional[Dict[str, float]] = None,
    feature_set: str = "core"
) -> Tuple[float, float, float, float]:
    """Predict point forecast median and 90% conformal prediction interval (lower, upper, width).

    Returns:
        Tuple of (s_median, s_lower, s_upper, width).
    """
    from ml.forecast.growth import predict_growth_curve
    s_growth = predict_growth_curve(s0=s0, dt_days=dt_days, t_mean=t_mean, rh_mean=rh_mean, params=growth_params)

    if s0 == 0.0:
        return 0.0, 0.0, 0.0, 0.0

    if not boosters or "med" not in boosters:
        return float(s_growth), float(max(s_growth - 0.05, 0.0)), float(min(s_growth + 0.05, 1.0)), 0.10

    if boosters and "med" in boosters and boosters["med"] is not None:
        feature_names = boosters["med"].feature_name()
    else:
        feature_names = CORE_FEATURES if feature_set.lower() == "core" else EXTENDED_FEATURES

    x_vec = np.array([[feat_dict.get(col, 0.0) for col in feature_names]], dtype=np.float32)

    z_growth = float(logit(np.array([s_growth]))[0])

    q_med_res = float(boosters["med"].predict(x_vec)[0])
    q_low_res = float(boosters["low"].predict(x_vec)[0]) if "low" in boosters else q_med_res - 0.2
    q_high_res = float(boosters["high"].predict(x_vec)[0]) if "high" in boosters else q_med_res + 0.2

    q_conf = cqr_calib.get("conformal_quantile", 0.0)

    # Conformalized bounds
    res_low_calib = q_low_res - q_conf
    res_high_calib = q_high_res + q_conf

    s_med = float(inv_logit(np.array([z_growth + q_med_res]))[0])
    s_low = float(inv_logit(np.array([z_growth + res_low_calib]))[0])
    s_high = float(inv_logit(np.array([z_growth + res_high_calib]))[0])

    # Enforce monotonicity: s_low <= s_med <= s_high
    s_low = min(s_low, s_med)
    s_high = max(s_high, s_med)
    width = s_high - s_low

    K = growth_params.get("K", 0.95) if growth_params else 0.95
    s_high = min(s_high, K)

    return float(s_med), float(s_low), float(s_high), float(width)


def export_cqr_artifacts(
    boosters: Dict[str, lgb.Booster],
    cqr_calib: Dict[str, Any],
    export_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """Export CQR LightGBM text models and cqr_calibration.json."""
    paths = get_paths()
    if export_dir is None:
        export_dir = paths.project_root / "models" / "forecast"

    export_dir.mkdir(parents=True, exist_ok=True)
    exported_files = {}

    for q_name, booster in boosters.items():
        fn = f"lgbm_q_{q_name}.txt"
        fp = export_dir / fn
        booster.save_model(str(fp))
        exported_files[q_name] = fp

    cal_path = export_dir / "cqr_calibration.json"
    cal_path.write_text(json.dumps(cqr_calib, indent=2), encoding="utf-8")
    exported_files["calibration"] = cal_path

    logger.info(f"Exported CQR models and calibration to {export_dir}")
    return exported_files


def main() -> int:
    return 0


if __name__ == "__main__":
    sys.exit(main())
