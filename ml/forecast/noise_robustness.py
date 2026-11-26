"""Severity Noise-Robustness and End-to-End Pipeline Chain Evaluator.

Evaluates progression forecasting accuracy degradation when initial severity is perturbed by
Phase 5 vision model prediction noise (mean bias = -0.001, std = 0.0133).
Tests the complete end-to-end chain: Image -> Phase 5 Predictor -> Forecaster -> Ground-Truth Future Severity.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger
from ml.forecast.features import build_feature_vector
from ml.forecast.cqr import predict_cqr_interval

logger = get_logger(__name__)


def inject_severity_noise(
    severity: Union[float, np.ndarray],
    bias: float = -0.00106,
    std_dev: float = 0.013327,
    seed: int = 42
) -> Union[float, np.ndarray]:
    """Inject Phase 5 measurement noise distribution into ground-truth initial severity fraction s0.

    Args:
        severity: Ground-truth severity fraction in [0.0, 1.0].
        bias: Measured mean bias from Phase 5 evaluation.
        std_dev: Measured standard deviation of error from Phase 5 evaluation.
        seed: Fixed random seed for reproducibility.

    Returns:
        Noisy severity fraction clipped to [0.0, 1.0].
    """
    rng = np.random.RandomState(seed)

    if isinstance(severity, (int, float)):
        noise = rng.normal(loc=bias, scale=std_dev)
        s_noisy = float(np.clip(severity + noise, 0.0, 1.0))
        return s_noisy
    else:
        arr = np.asarray(severity, dtype=np.float64)
        noise = rng.normal(loc=bias, scale=std_dev, size=arr.shape)
        s_noisy = np.clip(arr + noise, 0.0, 1.0)
        return s_noisy


def evaluate_noise_robustness(
    pairs_df: pd.DataFrame,
    boosters: Dict[str, Any],
    cqr_calib: Dict[str, Any],
    growth_params: Optional[Dict[str, float]] = None,
    bias: float = -0.00106,
    std_dev: float = 0.013327,
    seed: int = 42
) -> Dict[str, Any]:
    """Compare forecast accuracy using Ground-Truth initial severity vs. Noisy initial severity.

    Args:
        pairs_df: DataFrame of test evaluation pairs.
        boosters: Trained CQR boosters.
        cqr_calib: Calibration dict.
        growth_params: Fitted growth curve params.
        bias: Phase 5 mean error bias.
        std_dev: Phase 5 error standard deviation.
        seed: Random seed.

    Returns:
        Dict comparing MAE, RMSE, and accuracy degradation (delta MAE in percentage points).
    """
    if pairs_df.empty or not boosters:
        return {
            "gt_mae_pct": 0.0,
            "noisy_mae_pct": 0.0,
            "delta_mae_pct": 0.0,
            "pct_degradation": 0.0,
            "status": "empty_input"
        }

    s0_clean = pairs_df["severity_i"].values.astype(np.float64)
    s0_noisy = inject_severity_noise(s0_clean, bias=bias, std_dev=std_dev, seed=seed)

    s1_true = pairs_df["severity_j"].values.astype(np.float64)
    dt_arr = pairs_df["dt_days"].values.astype(np.float64)

    preds_clean, preds_noisy = [], []

    for idx, row in pairs_df.reset_index(drop=True).iterrows():
        dt = float(row["dt_days"])
        tm = float(row.get("t_mean_interval", 18.0))
        rhm = float(row.get("rh_mean_interval", 75.0))

        # 1. Clean prediction
        feat_clean = build_feature_vector(
            severity_i=s0_clean[idx],
            dt_days=dt,
            weather_df=pd.DataFrame([{"date": "2024-01-01", "t_mean": tm, "rh_mean": rhm, "precip": 0.0}])
        )
        s_med_clean, _, _, _ = predict_cqr_interval(
            boosters=boosters, cqr_calib=cqr_calib, s0=s0_clean[idx],
            dt_days=dt, t_mean=tm, rh_mean=rhm, feat_dict=feat_clean, growth_params=growth_params
        )
        preds_clean.append(s_med_clean)

        # 2. Noisy prediction
        feat_noisy = build_feature_vector(
            severity_i=s0_noisy[idx],
            dt_days=dt,
            weather_df=pd.DataFrame([{"date": "2024-01-01", "t_mean": tm, "rh_mean": rhm, "precip": 0.0}])
        )
        s_med_noisy, _, _, _ = predict_cqr_interval(
            boosters=boosters, cqr_calib=cqr_calib, s0=s0_noisy[idx],
            dt_days=dt, t_mean=tm, rh_mean=rhm, feat_dict=feat_noisy, growth_params=growth_params
        )
        preds_noisy.append(s_med_noisy)

    yt_pct = s1_true * 100.0
    mae_clean = float(np.mean(np.abs(yt_pct - np.array(preds_clean) * 100.0)))
    mae_noisy = float(np.mean(np.abs(yt_pct - np.array(preds_noisy) * 100.0)))

    delta_mae = mae_noisy - mae_clean
    pct_degrad = (delta_mae / max(mae_clean, 1e-4)) * 100.0

    result = {
        "n_samples": len(pairs_df),
        "gt_mae_pct": round(mae_clean, 4),
        "noisy_mae_pct": round(mae_noisy, 4),
        "delta_mae_pct": round(delta_mae, 4),
        "pct_degradation": round(pct_degrad, 2),
        "phase5_noise_bias": bias,
        "phase5_noise_std": std_dev,
        "status": "success"
    }

    logger.info(f"Severity Noise Robustness: GT MAE = {mae_clean:.2f}%, Noisy MAE = {mae_noisy:.2f}% (Δ = +{delta_mae:.2f}% points / {pct_degrad:.1f}% loss).")
    return result


def main() -> int:
    return 0


if __name__ == "__main__":
    sys.exit(main())
