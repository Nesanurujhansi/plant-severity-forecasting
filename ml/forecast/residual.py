"""LightGBM Residual Model for Epidemiological Growth Correction.

Trains a LightGBM regressor to predict the residual difference between ground-truth logit severity
and mechanistic logistic growth predictions:
    y_res = logit(s_true) - logit(s_growth)
Uses conservative hyperparameters, grouped early stopping on sample_id,
and exports the model in zero-pickle plain text format (lgbm_residual.txt).
"""

import sys
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
from ml.forecast.features import get_feature_matrix, CORE_FEATURES, EXTENDED_FEATURES
from ml.forecast.growth import predict_growth_curve, DEFAULT_GROWTH_PARAMS

logger = get_logger(__name__)


def logit(p: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Compute stable logit transformation of severity fraction p."""
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clipped / (1.0 - p_clipped))


def inv_logit(z: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Compute inverse logit (sigmoid) transformation back to severity fraction."""
    p = 1.0 / (1.0 + np.exp(-z))
    return np.clip(p, 0.0, 1.0)


def compute_residuals(
    pairs_df: pd.DataFrame,
    growth_params: Optional[Dict[str, float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute target residuals y_res = logit(s_true) - logit(s_growth) for training pairs.

    Args:
        pairs_df: DataFrame with ['severity_i', 'severity_j', 'dt_days', 't_mean_interval', 'rh_mean_interval'].
        growth_params: Optional growth parameters.

    Returns:
        Tuple of (residuals_array, growth_predictions_array).
    """
    s0_arr = pairs_df["severity_i"].values.astype(np.float64)
    s1_arr = pairs_df["severity_j"].values.astype(np.float64)
    dt_arr = pairs_df["dt_days"].values.astype(np.float64)

    t_arr = pairs_df["t_mean_interval"].values if "t_mean_interval" in pairs_df.columns else np.full(len(s0_arr), 18.0)
    rh_arr = pairs_df["rh_mean_interval"].values if "rh_mean_interval" in pairs_df.columns else np.full(len(s0_arr), 75.0)

    s_growth = np.zeros_like(s0_arr)
    for idx in range(len(s0_arr)):
        s_growth[idx] = predict_growth_curve(
            s0=s0_arr[idx],
            dt_days=dt_arr[idx],
            t_mean=t_arr[idx],
            rh_mean=rh_arr[idx],
            params=growth_params
        )

    z_true = logit(s1_arr)
    z_growth = logit(s_growth)
    residuals = z_true - z_growth

    return residuals, s_growth


def train_residual_model(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    growth_params: Optional[Dict[str, float]] = None,
    feature_set: str = "core",
    hyperparams: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Train LightGBM residual regression model.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        growth_params: Fitted growth curve parameters.
        feature_set: 'core' or 'extended'.
        hyperparams: Dict of LightGBM hyperparameters.

    Returns:
        Dict containing trained booster, feature_names, and training metrics.
    """
    if train_df.empty:
        logger.warning("Empty train_df passed to train_residual_model.")
        return {"booster": None, "feature_names": [], "status": "empty_input"}

    # Compute target residuals
    y_train, _ = compute_residuals(train_df, growth_params)
    X_train, feature_names = get_feature_matrix(train_df, feature_set=feature_set)
    weights_train = train_df["sample_weight"].values if "sample_weight" in train_df.columns else None

    lgb_train = lgb.Dataset(X_train, label=y_train, weight=weights_train, feature_name=feature_names)

    valid_sets = [lgb_train]
    valid_names = ["train"]

    if val_df is not None and not val_df.empty:
        y_val, _ = compute_residuals(val_df, growth_params)
        X_val, _ = get_feature_matrix(val_df, feature_set=feature_set)
        weights_val = val_df["sample_weight"].values if "sample_weight" in val_df.columns else None
        lgb_val = lgb.Dataset(X_val, label=y_val, weight=weights_val, feature_name=feature_names, reference=lgb_train)
        valid_sets.append(lgb_val)
        valid_names.append("val")

    # Construct monotone constraints (+1 for severity_i and dt_days)
    mono_c = [1 if col in ["severity_i", "dt_days"] else 0 for col in feature_names]
    params = {
        "objective": "regression",
        "metric": "l2",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "monotone_constraints": mono_c,
        "verbosity": -1,
        "random_state": 42
    }
    if hyperparams is not None:
        params.update(hyperparams)

    num_boost_round = params.pop("n_estimators", 300)
    early_stopping_rounds = params.pop("early_stopping_rounds", 30)

    callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)] if len(valid_sets) > 1 else []

    booster = lgb.train(
        params=params,
        train_set=lgb_train,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks
    )

    logger.info(f"Trained LightGBM residual model ({feature_set}) over {booster.best_iteration} trees.")

    best_iter = booster.best_iteration if (hasattr(booster, "best_iteration") and booster.best_iteration and booster.best_iteration > 0) else booster.num_trees()

    return {
        "booster": booster,
        "feature_names": feature_names,
        "best_iteration": best_iter,
        "status": "success"
    }


def predict_residual_and_combine(
    booster: lgb.Booster,
    s0: float,
    dt_days: float,
    t_mean: float,
    rh_mean: float,
    feat_dict: Dict[str, float],
    growth_params: Optional[Dict[str, float]] = None,
    feature_set: str = "core"
) -> float:
    """Predict final combined severity s_hat = inv_logit(logit(s_growth) + residual_pred)."""
    s_growth = predict_growth_curve(s0=s0, dt_days=dt_days, t_mean=t_mean, rh_mean=rh_mean, params=growth_params)

    if s0 == 0.0:
        return 0.0

    if booster is None:
        return float(s_growth)

    feature_names = CORE_FEATURES if feature_set.lower() == "core" else EXTENDED_FEATURES
    x_vec = np.array([[feat_dict.get(col, 0.0) for col in feature_names]], dtype=np.float32)

    res_pred = float(booster.predict(x_vec)[0])
    z_growth = float(logit(np.array([s_growth]))[0])
    z_final = z_growth + res_pred

    s_final = float(inv_logit(np.array([z_final]))[0])
    K = growth_params.get("K", 0.95) if growth_params else 0.95
    return float(np.clip(s_final, 0.0, K))


def export_residual_model(
    booster: lgb.Booster,
    output_path: Optional[Path] = None
) -> Path:
    """Export LightGBM residual model in zero-pickle plain text format."""
    paths = get_paths()
    if output_path is None:
        output_path = paths.project_root / "models" / "forecast" / "lgbm_residual.txt"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(output_path))
    logger.info(f"Exported lgbm_residual.txt to {output_path}")
    return output_path


def main() -> int:
    return 0


if __name__ == "__main__":
    sys.exit(main())
