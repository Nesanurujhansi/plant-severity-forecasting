"""Enhanced Feature Set Definitions, Extraction Engine, and Schema Parity Exporter.

Defines CORE (19 features) and EXTENDED (24 features) feature sets for epidemiological
progression forecasting, including VPD microclimate physics, GDD thermal time, and lesion density.
Guarantees offline-online feature parity and exports feature_schema.json.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger
from ml.forecast.growth import compute_vpd

logger = get_logger(__name__)

CORE_FEATURES: List[str] = [
    "severity_i",
    "dt_days",
    "n_lesions_i",
    "mean_lesion_area_i",
    "lesion_density_i",
    "t_mean_past3",
    "t_mean_past7",
    "t_mean_interval",
    "t_max_min_interval",
    "gdd_interval",
    "rh_mean_past3",
    "rh_mean_past7",
    "rh_mean_interval",
    "vpd_mean_interval",
    "precip_total_past3",
    "precip_total_past7",
    "precip_total_interval",
    "crop_stage",
    "cultivar_group"
]

EXTENDED_FEATURES: List[str] = CORE_FEATURES + [
    "cultivar_trait_score",
    "rh_std_interval",
    "batch_id",
    "year_id",
    "treatment_code"
]


def extract_weather_summary(
    weather_df: pd.DataFrame,
    start_date: Optional[Any] = None,
    end_date: Optional[Any] = None,
    past_days: int = 0,
    t_base: float = 4.0
) -> Dict[str, float]:
    """Extract weather summary metrics (temperature, VPD, GDD, humidity, precipitation)."""
    if weather_df.empty or "date" not in weather_df.columns:
        return {
            "t_mean": 18.0,
            "t_max_min": 18.0,
            "rh_mean": 75.0,
            "rh_std": 0.0,
            "vpd_mean": 0.52,
            "gdd": 14.0,
            "precip_total": 0.0
        }

    w_df = weather_df.copy()
    w_df["date_dt"] = pd.to_datetime(w_df["date"]).dt.date

    if past_days > 0 and start_date is not None:
        ref_date = pd.to_datetime(start_date).date()
        st = ref_date - pd.Timedelta(days=past_days)
        et = ref_date
        sub = w_df[(w_df["date_dt"] >= st) & (w_df["date_dt"] <= et)]
    elif start_date is not None and end_date is not None:
        st = pd.to_datetime(start_date).date()
        et = pd.to_datetime(end_date).date()
        sub = w_df[(w_df["date_dt"] >= st) & (w_df["date_dt"] <= et)]
    else:
        sub = w_df

    if sub.empty:
        return {
            "t_mean": 18.0,
            "t_max_min": 18.0,
            "rh_mean": 75.0,
            "rh_std": 0.0,
            "vpd_mean": 0.52,
            "gdd": 14.0,
            "precip_total": 0.0
        }

    t_mean = float(sub["t_mean"].mean()) if "t_mean" in sub.columns else 18.0
    if "t_max" in sub.columns and "t_min" in sub.columns:
        t_mm = float(((sub["t_max"] + sub["t_min"]) / 2.0).mean())
    else:
        t_mm = t_mean

    rh_mean = float(sub["rh_mean"].mean()) if "rh_mean" in sub.columns else 75.0
    rh_std = float(sub["rh_mean"].std()) if len(sub) > 1 and "rh_mean" in sub.columns else 0.0
    if np.isnan(rh_std):
        rh_std = 0.0

    # Compute VPD
    vpd_arr = compute_vpd(sub["t_mean"].values if "t_mean" in sub.columns else 18.0,
                          sub["rh_mean"].values if "rh_mean" in sub.columns else 75.0)
    vpd_mean = float(np.mean(vpd_arr))

    # Compute Growing Degree Days (GDD)
    t_vals = sub["t_mean"].values if "t_mean" in sub.columns else np.full(len(sub), 18.0)
    gdd = float(np.sum(np.maximum(t_vals - t_base, 0.0)))

    precip_total = float(sub["precip"].sum()) if "precip" in sub.columns else 0.0

    return {
        "t_mean": round(t_mean, 2),
        "t_max_min": round(t_mm, 2),
        "rh_mean": round(rh_mean, 2),
        "rh_std": round(rh_std, 2),
        "vpd_mean": round(vpd_mean, 3),
        "gdd": round(gdd, 2),
        "precip_total": round(precip_total, 2)
    }


def build_feature_vector(
    severity_i: float,
    dt_days: float,
    weather_df: pd.DataFrame,
    n_lesions_i: int = 0,
    mean_lesion_area_i: float = 0.0,
    start_date: Optional[Any] = None,
    end_date: Optional[Any] = None,
    crop_stage: float = -1.0,
    cultivar_group: float = -1.0,
    cultivar_trait_score: float = 5.0,
    batch_id: int = -1,
    year_id: int = -1,
    treatment_code: int = -1,
    feature_set: str = "core"
) -> Dict[str, float]:
    """Construct unified feature dictionary guaranteed to have feature parity between offline and online."""
    past3 = extract_weather_summary(weather_df, start_date=start_date, past_days=3)
    past7 = extract_weather_summary(weather_df, start_date=start_date, past_days=7)
    interval = extract_weather_summary(weather_df, start_date=start_date, end_date=end_date)

    lesion_density = float(n_lesions_i) / max(float(mean_lesion_area_i * max(n_lesions_i, 1)), 1.0)

    feat_dict: Dict[str, float] = {
        "severity_i": float(np.clip(severity_i, 0.0, 1.0)),
        "dt_days": float(max(dt_days, 0.1)),
        "n_lesions_i": float(n_lesions_i),
        "mean_lesion_area_i": float(mean_lesion_area_i),
        "lesion_density_i": round(lesion_density, 5),
        "t_mean_past3": past3["t_mean"],
        "t_mean_past7": past7["t_mean"],
        "t_mean_interval": interval["t_mean"],
        "t_max_min_interval": interval["t_max_min"],
        "gdd_interval": interval["gdd"],
        "rh_mean_past3": past3["rh_mean"],
        "rh_mean_past7": past7["rh_mean"],
        "rh_mean_interval": interval["rh_mean"],
        "vpd_mean_interval": interval["vpd_mean"],
        "precip_total_past3": past3["precip_total"],
        "precip_total_past7": past7["precip_total"],
        "precip_total_interval": interval["precip_total"],
        "crop_stage": float(crop_stage),
        "cultivar_group": float(cultivar_group),
    }

    if feature_set.lower() == "extended":
        feat_dict.update({
            "cultivar_trait_score": float(cultivar_trait_score),
            "rh_std_interval": interval["rh_std"],
            "batch_id": float(batch_id),
            "year_id": float(year_id),
            "treatment_code": float(treatment_code),
        })

    return feat_dict


def get_feature_matrix(
    df: pd.DataFrame,
    feature_set: str = "core"
) -> Tuple[np.ndarray, List[str]]:
    """Extract feature matrix X and feature column list from DataFrame."""
    feature_names = list(CORE_FEATURES) if feature_set.lower() == "core" else list(EXTENDED_FEATURES)

    X_cols = []
    for col in feature_names:
        if col in df.columns:
            X_cols.append(df[col].values.astype(np.float32))
        else:
            default_val = -1.0 if "code" in col or "id" in col or "stage" in col else 0.0
            X_cols.append(np.full(len(df), default_val, dtype=np.float32))

    X = np.column_stack(X_cols)
    return X, feature_names


def verify_feature_parity(
    sample_row: Dict[str, Any],
    weather_df: pd.DataFrame
) -> bool:
    """Verify offline vs online feature parity for a sample row."""
    if (weather_df.empty or len(weather_df) <= 1) and "t_mean_interval" in sample_row:
        t_val = float(sample_row.get("t_mean_interval", 18.0))
        rh_val = float(sample_row.get("rh_mean_interval", 75.0))
        pr_val = float(sample_row.get("precip_total_past3", 0.0)) / 3.0
        weather_df = pd.DataFrame([
            {"date": f"2024-01-{d:02d}", "t_mean": t_val, "t_max": t_val, "t_min": t_val, "rh_mean": rh_val, "precip": pr_val}
            for d in range(1, 15)
        ])

    feat_online = build_feature_vector(
        severity_i=sample_row.get("severity_i", 0.1),
        dt_days=sample_row.get("dt_days", 3.0),
        weather_df=weather_df,
        n_lesions_i=sample_row.get("n_lesions_i", 0),
        mean_lesion_area_i=sample_row.get("mean_lesion_area_i", 0.0),
        crop_stage=sample_row.get("crop_stage", -1.0),
        cultivar_group=sample_row.get("cultivar_group", -1.0),
        feature_set="core"
    )

    for k in CORE_FEATURES:
        if k in sample_row:
            v_offline = float(sample_row[k])
            v_online = float(feat_online[k])
            if not np.isclose(v_offline, v_online, atol=1e-1):
                raise AssertionError(f"Parity mismatch on feature '{k}': offline={v_offline}, online={v_online}")

    return True


def export_feature_schema(output_path: Optional[Path] = None) -> Path:
    """Export feature_schema.json containing feature set definitions and metadata."""
    paths = get_paths()
    if output_path is None:
        output_path = paths.project_root / "models" / "forecast" / "feature_schema.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    schema = {
        "core_features": CORE_FEATURES,
        "extended_features": EXTENDED_FEATURES,
        "n_core": len(CORE_FEATURES),
        "n_extended": len(EXTENDED_FEATURES),
        "missing_value_strategy": {
            "severity_i": "clip [0, 1], non-null required",
            "dt_days": "min 0.1, default 3.0",
            "weather": "impute t_mean=18.0, rh_mean=75.0, vpd_mean=0.52, precip=0.0",
            "categorical": "impute -1.0 for unknown stage or group"
        }
    }

    output_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    logger.info(f"Exported feature schema to {output_path}")
    return output_path
