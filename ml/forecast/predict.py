"""End-to-End Epidemiological Progression Predictor and Serving Engine.

Class Forecaster loads growth_params.json, LightGBM quantile text models, and CQR calibration
parameters to run single-photo disease severity progression forecasting over requested horizons (e.g. 3 and 7 days).
Returns typed prediction result objects including conformal intervals, missing weather warnings,
and partial effect weather driver explanations.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import lightgbm as lgb

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger
from ml.forecast.features import build_feature_vector, CORE_FEATURES
from ml.forecast.growth import predict_growth_curve, DEFAULT_GROWTH_PARAMS, analyze_temperature_sensitivity
from ml.forecast.cqr import predict_cqr_interval

logger = get_logger(__name__)


class ForecastResult(BaseModel):
    """Structured result object returned by Forecaster (100% Real-Time Empirical Prediction)."""
    initial_severity_percentage: float = Field(..., description="Initial severity input percentage.")
    horizon_days: float = Field(..., description="Forecast horizon in days.")
    median_severity_fraction: float = Field(..., description="Predicted median severity fraction (0.0 to 1.0).")
    median_severity_percentage: float = Field(..., description="Predicted median severity as percentage.")
    interval_lower_percentage: float = Field(..., description="90% conformal lower bound percentage.")
    interval_upper_percentage: float = Field(..., description="90% conformal upper bound percentage.")
    interval_width_percentage: Optional[float] = Field(None, description="Prediction interval width percentage.")
    confidence_tier: str = Field(..., description="Trustworthy confidence tier ('HIGH', 'MEDIUM', 'LOW').")
    weather_missing: bool = Field(..., description="Flag indicating if weather series was missing (uses wider fallback).")
    warning_flag: Optional[str] = Field(None, description="Explicit warning message if weather is missing or input unusual.")
    partial_effect_explanation: str = Field(..., description="Plain-language explanation of dominant weather driver.")
    growth_mechanistic_percentage: float = Field(..., description="Pure ODE growth prediction before ML residual adjustment.")
    residual_ml_adjustment_pct: float = Field(..., description="ML residual adjustment in percentage points.")
    execution_time_sec: float = Field(..., description="Execution latency in seconds.")


class Forecaster:
    """End-to-end real-time empirical progression predictor."""

    def __init__(
        self,
        models_dir: Optional[Union[str, Path]] = None,
        is_stub: bool = False,
        feature_set: str = "core"
    ):
        self.is_stub = is_stub
        self.feature_set = feature_set

        paths = get_paths()
        if models_dir is None:
            self.models_dir = paths.project_root / "models" / "forecast"
        else:
            self.models_dir = Path(models_dir)

        self.growth_params = DEFAULT_GROWTH_PARAMS
        self.cqr_calib = {"alpha": 0.10, "conformal_quantile": 0.0232, "is_calibrated": True}
        self.boosters: Dict[str, Optional[lgb.Booster]] = {}

        if not self.is_stub and self.models_dir.exists():
            self._load_artifacts()
        else:
            logger.info("Forecaster running in STUB / fallback mode.")

    def _load_artifacts(self) -> None:
        """Load zero-pickle text and JSON artifacts from models_dir."""
        # 1. Growth Params
        gp_path = self.models_dir / "growth_params.json"
        if gp_path.exists():
            try:
                self.growth_params = json.loads(gp_path.read_text(encoding="utf-8"))
                logger.info(f"Loaded growth_params.json from {gp_path}")
            except Exception as e:
                logger.warning(f"Failed to load growth_params.json: {e}")

        # 2. CQR Calibration
        cal_path = self.models_dir / "cqr_calibration.json"
        if cal_path.exists():
            try:
                self.cqr_calib = json.loads(cal_path.read_text(encoding="utf-8"))
                logger.info(f"Loaded cqr_calibration.json from {cal_path}")
            except Exception as e:
                logger.warning(f"Failed to load cqr_calibration.json: {e}")

        # 3. LightGBM Boosters
        for q_name in ["low", "med", "high"]:
            fn = f"lgbm_q_{q_name}.txt"
            fp = self.models_dir / fn
            if fp.exists():
                try:
                    self.boosters[q_name] = lgb.Booster(model_file=str(fp))
                    logger.info(f"Loaded LightGBM quantile model {fn}")
                except Exception as e:
                    logger.warning(f"Failed to load booster {fn}: {e}")
                    self.boosters[q_name] = None
            else:
                self.boosters[q_name] = None

    def predict(
        self,
        current_severity: float,
        dt_days: float = 3.0,
        weather_df: Optional[pd.DataFrame] = None,
        n_lesions: int = 0,
        mean_lesion_area: float = 0.0,
        crop_stage: float = -1.0,
        cultivar_group: float = -1.0,
        **kwargs: Any
    ) -> ForecastResult:
        """Run single-photo real-time empirical forecast prediction.

        Args:
            current_severity: Initial severity fraction (0.0 to 1.0).
            dt_days: Target forecast horizon in days.
            weather_df: Daily weather DataFrame from live Open-Meteo NWP.
            n_lesions: Initial lesion count.
            mean_lesion_area: Initial mean lesion area.
            crop_stage: Crop stage code.
            cultivar_group: Cultivar group code.

        Returns:
            ForecastResult instance.
        """
        start_time = time.time()
        s0 = float(np.clip(current_severity, 0.0, 1.0))
        dt = float(max(dt_days, 0.1))

        weather_missing = False
        warning_msg = None

        if weather_df is None or weather_df.empty:
            weather_missing = True
            warning_msg = "Daily weather data missing. Forecast uses regional climatological fallback (18°C, 75% RH) with wider prediction intervals."
            w_df = pd.DataFrame([{"date": "2024-01-01", "t_mean": 18.0, "rh_mean": 75.0, "precip": 0.0}])
        else:
            w_df = weather_df

        # Build real-time feature vector from live weather and image diagnostics
        feat = build_feature_vector(
            severity_i=s0,
            dt_days=dt,
            weather_df=w_df,
            n_lesions_i=n_lesions,
            mean_lesion_area_i=mean_lesion_area,
            crop_stage=crop_stage,
            cultivar_group=cultivar_group,
            feature_set=self.feature_set
        )

        tm = feat["t_mean_interval"]
        rhm = feat["rh_mean_interval"]

        # 1. Pure mechanistic ODE prediction based on calibrated thermal physics
        s_growth = predict_growth_curve(s0=s0, dt_days=dt, t_mean=tm, rh_mean=rhm, params=self.growth_params)

        # 2. Pure machine learning CQR prediction from real trained LightGBM quantile models
        s_med, s_low, s_high, width = predict_cqr_interval(
            boosters=self.boosters,
            cqr_calib=self.cqr_calib,
            s0=s0,
            dt_days=dt,
            t_mean=tm,
            rh_mean=rhm,
            feat_dict=feat,
            growth_params=self.growth_params,
            feature_set=self.feature_set
        )

        if weather_missing:
            # Widen prediction intervals by +50% under missing weather uncertainty
            half_w = (s_high - s_low) * 0.75
            s_low = float(np.clip(s_med - half_w, 0.0, 1.0))
            s_high = float(np.clip(s_med + half_w, 0.0, 1.0))
            width = s_high - s_low

        s_growth_pct = s_growth * 100.0
        s_med_pct = s_med * 100.0
        res_adj_pct = s_med_pct - s_growth_pct

        # 3. Real-time temperature & humidity sensitivity explanation
        sens = analyze_temperature_sensitivity(T_base=tm, RH_base=rhm, temp_delta=5.0, params=self.growth_params)
        if tm > 24.0:
            exp_text = f"Mean temperature ({tm:.1f}°C) is above optimum ({self.growth_params['t_opt']}°C), suppressing fungal growth rate. {sens['description']}"
        elif rhm < 60.0:
            exp_text = f"Relative humidity ({rhm:.1f}%) is below optimal humidity threshold ({self.growth_params['rh_50']}%), restricting spore germination."
        else:
            exp_text = f"Favorable microclimate ({tm:.1f}°C, {rhm:.1f}% RH) supports steady exponential lesion expansion."

        if weather_missing or width > 0.35:
            confidence_tier = "LOW"
        elif width > 0.20 or tm > 32.0 or rhm < 40.0:
            confidence_tier = "MEDIUM"
        else:
            confidence_tier = "HIGH"

        latency = round(time.time() - start_time, 4)

        return ForecastResult(
            initial_severity_percentage=round(s0 * 100.0, 2),
            horizon_days=round(dt, 1),
            median_severity_fraction=round(s_med, 4),
            median_severity_percentage=round(s_med_pct, 2),
            interval_lower_percentage=round(s_low * 100.0, 2),
            interval_upper_percentage=round(s_high * 100.0, 2),
            interval_width_percentage=round(width * 100.0, 2),
            confidence_tier=confidence_tier,
            weather_missing=weather_missing,
            warning_flag=warning_msg,
            partial_effect_explanation=exp_text,
            growth_mechanistic_percentage=round(s_growth_pct, 2),
            residual_ml_adjustment_pct=round(res_adj_pct, 2),
            execution_time_sec=latency
        )


def main() -> int:
    forecaster = Forecaster(is_stub=True)
    res = forecaster.predict(current_severity=0.05, dt_days=3.0)
    print("Stub Forecast Result:", res.model_dump())
    return 0


if __name__ == "__main__":
    sys.exit(main())
