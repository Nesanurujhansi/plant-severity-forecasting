"""Epidemiological Progression Benchmark Baselines.

Implements benchmark baseline models to compare against the semi-mechanistic CQR model:
1. PersistenceBaseline (s_{t+dt} = s_t)
2. CultivarMeanGrowthBaseline (empirical mean rate per cultivar)
3. GrowthCurveOnlyBaseline (pure mechanistic logistic growth ODE without ML)
4. LightGBMOnlyBaseline (direct LightGBM regression without growth curve)
5. NaiveLateFusionBaseline (direct concatenation of growth curve output into LightGBM)
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
from ml.forecast.growth import predict_growth_curve, DEFAULT_GROWTH_PARAMS
from ml.forecast.features import get_feature_matrix

logger = get_logger(__name__)


class PersistenceBaseline:
    """Naive persistence baseline: future severity equals initial severity (s_{t+dt} = s_t)."""

    def predict(self, s0: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.clip(s0, 0.0, 1.0)


class CultivarMeanGrowthBaseline:
    """Cultivar-mean baseline: predicts growth based on average historical growth rate per cultivar."""

    def __init__(self) -> None:
        self.cultivar_rates: Dict[str, float] = {}
        self.global_mean_rate: float = 0.01  # Default rate per day

    def fit(self, pairs_df: pd.DataFrame) -> "CultivarMeanGrowthBaseline":
        if pairs_df.empty:
            return self

        rates = []
        df = pairs_df.copy()
        df["rate"] = (df["severity_j"] - df["severity_i"]) / np.maximum(df["dt_days"], 0.1)

        self.global_mean_rate = float(df["rate"].mean())

        if "cultivar" in df.columns:
            for cv, grp in df.groupby("cultivar"):
                self.cultivar_rates[str(cv)] = float(grp["rate"].mean())

        return self

    def predict(
        self,
        s0: Union[float, np.ndarray],
        dt_days: Union[float, np.ndarray],
        cultivar: Optional[str] = None
    ) -> Union[float, np.ndarray]:
        rate = self.cultivar_rates.get(str(cultivar), self.global_mean_rate) if cultivar else self.global_mean_rate
        pred = s0 + rate * dt_days
        return np.clip(pred, 0.0, 1.0)


class GrowthCurveOnlyBaseline:
    """Pure mechanistic logistic growth baseline without machine learning residual correction."""

    def __init__(self, params: Optional[Dict[str, float]] = None) -> None:
        self.params = DEFAULT_GROWTH_PARAMS if params is None else {**DEFAULT_GROWTH_PARAMS, **params}

    def predict(
        self,
        s0: float,
        dt_days: float,
        t_mean: float = 18.0,
        rh_mean: float = 75.0
    ) -> float:
        return predict_growth_curve(s0=s0, dt_days=dt_days, t_mean=t_mean, rh_mean=rh_mean, params=self.params)


class LightGBMOnlyBaseline:
    """Direct machine learning baseline: predicts future severity directly using LightGBM without ODE growth curve."""

    def __init__(self) -> None:
        self.booster: Optional[lgb.Booster] = None

    def fit(self, train_df: pd.DataFrame, feature_set: str = "core") -> "LightGBMOnlyBaseline":
        if train_df.empty:
            return self

        X_train, feature_names = get_feature_matrix(train_df, feature_set=feature_set)
        y_train = train_df["severity_j"].values.astype(np.float32)

        lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        params = {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 4,
            "verbosity": -1,
            "random_state": 42
        }

        self.booster = lgb.train(params=params, train_set=lgb_train, num_boost_round=200)
        return self

    def predict(self, feat_dict: Dict[str, float], feature_set: str = "core") -> float:
        if self.booster is None:
            return float(feat_dict.get("severity_i", 0.0))

        X = np.array([[feat_dict.get(k, 0.0) for k in self.booster.feature_name()]], dtype=np.float32)
        pred = float(self.booster.predict(X)[0])
        return float(np.clip(pred, 0.0, 1.0))


class NaiveLateFusionBaseline:
    """Naive late fusion baseline: appends raw growth curve prediction as a feature directly to LightGBM input."""

    def __init__(self) -> None:
        self.booster: Optional[lgb.Booster] = None

    def fit(self, train_df: pd.DataFrame, feature_set: str = "core") -> "NaiveLateFusionBaseline":
        if train_df.empty:
            return self

        df = train_df.copy()
        s_growth_list = []
        for _, row in df.iterrows():
            sg = predict_growth_curve(
                s0=row["severity_i"],
                dt_days=row["dt_days"],
                t_mean=row.get("t_mean_interval", 18.0),
                rh_mean=row.get("rh_mean_interval", 75.0)
            )
            s_growth_list.append(sg)

        df["raw_growth_pred"] = s_growth_list

        X_train, fnames = get_feature_matrix(df, feature_set=feature_set)
        X_train = np.column_stack([X_train, df["raw_growth_pred"].values.astype(np.float32)])
        fnames = list(fnames) + ["raw_growth_pred"]

        y_train = df["severity_j"].values.astype(np.float32)

        lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=fnames)
        params = {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 4,
            "verbosity": -1,
            "random_state": 42
        }

        self.booster = lgb.train(params=params, train_set=lgb_train, num_boost_round=200)
        return self

    def predict(
        self,
        s0: float,
        dt_days: float,
        t_mean: float,
        rh_mean: float,
        feat_dict: Dict[str, float],
        feature_set: str = "core"
    ) -> float:
        if self.booster is None:
            return float(s0)

        sg = predict_growth_curve(s0=s0, dt_days=dt_days, t_mean=t_mean, rh_mean=rh_mean)
        base_fnames = self.booster.feature_name()[:-1]

        x_row = [feat_dict.get(k, 0.0) for k in base_fnames] + [sg]
        X = np.array([x_row], dtype=np.float32)

        pred = float(self.booster.predict(X)[0])
        return float(np.clip(pred, 0.0, 1.0))
