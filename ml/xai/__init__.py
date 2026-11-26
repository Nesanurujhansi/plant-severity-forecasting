"""Explainable AI (XAI) Module for Plant Disease Severity.
Implements:
1. DINOv2 Vision Transformer Attention Rollout for spatial foliar saliency.
2. Exact TreeSHAP feature attributions for LightGBM progression forecasting.
"""

from ml.xai.attention_rollout import compute_attention_rollout, generate_attention_saliency_map
from ml.xai.forecast_shap import compute_forecast_shap, ForecastShapExplanation

__all__ = [
    "compute_attention_rollout",
    "generate_attention_saliency_map",
    "compute_forecast_shap",
    "ForecastShapExplanation",
]
