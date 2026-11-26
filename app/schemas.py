"""
Phase 10: Pydantic v2 Schemas for FastAPI Endpoints & Pipeline Output Payloads
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="Server liveness status")
    version: str = Field(..., description="Application version")
    is_stub: bool = Field(..., description="True if dev stub models are active")
    bundle_version: str = Field(..., description="Model bundle version")


class ModelInfoResponse(BaseModel):
    bundle_version: str = Field(..., description="Semantic version of active bundle")
    created_at: str = Field(..., description="ISO creation timestamp")
    git_commit: str = Field(..., description="Git commit SHA")
    is_stub: bool = Field(..., description="True if dev stub models are active")
    stub_banner: Optional[str] = Field(None, description="Warning banner if stub mode active")
    metrics_summary: Dict[str, Any] = Field(default_factory=dict, description="Headline evaluation metrics")
    dataset_identifiers: List[Dict[str, str]] = Field(default_factory=list, description="Dataset sources")
    disclaimer: str = Field(..., description="Non-prescription scientific disclaimer")
    known_limitations: List[str] = Field(default_factory=list, description="Model limits")


class BundleInfoBlock(BaseModel):
    version: str = Field(..., description="Bundle semantic version")
    created_at: Optional[str] = Field(None, description="ISO creation timestamp")
    git_commit: Optional[str] = Field(None, description="Git commit SHA")
    is_stub: bool = Field(..., description="True if stub mode active")
    stub_banner: Optional[str] = Field(None, description="Stub mode banner")


class SafetyBlock(BaseModel):
    confidence_tier: str = Field(..., description="Confidence rating: HIGH, MEDIUM, LOW")
    should_abstain: bool = Field(..., description="True if app should abstain from prediction")
    quality_passed: bool = Field(..., description="True if photo passed sharpness/exposure checks")
    ood_status: str = Field(..., description="OOD status: IN_DISTRIBUTION or OUT_OF_DISTRIBUTION")
    user_message: str = Field(..., description="Actionable user message / recommendation")
    module_gating: Dict[str, bool] = Field(..., description="Allowed downstream modules")
    failure_reasons: List[str] = Field(default_factory=list, description="Specific quality or OOD failures")


class SeverityBlock(BaseModel):
    severity_pct: float = Field(..., description="Predicted leaf severity percentage (0.0 to 100.0%)")
    severity_interval: List[float] = Field(..., description="90% Mondrian conformal prediction interval [low, high]")
    severity_class: str = Field(..., description="Ordinal severity bin label")
    lesion_count: int = Field(..., description="Count of distinct connected lesion spots")
    mean_lesion_size_px: float = Field(..., description="Average lesion spot area in pixels")
    senescence_pct: Optional[float] = Field(None, description="Physiological senescence / chlorosis leaf percentage (0.0 to 100.0%)")
    necrotic_pct: Optional[float] = Field(None, description="Highly affected necrotic/rust core leaf percentage (0.0 to 100.0%)")
    chlorotic_pct: Optional[float] = Field(None, description="Yellow chlorosis / halo margin leaf percentage (0.0 to 100.0%)")
    leaf_mask_base64: Optional[str] = Field(None, description="Base64 encoded PNG leaf mask")
    lesion_mask_base64: Optional[str] = Field(None, description="Base64 encoded PNG lesion mask")
    overlay_base64: Optional[str] = Field(None, description="Base64 encoded PNG transparent composite overlay")
    composite_image_base64: Optional[str] = Field(None, description="Base64 encoded PNG pre-composited image (leaf + overlay baked together)")
    projected_7d_overlay_base64: Optional[str] = Field(None, description="Base64 encoded PNG 7-day projected lesion expansion overlay")
    treated_7d_overlay_base64: Optional[str] = Field(None, description="Base64 encoded PNG 7-day treated spread reduction overlay")
    treated_7d_severity_pct: Optional[float] = Field(None, description="Predicted 7-day severity percentage under favorable spray intervention")
    spread_reduction_pct: Optional[float] = Field(None, description="Relative disease spread reduction percentage under spray intervention")


class HorizonForecast(BaseModel):
    horizon_days: int = Field(..., description="Forecast horizon in days (e.g. 3 or 7)")
    severity_pct: float = Field(..., description="Forecasted severity percentage")
    interval: List[float] = Field(..., description="90% CQR conformal interval [low, high]")
    drivers: List[str] = Field(default_factory=list, description="Microclimate drivers affecting progression")


class ForecastBlock(BaseModel):
    horizons: Dict[str, HorizonForecast] = Field(..., description="Forecast results keyed by horizon (e.g. '3d', '7d')")
    chart_base64: Optional[str] = Field(None, description="Rendered Matplotlib forecast progression chart PNG")
    projected_7d_overlay_base64: Optional[str] = Field(None, description="Base64 encoded PNG 7-day projected lesion expansion overlay")
    treated_7d_overlay_base64: Optional[str] = Field(None, description="Base64 encoded PNG 7-day treated spread reduction overlay")
    treated_7d_severity_pct: Optional[float] = Field(None, description="Predicted 7-day severity percentage under favorable spray intervention")
    spread_reduction_pct: Optional[float] = Field(None, description="Relative disease spread reduction percentage under spray intervention")


class ScenarioBlock(BaseModel):
    unprotected_7d_pct: float = Field(..., description="Predicted 7-day severity if left untreated")
    protected_7d_pct: float = Field(..., description="Predicted 7-day severity under fungicide protection")
    relative_reduction_pct: float = Field(..., description="Percentage severity reduction due to regime")
    interval: List[float] = Field(..., description="Cluster bootstrap confidence interval for difference")
    label: str = Field(..., description="Descriptive scenario comparison text")
    disclaimer: str = Field(..., description="Scenario limitation disclaimer")
    chart_base64: Optional[str] = Field(None, description="Rendered Matplotlib treatment scenario chart PNG")


class WeatherBlock(BaseModel):
    source: str = Field(..., description="Weather data source description")
    preset: str = Field(..., description="Future weather scenario preset applied")
    table_summary: List[Dict[str, Any]] = Field(default_factory=list, description="Summary table of daily microclimate values")
    latitude: Optional[float] = Field(None, description="Queried or detected field latitude")
    longitude: Optional[float] = Field(None, description="Queried or detected field longitude")
    current_temp_c: Optional[float] = Field(None, description="Current ambient mean temperature in °C")
    current_rh_pct: Optional[float] = Field(None, description="Current ambient relative humidity %")
    current_vpd_kpa: Optional[float] = Field(None, description="Current vapor pressure deficit in kPa")
    is_default_location: Optional[bool] = Field(False, description="Whether reference agricultural coordinates were used")


class TimingsBlock(BaseModel):
    total_ms: float = Field(..., description="Total pipeline execution time in milliseconds")
    stages_ms: Dict[str, float] = Field(default_factory=dict, description="Stage-by-stage execution durations")


class FactorAttribution(BaseModel):
    factor_id: str = Field(..., description="Unique factor identifier")
    factor_name: str = Field(..., description="Human-readable factor title")
    raw_value: str = Field(..., description="Observed input value string")
    delta_pct: float = Field(..., description="Percentage point impact on 7-day severity")
    impact_direction: str = Field(..., description="'ACCELERATING' or 'RETARDING'")
    color: str = Field(..., description="Visual badge color hex code")
    description: str = Field(..., description="Agronomic causal explanation")


class XAIBlock(BaseModel):
    attention_saliency_base64: Optional[str] = Field(None, description="Base64 PNG DINOv2 ViT attention saliency heatmap")
    baseline_severity_pct: float = Field(..., description="Starting Day 0 severity percentage")
    forecasted_severity_pct: float = Field(..., description="Target 7-day forecasted median severity percentage")
    net_change_pct: float = Field(..., description="Net change over the forecast horizon")
    factors: List[FactorAttribution] = Field(default_factory=list, description="Ranked agronomic factor contributions")
    narrative: str = Field(..., description="Plain-language agronomic summary narrative")


class AnalyzeResponse(BaseModel):
    bundle_info: BundleInfoBlock = Field(..., description="Active bundle details")
    safety: SafetyBlock = Field(..., description="Safety, quality & OOD evaluation")
    severity: Optional[SeverityBlock] = Field(None, description="Severity assessment (if permitted by safety)")
    forecast: Optional[ForecastBlock] = Field(None, description="Progression forecast (if permitted by safety)")
    scenario: Optional[ScenarioBlock] = Field(None, description="Treatment scenario comparison (if permitted by safety)")
    weather: WeatherBlock = Field(..., description="Microclimate weather summary")
    xai: Optional[XAIBlock] = Field(None, description="Explainable AI diagnostic saliency and factor breakdown")
    warnings: List[str] = Field(default_factory=list, description="Structured user warnings")
    timings: TimingsBlock = Field(..., description="Performance timings")
    disclaimer: str = Field(..., description="Mandatory scientific non-prescription disclaimer")


class WeatherQueryResponse(BaseModel):
    latitude: float = Field(..., description="Query latitude")
    longitude: float = Field(..., description="Query longitude")
    source: str = Field(..., description="Data source description")
    days_count: int = Field(..., description="Count of daily weather records returned")
    records: List[Dict[str, Any]] = Field(default_factory=list, description="Daily meteorological records")


class ErrorResponse(BaseModel):
    error_code: str = Field(..., description="Machine-readable error identifier")
    message: str = Field(..., description="Human-readable error explanation")
    hint: str = Field(..., description="Actionable hint for resolving error")
