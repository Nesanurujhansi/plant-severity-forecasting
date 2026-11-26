"""Configuration loader and validator using Pydantic v2."""

from pathlib import Path
from typing import Any, Optional, Union
from pydantic import BaseModel, Field, field_validator
from ml.utils.io import read_yaml

class AppConfig(BaseModel):
    """Main application and ML pipeline configuration schema."""
    
    seed: int = Field(default=42, description="Global random seed for reproducibility.")
    severity_bins: list[float] = Field(
        default=[0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00],
        description="Ordinal severity class bin boundaries (fractions between 0 and 1)."
    )
    conformal_alpha: float = Field(
        default=0.10,
        gt=0.0,
        lt=1.0,
        description="Conformal error rate alpha (e.g., 0.10 yields 90% coverage intervals)."
    )
    forecast_horizons: list[int] = Field(
        default=[3, 7],
        description="Forecast time steps in days."
    )
    backbone_id: str = Field(
        default="",
        description="Hugging Face vision backbone repository ID."
    )
    image_sizes: list[int] = Field(
        default=[448, 672, 896],
        description="Multi-scale crop/resolution sizes for vision feature extraction."
    )
    data_root_override: Optional[str] = Field(
        default=None,
        description="Optional custom root directory for datasets."
    )

    @field_validator("severity_bins")
    @classmethod
    def validate_severity_bins(cls, v: list[float]) -> list[float]:
        if len(v) < 2:
            raise ValueError("severity_bins must contain at least 2 boundaries.")
        if v[0] != 0.0 or v[-1] != 1.0:
            raise ValueError("severity_bins must start with 0.0 and end with 1.0.")
        if any(v[i] >= v[i+1] for i in range(len(v)-1)):
            raise ValueError("severity_bins must be strictly increasing.")
        return v

def _apply_override(config_dict: dict[str, Any], key: str, value: Any) -> None:
    """Apply key=value overrides supporting dot-notation nested keys."""
    keys = key.split(".")
    curr = config_dict
    for k in keys[:-1]:
        curr = curr.setdefault(k, {})
    
    # Parse basic type conversions if value is string
    if isinstance(value, str):
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass
                
    curr[keys[-1]] = value

def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Union[list[str], dict[str, Any]]] = None
) -> AppConfig:
    """Load, merge, and validate YAML configuration file with optional overrides.
    
    Args:
        config_path: Path to YAML config file. Defaults to ml/configs/base.yaml if None.
        overrides: List of "key=value" strings or dictionary of overrides.
        
    Returns:
        Validated AppConfig instance.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
        
    path = Path(config_path)
    config_data = read_yaml(path) if path.exists() else {}
    
    if overrides:
        if isinstance(overrides, dict):
            for k, v in overrides.items():
                _apply_override(config_data, k, v)
        elif isinstance(overrides, list):
            for item in overrides:
                if "=" in item:
                    k, v = item.split("=", 1)
                    _apply_override(config_data, k.strip(), v.strip())
                    
    return AppConfig(**config_data)
