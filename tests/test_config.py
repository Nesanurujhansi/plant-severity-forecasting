"""Unit tests for configuration loading, validation, and overrides."""

import pytest
from ml.utils.config import load_config, AppConfig

def test_load_default_config():
    config = load_config()
    assert isinstance(config, AppConfig)
    assert config.seed == 42
    assert config.conformal_alpha == 0.10
    assert config.forecast_horizons == [3, 7]
    assert config.severity_bins == [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00]

def test_config_overrides():
    config = load_config(overrides=["seed=1234", "conformal_alpha=0.05"])
    assert config.seed == 1234
    assert config.conformal_alpha == 0.05

def test_invalid_severity_bins():
    with pytest.raises(ValueError):
        # Must start with 0.0 and end with 1.0
        AppConfig(severity_bins=[0.1, 0.5, 0.9])
        
    with pytest.raises(ValueError):
        # Must be strictly increasing
        AppConfig(severity_bins=[0.0, 0.5, 0.3, 1.0])
