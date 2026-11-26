"""Unit tests for NASA POWER weather API client using mocked HTTP responses."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd

from ml.data.weather_nasa_power import fetch_nasa_power_daily, NASAPowerError

MOCK_NASA_RESPONSE = {
    "properties": {
        "parameter": {
            "T2M": {"20230501": 18.5, "20230502": 19.2},
            "T2M_MAX": {"20230501": 22.0, "20230502": 23.5},
            "T2M_MIN": {"20230501": 14.1, "20230502": 15.0},
            "RH2M": {"20230501": 78.0, "20230502": 82.5},
            "PRECTOTCORR": {"20230501": 0.0, "20230502": 3.4},
        }
    }
}

@patch("requests.get")
def test_fetch_nasa_power_daily_mocked(mock_get):
    # Mock successful HTTP 200 response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_NASA_RESPONSE
    mock_get.return_value = mock_response

    with tempfile.TemporaryDirectory() as tmpdir:
        df = fetch_nasa_power_daily(
            latitude=47.449,
            longitude=8.682,
            start_date="20230501",
            end_date="20230502",
            cache_dir=tmpdir
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["date", "t_mean", "t_max", "t_min", "rh_mean", "precip_mm"]
        assert df["t_mean"].iloc[0] == 18.5
        assert df["rh_mean"].iloc[1] == 82.5

        # Verify disk cache creation
        cache_files = list(Path(tmpdir).glob("*.json"))
        assert len(cache_files) == 1

        # Second call should read from disk cache without making another HTTP request
        mock_get.reset_mock()
        df_cached = fetch_nasa_power_daily(
            latitude=47.449,
            longitude=8.682,
            start_date="20230501",
            end_date="20230502",
            cache_dir=tmpdir
        )
        mock_get.assert_not_called()
        assert len(df_cached) == 2

@patch("requests.get")
def test_fetch_nasa_power_error_handling(mock_get):
    # Mock HTTP 500 error
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_get.return_value = mock_response

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(NASAPowerError):
            fetch_nasa_power_daily(
                latitude=47.449,
                longitude=8.682,
                start_date="20230501",
                end_date="20230502",
                cache_dir=tmpdir,
                max_retries=1
            )
