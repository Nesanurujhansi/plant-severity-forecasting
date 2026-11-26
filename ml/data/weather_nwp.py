"""Numerical Weather Prediction (NWP) High-Resolution Client for Forward Forecasting.

Fetches forward 7-day hourly numerical weather predictions from Open-Meteo API
(ECMWF IFS / GFS 0.1° ~9 km spatial resolution) with zero publication lag,
local disk caching, and offline fallback.
"""

import sys
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import requests
import pandas as pd
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger
from ml.utils.io import read_json, write_json

logger = get_logger(__name__)

OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
NWP_CACHE_TTL_SECONDS = 7200  # 2 hours Time-To-Live for fresh weather prediction


class HourlyWeatherForecast(BaseModel):
    """Pydantic model for parsed hourly numerical forecast records."""
    latitude: float
    longitude: float
    elevation_m: float = 0.0
    timezone: str = "UTC"
    hourly_records: List[Dict[str, Any]] = Field(default_factory=list)


def _get_nwp_cache_key(lat: float, lon: float, days: int) -> str:
    rounded_lat = round(lat, 2)
    rounded_lon = round(lon, 2)
    raw_key = f"nwp_{rounded_lat}_{rounded_lon}_{days}"
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:12]
    return f"nwp_forecast_{rounded_lat}_{rounded_lon}_{digest}.json"


def fetch_open_meteo_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 7,
    cache_dir: Optional[Union[str, Path]] = None,
    timeout: float = 8.0,
    force_refresh: bool = False
) -> pd.DataFrame:
    """Fetch high-resolution hourly forward numerical weather prediction.

    Args:
        latitude: Target latitude (-90 to 90).
        longitude: Target longitude (-180 to 180).
        forecast_days: Horizon in days (1 to 14, default 7).
        cache_dir: Directory to cache forecast responses (default data/interim/weather_cache).
        timeout: HTTP timeout in seconds.
        force_refresh: If True, bypasses local disk cache.

    Returns:
        DataFrame with columns: ['datetime', 'temp_2m', 'rh_2m', 'dew_point_2m', 'precip_mm', 'pressure_hpa', 'cloud_cover_pct']
    """
    if cache_dir is None:
        try:
            cache_dir = get_paths().interim_data / "weather_cache"
        except Exception:
            cache_dir = Path("data/interim/weather_cache")
    else:
        cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / _get_nwp_cache_key(latitude, longitude, forecast_days)

    # 1. Check TTL Cache
    if not force_refresh and cache_file.exists():
        try:
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age < NWP_CACHE_TTL_SECONDS:
                logger.info(f"Serving fresh NWP forecast from cache (age {file_age/60:.1f}m): {cache_file.name}")
                payload = read_json(cache_file)
                return _parse_open_meteo_payload(payload)
            else:
                logger.info(f"NWP cache expired (age {file_age/60:.1f}m > {NWP_CACHE_TTL_SECONDS/60:.0f}m). Fetching fresh NWP data.")
        except Exception as e:
            logger.warning(f"Error checking cache timestamp: {e}")

    params = {
        "latitude": round(latitude, 2),
        "longitude": round(longitude, 2),
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,surface_pressure,cloud_cover,wind_speed_10m,wind_gusts_10m",
        "forecast_days": forecast_days,
        "timezone": "UTC"
    }

    try:
        resp = requests.get(OPEN_METEO_BASE_URL, params=params, timeout=timeout)
        if resp.status_code == 200:
            payload = resp.json()
            write_json(payload, cache_file)
            logger.info(f"Saved fresh NWP forecast to cache: {cache_file.name}")
            return _parse_open_meteo_payload(payload)
        else:
            logger.warning(f"Open-Meteo returned status {resp.status_code}. Attempting stale cache fallback.")
            if cache_file.exists():
                return _parse_open_meteo_payload(read_json(cache_file))
            return _generate_fallback_nwp(latitude, longitude, forecast_days)
    except Exception as e:
        logger.warning(f"Failed to query Open-Meteo ({e}). Attempting stale cache fallback.")
        if cache_file.exists():
            return _parse_open_meteo_payload(read_json(cache_file))
        return _generate_fallback_nwp(latitude, longitude, forecast_days)


def _parse_open_meteo_payload(payload: Dict[str, Any]) -> pd.DataFrame:
    """Parse raw Open-Meteo JSON into a tidy pandas DataFrame."""
    hourly = payload.get("hourly", {})
    timestamps = hourly.get("time", [])

    df = pd.DataFrame({
        "datetime": pd.to_datetime(timestamps, utc=True),
        "temp_2m": hourly.get("temperature_2m", []),
        "rh_2m": hourly.get("relative_humidity_2m", []),
        "dew_point_2m": hourly.get("dew_point_2m", []),
        "precip_mm": hourly.get("precipitation", []),
        "pressure_hpa": hourly.get("surface_pressure", [1013.25] * len(timestamps)),
        "cloud_cover_pct": hourly.get("cloud_cover", [50.0] * len(timestamps)),
        "wind_speed_10m": hourly.get("wind_speed_10m", [8.0] * len(timestamps)),
        "wind_gusts_10m": hourly.get("wind_gusts_10m", [12.0] * len(timestamps))
    })
    return df


def _generate_fallback_nwp(latitude: float, longitude: float, days: int) -> pd.DataFrame:
    """Deterministic fallback NWP forecast for offline environments."""
    now = pd.Timestamp.now(tz="UTC").floor("h")
    datetimes = [now + pd.Timedelta(hours=i) for i in range(days * 24)]
    
    # Realistic diurnal cycle: T peaks at 14:00, RH troughs at 14:00
    temps = [18.0 + 4.5 * (1.0 - (abs((t.hour - 14) % 24 - 12) / 12.0)) for t in datetimes]
    rhs = [75.0 - 20.0 * (1.0 - (abs((t.hour - 14) % 24 - 12) / 12.0)) for t in datetimes]
    dew_pts = [t - ((100.0 - rh) / 5.0) for t, rh in zip(temps, rhs)]
    precips = [0.0] * len(datetimes)
    winds = [8.0 + 3.0 * math.sin(i * 0.26) for i in range(len(datetimes))]
    gusts = [w + 5.0 for w in winds]

    return pd.DataFrame({
        "datetime": datetimes,
        "temp_2m": [round(t, 2) for t in temps],
        "rh_2m": [round(rh, 1) for rh in rhs],
        "dew_point_2m": [round(d, 2) for d in dew_pts],
        "precip_mm": precips,
        "pressure_hpa": [1013.25] * len(datetimes),
        "cloud_cover_pct": [40.0] * len(datetimes),
        "wind_speed_10m": [round(w, 1) for w in winds],
        "wind_gusts_10m": [round(g, 1) for g in gusts]
    })
