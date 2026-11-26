"""NASA POWER Agroclimatology Daily Point API Client.

Fetches daily meteorological observations (T2M, T2M_MAX, T2M_MIN, RH2M, PRECTOTCORR)
for agricultural risk modeling. Implements mandatory disk caching, polite retries,
and exponential backoff.

IMPORTANT NOTICE:
- NASA POWER provides HISTORICAL and NEAR-REAL-TIME observations only, NOT future forecasts.
- Spatial resolution is ~0.5° x 0.625° (~50 km grid cell size).
- Responses are strictly cached locally to avoid server blocking.
"""

import sys
import time
import hashlib
from pathlib import Path
from typing import Optional, Union
import requests
import pandas as pd

# Ensure project root is in sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger
from ml.utils.io import read_json, write_json

logger = get_logger(__name__)

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

class NASAPowerError(Exception):
    """Exception raised when NASA POWER API service fails or returns an error."""
    pass

def _get_cache_key(lat: float, lon: float, start_date: str, end_date: str, params: list[str]) -> str:
    """Generate a deterministic filename key based on rounded location and date range."""
    # Round coordinates to 2 decimal places matching grid resolution
    rounded_lat = round(lat, 2)
    rounded_lon = round(lon, 2)
    param_str = ",".join(sorted(params))
    raw_key = f"{rounded_lat}_{rounded_lon}_{start_date}_{end_date}_{param_str}"
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
    return f"nasa_power_{rounded_lat}_{rounded_lon}_{start_date}_{end_date}_{digest}.json"

def fetch_nasa_power_daily(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    parameters: Optional[list[str]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    max_retries: int = 3,
    backoff_factor: float = 2.0
) -> pd.DataFrame:
    """Fetch daily weather observations from NASA POWER API with disk caching and retries.
    
    Args:
        latitude: Target point latitude (-90 to 90).
        longitude: Target point longitude (-180 to 180).
        start_date: Start date string formatted as 'YYYYMMDD' (e.g. '20230101').
        end_date: End date string formatted as 'YYYYMMDD' (e.g. '20231231').
        parameters: List of NASA POWER parameters. Defaults to T2M, T2M_MAX, T2M_MIN, RH2M, PRECTOTCORR.
        cache_dir: Custom directory for cache JSON files. Defaults to data/interim/weather_cache.
        max_retries: Maximum number of HTTP request retries.
        backoff_factor: Multiplier for exponential backoff sleep interval.
        
    Returns:
        Tidy pandas DataFrame with columns: date, t_mean, t_max, t_min, rh_mean, precip_mm.
    """
    if parameters is None:
        parameters = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR"]
        
    if cache_dir is None:
        cache_dir = get_paths().interim_data / "weather_cache"
    else:
        cache_dir = Path(cache_dir)
        
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / _get_cache_key(latitude, longitude, start_date, end_date, parameters)

    # Check local cache first
    if cache_file.exists():
        logger.info(f"Loading NASA POWER data from local cache: {cache_file.name}")
        payload = read_json(cache_file)
        return _parse_nasa_power_payload(payload)

    # Make HTTP request with exponential backoff
    query_params = {
        "parameters": ",".join(parameters),
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }

    payload = None
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Fetching NASA POWER API (attempt {attempt}/{max_retries}) for ({latitude}, {longitude})...")
            response = requests.get(BASE_URL, params=query_params, timeout=15)
            
            if response.status_code == 200:
                payload = response.json()
                break
            else:
                last_err = f"HTTP status {response.status_code}: {response.text[:200]}"
                logger.warning(f"NASA POWER API returned error: {last_err}")
                
        except requests.RequestException as e:
            last_err = str(e)
            logger.warning(f"Request failed on attempt {attempt}: {e}")

        if attempt < max_retries:
            sleep_time = backoff_factor ** attempt
            time.sleep(sleep_time)

    if payload is None:
        raise NASAPowerError(
            f"Failed to fetch NASA POWER weather data after {max_retries} attempts. Last error: {last_err}. "
            "Please check network connection or NASA POWER API service status."
        )

    # Save response to local disk cache
    write_json(payload, cache_file)
    logger.info(f"Saved response to cache: {cache_file.name}")

    return _parse_nasa_power_payload(payload)

def _parse_nasa_power_payload(payload: dict) -> pd.DataFrame:
    """Parse raw NASA POWER JSON payload into a tidy pandas DataFrame."""
    try:
        params_data = payload["properties"]["parameter"]
        
        # Get list of date strings (keys in parameter dictionaries)
        t_mean_dict = params_data.get("T2M", {})
        dates = sorted(t_mean_dict.keys())
        
        records = []
        for d in dates:
            records.append({
                "date": pd.to_datetime(d, format="%Y%m%d"),
                "t_mean": params_data.get("T2M", {}).get(d),
                "t_max": params_data.get("T2M_MAX", {}).get(d),
                "t_min": params_data.get("T2M_MIN", {}).get(d),
                "rh_mean": params_data.get("RH2M", {}).get(d),
                "precip_mm": params_data.get("PRECTOTCORR", {}).get(d),
            })
            
        df = pd.DataFrame(records)
        # Handle NASA missing value indicator (-999 or -999.0)
        df = df.replace(-999.0, None).replace(-999, None)
        return df
        
    except KeyError as e:
        raise NASAPowerError(f"Unexpected JSON response structure from NASA POWER API: missing key {e}")
