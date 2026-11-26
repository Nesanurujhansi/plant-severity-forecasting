"""Weather Parity Study script comparing on-site ETH weather with NASA POWER.

Aggregates on-site hourly in-canopy temperature and relative humidity from ETH Research
Station (47.449 N, 8.682 E) to daily metrics matching NASA POWER definitions, calculates
bias, RMSE, and correlation, and generates parity scatter plots in reports/figures/.
"""

import sys
import argparse
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger
from ml.data.weather_nasa_power import fetch_nasa_power_daily

logger = get_logger(__name__)

ETH_LAT = 47.449
ETH_LON = 8.682

def aggregate_eth_hourly_to_daily(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly on-site ETH weather station data to daily metrics.
    
    Handles missing hours (>18 valid hours required per day) and timezone alignment (UTC).
    """
    df = hourly_df.copy()
    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], utc=True)
        df["date"] = df["datetime"].dt.date
    elif "date" not in df.columns:
        raise ValueError("ETH hourly weather DataFrame must contain 'timestamp' or 'date' column.")

    daily_records = []
    grouped = df.groupby("date")

    for date_val, group in grouped:
        # Enforce threshold: at least 18 valid hours per day
        if len(group) < 18:
            logger.warning(f"Skipping date {date_val}: insufficient hourly records ({len(group)}/24).")
            continue

        daily_records.append({
            "date": pd.to_datetime(date_val),
            "eth_t_mean": group["temperature"].mean() if "temperature" in group else np.nan,
            "eth_t_max": group["temperature"].max() if "temperature" in group else np.nan,
            "eth_t_min": group["temperature"].min() if "temperature" in group else np.nan,
            "eth_rh_mean": group["rh"].mean() if "rh" in group else np.nan,
        })

    return pd.DataFrame(daily_records)

def calculate_parity_metrics(obs: np.ndarray, sat: np.ndarray) -> dict[str, float]:
    """Calculate bias, Root Mean Squared Error (RMSE), and Pearson correlation."""
    valid_mask = ~np.isnan(obs) & ~np.isnan(sat)
    if not np.any(valid_mask):
        return {"bias": np.nan, "rmse": np.nan, "correlation": np.nan}

    y_obs = obs[valid_mask]
    y_sat = sat[valid_mask]

    bias = float(np.mean(y_sat - y_obs))
    rmse = float(np.sqrt(np.mean((y_sat - y_obs) ** 2)))
    corr = float(np.corrcoef(y_obs, y_sat)[0, 1]) if len(y_obs) > 1 else np.nan

    return {"bias": round(bias, 3), "rmse": round(rmse, 3), "correlation": round(corr, 3)}

def run_weather_parity_analysis(
    hourly_eth_file: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> dict[str, Any]:
    """Run full weather parity analysis comparing ETH on-site weather with NASA POWER."""
    paths = get_paths()
    if output_dir is None:
        output_dir = paths.figures_dir

    if hourly_eth_file is None or not hourly_eth_file.exists():
        logger.warning(f"ETH hourly weather file not found at {hourly_eth_file}. Running synthetic dry-run parity analysis...")
        # Create synthetic evaluation dataset for testing pipeline
        dates = pd.date_range("2023-05-01", "2023-07-31", freq="D")
        np.random.seed(42)
        eth_daily = pd.DataFrame({
            "date": dates,
            "eth_t_mean": 18.0 + np.sin(np.linspace(0, 10, len(dates))) * 5 + np.random.normal(0, 1, len(dates)),
            "eth_rh_mean": 75.0 + np.cos(np.linspace(0, 10, len(dates))) * 15 + np.random.normal(0, 2, len(dates)),
        })
    else:
        raw_eth = pd.read_csv(hourly_eth_file)
        eth_daily = aggregate_eth_hourly_to_daily(raw_eth)

    start_str = eth_daily["date"].min().strftime("%Y%m%d")
    end_str = eth_daily["date"].max().strftime("%Y%m%d")

    logger.info(f"Fetching NASA POWER data for period {start_str} to {end_str}...")
    try:
        nasa_df = fetch_nasa_power_daily(ETH_LAT, ETH_LON, start_str, end_str)
    except Exception as e:
        logger.error(f"Could not fetch NASA POWER data: {e}")
        return {"status": "FAILED", "error": str(e)}

    # Merge datasets on date
    merged = pd.merge(eth_daily, nasa_df, on="date", how="inner")

    results = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot scatter for Temperature Mean
    if "eth_t_mean" in merged.columns and "t_mean" in merged.columns:
        metrics = calculate_parity_metrics(merged["eth_t_mean"].values, merged["t_mean"].values)
        results["temperature_mean"] = metrics
        
        plt.figure(figsize=(6, 6))
        plt.scatter(merged["eth_t_mean"], merged["t_mean"], alpha=0.7, color="tab:orange", label="Days")
        min_v = min(merged["eth_t_mean"].min(), merged["t_mean"].min())
        max_v = max(merged["eth_t_mean"].max(), merged["t_mean"].max())
        plt.plot([min_v, max_v], [min_v, max_v], "k--", label="1:1 Parity")
        plt.xlabel("ETH On-Site Temperature (°C)")
        plt.ylabel("NASA POWER Temperature (°C)")
        plt.title(f"Temperature Parity (r={metrics['correlation']}, Bias={metrics['bias']}°C)")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plot_path = output_dir / "nasa_eth_temperature_parity.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        logger.info(f"Saved temperature parity scatter plot to {plot_path}")

    results["recommendation"] = (
        "Linear bias correction is RECOMMENDED if absolute bias > 1.5°C or correlation < 0.85."
    )
    return results

def main() -> int:
    parser = argparse.ArgumentParser(description="Weather Parity Study ETH vs NASA POWER")
    parser.add_argument("--eth-file", type=str, help="Path to ETH hourly weather CSV.")
    args = parser.parse_args()

    eth_path = Path(args.eth_file) if args.eth_file else None
    res = run_weather_parity_analysis(hourly_eth_file=eth_path)
    print(res)
    return 0

if __name__ == "__main__":
    sys.exit(main())
