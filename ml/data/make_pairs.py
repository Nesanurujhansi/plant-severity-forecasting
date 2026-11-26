"""Sequence image pair generator for epidemiological progression forecasting.

Pairs ordered image frames (i, j) within each leaf sequence for elapsed time intervals dt in [1, 10] days.
Computes interval weather statistics, attaches sample weights to balance sequence lengths,
and exports to data/processed/eth/pairs.parquet.
"""

import sys
import argparse
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger
from ml.utils.io import read_parquet, write_parquet

logger = get_logger(__name__)

def generate_pairs_from_images(
    images_df: pd.DataFrame,
    weather_d_df: pd.DataFrame,
    min_dt_days: float = 1.0,
    max_dt_days: float = 10.0
) -> pd.DataFrame:
    """Generate image pairs (i, j) for elapsed time dt in [min_dt_days, max_dt_days]."""
    if images_df.empty:
        logger.warning("Empty images_df passed to generate_pairs_from_images.")
        return pd.DataFrame()

    pair_records = []

    # Ensure timestamp is datetime
    df = images_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    # Group images by sample_id
    for sample_id, group in df.groupby("sample_id"):
        # Sort images chronologically
        sorted_imgs = group.sort_values("timestamp").to_dict("records")
        n_imgs = len(sorted_imgs)

        sample_pairs = []
        for i in range(n_imgs):
            rec_i = sorted_imgs[i]
            ts_i = rec_i["timestamp"]

            for j in range(i + 1, n_imgs):
                rec_j = sorted_imgs[j]
                ts_j = rec_j["timestamp"]

                dt_days = (ts_j - ts_i).total_seconds() / 86400.0

                # Filter by allowed elapsed time window
                if min_dt_days <= dt_days <= max_dt_days:
                    # Calculate interval weather statistics
                    date_i = ts_i.date()
                    date_j = ts_j.date()

                    if not weather_d_df.empty and "date" in weather_d_df.columns:
                        w_sub = weather_d_df[
                            (pd.to_datetime(weather_d_df["date"]).dt.date >= date_i) &
                            (pd.to_datetime(weather_d_df["date"]).dt.date <= date_j)
                        ]
                        t_mean_int = float(w_sub["t_mean"].mean()) if not w_sub.empty else 18.0
                        rh_mean_int = float(w_sub["rh_mean"].mean()) if not w_sub.empty else 75.0
                    else:
                        t_mean_int, rh_mean_int = 18.0, 75.0

                    sample_pairs.append({
                        "sample_id": sample_id,
                        "image_id_i": rec_i["image_id"],
                        "image_id_j": rec_j["image_id"],
                        "timestamp_i": ts_i,
                        "timestamp_j": ts_j,
                        "dt_days": round(dt_days, 3),
                        "severity_i": rec_i["severity"],
                        "severity_j": rec_j["severity"],
                        "n_lesions_i": rec_i.get("n_lesions", 0),
                        "mean_lesion_area_i": rec_i.get("mean_lesion_area", 0.0),
                        "t_mean_interval": round(t_mean_int, 2),
                        "rh_mean_interval": round(rh_mean_int, 2),
                        "cultivar": rec_i["cultivar"],
                        "treatment": rec_i["treatment"],
                        "batch": rec_i["batch"],
                        "year": rec_i["year"],
                    })

        # Calculate reciprocal sample weights to prevent long sequences from dominating
        n_pairs = len(sample_pairs)
        weight = 1.0 / max(n_pairs, 1)
        for pair in sample_pairs:
            pair["sample_weight"] = round(weight, 5)
            pair_records.append(pair)

    pairs_df = pd.DataFrame(pair_records)
    return pairs_df

def process_pairs(processed_dir: Optional[Path] = None) -> dict[str, Any]:
    """Load processed ETH Parquet files and generate pairs.parquet."""
    paths = get_paths()
    if processed_dir is None:
        processed_dir = paths.project_root / "data" / "processed" / "eth"

    images_path = processed_dir / "images_long.parquet"
    weather_path = processed_dir / "weather_daily.parquet"

    if not images_path.exists():
        logger.warning("images_long.parquet not found. Running ETH dataset parser first...")
        from ml.data.eth_wheat import process_eth_dataset
        process_eth_dataset(processed_dir=processed_dir)

    images_df = read_parquet(images_path)
    weather_d_df = read_parquet(weather_path) if weather_path.exists() else pd.DataFrame()

    pairs_df = generate_pairs_from_images(images_df, weather_d_df)

    # Save to pairs.parquet
    out_path = processed_dir / "pairs.parquet"
    write_parquet(pairs_df, out_path)

    # Report pair counts for key horizons
    if not pairs_df.empty:
        pairs_3d = pairs_df[(pairs_df["dt_days"] >= 2.0) & (pairs_df["dt_days"] <= 4.0)]
        pairs_7d = pairs_df[(pairs_df["dt_days"] >= 5.0) & (pairs_df["dt_days"] <= 9.0)]
        logger.info(f"Generated {len(pairs_df)} total pairs.")
        logger.info(f"  Pairs in 3-day window (2-4 days): {len(pairs_3d)}")
        logger.info(f"  Pairs in 7-day window (5-9 days): {len(pairs_7d)}")
    else:
        pairs_3d, pairs_7d = pd.DataFrame(), pd.DataFrame()

    return {
        "total_pairs": len(pairs_df),
        "pairs_3d_window": len(pairs_3d),
        "pairs_7d_window": len(pairs_7d),
    }

def main() -> int:
    process_pairs()
    return 0

if __name__ == "__main__":
    sys.exit(main())
