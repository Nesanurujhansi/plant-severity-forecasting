"""Exploratory Data Analysis (EDA) tool for ETH Zurich Wheat Leaf dataset.

Generates trajectory plots, distribution histograms, and summary tables in reports/figures/ and reports/tables/.
"""

import sys
import argparse
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger
from ml.utils.io import read_parquet

logger = get_logger(__name__)

def run_eth_eda(
    processed_dir: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
    tables_dir: Optional[Path] = None
) -> dict[str, Any]:
    """Generate ETH dataset EDA plots and statistical summary tables."""
    paths = get_paths()
    if processed_dir is None:
        processed_dir = paths.project_root / "data" / "processed" / "eth"
    if figures_dir is None:
        figures_dir = paths.figures_dir
    if tables_dir is None:
        tables_dir = paths.tables_dir

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    images_path = processed_dir / "images_long.parquet"
    sequences_path = processed_dir / "sequences.parquet"
    weather_path = processed_dir / "weather_daily.parquet"

    if not images_path.exists():
        logger.warning("Processed ETH tables not found. Running parser first...")
        from ml.data.eth_wheat import process_eth_dataset
        process_eth_dataset(processed_dir=processed_dir)

    images_df = read_parquet(images_path)
    sequences_df = read_parquet(sequences_path) if sequences_path.exists() else pd.DataFrame()
    weather_df = read_parquet(weather_path) if weather_path.exists() else pd.DataFrame()

    results: dict[str, Any] = {"figures": [], "tables": []}

    # 1. Severity Trajectories by Treatment
    if "treatment" in images_df.columns and "hours_since_first" in images_df.columns:
        plt.figure(figsize=(8, 5))
        for trt, grp in images_df.groupby("treatment"):
            # Plot individual sample trajectories
            for sample_id, s_grp in grp.groupby("sample_id"):
                plt.plot(s_grp["hours_since_first"] / 24.0, s_grp["severity"], alpha=0.3, linewidth=1)
        plt.xlabel("Elapsed Time (Days)")
        plt.ylabel("Disease Severity (Fraction)")
        plt.title("ETH Wheat Disease Severity Trajectories by Treatment")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        fig_path = figures_dir / "eth_severity_trajectories_by_treatment.png"
        plt.savefig(fig_path, dpi=200)
        plt.close()
        results["figures"].append(str(fig_path))

    # 2. Sequence Length Distribution
    if not sequences_df.empty and "n_images" in sequences_df.columns:
        plt.figure(figsize=(6, 4))
        plt.hist(sequences_df["n_images"], bins=range(1, sequences_df["n_images"].max() + 2), color="tab:blue", edgecolor="black", alpha=0.7)
        plt.xlabel("Number of Images per Sequence")
        plt.ylabel("Count of Leaf Sequences")
        plt.title("ETH Wheat Sequence Length Distribution")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        fig_path = figures_dir / "eth_sequence_length_distribution.png"
        plt.savefig(fig_path, dpi=200)
        plt.close()
        results["figures"].append(str(fig_path))

    # 3. Sequences per Cultivar Histogram
    if not sequences_df.empty and "cultivar" in sequences_df.columns:
        plt.figure(figsize=(8, 4))
        counts = sequences_df["cultivar"].value_counts()
        counts.plot(kind="bar", color="tab:green", edgecolor="black", alpha=0.7)
        plt.xlabel("Cultivar")
        plt.ylabel("Number of Leaf Sequences")
        plt.title("ETH Wheat Sequences per Cultivar")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        fig_path = figures_dir / "eth_sequences_per_cultivar.png"
        plt.savefig(fig_path, dpi=200)
        plt.close()
        results["figures"].append(str(fig_path))

    # 4. Summary Table Exports
    if not sequences_df.empty:
        trt_summary = sequences_df.groupby("treatment").agg(
            n_sequences=("sample_id", "count"),
            avg_n_images=("n_images", "mean"),
            avg_span_days=("span_days", "mean")
        ).reset_index()
        trt_summary_path = tables_dir / "eth_treatment_summary.csv"
        trt_summary.to_csv(trt_summary_path, index=False)
        results["tables"].append(str(trt_summary_path))

        cult_summary = sequences_df.groupby("cultivar").agg(
            n_sequences=("sample_id", "count"),
            avg_n_images=("n_images", "mean")
        ).reset_index()
        cult_summary_path = tables_dir / "eth_cultivar_summary.csv"
        cult_summary.to_csv(cult_summary_path, index=False)
        results["tables"].append(str(cult_summary_path))

    logger.info(f"Generated {len(results['figures'])} figures and {len(results['tables'])} tables.")
    return results

def main() -> int:
    run_eth_eda()
    return 0

if __name__ == "__main__":
    sys.exit(main())
