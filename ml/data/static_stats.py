"""Static Dataset Exploratory Data Analysis & Statistics Tool.

Generates summary CSV tables in reports/tables/ and severity distribution histograms
in reports/figures/ for all static in-the-wild datasets.
"""

import sys
import argparse
from pathlib import Path
from typing import Any, Optional, Dict, List
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

def run_static_stats(
    manifest_df: Optional[pd.DataFrame] = None,
    reports_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Compute dataset counts, severity distributions, and export reports."""
    paths = get_paths()
    if reports_dir is None:
        reports_dir = paths.reports_dir

    figures_dir = paths.figures_dir
    tables_dir = paths.tables_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    if manifest_df is None or manifest_df.empty:
        manifest_p = paths.project_root / "data" / "processed" / "static" / "manifest.parquet"
        if manifest_p.exists():
            manifest_df = read_parquet(manifest_p)
        else:
            logger.warning("Manifest file not found. Running manifest builder first...")
            from ml.data.build_static_manifest import process_static_manifest
            manifest_df = process_static_manifest()

    results: Dict[str, Any] = {"figures": [], "tables": []}

    # 1. Source & Crop Summary Table
    if "source" in manifest_df.columns:
        summary_table = manifest_df.groupby("source").agg(
            total_images=("image_id", "count"),
            reviewed_leaf_masks=("leaf_mask_origin", lambda s: (s == "sam2_reviewed").sum()),
            auto_leaf_masks=("leaf_mask_origin", lambda s: (s == "sam2_auto").sum()),
            valid_images=("is_quality_valid", lambda s: s.sum() if "is_quality_valid" in manifest_df.columns else len(s))
        ).reset_index()

        out_summary_p = tables_dir / "static_dataset_summary.csv"
        summary_table.to_csv(out_summary_p, index=False)
        results["tables"].append(str(out_summary_p))

    # 2. Severity Distribution Histogram Plot
    if "severity" in manifest_df.columns:
        plt.figure(figsize=(7, 4))
        plt.hist(manifest_df["severity"].dropna(), bins=20, color="tab:orange", edgecolor="black", alpha=0.75)
        plt.xlabel("Disease Severity Fraction (Lesion / Leaf)")
        plt.ylabel("Image Count")
        plt.title("Static In-The-Wild Disease Severity Distribution")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        out_fig_p = figures_dir / "static_severity_distribution.png"
        plt.savefig(out_fig_p, dpi=200)
        plt.close()
        results["figures"].append(str(out_fig_p))

        # Severity Bin Table
        if "severity_bin_label" in manifest_df.columns:
            bin_table = manifest_df["severity_bin_label"].value_counts().reset_index()
            bin_table.columns = ["severity_bin", "image_count"]
            out_bin_p = tables_dir / "static_severity_distribution.csv"
            bin_table.to_csv(out_bin_p, index=False)
            results["tables"].append(str(out_bin_p))

    logger.info(f"Generated {len(results['figures'])} static figures and {len(results['tables'])} summary tables.")
    return results

def main() -> int:
    run_static_stats()
    return 0

if __name__ == "__main__":
    sys.exit(main())
