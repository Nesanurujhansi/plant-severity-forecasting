"""Leak-free evaluation splits generator for static in-the-wild datasets.

Groups samples strictly by near-duplicate cluster_id and dataset source to prevent leakage.
Reserves independent evaluation sets (e.g. coffee rust 100 eval masks) as external_test.
Creates train, calibration, val, and test partitions, and exports to static_splits.json.
"""

import sys
import argparse
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger
from ml.utils.io import read_parquet, write_parquet, write_json

logger = get_logger(__name__)

def create_static_splits(
    manifest_df: pd.DataFrame,
    seed: int = 42,
    cal_fraction: float = 0.15,
    val_fraction: float = 0.10,
    test_fraction: float = 0.15
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Assign leak-free split labels (train, calibration, val, test, external_test) grouped by cluster_id."""
    if manifest_df.empty:
        logger.warning("Empty manifest_df passed to create_static_splits.")
        return manifest_df, {"splits": {}}

    np.random.seed(seed)
    df = manifest_df.copy()

    # Default split assignment
    df["split"] = "train"

    # 1. Identify external test samples (e.g. Coffee Rust 100 eval set)
    is_ext_eval = df["is_external_eval"].fillna(False) if "is_external_eval" in df.columns else pd.Series(False, index=df.index)
    df.loc[is_ext_eval, "split"] = "external_test"

    # Process remaining non-external samples grouped by cluster_id
    main_mask = ~is_ext_eval
    main_df = df[main_mask].copy()

    unique_clusters = main_df["cluster_id"].unique()
    np.random.shuffle(unique_clusters)

    n_clusters = len(unique_clusters)
    n_test = int(n_clusters * test_fraction)
    n_val = int(n_clusters * val_fraction)
    n_cal = int(n_clusters * cal_fraction)

    test_clusters = set(unique_clusters[:n_test])
    val_clusters = set(unique_clusters[n_test : n_test + n_val])
    cal_clusters = set(unique_clusters[n_test + n_val : n_test + n_val + n_cal])
    train_clusters = set(unique_clusters[n_test + n_val + n_cal:])

    # Assign split labels
    df.loc[main_mask & df["cluster_id"].isin(test_clusters), "split"] = "test"
    df.loc[main_mask & df["cluster_id"].isin(val_clusters), "split"] = "val"
    df.loc[main_mask & df["cluster_id"].isin(cal_clusters), "split"] = "calibration"
    df.loc[main_mask & df["cluster_id"].isin(train_clusters), "split"] = "train"

    # Build summary metadata dictionary
    split_summary = {
        "metadata": {
            "total_images": len(df),
            "seed": seed,
            "test_fraction": test_fraction,
            "cal_fraction": cal_fraction,
            "val_fraction": val_fraction
        },
        "split_counts": df["split"].value_counts().to_dict(),
        "source_breakdown": df.groupby(["source", "split"]).size().unstack(fill_value=0).to_dict()
    }

    return df, split_summary

def verify_static_zero_leakage(manifest_df: pd.DataFrame) -> bool:
    """Verify that no near-duplicate cluster_id appears in multiple splits."""
    if manifest_df.empty:
        return True

    # Check pairwise cluster overlap among non-external splits
    splits = ["train", "calibration", "val", "test"]
    cluster_sets = {}
    for s in splits:
        cluster_sets[s] = set(manifest_df[manifest_df["split"] == s]["cluster_id"].unique())

    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            overlap = cluster_sets[s1].intersection(cluster_sets[s2])
            if overlap:
                logger.error(f"Cluster leakage detected between split '{s1}' and '{s2}': overlap={overlap}")
                return False

    return True

def process_static_splits(processed_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load manifest.parquet, generate leak-free splits, and save static_splits.json."""
    paths = get_paths()
    if processed_dir is None:
        processed_dir = paths.project_root / "data" / "processed" / "static"

    manifest_p = processed_dir / "manifest.parquet"
    if not manifest_p.exists():
        from ml.data.build_static_manifest import process_static_manifest
        manifest_df = process_static_manifest()
    else:
        manifest_df = read_parquet(manifest_p)

    updated_df, split_summary = create_static_splits(manifest_df)

    if not verify_static_zero_leakage(updated_df):
        raise ValueError("Data leakage assertion failed during static split generation!")

    out_manifest = processed_dir / "manifest.parquet"
    write_parquet(updated_df, out_manifest)

    out_splits_json = processed_dir / "static_splits.json"
    write_json(split_summary, out_splits_json)

    logger.info(f"Exported static splits summary to {out_splits_json}")
    return updated_df, split_summary

def main() -> int:
    process_static_splits()
    return 0

if __name__ == "__main__":
    sys.exit(main())
