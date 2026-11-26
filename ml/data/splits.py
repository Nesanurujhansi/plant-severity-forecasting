"""Leak-free evaluation splits generator for ETH wheat dataset.

Generates leave-cultivar-out, leave-batch-out, and leave-year-out cross-validation folds.
Ensures leaf-level sequences (sample_id) strictly remain within a single split (train, calibration, test)
with zero sample leakage. Exports to data/processed/eth/splits.json.
"""

import sys
import argparse
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger
from ml.utils.io import read_parquet, write_json

logger = get_logger(__name__)

def create_leak_free_splits(
    sequences_df: pd.DataFrame,
    n_folds: int = 5,
    cal_fraction: float = 0.20,
    seed: int = 42
) -> dict[str, Any]:
    """Create cross-validation folds and calibration splits grouped strictly by sample_id."""
    if sequences_df.empty:
        logger.warning("Empty sequences_df passed to create_leak_free_splits.")
        return {"leave_cultivar_out": [], "leave_batch_out": [], "leave_year_out": []}

    np.random.seed(seed)
    df = sequences_df.copy()
    unique_samples = df["sample_id"].unique().tolist()

    splits_dict: dict[str, Any] = {
        "metadata": {
            "total_samples": len(unique_samples),
            "n_folds": n_folds,
            "cal_fraction": cal_fraction,
            "seed": seed
        },
        "leave_cultivar_out": [],
        "leave_batch_out": [],
        "leave_year_out": []
    }

    # 1. Leave-Cultivar-Out Folds
    if "cultivar" in df.columns:
        cultivars = df["cultivar"].unique()
        if len(cultivars) >= 2:
            gkf = GroupKFold(n_splits=min(n_folds, len(cultivars)))
            for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df["cultivar"])):
                train_samples = df.iloc[train_idx]["sample_id"].unique().tolist()
                test_samples = df.iloc[test_idx]["sample_id"].unique().tolist()
                
                n_cal = int(len(train_samples) * cal_fraction)
                np.random.shuffle(train_samples)
                cal_samples = train_samples[:n_cal]
                fit_train_samples = train_samples[n_cal:]

                splits_dict["leave_cultivar_out"].append({
                    "fold": fold_idx,
                    "held_out_group": str(df.iloc[test_idx]["cultivar"].iloc[0]),
                    "train_samples": fit_train_samples,
                    "calibration_samples": cal_samples,
                    "test_samples": test_samples
                })
        else:
            logger.warning("Fewer than 2 unique cultivars; skipping leave-cultivar-out splits.")

    # 2. Leave-Batch-Out Folds
    if "batch" in df.columns:
        batches = df["batch"].unique()
        if len(batches) >= 2:
            gkf = GroupKFold(n_splits=min(n_folds, len(batches)))
            for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df["batch"])):
                train_samples = df.iloc[train_idx]["sample_id"].unique().tolist()
                test_samples = df.iloc[test_idx]["sample_id"].unique().tolist()
                
                n_cal = int(len(train_samples) * cal_fraction)
                np.random.shuffle(train_samples)
                cal_samples = train_samples[:n_cal]
                fit_train_samples = train_samples[n_cal:]

                splits_dict["leave_batch_out"].append({
                    "fold": fold_idx,
                    "held_out_group": int(df.iloc[test_idx]["batch"].iloc[0]),
                    "train_samples": fit_train_samples,
                    "calibration_samples": cal_samples,
                    "test_samples": test_samples
                })
        else:
            logger.warning("Fewer than 2 unique batches; skipping leave-batch-out splits.")

    # 3. Leave-Year-Out Folds
    if "year" in df.columns:
        years = df["year"].unique()
        if len(years) >= 2:
            gkf = GroupKFold(n_splits=min(n_folds, len(years)))
            for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df["year"])):
                train_samples = df.iloc[train_idx]["sample_id"].unique().tolist()
                test_samples = df.iloc[test_idx]["sample_id"].unique().tolist()
                
                n_cal = int(len(train_samples) * cal_fraction)
                np.random.shuffle(train_samples)
                cal_samples = train_samples[:n_cal]
                fit_train_samples = train_samples[n_cal:]

                splits_dict["leave_year_out"].append({
                    "fold": fold_idx,
                    "held_out_group": int(df.iloc[test_idx]["year"].iloc[0]),
                    "train_samples": fit_train_samples,
                    "calibration_samples": cal_samples,
                    "test_samples": test_samples
                })
        else:
            logger.warning("Fewer than 2 unique years; skipping leave-year-out splits.")

    return splits_dict

def verify_zero_leakage(splits_dict: dict[str, Any]) -> bool:
    """Verify that no sample_id appears in multiple split partitions within the same fold."""
    for split_type in ["leave_cultivar_out", "leave_batch_out", "leave_year_out"]:
        folds = splits_dict.get(split_type, [])
        for fold in folds:
            tr = set(fold["train_samples"])
            cal = set(fold["calibration_samples"])
            te = set(fold["test_samples"])
            
            # Check pairwise intersections
            if tr.intersection(cal):
                logger.error(f"Leakage detected in {split_type} fold {fold['fold']}: train & cal overlap!")
                return False
            if tr.intersection(te):
                logger.error(f"Leakage detected in {split_type} fold {fold['fold']}: train & test overlap!")
                return False
            if cal.intersection(te):
                logger.error(f"Leakage detected in {split_type} fold {fold['fold']}: cal & test overlap!")
                return False
    return True

def process_splits(processed_dir: Optional[Path] = None) -> dict[str, Any]:
    """Load sequences.parquet and generate splits.json."""
    paths = get_paths()
    if processed_dir is None:
        processed_dir = paths.project_root / "data" / "processed" / "eth"

    seq_path = processed_dir / "sequences.parquet"
    if not seq_path.exists():
        logger.warning("sequences.parquet not found. Running ETH dataset parser first...")
        from ml.data.eth_wheat import process_eth_dataset
        process_eth_dataset(processed_dir=processed_dir)

    sequences_df = read_parquet(seq_path)
    splits_dict = create_leak_free_splits(sequences_df)

    if not verify_zero_leakage(splits_dict):
        raise ValueError("Data leakage assertion failed during split generation!")

    out_path = processed_dir / "splits.json"
    write_json(splits_dict, out_path)

    logger.info(f"Exported leak-free splits to {out_path}")
    return splits_dict

def main() -> int:
    process_splits()
    return 0

if __name__ == "__main__":
    sys.exit(main())
