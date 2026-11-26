"""Static In-The-Wild Image Severity Calculator.

Computes disease severity fraction S = lesion_pixels / leaf_pixels on static images,
enforcing the union rule (lesion_pixels <= leaf_pixels) and clipping to [0, 1].
Assigns ordinal severity bin classes and attaches severity metrics to manifest.parquet.
"""

import sys
import argparse
from pathlib import Path
from typing import Any, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger
from ml.utils.io import read_parquet, write_parquet

logger = get_logger(__name__)

SEVERITY_BINS = [
    (0, 0.0, 0.01, "<1%"),
    (1, 0.01, 0.05, "1-5%"),
    (2, 0.05, 0.10, "5-10%"),
    (3, 0.10, 0.25, "10-25%"),
    (4, 0.25, 0.50, "25-50%"),
    (5, 0.50, 1.00, ">50%"),
]

def calculate_severity_from_masks(
    lesion_mask_array: np.ndarray,
    leaf_mask_array: np.ndarray
) -> Tuple[float, int, str]:
    """Calculate severity fraction, ordinal bin index, and bin label.
    
    Union Rule Enforced: leaf_mask = leaf_mask | lesion_mask
    """
    lesion_arr = (lesion_mask_array > 0).astype(bool)
    leaf_arr = (leaf_mask_array > 0).astype(bool) | lesion_arr # Enforce union rule

    lesion_cnt = int(np.sum(lesion_arr))
    leaf_cnt = int(np.sum(leaf_arr))

    if leaf_cnt == 0:
        return 0.0, 0, "<1%"

    severity = float(np.clip(lesion_cnt / float(leaf_cnt), 0.0, 1.0))

    # Assign ordinal severity bin
    bin_idx = 0
    bin_label = "<1%"
    for idx, low, high, label in SEVERITY_BINS:
        if low <= severity <= high:
            bin_idx = idx
            bin_label = label
            break

    return round(severity, 4), bin_idx, bin_label

def process_static_severity(
    manifest_df: pd.DataFrame,
    processed_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Compute severity fraction and ordinal class for all static manifest rows."""
    paths = get_paths()
    if processed_dir is None:
        processed_dir = paths.project_root / "data" / "processed" / "static"

    if manifest_df.empty:
        logger.warning("Empty manifest_df passed to process_static_severity.")
        return manifest_df

    updated_records = []

    for row in manifest_df.to_dict("records"):
        rel_lesion = row["rel_lesion_mask_path"]
        rel_leaf = row.get("rel_leaf_mask_path")

        lesion_p = paths.project_root / rel_lesion if rel_lesion else None
        leaf_p = paths.project_root / rel_leaf if rel_leaf else None

        if lesion_p and lesion_p.exists():
            lesion_arr = np.array(Image.open(lesion_p))
            if leaf_p and leaf_p.exists():
                leaf_arr = np.array(Image.open(leaf_p))
            else:
                leaf_arr = np.ones_like(lesion_arr) * 255 # Fallback whole frame

            sev, bin_idx, bin_label = calculate_severity_from_masks(lesion_arr, leaf_arr)
        else:
            sev, bin_idx, bin_label = 0.0, 0, "<1%"

        row["severity"] = sev
        row["severity_bin_idx"] = bin_idx
        row["severity_bin_label"] = bin_label
        updated_records.append(row)

    updated_df = pd.DataFrame(updated_records)
    out_path = processed_dir / "manifest.parquet"
    write_parquet(updated_df, out_path)
    logger.info(f"Calculated static severity metrics for {len(updated_df)} images at {out_path}")

    return updated_df

def main() -> int:
    paths = get_paths()
    manifest_p = paths.project_root / "data" / "processed" / "static" / "manifest.parquet"
    if not manifest_p.exists():
        from ml.data.build_static_manifest import process_static_manifest
        df = process_static_manifest()
    else:
        df = read_parquet(manifest_p)

    process_static_severity(df)
    return 0

if __name__ == "__main__":
    sys.exit(main())
