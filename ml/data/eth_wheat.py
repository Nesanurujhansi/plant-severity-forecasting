"""ETH Zurich Wheat Dataset Parser and Feature Processor.

Parses experimental design metadata, hourly weather logs, and symptom segmentation masks.
Outputs standardized Parquet tables: images_long.parquet, sequences.parquet,
weather_hourly.parquet, and weather_daily.parquet.
"""

import sys
import re
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
from ml.utils.io import write_parquet

logger = get_logger(__name__)

def parse_image_id(filename_stem: str) -> dict[str, Any]:
    """Parse image ID string into metadata attributes.
    
    Expected format: YYYYMMDD_HHMMSS_<plot_id>_<leaf_no>
    Example: 20230515_103000_ESWW001_L1
    """
    pattern = r"^(\d{8}_\d{6})_([A-Z0-9]+)_(L\d+)$"
    match = re.match(pattern, filename_stem)
    if match:
        ts_str, plot_id, leaf_no = match.groups()
        timestamp = pd.to_datetime(ts_str, format="%Y%m%d_%H%M%S")
        sample_id = f"{plot_id}_{leaf_no}"
        return {
            "image_id": filename_stem,
            "timestamp": timestamp,
            "plot_id": plot_id,
            "leaf_no": leaf_no,
            "sample_id": sample_id
        }
    else:
        # Generic fallback parsing
        parts = filename_stem.split("_")
        sample_id = f"{parts[2] if len(parts)>2 else 'P0'}_{parts[3] if len(parts)>3 else 'L1'}"
        return {
            "image_id": filename_stem,
            "timestamp": pd.Timestamp.now(),
            "plot_id": parts[2] if len(parts)>2 else "P0",
            "leaf_no": parts[3] if len(parts)>3 else "L1",
            "sample_id": sample_id
        }

def compute_mask_metrics(mask_array: np.ndarray) -> dict[str, Any]:
    """Compute per-image mask pixel counts, disease ratios, and lesion stats.
    
    Mask Pixel Encoding:
      0: Background
      1: Healthy Green
      2: Chlorosis
      3: Necrosis / Septoria
      4: Rust Pustules
      5: Ink Markers
    """
    unique_vals, counts = np.unique(mask_array, return_counts=True)
    val_map = dict(zip(unique_vals, counts))

    healthy_cnt = val_map.get(1, 0)
    chlorosis_cnt = val_map.get(2, 0)
    necrosis_cnt = val_map.get(3, 0)
    pustule_cnt = val_map.get(4, 0)

    diseased_pixels = necrosis_cnt + pustule_cnt
    leaf_pixels = healthy_cnt + chlorosis_cnt + diseased_pixels

    # Compute lesion connected components (classes 3 and 4)
    diseased_mask = (mask_array == 3) | (mask_array == 4)
    n_lesions, mean_lesion_area = _count_connected_lesions(diseased_mask)

    return {
        "diseased_pixels": int(diseased_pixels),
        "leaf_pixels_frame": int(leaf_pixels),
        "necrosis_pixels": int(necrosis_cnt),
        "pustule_pixels": int(pustule_cnt),
        "n_lesions": int(n_lesions),
        "mean_lesion_area": float(mean_lesion_area)
    }

def _count_connected_lesions(binary_mask: np.ndarray) -> Tuple[int, float]:
    """Count connected components and mean component size in a binary mask."""
    if not np.any(binary_mask):
        return 0, 0.0
    try:
        from scipy.ndimage import label
        labeled, num_features = label(binary_mask)
        if num_features == 0:
            return 0, 0.0
        sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
        return num_features, float(np.mean(sizes))
    except Exception:
        # Simple fallback without scipy
        total = np.sum(binary_mask)
        return (1 if total > 0 else 0), float(total)

def generate_synthetic_eth_dataset(raw_dir: Path) -> None:
    """Generate a lightweight synthetic ETH dataset structure for testing when real images are missing."""
    logger.info(f"Generating synthetic ETH dataset tree at {raw_dir}...")
    meta_dir = raw_dir / "meta"
    ts_dir = raw_dir / "processed" / "ts"
    meta_dir.mkdir(parents=True, exist_ok=True)
    ts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Experimental Design CSV
    exp_design = []
    cultivars = ["Arina", "CH_Claro", "Titan", "Runal", "Forel"]
    treatments = ["FI2", "0FI1", "F0I", "0F0I"]
    for i in range(1, 11):
        plot_id = f"ESWW{i:03d}"
        exp_design.append({
            "plot_id": plot_id,
            "cultivar": cultivars[(i - 1) % len(cultivars)],
            "treatment": treatments[(i - 1) % len(treatments)],
            "batch": 1 if i <= 5 else 2,
            "year": 2023,
            "heading_date": "2023-05-15"
        })
    pd.DataFrame(exp_design).to_csv(meta_dir / "experimental_design.csv", index=False)

    # 2. Weather Hourly CSV
    dates = pd.date_range("2023-05-01", "2023-06-30", freq="h")
    np.random.seed(42)
    weather_data = []
    for d in dates:
        weather_data.append({
            "timestamp": d.strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": 15.0 + 5.0 * np.sin(d.hour / 24.0 * 2 * np.pi) + np.random.normal(0, 0.5),
            "relative_humidity": 75.0 + 15.0 * np.cos(d.hour / 24.0 * 2 * np.pi) + np.random.normal(0, 1.0)
        })
    pd.DataFrame(weather_data).to_csv(meta_dir / "weather_station_hourly.csv", index=False)

    # 3. Create 5 sample sequence folders with synthetic mask images
    for i in range(1, 6):
        plot_id = f"ESWW{i:03d}"
        sample_id = f"{plot_id}_L1"
        sample_dir = ts_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        start_time = pd.Timestamp("2023-05-10 10:00:00")
        for step in range(5):
            ts = start_time + pd.Timedelta(days=step * 2)
            ts_str = ts.strftime("%Y%m%d_%H%M%S")
            img_id = f"{ts_str}_{plot_id}_L1"

            # Create synthetic mask (100x100 pixels)
            mask = np.ones((100, 100), dtype=np.uint8) # 1: Healthy
            # Add increasing lesion area (3: Necrosis)
            lesion_size = 5 + step * 8
            mask[10:10+lesion_size, 10:10+lesion_size] = 3
            # Add ink markers (5)
            mask[0:2, 0:2] = 5

            Image.fromarray(mask).save(sample_dir / f"{img_id}_mask.png")
            # Create synthetic rgb leaf image
            rgb = np.stack([mask * 30, mask * 40 + 50, mask * 20], axis=-1).astype(np.uint8)
            Image.fromarray(rgb).save(sample_dir / f"{img_id}.jpg")

    logger.info("Synthetic ETH dataset generation complete.")

def process_eth_dataset(raw_dir: Optional[Path] = None, processed_dir: Optional[Path] = None) -> dict[str, Any]:
    """Parse ETH dataset files and export processed Parquet tables."""
    paths = get_paths()
    if raw_dir is None:
        raw_dir = paths.raw_data / "eth_wheat"
    if processed_dir is None:
        processed_dir = paths.project_root / "data" / "processed" / "eth"

    processed_dir.mkdir(parents=True, exist_ok=True)

    meta_dir = raw_dir / "meta"
    ts_dir = raw_dir / "processed" / "ts"

    # If raw data does not exist, generate synthetic test data
    if not meta_dir.exists() or not (meta_dir / "experimental_design.csv").exists():
        logger.warning("ETH metadata CSV files not found. Creating synthetic ETH tree...")
        generate_synthetic_eth_dataset(raw_dir)

    # 1. Read Experimental Design
    exp_df = pd.read_csv(meta_dir / "experimental_design.csv")

    # 2. Process hourly weather
    weather_csv = meta_dir / "weather_station_hourly.csv"
    if weather_csv.exists():
        weather_h = pd.read_csv(weather_csv)
        weather_h["timestamp"] = pd.to_datetime(weather_h["timestamp"], utc=True)
        weather_h["date"] = weather_h["timestamp"].dt.date

        # Aggregate to daily
        daily_list = []
        for date_val, grp in weather_h.groupby("date"):
            daily_list.append({
                "date": pd.to_datetime(date_val),
                "t_mean": grp["temperature"].mean(),
                "t_max": grp["temperature"].max(),
                "t_min": grp["temperature"].min(),
                "rh_mean": grp["relative_humidity"].mean(),
                "hours_above_90_rh": int((grp["relative_humidity"] >= 90.0).sum())
            })
        weather_d = pd.DataFrame(daily_list)
    else:
        weather_h = pd.DataFrame(columns=["timestamp", "temperature", "relative_humidity"])
        weather_d = pd.DataFrame(columns=["date", "t_mean", "t_max", "t_min", "rh_mean"])

    write_parquet(weather_h, processed_dir / "weather_hourly.parquet")
    write_parquet(weather_d, processed_dir / "weather_daily.parquet")

    # 3. Process time series mask files
    image_records = []
    first_frame_ref_area: dict[str, int] = {}

    # Gather mask images sorted by timestamp per sample
    sample_mask_paths: dict[str, list[Path]] = {}
    for mask_path in sorted(ts_dir.rglob("*_mask.png")):
        sample_id = mask_path.parent.name
        sample_mask_paths.setdefault(sample_id, []).append(mask_path)

    # First pass: Determine reference leaf area at frame 1 for each sample
    for sample_id, mask_list in sample_mask_paths.items():
        if not mask_list:
            continue
        first_mask_path = mask_list[0]
        try:
            arr = np.array(Image.open(first_mask_path))
            m = compute_mask_metrics(arr)
            ref_area = max(m["leaf_pixels_frame"], 1) # Avoid division by zero
            first_frame_ref_area[sample_id] = ref_area
        except Exception:
            first_frame_ref_area[sample_id] = 10000

    # Second pass: Compute per-image long metrics
    sequence_records = []

    for sample_id, mask_list in sample_mask_paths.items():
        ref_area = first_frame_ref_area[sample_id]
        plot_id = sample_id.split("_")[0]

        # Match metadata plot details
        plot_meta = exp_df[exp_df["plot_id"] == plot_id]
        if not plot_meta.empty:
            cultivar = plot_meta.iloc[0]["cultivar"]
            treatment = plot_meta.iloc[0]["treatment"]
            batch = plot_meta.iloc[0]["batch"]
            year = plot_meta.iloc[0]["year"]
        else:
            cultivar, treatment, batch, year = "Unknown", "Untreated", 1, 2023

        sample_images = []
        first_timestamp = None

        for mask_path in mask_list:
            stem = mask_path.name.replace("_mask.png", "")
            meta = parse_image_id(stem)
            
            if first_timestamp is None:
                first_timestamp = meta["timestamp"]
            hours_since_first = (meta["timestamp"] - first_timestamp).total_seconds() / 3600.0

            arr = np.array(Image.open(mask_path))
            m = compute_mask_metrics(arr)

            # Primary severity formulation: diseased / first frame leaf area
            severity = float(np.clip(m["diseased_pixels"] / float(ref_area), 0.0, 1.0))
            
            # Alternative metrics
            per_frame_leaf = max(m["leaf_pixels_frame"], 1)
            severity_per_frame = float(np.clip(m["diseased_pixels"] / float(per_frame_leaf), 0.0, 1.0))
            necrosis_frac = float(m["necrosis_pixels"] / float(ref_area))
            pustule_density = float(m["pustule_pixels"] / float(ref_area))

            try:
                rel_img_str = str(mask_path.relative_to(paths.project_root)).replace("_mask.png", ".jpg")
            except ValueError:
                rel_img_str = str(mask_path).replace("_mask.png", ".jpg")

            rec = {
                "image_id": meta["image_id"],
                "sample_id": sample_id,
                "plot_id": plot_id,
                "leaf_no": meta["leaf_no"],
                "cultivar": cultivar,
                "treatment": treatment,
                "batch": batch,
                "year": year,
                "timestamp": meta["timestamp"],
                "hours_since_first": round(hours_since_first, 2),
                "rel_image_path": rel_img_str,
                "path": rel_img_str,
                "severity": severity,
                "severity_per_frame": severity_per_frame,
                "necrosis_fraction": necrosis_frac,
                "pustule_density": pustule_density,
                "n_lesions": m["n_lesions"],
                "mean_lesion_area": m["mean_lesion_area"],
                "diseased_pixels": m["diseased_pixels"],
                "reference_leaf_area": ref_area,
                "missing_alignment": False,
                "washed_off_markers": False,
                "missing_masks": False
            }
            image_records.append(rec)
            sample_images.append(rec)

        if sample_images:
            span_days = (sample_images[-1]["timestamp"] - sample_images[0]["timestamp"]).total_seconds() / 86400.0
            sequence_records.append({
                "sample_id": sample_id,
                "n_images": len(sample_images),
                "span_days": round(span_days, 2),
                "cultivar": cultivar,
                "plot_id": plot_id,
                "treatment": treatment,
                "batch": batch,
                "year": year,
                "dominant_disease": "Septoria / Rust"
            })

    images_df = pd.DataFrame(image_records)
    sequences_df = pd.DataFrame(sequence_records)

    write_parquet(images_df, processed_dir / "images_long.parquet")
    write_parquet(sequences_df, processed_dir / "sequences.parquet")

    logger.info(f"Exported {len(images_df)} images and {len(sequences_df)} sequences to {processed_dir}")
    return {
        "images_count": len(images_df),
        "sequences_count": len(sequences_df),
        "weather_days": len(weather_d)
    }

def main() -> int:
    process_eth_dataset()
    return 0

if __name__ == "__main__":
    sys.exit(main())
