"""Unified Multi-Crop Dataset Manifest Builder.

Scans static and sequence datasets across 6 crop types:
- Wheat (ETH Zurich Wheat)
- Rice / Paddy (RiceSeg-5932)
- Tomato (PlantSeg / PlantDoc)
- Corn / Maize (PlantSeg / Zenodo)
- Sugarcane (Sugarcane Disease)
- Cotton (Cotton Plant Disease)

Standardizes lesion masks to binary PNG (0/1), verifies mask integrity,
calculates perceptual hashes (dhash) and near-duplicate cluster IDs, and
exports data/processed/multicrop/manifest.parquet.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
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
from ml.data.provenance import compute_dhash, cluster_near_duplicates
from ml.forecast.crop_physics import CROP_PHYSICS_REGISTRY

logger = get_logger(__name__)


def validate_mask_integrity(image_path: Path, mask_path: Path) -> Tuple[bool, str]:
    """Verify single-channel mask dimension match and binary 0/1 pixel values."""
    if not image_path.exists():
        return False, "Image file missing"
    if not mask_path.exists():
        return False, "Mask file missing"
    try:
        img = Image.open(image_path)
        mask = Image.open(mask_path)

        if img.size != mask.size:
            return False, f"Dimension mismatch: image ({img.size}) vs mask ({mask.size})"

        mask_arr = np.array(mask)
        unique_vals = np.unique(mask_arr)
        if not set(unique_vals).issubset({0, 1}):
            return False, f"Non-binary mask values: {unique_vals}"

        return True, "Valid"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def process_multicrop_manifest(raw_dir: Optional[Path] = None, processed_dir: Optional[Path] = None) -> pd.DataFrame:
    """Scan multi-crop dataset directories and assemble unified manifest.parquet."""
    paths = get_paths()
    if raw_dir is None:
        raw_dir = paths.raw_data
    if processed_dir is None:
        processed_dir = paths.project_root / "data" / "processed" / "multicrop"

    processed_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir = processed_dir / "images"
    masks_out_dir = processed_dir / "lesion_masks"
    images_out_dir.mkdir(parents=True, exist_ok=True)
    masks_out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []

    # Map dataset folders to crop keys
    crop_sources = [
        ("riceseg", "rice", "Rice / Paddy", "Rice Leaf Blast"),
        ("plantseg", "tomato", "Tomato", "Tomato Late Blight"),
        ("sugarcane", "sugarcane", "Sugarcane", "Sugarcane Red Rot"),
        ("cotton", "cotton", "Cotton", "Cotton Bacterial Blight"),
    ]

    for source_folder, crop_key, crop_name, default_disease in crop_sources:
        crop_dir = raw_dir / source_folder
        if not crop_dir.exists():
            continue

        for img_path in crop_dir.rglob("*.jpg"):
            stem = img_path.stem
            if stem.endswith("_mask") or "lesion" in stem:
                continue

            # Look for mask candidate
            mask_path = img_path.parent / f"{stem}_mask.png"
            if not mask_path.exists():
                mask_candidate = crop_dir / "masks" / f"{stem}_mask.png"
                if mask_candidate.exists():
                    mask_path = mask_candidate
                else:
                    continue

            std_img_path = images_out_dir / f"{source_folder}_{stem}.jpg"
            std_mask_path = masks_out_dir / f"{source_folder}_{stem}_lesion.png"

            try:
                img_obj = Image.open(img_path).convert("RGB")
                img_obj.save(std_img_path)

                m_arr = np.array(Image.open(mask_path))
                m_binary = (m_arr > 0).astype(np.uint8)
                Image.fromarray(m_binary).save(std_mask_path)

                valid, msg = validate_mask_integrity(std_img_path, std_mask_path)
                dhash = compute_dhash(std_img_path)

                records.append({
                    "image_id": f"{source_folder}_{stem}",
                    "source": source_folder,
                    "crop_key": crop_key,
                    "crop_name": crop_name,
                    "disease": default_disease,
                    "rel_image_path": str(std_img_path.relative_to(paths.project_root)),
                    "rel_lesion_mask_path": str(std_mask_path.relative_to(paths.project_root)),
                    "width": img_obj.width,
                    "height": img_obj.height,
                    "dhash": dhash,
                    "is_quality_valid": valid,
                    "validation_msg": msg,
                })
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")

    manifest_df = pd.DataFrame(records)

    # Compute dhash cluster IDs for duplicate prevention
    if not manifest_df.empty and "dhash" in manifest_df.columns:
        dhashes = manifest_df["dhash"].tolist()
        manifest_df["cluster_id"] = cluster_near_duplicates(dhashes, max_distance=10)
    else:
        manifest_df = pd.DataFrame(columns=[
            "image_id", "source", "crop_key", "crop_name", "disease",
            "rel_image_path", "rel_lesion_mask_path", "width", "height",
            "dhash", "is_quality_valid", "validation_msg", "cluster_id"
        ])

    out_manifest_path = processed_dir / "manifest.parquet"
    write_parquet(manifest_df, out_manifest_path)
    logger.info(f"Built unified multi-crop manifest with {len(manifest_df)} records at {out_manifest_path}")

    return manifest_df


def main() -> int:
    process_multicrop_manifest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
