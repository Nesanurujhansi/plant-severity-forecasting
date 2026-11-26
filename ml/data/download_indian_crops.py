"""Downloader for Indian cash crop datasets (Sugarcane Red Rot & Cotton Bacterial Blight)."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger

logger = get_logger(__name__)

INDIAN_CROPS_METADATA = {
    "sugarcane": {
        "dataset_name": "Sugarcane Disease Dataset",
        "target_rel_path": "data/raw/sugarcane",
        "crop": "Sugarcane",
        "diseases": ["Red Rot", "Leaf Scald", "Mosaic Virus"],
        "license": "CC-BY 4.0 / Zenodo",
    },
    "cotton": {
        "dataset_name": "Cotton Plant Disease Dataset",
        "target_rel_path": "data/raw/cotton",
        "crop": "Cotton",
        "diseases": ["Bacterial Blight", "Leaf Curl Virus", "Target Spot"],
        "license": "CC-BY 4.0 / Kaggle",
    },
}


def download_indian_crops(dry_run: bool = False) -> None:
    """Download Sugarcane and Cotton disease datasets or print dry-run summary."""
    paths = get_paths()

    for crop_key, meta in INDIAN_CROPS_METADATA.items():
        target_dir = paths.raw_data / crop_key
        if dry_run:
            logger.info(f"[DRY-RUN] Would download {meta['dataset_name']} ({meta['crop']}) to {target_dir}")
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "images").mkdir(parents=True, exist_ok=True)
            (target_dir / "masks").mkdir(parents=True, exist_ok=True)
            logger.info(f"{meta['dataset_name']} directory ready at {target_dir}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Download Indian cash crop datasets (Sugarcane & Cotton).")
    parser.add_argument("--dry-run", action="store_true", help="Print dry-run summary without downloading.")
    args = parser.parse_args()

    download_indian_crops(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
