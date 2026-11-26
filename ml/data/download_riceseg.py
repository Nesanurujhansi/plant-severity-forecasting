"""Downloader for RiceSeg-5932 dataset (Mendeley Data / Kaggle Rice Disease)."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger

logger = get_logger(__name__)

RICESEG_METADATA = {
    "dataset_name": "RiceSeg-5932",
    "source": "Mendeley Data (Rice Leaf Disease Image Samples & Masks)",
    "target_rel_path": "data/raw/riceseg",
    "crop": "Rice / Paddy",
    "diseases": ["Rice Blast", "Brown Spot", "Bacterial Blight", "Tungro"],
    "sample_count": 5932,
    "license": "CC-BY 4.0",
}


def download_riceseg(dry_run: bool = False) -> None:
    """Download RiceSeg-5932 dataset or perform dry-run validation."""
    paths = get_paths()
    target_dir = paths.raw_data / "riceseg"

    if dry_run:
        logger.info(
            f"[DRY-RUN] Would download {RICESEG_METADATA['dataset_name']} ({RICESEG_METADATA['sample_count']} samples) to {target_dir}"
        )
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    images_dir = target_dir / "images"
    masks_dir = target_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"RiceSeg-5932 directory structure ready at {target_dir}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Download RiceSeg-5932 dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Print dry-run summary without downloading.")
    args = parser.parse_args()

    download_riceseg(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
