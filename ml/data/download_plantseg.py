"""Downloader for PlantSeg in-the-wild disease segmentation dataset."""

import sys
import argparse
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger

logger = get_logger(__name__)

HF_PLANTSEG_REPO = "Voxel51/PlantSeg-Test"

def download_plantseg(dry_run: bool = False) -> None:
    """Download PlantSeg dataset from Hugging Face or print dry-run summary."""
    paths = get_paths()
    target_dir = paths.raw_data / "plantseg"

    if dry_run:
        logger.info("[DRY RUN] PlantSeg Dataset")
        logger.info(f"Target Directory: {target_dir}")
        logger.info(f"Source: Hugging Face dataset '{HF_PLANTSEG_REPO}'")
        logger.info("Estimated Download Size: ~500 MB (1,200 sample evaluation subset)")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading PlantSeg test subset ({HF_PLANTSEG_REPO}) to {target_dir}...")
    
    try:
        from datasets import load_dataset
        ds = load_dataset(HF_PLANTSEG_REPO)
        # Save split info
        ds.save_to_disk(str(target_dir))
        logger.info(f"Successfully downloaded PlantSeg dataset: {len(ds)} samples.")
    except Exception as e:
        logger.error(f"Failed to download PlantSeg dataset automatically: {e}")
        logger.info("Fallback: Manual download available via Zenodo link in docs/data_cards/plantseg.md")

def main() -> int:
    parser = argparse.ArgumentParser(description="PlantSeg Dataset Downloader")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without downloading.")
    args = parser.parse_args()

    download_plantseg(dry_run=args.dry_run)
    return 0

if __name__ == "__main__":
    sys.exit(main())
