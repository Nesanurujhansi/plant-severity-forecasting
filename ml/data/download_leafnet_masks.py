"""Downloader for LeafNet75 segmentation mask datasets."""

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

HF_LESION_MASKS = "LeafNet75/Leaf-Disease-Segmentation-masks"
HF_WILD_MASKS = "LeafNet75/In_the_Wild_masks600"

def download_leafnet_masks(dry_run: bool = False) -> None:
    """Download LeafNet75 mask repositories or print dry-run summary."""
    paths = get_paths()
    target_dir = paths.raw_data / "leafnet_masks"

    if dry_run:
        logger.info("[DRY RUN] LeafNet75 Mask Datasets")
        logger.info(f"Target Directory: {target_dir}")
        logger.info(f"Sources: '{HF_LESION_MASKS}' and '{HF_WILD_MASKS}'")
        logger.info("Estimated Download Size: ~54 MB + ~120 MB")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading LeafNet75 mask datasets to {target_dir}...")
    
    try:
        from datasets import load_dataset
        ds_lesion = load_dataset(HF_LESION_MASKS)
        ds_lesion.save_to_disk(str(target_dir / "lesion_masks"))
        ds_wild = load_dataset(HF_WILD_MASKS)
        ds_wild.save_to_disk(str(target_dir / "wild_masks"))
        logger.info("Successfully downloaded LeafNet75 mask datasets.")
    except Exception as e:
        logger.error(f"Failed to download LeafNet75 datasets automatically: {e}")

def main() -> int:
    parser = argparse.ArgumentParser(description="LeafNet75 Mask Datasets Downloader")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without downloading.")
    args = parser.parse_args()

    download_leafnet_masks(dry_run=args.dry_run)
    return 0

if __name__ == "__main__":
    sys.exit(main())
