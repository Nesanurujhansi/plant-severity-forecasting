"""Optional downloader for PlantVillage dataset (mohanty/PlantVillage)."""

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

HF_PLANTVILLAGE_REPO = "mohanty/PlantVillage"

def download_plantvillage(dry_run: bool = False) -> None:
    """Download PlantVillage dataset or print dry-run summary."""
    paths = get_paths()
    target_dir = paths.raw_data / "plantvillage"

    if dry_run:
        logger.info("[DRY RUN] PlantVillage Dataset (Optional)")
        logger.info(f"Target Directory: {target_dir}")
        logger.info(f"Source: Hugging Face dataset '{HF_PLANTVILLAGE_REPO}'")
        logger.info("Estimated Download Size: ~800 MB (54,303 lab-style images)")
        logger.info("POLICY REMINDER: Strictly for pretraining/sanity check. NEVER for headline claims.")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading PlantVillage dataset ({HF_PLANTVILLAGE_REPO}) to {target_dir}...")
    
    try:
        from datasets import load_dataset
        ds = load_dataset(HF_PLANTVILLAGE_REPO, name="default")
        ds.save_to_disk(str(target_dir))
        logger.info("Successfully downloaded PlantVillage dataset.")
    except Exception as e:
        logger.warning(f"Failed with config name='default': {e}. Trying default configuration...")
        try:
            from datasets import load_dataset
            ds = load_dataset(HF_PLANTVILLAGE_REPO)
            ds.save_to_disk(str(target_dir))
            logger.info("Successfully downloaded PlantVillage dataset.")
        except Exception as err:
            logger.error(f"Failed to download PlantVillage dataset automatically: {err}")

def main() -> int:
    parser = argparse.ArgumentParser(description="PlantVillage Dataset Downloader")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without downloading.")
    args = parser.parse_args()

    download_plantvillage(dry_run=args.dry_run)
    return 0

if __name__ == "__main__":
    sys.exit(main())
