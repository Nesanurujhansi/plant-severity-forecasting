"""Downloader for Coffee Leaf Rust severity dataset (MaryPazRB/Paper_CLR_CV)."""

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

HF_COFFEE_REPO = "MaryPazRB/Paper_CLR_CV"

def download_coffee_rust(dry_run: bool = False) -> None:
    """Download Coffee Leaf Rust dataset from Hugging Face or print dry-run summary."""
    paths = get_paths()
    target_dir = paths.raw_data / "coffee_rust"

    if dry_run:
        logger.info("[DRY RUN] Coffee Leaf Rust Severity Dataset")
        logger.info(f"Target Directory: {target_dir}")
        logger.info(f"Source: Hugging Face dataset '{HF_COFFEE_REPO}'")
        logger.info("Estimated Download Size: ~600 MB (1,285 images, 606 lesion masks)")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading Coffee Leaf Rust repository ({HF_COFFEE_REPO}) to {target_dir}...")
    
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=HF_COFFEE_REPO,
            repo_type="model",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False
        )
        logger.info("Successfully downloaded Coffee Leaf Rust repository.")
    except Exception as e:
        logger.warning(f"Failed via snapshot_download model type: {e}. Trying dataset load_dataset fallback...")
        try:
            from datasets import load_dataset
            ds = load_dataset(HF_COFFEE_REPO)
            ds.save_to_disk(str(target_dir))
            logger.info("Successfully downloaded Coffee Leaf Rust dataset.")
        except Exception as err:
            logger.error(f"Failed to download Coffee Leaf Rust repository automatically: {err}")
            logger.info("Fallback: Direct link available in docs/data_cards/coffee_leaf_rust.md")

def main() -> int:
    parser = argparse.ArgumentParser(description="Coffee Leaf Rust Dataset Downloader")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without downloading.")
    args = parser.parse_args()

    download_coffee_rust(dry_run=args.dry_run)
    return 0

if __name__ == "__main__":
    sys.exit(main())
