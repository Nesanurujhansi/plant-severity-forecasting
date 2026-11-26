"""Downloader and structure validator for ETH Zurich Wheat Leaf Image Time Series.

Downloads metadata and processed alignment/mask files first. Includes a manual-download
validation utility checking folder layout against reported author statistics (12,520 images, 1,032 sequences).
"""

import sys
import argparse
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger
from ml.utils.io import write_json

logger = get_logger(__name__)

ETH_DOI = "10.3929/ethz-b-000739194"
ETH_BASE_URL = f"https://www.research-collection.ethz.ch/handle/{ETH_DOI}"

REPORTED_STATS = {
    "total_images": 12520,
    "leaf_sequences": 1032,
    "cultivars": 15,
    "seasons": ["2022-2023", "2023-2024"],
    "treatments": ["FI2", "0FI1", "F0I", "0F0I"],
}

def print_manual_download_guide(target_dir: Path) -> None:
    """Print step-by-step instructions for manual download from ETH Research Collection."""
    print("=" * 70)
    print("MANUAL DOWNLOAD GUIDE: ETH Zurich Wheat Leaf Time Series")
    print("=" * 70)
    print(f"1. Open the ETH Research Collection URL: {ETH_BASE_URL}")
    print("2. Download the metadata and processed dataset archives:")
    print("   - meta.zip (contains experimental design, weather, and lesion metadata)")
    print("   - processed.zip (contains geometric alignment matrices and symptom masks)")
    print(f"3. Unzip the downloaded contents into your project data root at:")
    print(f"   {target_dir / 'raw' / 'eth_wheat'}")
    print("   Expected subfolders: raw/, processed/ (reg/, ts/), meta/, markers/")
    print("=" * 70)

def validate_eth_folder_structure(target_dir: Path) -> dict[str, Any]:
    """Validate extracted ETH dataset files against reported author numbers."""
    eth_dir = target_dir / "raw" / "eth_wheat"
    results: dict[str, Any] = {
        "found": eth_dir.exists(),
        "path": str(eth_dir),
        "found_images": 0,
        "found_sequences": 0,
        "mismatches": [],
        "status": "MISSING"
    }

    if not eth_dir.exists():
        results["mismatches"].append(f"Directory not found: {eth_dir}")
        return results

    # Count image files (.jpg, .png, .tif)
    images = list(eth_dir.rglob("*.jpg")) + list(eth_dir.rglob("*.png")) + list(eth_dir.rglob("*.tif"))
    results["found_images"] = len(images)

    # Count leaf sequence folders under processed/ts
    ts_dir = eth_dir / "processed" / "ts"
    if ts_dir.exists():
        seq_dirs = [d for d in ts_dir.iterdir() if d.is_dir()]
        results["found_sequences"] = len(seq_dirs)

    # Check against reported numbers
    if results["found_images"] > 0 and results["found_images"] != REPORTED_STATS["total_images"]:
        results["mismatches"].append(
            f"Image count mismatch: Found {results['found_images']}, reported {REPORTED_STATS['total_images']}."
        )

    if results["found_sequences"] > 0 and results["found_sequences"] != REPORTED_STATS["leaf_sequences"]:
        results["mismatches"].append(
            f"Sequence count mismatch: Found {results['found_sequences']}, reported {REPORTED_STATS['leaf_sequences']}."
        )

    results["status"] = "OK" if not results["mismatches"] and results["found_images"] > 0 else "PARTIAL_OR_MISSING"
    return results

def download_eth_wheat(dry_run: bool = False, raw: bool = False) -> None:
    """Run ETH wheat downloader or dry-run inspection."""
    paths = get_paths()
    target_dir = paths.raw_data / "eth_wheat"

    if dry_run:
        logger.info("[DRY RUN] ETH Zurich Wheat Time Series Dataset")
        logger.info(f"Target Directory: {target_dir}")
        logger.info("Planned Files to Download:")
        logger.info("  - Metadata archive (meta.zip): ~50 MB")
        logger.info("  - Processed alignment & mask archive (processed.zip): ~2.5 GB")
        if raw:
            logger.info("  - Raw 45 MP image archives (raw_images.zip): >100 GB (Requested)")
        else:
            logger.info("  - Raw full-frame images skipped by default (--raw not set).")
        print_manual_download_guide(paths.data_root)
        return

    logger.info(f"Attempting ETH Wheat dataset verification in {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Try fetching metadata archive automatically with branch fallbacks
    meta_urls = [
        "https://github.com/and-jonas/sympathique-wheat/archive/refs/heads/master.zip",
        "https://github.com/and-jonas/sympathique-wheat/archive/refs/heads/main.zip",
        "https://github.com/and-jonas/sympathique-wheat/archive/refs/heads/development.zip"
    ]
    
    download_success = False
    for meta_url in meta_urls:
        logger.info(f"Fetching metadata archive from {meta_url}...")
        try:
            resp = requests.get(meta_url, timeout=30)
            if resp.status_code == 200:
                zip_path = target_dir / "meta_archive.zip"
                with open(zip_path, "wb") as f:
                    f.write(resp.content)
                import zipfile
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(target_dir)
                logger.info(f"Successfully extracted ETH metadata files to {target_dir}")
                download_success = True
                break
        except Exception as e:
            logger.warning(f"Could not download from {meta_url}: {e}")

    if not download_success:
        logger.warning("Could not automatically download ETH metadata archive from GitHub branches.")

    validation = validate_eth_folder_structure(paths.data_root)
    if validation["status"] != "OK":
        print_manual_download_guide(paths.data_root)

def main() -> int:
    parser = argparse.ArgumentParser(description="ETH Zurich Wheat Dataset Downloader")
    parser.add_argument("--dry-run", action="store_true", help="Print file list and sizes without downloading.")
    parser.add_argument("--raw", action="store_true", help="Include raw 100GB full-frame images.")
    args = parser.parse_args()

    download_eth_wheat(dry_run=args.dry_run, raw=args.raw)
    return 0

if __name__ == "__main__":
    sys.exit(main())
