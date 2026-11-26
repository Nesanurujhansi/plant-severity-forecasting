"""Validation script assessing downloaded datasets integrity and file statistics.

Produces JSON and Markdown summary reports at data/interim/download_validation.json and .md.
"""

import sys
import json
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

DATASET_CONFIGS = [
    {
        "name": "eth_wheat",
        "subpath": "eth_wheat",
        "expected_images": 12520,
        "licence": "CC BY 4.0",
        "required_subdirs": ["meta", "processed"]
    },
    {
        "name": "plantseg",
        "subpath": "plantseg",
        "expected_images": 1200,
        "licence": "Metadata per-image",
        "required_subdirs": []
    },
    {
        "name": "coffee_rust",
        "subpath": "coffee_rust",
        "expected_images": 1285,
        "licence": "CC BY 4.0",
        "required_subdirs": []
    },
    {
        "name": "leafnet_masks",
        "subpath": "leafnet_masks",
        "expected_images": 600,
        "licence": "Apache-2.0 / MIT",
        "required_subdirs": []
    },
    {
        "name": "plantvillage",
        "subpath": "plantvillage",
        "expected_images": 54303,
        "licence": "CC0 1.0",
        "required_subdirs": []
    }
]

def scan_dataset(data_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Scan a single dataset folder and count images, masks, and total bytes."""
    raw_dir = data_root / "raw" / config["subpath"]
    
    result: dict[str, Any] = {
        "dataset_name": config["name"],
        "path": str(raw_dir),
        "exists": raw_dir.exists(),
        "total_files": 0,
        "image_count": 0,
        "mask_count": 0,
        "total_size_mb": 0.0,
        "licence": config["licence"],
        "missing_pieces": [],
        "status": "MISSING"
    }

    if not raw_dir.exists():
        result["missing_pieces"].append(f"Directory not found: {raw_dir}")
        return result

    # Calculate total size and counts
    total_bytes = 0
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    
    for item in raw_dir.rglob("*"):
        if item.is_file():
            result["total_files"] += 1
            total_bytes += item.stat().st_size
            ext = item.suffix.lower()
            if ext in image_exts:
                if "mask" in item.name.lower():
                    result["mask_count"] += 1
                else:
                    result["image_count"] += 1

    result["total_size_mb"] = round(total_bytes / (1024 * 1024), 2)

    # Check required subdirectories
    for subdir in config.get("required_subdirs", []):
        if not (raw_dir / subdir).exists():
            result["missing_pieces"].append(f"Missing required subdirectory: {subdir}")

    if result["total_files"] > 0 and not result["missing_pieces"]:
        result["status"] = "OK"
    elif result["total_files"] > 0:
        result["status"] = "PARTIAL"

    return result

def generate_validation_reports(data_root: Path, output_dir: Path) -> dict[str, Any]:
    """Scan all candidate datasets and generate JSON and Markdown reports."""
    summary: dict[str, Any] = {
        "datasets": [],
        "total_datasets_checked": len(DATASET_CONFIGS),
        "total_datasets_found": 0,
        "total_images_found": 0,
        "total_size_mb": 0.0
    }

    markdown_lines = [
        "# Download Validation Summary Report",
        "",
        "| Dataset Name | Status | Files Found | Image Count | Mask Count | Size (MB) | Licence | Missing Pieces |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for config in DATASET_CONFIGS:
        res = scan_dataset(data_root, config)
        summary["datasets"].append(res)
        
        if res["exists"]:
            summary["total_datasets_found"] += 1
            summary["total_images_found"] += res["image_count"]
            summary["total_size_mb"] += res["total_size_mb"]
            
        missing_str = ", ".join(res["missing_pieces"]) if res["missing_pieces"] else "None"
        markdown_lines.append(
            f"| **{res['dataset_name']}** | {res['status']} | {res['total_files']} | "
            f"{res['image_count']} | {res['mask_count']} | {res['total_size_mb']} | "
            f"{res['licence']} | {missing_str} |"
        )

    # Save JSON report
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "download_validation.json"
    write_json(summary, json_path)

    # Save Markdown report
    md_path = output_dir / "download_validation.md"
    md_path.write_text("\n".join(markdown_lines), encoding="utf-8")

    logger.info(f"Saved validation reports to {json_path} and {md_path}")
    return summary

def main() -> int:
    paths = get_paths()
    out_dir = paths.interim_data
    generate_validation_reports(paths.data_root, out_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())
