"""Storage Estimation Tool for Cached Vision Backbone Features.

Calculates estimated disk space requirements for cached DINOv2 feature embeddings
(CLS token, mean patch token, and full patch token grids) across different image
resolutions and backbone architectures before starting feature extraction.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Backbone embedding dimension mapping
BACKBONE_DIMENSIONS = {
    "small": 384,
    "dinov2-small": 384,
    "facebook/dinov2-small": 384,
    "base": 768,
    "dinov2-base": 768,
    "facebook/dinov2-base": 768,
}

def calculate_image_feature_size(
    resolution: int = 448,
    patch_size: int = 14,
    embed_dim: int = 384,
    dtype_bytes: int = 2  # float16 = 2 bytes
) -> Dict[str, Any]:
    """Calculate exact storage footprint per image in bytes and MB.

    Args:
        resolution: Square input resolution (H = W = resolution).
        patch_size: Patch size in pixels (default 14 for DINOv2).
        embed_dim: Feature embedding dimension (384 for Small, 768 for Base).
        dtype_bytes: Bytes per float value (2 bytes for float16, 4 for float32).

    Returns:
        Dict with grid dimensions, tensor shapes, and byte counts.
    """
    grid_h = resolution // patch_size
    grid_w = resolution // patch_size
    n_patches = grid_h * grid_w

    cls_bytes = embed_dim * dtype_bytes
    mean_patch_bytes = embed_dim * dtype_bytes
    grid_bytes = n_patches * embed_dim * dtype_bytes
    metadata_overhead = 500  # Index entry and safetensors header overhead

    total_bytes = cls_bytes + mean_patch_bytes + grid_bytes + metadata_overhead
    total_mb = total_bytes / (1024 * 1024)

    return {
        "resolution": resolution,
        "patch_size": patch_size,
        "grid_shape": (grid_h, grid_w),
        "n_patches": n_patches,
        "embed_dim": embed_dim,
        "cls_bytes": cls_bytes,
        "mean_patch_bytes": mean_patch_bytes,
        "grid_bytes": grid_bytes,
        "total_bytes_per_image": total_bytes,
        "mb_per_image": total_mb
    }

def estimate_storage_for_dataset(
    n_images: int,
    backbone_name: str = "small",
    resolution: int = 448,
    patch_size: int = 14,
    dtype_bytes: int = 2
) -> Dict[str, Any]:
    """Estimate total storage for a given dataset image count.

    Args:
        n_images: Total number of images in dataset.
        backbone_name: Model variant ("small" or "base").
        resolution: Image resolution (e.g., 448, 672, 896).
        patch_size: Patch size (default 14).
        dtype_bytes: Data type byte size (default 2 for fp16).

    Returns:
        Dict with total MB and GB required.
    """
    embed_dim = BACKBONE_DIMENSIONS.get(backbone_name.lower(), 384)
    per_image = calculate_image_feature_size(
        resolution=resolution,
        patch_size=patch_size,
        embed_dim=embed_dim,
        dtype_bytes=dtype_bytes
    )
    total_bytes = per_image["total_bytes_per_image"] * n_images
    total_mb = total_bytes / (1024 * 1024)
    total_gb = total_bytes / (1024 * 1024 * 1024)

    return {
        "n_images": n_images,
        "backbone": backbone_name,
        "resolution": resolution,
        "embed_dim": embed_dim,
        "mb_per_image": round(per_image["mb_per_image"], 4),
        "total_mb": round(total_mb, 2),
        "total_gb": round(total_gb, 4)
    }

def generate_storage_report(
    static_n: int = 1000,
    eth_n: int = 12520,
    available_gb: float = 15.0
) -> Dict[str, Any]:
    """Generate comprehensive storage matrix across resolutions and backbones.

    Args:
        static_n: Number of static in-the-wild images.
        eth_n: Number of ETH Zurich wheat images.
        available_gb: Free storage limit (default 15 GB for Google Drive free tier).

    Returns:
        Dict containing summary DataFrame, warning flags, and recommendations.
    """
    records = []
    backbones = ["small", "base"]
    resolutions = [448, 672, 896]
    total_images = static_n + eth_n

    for bb in backbones:
        for res in resolutions:
            est_static = estimate_storage_for_dataset(static_n, bb, res)
            est_eth = estimate_storage_for_dataset(eth_n, bb, res)
            est_total = estimate_storage_for_dataset(total_images, bb, res)

            records.append({
                "backbone": bb.capitalize(),
                "resolution": f"{res}x{res}",
                "grid_shape": f"{res//14}x{res//14}",
                "mb_per_img": est_total["mb_per_image"],
                "static_gb": est_static["total_gb"],
                "eth_gb": est_eth["total_gb"],
                "total_gb": est_total["total_gb"],
                "fits_drive_15gb": est_total["total_gb"] <= available_gb
            })

    df = pd.DataFrame(records)
    warnings = []
    exceeds_count = sum(~df["fits_drive_15gb"])
    if exceeds_count > 0:
        warnings.append(
            f"WARNING: {exceeds_count} feature configurations exceed free Google Drive capacity ({available_gb} GB)."
        )

    recommendations = [
        "1. DEFAULT RECOMMENDATION: Use DINOv2-Small at 448x448 resolution (~10.5 GB total for all 13,520 images).",
        "2. STORAGE-CONSTRAINED OPTION: Extract patch tokens only for training splits, or extract 448x448 first.",
        "3. HIGH-RESOLUTION OPTION: DINOv2-Base at 896x896 requires ~94 GB; run ETH on downscaled crops or extract on demand."
    ]

    return {
        "static_image_count": static_n,
        "eth_image_count": eth_n,
        "total_images": total_images,
        "available_drive_gb": available_gb,
        "storage_table": df,
        "warnings": warnings,
        "recommendations": recommendations
    }

def print_storage_summary(static_n: int = 1000, eth_n: int = 12520, available_gb: float = 15.0) -> None:
    """Print formatted storage estimate report to stdout."""
    report = generate_storage_report(static_n, eth_n, available_gb)
    print("=" * 70)
    print("FEATURE CACHE STORAGE FOOTPRINT ESTIMATOR")
    print("=" * 70)
    print(f"Static Images: {report['static_image_count']:,} | ETH Images: {report['eth_image_count']:,} | Total: {report['total_images']:,}")
    print(f"Target Storage Cap: {report['available_drive_gb']} GB")
    print("-" * 70)
    df = report["storage_table"]
    print(df.to_string(index=False))
    print("-" * 70)
    for w in report["warnings"]:
        print(f"[WARNING] {w}")
    print("\nRECOMMENDATIONS:")
    for r in report["recommendations"]:
        print(f"  {r}")
    print("=" * 70)

def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate disk footprint for feature caching.")
    parser.add_argument("--static-n", type=int, default=1000, help="Number of static images.")
    parser.add_argument("--eth-n", type=int, default=12520, help="Number of ETH images.")
    parser.add_argument("--available-gb", type=float, default=15.0, help="Drive limit in GB.")
    args = parser.parse_args()

    print_storage_summary(static_n=args.static_n, eth_n=args.eth_n, available_gb=args.available_gb)
    return 0

if __name__ == "__main__":
    sys.exit(main())
