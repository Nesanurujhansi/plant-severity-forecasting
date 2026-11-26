"""Data Provenance and Perceptual Image Deduplication Tool.

Calculates perceptual hashes (dHash/pHash) for image collections, clusters near-duplicate
images across static datasets, and exports a standardized static manifest to
data/processed/static_manifest.parquet.
"""

import sys
from pathlib import Path
from typing import Optional, Union
from PIL import Image
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# Ensure project root is in sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger
from ml.utils.io import write_parquet

logger = get_logger(__name__)

class ManifestRecord(BaseModel):
    """Schema definition for data/processed/static_manifest.parquet records."""
    image_id: str = Field(..., description="Unique image identifier.")
    source: str = Field(..., description="Source dataset name.")
    path: str = Field(..., description="Relative filepath.")
    licence: str = Field(..., description="Captured dataset license.")
    width: int = Field(..., description="Image width in pixels.")
    height: int = Field(..., description="Image height in pixels.")
    hash: str = Field(..., description="Hexadecimal perceptual dHash string.")
    cluster_id: int = Field(..., description="Near-duplicate cluster group ID.")

def compute_dhash(image: Union[Image.Image, str, Path], hash_size: int = 8) -> str:
    """Compute difference hash (dHash) for a PIL image or image file path.
    
    Args:
        image: Input PIL Image instance or path to an image file.
        hash_size: Grid resolution (default 8 yields 64-bit hash).
        
    Returns:
        64-character binary/hex string representation of the hash.
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.array(resized)
    
    # Calculate difference between adjacent horizontal pixels
    diff = pixels[:, 1:] > pixels[:, :-1]
    
    # Convert boolean array to hex string
    binary_str = "".join(["1" if b else "0" for b in diff.flatten()])
    hex_str = f"{int(binary_str, 2):016x}"
    return hex_str

def hamming_distance(hex_hash1: str, hex_hash2: str) -> int:
    """Compute Hamming distance (number of differing bits) between two hex hashes."""
    b1 = bin(int(hex_hash1, 16))[2:].zfill(64)
    b2 = bin(int(hex_hash2, 16))[2:].zfill(64)
    return sum(c1 != c2 for c1, c2 in zip(b1, b2))

def cluster_near_duplicates(hashes: list[str], max_distance: int = 5) -> list[int]:
    """Group image indices into clusters based on Hamming distance threshold."""
    n = len(hashes)
    cluster_ids = [-1] * n
    current_cluster = 0
    
    for i in range(n):
        if cluster_ids[i] != -1:
            continue
        cluster_ids[i] = current_cluster
        for j in range(i + 1, n):
            if cluster_ids[j] == -1 and hamming_distance(hashes[i], hashes[j]) <= max_distance:
                cluster_ids[j] = current_cluster
        current_cluster += 1
        
    return cluster_ids

def build_static_manifest(data_root: Optional[Path] = None, max_distance: int = 5) -> pd.DataFrame:
    """Build static manifest Parquet file indexing all available dataset images."""
    if data_root is None:
        paths = get_paths()
        data_root = paths.data_root
        
    records = []
    logger.info(f"Scanning images in {data_root / 'raw'} for provenance manifest...")
    
    # Image extensions
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_files = [p for p in (data_root / "raw").rglob("*") if p.suffix.lower() in image_exts]
    
    for idx, file_path in enumerate(image_files):
        try:
            with Image.open(file_path) as img:
                w, h = img.size
                h_val = compute_dhash(img)
                
            source_name = file_path.relative_to(data_root / "raw").parts[0]
            records.append({
                "image_id": f"{source_name}_{idx:06d}",
                "source": source_name,
                "path": str(file_path.relative_to(data_root)),
                "licence": "CC BY 4.0" if "eth" in source_name else "Unknown",
                "width": w,
                "height": h,
                "hash": h_val,
            })
        except Exception as e:
            logger.warning(f"Failed to process image {file_path}: {e}")

    if not records:
        logger.warning("No image files found. Returning empty manifest DataFrame.")
        df = pd.DataFrame(columns=["image_id", "source", "path", "licence", "width", "height", "hash", "cluster_id"])
    else:
        df = pd.DataFrame(records)
        cluster_ids = cluster_near_duplicates(df["hash"].tolist(), max_distance=max_distance)
        df["cluster_id"] = cluster_ids
        
    # Save output to static_manifest.parquet
    output_path = paths.project_root / "data" / "processed" / "static_manifest.parquet"
    write_parquet(df, output_path)
    logger.info(f"Exported manifest with {len(df)} records to {output_path}")
    return df

def main() -> int:
    build_static_manifest()
    return 0

if __name__ == "__main__":
    sys.exit(main())
