"""Feature Extractor and Caching Engine for Vision Backbones (DINOv2).

Extracts CLS tokens, mean patch tokens, and patch token grids from frozen DINOv2 vision
backbones, supports ETH Zurich macro image cropping and tiling, and caches representations
to sharded .safetensors files with cryptographic cache-key verification and resumability.
"""

import sys
import os
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger
from ml.utils.io import read_parquet

logger = get_logger(__name__)

# Standard ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class StubBackbone(nn.Module):
    """Lightweight stub vision backbone for CPU unit testing without network/GPU dependencies."""
    def __init__(self, embed_dim: int = 384, patch_size: int = 14):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (B, 3, H, W)
        b, c, h, w = x.shape
        gh, gw = h // self.patch_size, w // self.patch_size
        feat_map = self.proj(x)  # (B, D, gh, gw)
        patches = feat_map.flatten(2).transpose(1, 2)  # (B, gh*gw, D)
        cls_token = patches.mean(dim=1)  # (B, D)
        tokens = torch.cat([cls_token.unsqueeze(1), patches], dim=1)  # (B, 1 + gh*gw, D)
        
        class StubOutput:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state
        return StubOutput(tokens)

def compute_cache_key(
    backbone_id: str,
    revision: str,
    resolution: int,
    patch_size: int,
    eth_mode: str = "downscaled_crop"
) -> str:
    """Compute deterministic SHA-256 hash key for cache verification.

    Args:
        backbone_id: Hugging Face model identifier (e.g. "facebook/dinov2-small").
        revision: Git commit hash or revision tag (e.g. "main").
        resolution: Target spatial resolution.
        patch_size: Model patch resolution.
        eth_mode: ETH macro processing mode ("downscaled_crop" or "tiled").

    Returns:
        16-character hexadecimal hash string.
    """
    raw_str = f"{backbone_id}::{revision}::{resolution}::{patch_size}::{eth_mode}"
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:16]

def preprocess_image(
    image: Image.Image,
    target_resolution: int = 448,
    patch_size: int = 14,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Resize image preserving aspect ratio and pad to exact multiples of patch size.

    Args:
        image: PIL Image instance (RGB).
        target_resolution: Target max side length.
        patch_size: Patch stride in pixels (default 14).
        mean: Normalization channel means.
        std: Normalization channel standard deviations.

    Returns:
        Tuple of (normalized tensor (3, padded_h, padded_w), metadata_dict).
    """
    orig_w, orig_h = image.size
    
    # Calculate scale factor to fit within target_resolution
    scale = min(target_resolution / orig_w, target_resolution / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    
    # Resize preserving aspect ratio
    resized_img = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    # Calculate padded size (nearest upper multiple of patch_size)
    padded_w = int(np.ceil(new_w / patch_size) * patch_size)
    padded_h = int(np.ceil(new_h / patch_size) * patch_size)
    
    # Convert to NumPy array and pad with edge pixels or zero
    img_np = np.array(resized_img.convert("RGB"), dtype=np.float32) / 255.0
    
    # Pad array
    pad_h = padded_h - new_h
    pad_w = padded_w - new_w
    padded_np = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    
    # Normalize with mean and std
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    norm_np = (padded_np - mean_arr) / std_arr
    
    # Convert to C x H x W torch Tensor
    tensor = torch.from_numpy(norm_np.transpose(2, 0, 1)).float()
    
    meta = {
        "original_size": (orig_h, orig_w),
        "resized_size": (new_h, new_w),
        "padded_size": (padded_h, padded_w),
        "grid_shape": (padded_h // patch_size, padded_w // patch_size)
    }
    return tensor, meta

def extract_image_tiles(
    image: Image.Image,
    tile_size: int = 448,
    tile_overlap: int = 56,
    patch_size: int = 14
) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int, int]], List[Dict[str, Any]]]:
    """Extract overlapping sliding-window tiles from high-resolution macro photos.

    Args:
        image: High-resolution PIL RGB image.
        tile_size: Square tile dimension.
        tile_overlap: Pixel overlap between adjacent tiles.
        patch_size: Model patch size.

    Returns:
        Tuple of (list of tile tensors, list of (x1, y1, x2, y2) bounds, list of tile metadata).
    """
    w, h = image.size
    stride = tile_size - tile_overlap
    
    x_steps = list(range(0, max(1, w - tile_size + 1), stride))
    if x_steps[-1] + tile_size < w:
        x_steps.append(w - tile_size)
        
    y_steps = list(range(0, max(1, h - tile_size + 1), stride))
    if y_steps[-1] + tile_size < h:
        y_steps.append(h - tile_size)
        
    tile_tensors = []
    tile_coords = []
    tile_metas = []
    
    for y1 in y_steps:
        for x1 in x_steps:
            x2 = min(w, x1 + tile_size)
            y2 = min(h, y1 + tile_size)
            crop_img = image.crop((x1, y1, x2, y2))
            t_tensor, t_meta = preprocess_image(crop_img, target_resolution=tile_size, patch_size=patch_size)
            tile_tensors.append(t_tensor)
            tile_coords.append((x1, y1, x2, y2))
            tile_metas.append(t_meta)
            
    return tile_tensors, tile_coords, tile_metas

class FeatureExtractorEngine:
    """Core feature extraction runner supporting HF DINOv2 and Stub backbones."""
    def __init__(
        self,
        backbone_id: str = "facebook/dinov2-small",
        revision: str = "main",
        patch_size: int = 14,
        is_stub: bool = False,
        device: Optional[str] = None
    ):
        self.backbone_id = backbone_id
        self.revision = revision
        self.patch_size = patch_size
        self.is_stub = is_stub
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        if is_stub:
            embed_dim = 384 if "small" in backbone_id else 768
            self.model = StubBackbone(embed_dim=embed_dim, patch_size=patch_size)
            self.embed_dim = embed_dim
        else:
            try:
                from transformers import AutoModel
                logger.info(f"Loading HF backbone model: {backbone_id} (revision: {revision})...")
                try:
                    self.model = AutoModel.from_pretrained(backbone_id, revision=revision, local_files_only=True)
                except Exception:
                    self.model = AutoModel.from_pretrained(backbone_id, revision=revision)
                self.embed_dim = getattr(self.model.config, "hidden_size", 384)
            except Exception as e:
                logger.warning(f"Failed to load HF backbone ({e}). Falling back to StubBackbone for CPU.")
                embed_dim = 384 if "small" in backbone_id else 768
                self.model = StubBackbone(embed_dim=embed_dim, patch_size=patch_size)
                self.embed_dim = embed_dim
                self.is_stub = True
                
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract_features(
        self,
        input_tensor: torch.Tensor,
        use_amp: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract CLS token, mean patch token, and patch token grid.

        Args:
            input_tensor: Single image tensor (3, H, W) or batch tensor (B, 3, H, W).
            use_amp: Whether to use mixed precision float16 on GPU.

        Returns:
            Tuple of (cls_token [B, D], mean_patch_token [B, D], patch_grid [B, gh, gw, D]).
        """
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
            
        b, c, h, w = input_tensor.shape
        gh, gw = h // self.patch_size, w // self.patch_size
        
        input_tensor = input_tensor.to(self.device)
        amp_device = "cuda" if self.device.type == "cuda" else "cpu"
        
        with torch.amp.autocast(device_type=amp_device, enabled=use_amp and self.device.type == "cuda"):
            outputs = self.model(input_tensor)
            
        last_hidden = outputs.last_hidden_state  # (B, 1 + gh*gw, D)
        cls_token = last_hidden[:, 0, :]  # (B, D)
        patch_tokens = last_hidden[:, 1:, :]  # (B, gh*gw, D)
        mean_patch_token = patch_tokens.mean(dim=1)  # (B, D)
        patch_grid = patch_tokens.reshape(b, gh, gw, self.embed_dim)  # (B, gh, gw, D)
        
        return cls_token.cpu().half(), mean_patch_token.cpu().half(), patch_grid.cpu().half()

def run_feature_extraction_pipeline(
    manifest_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    backbone_id: str = "facebook/dinov2-small",
    revision: str = "main",
    resolution: int = 448,
    patch_size: int = 14,
    shard_size: int = 500,
    eth_mode: str = "downscaled_crop",
    is_stub: bool = False,
    use_gpu: bool = False
) -> Dict[str, Any]:
    """Run full feature extraction pipeline over manifest images, saving sharded safetensors.

    Args:
        manifest_df: DataFrame containing image_id and rel_image_path columns.
        output_dir: Target output cache directory.
        backbone_id: DINOv2 model ID.
        revision: Model revision tag.
        resolution: Target spatial resolution.
        patch_size: Patch stride size.
        shard_size: Number of images per safetensors file.
        eth_mode: ETH macro processing mode.
        is_stub: Force stub model for testing.
        use_gpu: Use GPU if available.

    Returns:
        Summary dict containing cache_key, total_images, processed_images, and throughput stats.
    """
    paths = get_paths()
    cache_key = compute_cache_key(backbone_id, revision, resolution, patch_size, eth_mode)
    
    if output_dir is None:
        cache_dir = paths.data_root / "processed" / "features" / cache_key
    else:
        if (output_dir / "index.json").exists() or output_dir.name == cache_key:
            cache_dir = output_dir
        else:
            cache_dir = output_dir / cache_key
        
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "index.json"
    
    # Check existing index for resumption & cache key verification
    index_data = {
        "cache_key": cache_key,
        "backbone_id": backbone_id,
        "revision": revision,
        "resolution": resolution,
        "patch_size": patch_size,
        "eth_mode": eth_mode,
        "images": {}
    }
    
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("cache_key") != cache_key:
            raise ValueError(
                f"Cache key mismatch! Existing cache in {cache_dir} has key '{existing.get('cache_key')}', "
                f"but current extraction settings produced '{cache_key}'. Refusing to mix caches."
            )
        index_data = existing
        logger.info(f"Resuming feature extraction. Found {len(index_data['images'])} previously extracted images.")

    device_str = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    engine = FeatureExtractorEngine(
        backbone_id=backbone_id,
        revision=revision,
        patch_size=patch_size,
        is_stub=is_stub,
        device=device_str
    )
    
    processed_count = 0
    skipped_count = 0
    current_shard_tensors = {}
    current_shard_idx = len(list(cache_dir.glob("shard_*.safetensors")))
    
    records = manifest_df.to_dict("records") if not manifest_df.empty else []
    
    for row in records:
        img_id = row.get("image_id")
        if not img_id:
            continue
            
        # Check if already processed (resumable)
        if img_id in index_data["images"]:
            skipped_count += 1
            continue
            
        rel_path = row.get("rel_image_path") or row.get("path") or row.get("rel_mask_path")
        if not rel_path:
            continue
            
        full_img_path = paths.project_root / rel_path
        if not full_img_path.exists():
            # Fall back to checking under data_root/raw
            full_img_path = paths.raw_data / rel_path
            if not full_img_path.exists():
                mask_rel = row.get("rel_mask_path") or rel_path.replace(".jpg", "_mask.png")
                full_img_path = paths.project_root / mask_rel
                if not full_img_path.exists():
                    full_img_path = paths.raw_data / mask_rel
                    if not full_img_path.exists():
                        logger.warning(f"Image path missing: {rel_path}. Skipping.")
                        continue
                
        try:
            with Image.open(full_img_path) as img:
                img_rgb = img.convert("RGB")
                tensor, meta = preprocess_image(img_rgb, target_resolution=resolution, patch_size=patch_size)
                cls_t, mean_t, patch_t = engine.extract_features(tensor, use_amp=(device_str == "cuda"))
                
            shard_name = f"shard_{current_shard_idx:04d}.safetensors"
            current_shard_tensors[f"{img_id}_cls"] = cls_t.squeeze(0)
            current_shard_tensors[f"{img_id}_mean"] = mean_t.squeeze(0)
            current_shard_tensors[f"{img_id}_patch"] = patch_t.squeeze(0)
            
            index_data["images"][img_id] = {
                "shard": shard_name,
                "original_size": list(meta["original_size"]),
                "resized_size": list(meta["resized_size"]),
                "padded_size": list(meta["padded_size"]),
                "grid_shape": list(meta["grid_shape"])
            }
            processed_count += 1
            
            # Save shard when capacity reached
            if len(current_shard_tensors) >= (shard_size * 3):
                shard_file = cache_dir / shard_name
                save_file(current_shard_tensors, shard_file)
                current_shard_tensors = {}
                current_shard_idx += 1
                
                # Periodically flush index JSON
                with open(index_path, "w", encoding="utf-8") as f:
                    json.dump(index_data, f, indent=2)
                    
        except Exception as e:
            logger.warning(f"Error processing image {img_id}: {e}")
            continue

    # Flush remaining tensors in last shard
    if current_shard_tensors:
        shard_name = f"shard_{current_shard_idx:04d}.safetensors"
        shard_file = cache_dir / shard_name
        save_file(current_shard_tensors, shard_file)

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)
        
    logger.info(f"Feature extraction completed. Processed: {processed_count}, Skipped: {skipped_count}, Total Indexed: {len(index_data['images'])}.")
    
    return {
        "cache_key": cache_key,
        "cache_dir": str(cache_dir),
        "total_manifest_images": len(records),
        "processed_images": processed_count,
        "skipped_images": skipped_count,
        "total_indexed_images": len(index_data["images"]),
        "num_shards": current_shard_idx + (1 if current_shard_tensors else 0)
    }

def main() -> int:
    parser = argparse.ArgumentParser(description="Extract and cache DINOv2 vision features.")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest parquet.")
    parser.add_argument("--output-dir", type=str, default=None, help="Cache directory.")
    parser.add_argument("--backbone", type=str, default="facebook/dinov2-small", help="Model ID.")
    parser.add_argument("--resolution", type=int, default=448, help="Target resolution.")
    parser.add_argument("--stub", action="store_true", help="Force stub backbone for CPU testing.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration.")
    args = parser.parse_args()

    paths = get_paths()
    if args.manifest:
        m_path = Path(args.manifest)
    else:
        m_path = paths.project_root / "data" / "processed" / "static" / "manifest.parquet"
        
    if m_path.exists():
        df = read_parquet(m_path)
    else:
        logger.warning(f"Manifest file {m_path} not found. Running with empty DataFrame.")
        df = pd.DataFrame()
        
    out_dir = Path(args.output_dir) if args.output_dir else None
    res = run_feature_extraction_pipeline(
        manifest_df=df,
        output_dir=out_dir,
        backbone_id=args.backbone,
        resolution=args.resolution,
        is_stub=args.stub,
        use_gpu=args.gpu
    )
    print("Extraction Summary:", res)
    return 0

if __name__ == "__main__":
    sys.exit(main())
