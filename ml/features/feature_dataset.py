"""PyTorch Dataset Class for Cached Vision Backbone Features.

Provides lazy, memory-efficient loading of cached DINOv2 feature embeddings (.safetensors shards)
along with ground truth masks (resized to patch grid spatial dimensions) for downstream head training.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from safetensors.torch import load_file

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger

logger = get_logger(__name__)

class CachedFeatureDataset(Dataset):
    """Lazy-loading PyTorch Dataset reading sharded .safetensors feature embeddings."""

    def __init__(
        self,
        cache_dir: Union[str, Path],
        manifest_df: Optional[pd.DataFrame] = None,
        split: Optional[str] = None,
        transform: Optional[Any] = None
    ):
        """Initialize dataset from feature cache directory and optional manifest filters.

        Args:
            cache_dir: Path to directory containing index.json and shard safetensors.
            manifest_df: Optional manifest DataFrame to filter image IDs or load mask paths.
            split: Optional split filter string ("train", "val", "test", "calibration").
            transform: Optional feature transform function.
        """
        self.cache_dir = Path(cache_dir)
        self.index_path = self.cache_dir / "index.json"
        
        if not self.index_path.exists():
            raise FileNotFoundError(f"Feature cache index.json missing at {self.index_path}")
            
        with open(self.index_path, "r", encoding="utf-8") as f:
            self.index_data = json.load(f)

        self.cache_key = self.index_data.get("cache_key", "")
        self.backbone_id = self.index_data.get("backbone_id", "")
        self.embed_dim = self.index_data.get("embed_dim", 384)
        self.patch_size = self.index_data.get("patch_size", 14)
        self.transform = transform
        
        all_images = self.index_data.get("images", {})
        
        # Merge with manifest DataFrame if provided
        self.items = []
        paths = get_paths()
        
        if manifest_df is not None and not manifest_df.empty:
            df = manifest_df.copy()
            if split is not None and "split" in df.columns:
                df = df[df["split"] == split]
                
            manifest_map = {row["image_id"]: row for row in df.to_dict("records") if "image_id" in row}
            
            for img_id, entry in all_images.items():
                if img_id in manifest_map:
                    m_row = manifest_map[img_id]
                    entry_copy = dict(entry)
                    entry_copy["image_id"] = img_id
                    entry_copy["rel_image_path"] = m_row.get("rel_image_path")
                    entry_copy["rel_lesion_mask_path"] = m_row.get("rel_lesion_mask_path")
                    entry_copy["rel_leaf_mask_path"] = m_row.get("rel_leaf_mask_path")
                    entry_copy["split"] = m_row.get("split", "train")
                    entry_copy["source"] = m_row.get("source", "unknown")
                    entry_copy["severity"] = m_row.get("severity", 0.0)
                    entry_copy["cluster_id"] = m_row.get("cluster_id", 0)
                    self.items.append(entry_copy)
        else:
            for img_id, entry in all_images.items():
                entry_copy = dict(entry)
                entry_copy["image_id"] = img_id
                self.items.append(entry_copy)
                
        # In-memory shard cache to minimize disk reads
        self._shard_cache: Dict[str, Dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _get_shard_tensors(self, shard_name: str) -> Dict[str, torch.Tensor]:
        """Lazy load and cache safetensors shard file."""
        if shard_name not in self._shard_cache:
            shard_path = self.cache_dir / shard_name
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard file {shard_path} does not exist.")
            self._shard_cache[shard_name] = load_file(shard_path)
        return self._shard_cache[shard_name]

    def _load_resized_mask(
        self,
        rel_mask_path: Optional[str],
        grid_shape: Tuple[int, int]
    ) -> Optional[torch.Tensor]:
        """Load ground truth binary mask and resize to patch grid resolution (0/1)."""
        if not rel_mask_path:
            return None
            
        paths = get_paths()
        full_path = paths.project_root / rel_mask_path
        if not full_path.exists():
            full_path = paths.raw_data / rel_mask_path
            if not full_path.exists():
                return None
                
        try:
            with Image.open(full_path) as m_img:
                m_arr = (np.array(m_img.convert("L")) > 0).astype(np.float32)
                
            gh, gw = grid_shape
            # Resize using PIL nearest-neighbor to preserve binary labels
            m_pil = Image.fromarray((m_arr * 255).astype(np.uint8))
            m_resized = m_pil.resize((gw, gh), Image.Resampling.NEAREST)
            m_bin = (np.array(m_resized) > 0).astype(np.float32)
            return torch.from_numpy(m_bin)
        except Exception as e:
            logger.warning(f"Failed to load mask {rel_mask_path}: {e}")
            return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Fetch item by index.

        Returns:
            Dict containing image_id, cls_token, mean_patch_token, patch_tokens,
            grid_shape, optional lesion_mask, optional leaf_mask, and metadata.
        """
        item = self.items[idx]
        img_id = item["image_id"]
        shard_name = item["shard"]
        grid_shape = tuple(item["grid_shape"])  # (gh, gw)
        
        tensors = self._get_shard_tensors(shard_name)
        
        cls_token = tensors.get(f"{img_id}_cls")
        mean_patch_token = tensors.get(f"{img_id}_mean")
        patch_tokens = tensors.get(f"{img_id}_patch")
        
        if cls_token is None or patch_tokens is None:
            raise KeyError(f"Feature tensors for image '{img_id}' not found in shard '{shard_name}'.")

        # Load ground truth masks if available
        lesion_mask = self._load_resized_mask(item.get("rel_lesion_mask_path"), grid_shape)
        leaf_mask = self._load_resized_mask(item.get("rel_leaf_mask_path"), grid_shape)

        sample = {
            "image_id": img_id,
            "cls_token": cls_token,
            "mean_patch_token": mean_patch_token if mean_patch_token is not None else patch_tokens.mean(dim=(0, 1)),
            "patch_tokens": patch_tokens,
            "grid_shape": torch.tensor(grid_shape, dtype=torch.int64),
            "lesion_mask": lesion_mask if lesion_mask is not None else torch.zeros(grid_shape, dtype=torch.float32),
            "leaf_mask": leaf_mask if leaf_mask is not None else torch.ones(grid_shape, dtype=torch.float32),
            "metadata": {
                "source": item.get("source", "unknown"),
                "split": item.get("split", "train"),
                "severity": item.get("severity", 0.0),
                "cluster_id": item.get("cluster_id", 0),
                "original_size": item.get("original_size", [448, 448]),
                "padded_size": item.get("padded_size", [448, 448])
            }
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
