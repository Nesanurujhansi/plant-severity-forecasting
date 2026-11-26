"""Unit Tests for Vision Backbone Feature Extraction, Storage Estimator, OOD Stats, and Feature Dataset.

Runs on CPU using a stub transformer backbone to test spatial padding math, cache key security,
shard generation, resumption, ETH tiling, Mahalanobis OOD calculations, and lazy dataset loading.
"""

import os
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import pytest
from safetensors.torch import load_file

from ml.features.storage_estimate import (
    calculate_image_feature_size,
    estimate_storage_for_dataset,
    generate_storage_report
)
from ml.features.extract_features import (
    compute_cache_key,
    preprocess_image,
    extract_image_tiles,
    FeatureExtractorEngine,
    run_feature_extraction_pipeline
)
from ml.features.ood_stats import (
    compute_mahalanobis_distance,
    compute_ood_statistics,
    process_ood_stats
)
from ml.features.feature_dataset import CachedFeatureDataset

def test_storage_estimate_calculations():
    """Test storage estimator spatial grid math and byte footprint calculations."""
    res = calculate_image_feature_size(resolution=448, patch_size=14, embed_dim=384, dtype_bytes=2)
    assert res["grid_shape"] == (32, 32)
    assert res["n_patches"] == 1024
    
    # 1024 patches * 384 * 2 bytes + 2 * 384 * 2 + 500 overhead = 787,968 bytes
    expected_bytes = 1024 * 384 * 2 + 2 * 384 * 2 + 500
    assert res["total_bytes_per_image"] == expected_bytes
    
    est_total = estimate_storage_for_dataset(n_images=1000, backbone_name="small", resolution=448)
    assert est_total["n_images"] == 1000
    assert est_total["total_mb"] > 0
    assert est_total["total_gb"] < 1.0  # ~0.73 GB
    
    report = generate_storage_report(static_n=100, eth_n=500, available_gb=15.0)
    assert "storage_table" in report
    assert len(report["storage_table"]) == 6  # 2 backbones x 3 resolutions

def test_preprocess_image_padding_math():
    """Test aspect-ratio preserving resize and spatial padding to patch_size multiples."""
    img = Image.fromarray(np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8))
    tensor, meta = preprocess_image(img, target_resolution=448, patch_size=14)
    
    padded_h, padded_w = meta["padded_size"]
    assert padded_h % 14 == 0
    assert padded_w % 14 == 0
    assert tensor.shape == (3, padded_h, padded_w)
    assert meta["grid_shape"] == (padded_h // 14, padded_w // 14)

def test_cache_key_hashing_and_mismatch_refusal():
    """Test cache key generation and verify refusal to mix mismatched cache keys."""
    key1 = compute_cache_key("facebook/dinov2-small", "main", 448, 14, "downscaled_crop")
    key2 = compute_cache_key("facebook/dinov2-small", "main", 672, 14, "downscaled_crop")
    assert key1 != key2
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cache_dir = tmp_path / key1
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Write dummy index with key1
        with open(cache_dir / "index.json", "w") as f:
            json.dump({"cache_key": key1, "images": {}}, f)
            
        dummy_df = pd.DataFrame([{"image_id": "test_01", "rel_image_path": "dummy.jpg"}])
        
        # Attempt extraction with settings producing key2 into key1 directory
        with pytest.raises(ValueError, match="Cache key mismatch"):
            run_feature_extraction_pipeline(
                manifest_df=dummy_df,
                output_dir=cache_dir,
                resolution=672,  # Different resolution -> key2
                is_stub=True
            )

def test_stub_feature_extractor():
    """Test feature extraction shapes using stub backbone on CPU."""
    engine = FeatureExtractorEngine(backbone_id="facebook/dinov2-small", is_stub=True, device="cpu")
    dummy_input = torch.randn(1, 3, 448, 448)
    
    cls_t, mean_t, patch_grid = engine.extract_features(dummy_input, use_amp=False)
    
    assert cls_t.shape == (1, 384)
    assert mean_t.shape == (1, 384)
    assert patch_grid.shape == (1, 32, 32, 384)

def test_pipeline_extraction_and_resumption():
    """Test end-to-end feature extraction pipeline and resumable execution."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        img_dir = tmp_path / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 3 synthetic image files
        records = []
        for i in range(1, 4):
            img_name = f"sample_{i:03d}.jpg"
            img_p = img_dir / img_name
            Image.fromarray(np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)).save(img_p)
            records.append({"image_id": f"sample_{i:03d}", "rel_image_path": str(img_p)})
            
        manifest_df = pd.DataFrame(records)
        
        # Run 1: Initial extraction
        res1 = run_feature_extraction_pipeline(
            manifest_df=manifest_df,
            output_dir=tmp_path,
            resolution=448,
            shard_size=2,
            is_stub=True
        )
        
        assert res1["processed_images"] == 3
        assert res1["skipped_images"] == 0
        assert res1["total_indexed_images"] == 3
        
        # Verify index.json and shard files exist
        cache_dir = Path(res1["cache_dir"])
        assert (cache_dir / "index.json").exists()
        assert len(list(cache_dir.glob("shard_*.safetensors"))) >= 1
        
        # Run 2: Resumption (should skip all 3)
        res2 = run_feature_extraction_pipeline(
            manifest_df=manifest_df,
            output_dir=tmp_path,
            resolution=448,
            shard_size=2,
            is_stub=True
        )
        
        assert res2["processed_images"] == 0
        assert res2["skipped_images"] == 3
        assert res2["total_indexed_images"] == 3

def test_eth_tiling_mode():
    """Test sliding-window macro image tile extraction."""
    large_img = Image.fromarray(np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8))
    tiles, coords, metas = extract_image_tiles(large_img, tile_size=448, tile_overlap=56)
    
    assert len(tiles) > 1
    assert len(coords) == len(tiles)
    assert len(metas) == len(tiles)
    
    # Check bounding box bounds
    for x1, y1, x2, y2 in coords:
        assert 0 <= x1 < x2 <= 800
        assert 0 <= y1 < y2 <= 600

def test_ood_stats_mahalanobis_calculation():
    """Test covariance shrinkage and Mahalanobis distance OOD calculation."""
    np.random.seed(42)
    train_cls = np.random.randn(50, 384).astype(np.float32)
    val_cls = np.random.randn(20, 384).astype(np.float32)
    
    stats = compute_ood_statistics(train_cls, val_cls, gamma=0.01, percentiles=[95.0, 99.0])
    
    assert stats["mean_vector"].shape == (384,)
    assert stats["precision_matrix"].shape == (384, 384)
    assert "p95" in stats["percentiles"]
    assert "p99" in stats["percentiles"]
    assert stats["percentiles"]["p99"] >= stats["percentiles"]["p95"]
    
    # Test distance computation
    dists = compute_mahalanobis_distance(val_cls, stats["mean_vector"], stats["precision_matrix"])
    assert len(dists) == 20
    assert np.all(dists >= 0.0)

def test_cached_feature_dataset_loading():
    """Test lazy Dataset loading from cached safetensors shards."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        img_dir = tmp_path / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        
        records = []
        for i in range(1, 3):
            img_p = img_dir / f"test_{i:02d}.jpg"
            Image.fromarray(np.random.randint(50, 200, (120, 120, 3), dtype=np.uint8)).save(img_p)
            records.append({
                "image_id": f"test_{i:02d}",
                "rel_image_path": str(img_p),
                "split": "train",
                "source": "test_src"
            })
            
        manifest_df = pd.DataFrame(records)
        res = run_feature_extraction_pipeline(
            manifest_df=manifest_df,
            output_dir=tmp_path,
            resolution=448,
            is_stub=True
        )
        
        cache_dir = Path(res["cache_dir"])
        ds = CachedFeatureDataset(cache_dir=cache_dir, manifest_df=manifest_df, split="train")
        
        assert len(ds) == 2
        sample = ds[0]
        
        assert "image_id" in sample
        assert sample["cls_token"].shape == (384,)
        assert sample["patch_tokens"].shape == (32, 32, 384)
        assert sample["grid_shape"].tolist() == [32, 32]
        assert sample["lesion_mask"].shape == (32, 32)
        assert sample["leaf_mask"].shape == (32, 32)

def test_zero_pickle_security():
    """Verify zero pickle usage and strict safetensors format in feature caching."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        img_p = tmp_path / "sec_test.jpg"
        Image.fromarray(np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)).save(img_p)
        manifest_df = pd.DataFrame([{"image_id": "sec_01", "rel_image_path": str(img_p)}])
        
        res = run_feature_extraction_pipeline(manifest_df=manifest_df, output_dir=tmp_path, is_stub=True)
        cache_dir = Path(res["cache_dir"])
        
        # Verify no .pkl files exist
        pkl_files = list(cache_dir.glob("*.pkl")) + list(cache_dir.glob("*.pickle"))
        assert len(pkl_files) == 0
        
        # Verify shard is valid safetensors
        shards = list(cache_dir.glob("*.safetensors"))
        assert len(shards) == 1
        tensors = load_file(shards[0])
        assert "sec_01_cls" in tensors
