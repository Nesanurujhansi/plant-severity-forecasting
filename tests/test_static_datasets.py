"""Unit tests for Phase 03 static dataset pipeline modules."""

import pytest
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

from ml.data.build_static_manifest import validate_mask_integrity
from ml.data.static_severity import calculate_severity_from_masks
from ml.data.static_splits import create_static_splits, verify_static_zero_leakage
from ml.data.make_leaf_pseudolabels_sam2 import compute_stub_leaf_mask

def test_validate_mask_integrity(tmp_path: Path):
    """Test mask validator on matching vs mismatched dimensions and binary values."""
    img_path = tmp_path / "test_img.jpg"
    mask_path = tmp_path / "test_mask.png"

    # Create matching 50x50 RGB image and 50x50 binary mask
    img = Image.fromarray(np.ones((50, 50, 3), dtype=np.uint8) * 100)
    mask = Image.fromarray(np.zeros((50, 50), dtype=np.uint8))
    img.save(img_path)
    mask.save(mask_path)

    valid, msg = validate_mask_integrity(img_path, mask_path)
    assert valid is True
    assert msg == "Valid"

    # Create dimension mismatch mask (40x40)
    mismatch_mask_path = tmp_path / "mismatch_mask.png"
    Image.fromarray(np.zeros((40, 40), dtype=np.uint8)).save(mismatch_mask_path)

    valid_m, msg_m = validate_mask_integrity(img_path, mismatch_mask_path)
    assert valid_m is False
    assert "Dimension mismatch" in msg_m

def test_calculate_severity_from_masks_union_rule():
    """Test severity calculation with union rule (lesion contained in leaf)."""
    # Create 100x100 toy arrays
    lesion_mask = np.zeros((100, 100), dtype=np.uint8)
    leaf_mask = np.zeros((100, 100), dtype=np.uint8)

    # 1000 leaf pixels, 50 lesion pixels strictly inside leaf
    leaf_mask[10:50, 10:35] = 1 # 40 * 25 = 1000
    lesion_mask[10:20, 10:15] = 1 # 10 * 5 = 50

    sev, bin_idx, label = calculate_severity_from_masks(lesion_mask, leaf_mask)
    assert sev == pytest.approx(0.05, abs=1e-3)
    assert bin_idx in [1, 2] # 5% boundary

    # Test lesion partly outside leaf -> union rule absorbs lesion into leaf
    lesion_outside = np.zeros((100, 100), dtype=np.uint8)
    lesion_outside[0:10, 0:10] = 1 # 100 pixels outside
    sev_u, _, _ = calculate_severity_from_masks(lesion_outside, leaf_mask)
    assert 0.0 < sev_u <= 1.0

def test_create_static_splits_zero_leakage():
    """Test static split generator and zero near-duplicate cluster leakage assertion."""
    records = []
    for i in range(20):
        records.append({
            "image_id": f"img_{i}",
            "source": "plantseg",
            "cluster_id": i // 2, # 10 clusters (2 images per cluster)
            "is_external_eval": False
        })

    df = pd.DataFrame(records)
    splits_df, meta = create_static_splits(df, seed=42)

    assert "split" in splits_df.columns
    assert verify_static_zero_leakage(splits_df) is True

def test_sam2_stub_leaf_generator():
    """Test CPU stub leaf generator returns valid binary array containing lesion."""
    img = Image.fromarray(np.ones((60, 60, 3), dtype=np.uint8) * 120)
    lesion = Image.fromarray(np.zeros((60, 60), dtype=np.uint8))
    lesion_arr = np.array(lesion)
    lesion_arr[20:30, 20:30] = 1
    lesion = Image.fromarray(lesion_arr)

    leaf_arr = compute_stub_leaf_mask(img, lesion)
    assert leaf_arr.shape == (60, 60)
    assert np.all(leaf_arr[20:30, 20:30] == 1) # Invariant: lesion contained inside leaf
