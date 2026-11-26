"""Unit tests for ETH wheat metadata parsing and severity calculation."""

import numpy as np
import pandas as pd
from ml.data.eth_wheat import parse_image_id, compute_mask_metrics

def test_parse_image_id():
    stem = "20230515_103000_ESWW001_L1"
    parsed = parse_image_id(stem)
    assert parsed["image_id"] == stem
    assert parsed["plot_id"] == "ESWW001"
    assert parsed["leaf_no"] == "L1"
    assert parsed["sample_id"] == "ESWW001_L1"
    assert parsed["timestamp"] == pd.Timestamp("2023-05-15 10:30:00")

def test_compute_mask_metrics_exact():
    # Create 100-pixel leaf with known mask pixels:
    # 80 healthy (1), 10 chlorosis (2), 10 necrosis (3) -> 100 total leaf pixels
    mask = np.ones((10, 10), dtype=np.uint8) # 100 healthy
    mask[0, :] = 3 # 10 pixels of necrosis

    metrics = compute_mask_metrics(mask)
    assert metrics["diseased_pixels"] == 10
    assert metrics["leaf_pixels_frame"] == 100
    assert metrics["necrosis_pixels"] == 10
    assert metrics["pustule_pixels"] == 0
    assert metrics["n_lesions"] == 1
    assert metrics["mean_lesion_area"] == 10.0
