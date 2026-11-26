"""Unit tests verifying zero sample leakage across evaluation splits."""

import pandas as pd
from ml.data.splits import create_leak_free_splits, verify_zero_leakage

def test_create_leak_free_splits_and_zero_leakage():
    records = []
    cultivars = ["Arina", "CH_Claro", "Titan", "Runal", "Forel"]
    for i in range(20):
        records.append({
            "sample_id": f"sample_{i:02d}",
            "cultivar": cultivars[i % len(cultivars)],
            "batch": 1 if i < 10 else 2,
            "year": 2022 if i < 10 else 2023,
            "n_images": 5,
            "span_days": 10.0
        })

    sequences_df = pd.DataFrame(records)
    splits = create_leak_free_splits(sequences_df, n_folds=3, cal_fraction=0.20, seed=42)

    assert verify_zero_leakage(splits)

def test_leakage_detector_fails_on_overlap():
    bad_splits = {
        "leave_cultivar_out": [
            {
                "fold": 0,
                "train_samples": ["s1", "s2"],
                "calibration_samples": ["s3"],
                "test_samples": ["s2", "s4"] # Overlap "s2" in train and test!
            }
        ]
    }
    assert not verify_zero_leakage(bad_splits)
