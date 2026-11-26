"""Unit tests for sequence image pair generation."""

import pandas as pd
from ml.data.make_pairs import generate_pairs_from_images

def test_generate_pairs_from_images():
    dates = pd.date_range("2023-05-01", "2023-05-15", freq="2D")
    records = []
    for idx, d in enumerate(dates):
        records.append({
            "image_id": f"img_{idx}",
            "sample_id": "sample_001",
            "timestamp": d,
            "severity": 0.05 + idx * 0.02,
            "n_lesions": idx + 1,
            "mean_lesion_area": 5.0,
            "cultivar": "Arina",
            "treatment": "FI2",
            "batch": 1,
            "year": 2023
        })

    images_df = pd.DataFrame(records)
    weather_df = pd.DataFrame({
        "date": pd.date_range("2023-05-01", "2023-05-20", freq="D"),
        "t_mean": 18.0,
        "rh_mean": 75.0
    })

    pairs_df = generate_pairs_from_images(images_df, weather_df, min_dt_days=1.0, max_dt_days=10.0)

    assert not pairs_df.empty
    assert (pairs_df["dt_days"] >= 1.0).all()
    assert (pairs_df["dt_days"] <= 10.0).all()
    assert "sample_weight" in pairs_df.columns
    # Sum of sample_weight should equal 1.0 per sample
    assert abs(pairs_df["sample_weight"].sum() - 1.0) < 1e-3
