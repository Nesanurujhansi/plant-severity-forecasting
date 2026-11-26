"""Unit tests for Multi-Crop & Indian Agriculture Pipeline (Phase 07)."""

import pytest
from pathlib import Path
import pandas as pd
from ml.data.download_riceseg import download_riceseg
from ml.data.download_indian_crops import download_indian_crops
from ml.forecast.crop_physics import get_crop_thermal_profile, CROP_PHYSICS_REGISTRY, CropThermalProfile
from ml.data.build_multicrop_manifest import process_multicrop_manifest
from ml.data.make_multicrop_pairs import generate_multicrop_pairs, INDIAN_AGRO_CLIMATE_ZONES


def test_download_riceseg_dry_run() -> None:
    """Test RiceSeg downloader dry-run execution."""
    download_riceseg(dry_run=True)


def test_download_indian_crops_dry_run() -> None:
    """Test Indian cash crop downloader dry-run execution."""
    download_indian_crops(dry_run=True)


def test_crop_thermal_physics_registry() -> None:
    """Test species thermal profile registry for all 6 crops."""
    expected_crops = ["wheat", "rice", "tomato", "corn", "sugarcane", "cotton"]
    for crop in expected_crops:
        profile = get_crop_thermal_profile(crop)
        assert isinstance(profile, CropThermalProfile)
        assert profile.t_min < profile.t_opt < profile.t_max
        assert 0.0 <= profile.yan_hunt_thermal_response(profile.t_opt) <= 1.0
        assert profile.yan_hunt_thermal_response(profile.t_min - 5.0) == 0.0
        assert profile.yan_hunt_thermal_response(profile.t_max + 5.0) == 0.0


def test_multicrop_manifest_builder(tmp_path: Path) -> None:
    """Test multi-crop manifest builder."""
    df_manifest = process_multicrop_manifest(raw_dir=tmp_path / "raw", processed_dir=tmp_path / "multicrop")
    assert isinstance(df_manifest, pd.DataFrame)
    assert (tmp_path / "multicrop" / "manifest.parquet").exists()


def test_generate_multicrop_pairs(tmp_path: Path) -> None:
    """Test multi-crop sequence pair generation."""
    out_pairs = tmp_path / "pairs.parquet"
    df_pairs = generate_multicrop_pairs(output_path=out_pairs, synthetic_pair_count=50)

    assert isinstance(df_pairs, pd.DataFrame)
    assert len(df_pairs) == 50
    assert out_pairs.exists()
    assert "crop_key" in df_pairs.columns
    assert "agro_climate_zone" in df_pairs.columns
    assert "s0_severity" in df_pairs.columns
    assert "st_severity" in df_pairs.columns

    # Verify severity bounds
    assert (df_pairs["s0_severity"] >= 0.0).all()
    assert (df_pairs["st_severity"] <= 1.0).all()
