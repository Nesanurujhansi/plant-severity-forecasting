"""Unit tests for dataset download validation using temporary fake directory trees."""

import tempfile
from pathlib import Path
from ml.data.validate_downloads import scan_dataset, generate_validation_reports

def test_scan_dataset_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_root = Path(tmpdir)
        config = {
            "name": "test_dataset",
            "subpath": "test_dataset",
            "licence": "MIT",
            "required_subdirs": ["meta"]
        }
        res = scan_dataset(fake_root, config)
        assert res["status"] == "MISSING"
        assert res["total_files"] == 0

def test_scan_dataset_present():
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_root = Path(tmpdir)
        ds_dir = fake_root / "raw" / "test_dataset"
        (ds_dir / "meta").mkdir(parents=True)
        (ds_dir / "img1.jpg").write_text("fake_image_bytes")
        (ds_dir / "mask1_mask.png").write_text("fake_mask_bytes")

        config = {
            "name": "test_dataset",
            "subpath": "test_dataset",
            "licence": "MIT",
            "required_subdirs": ["meta"]
        }
        res = scan_dataset(fake_root, config)
        assert res["status"] == "OK"
        assert res["total_files"] == 2
        assert res["image_count"] == 1
        assert res["mask_count"] == 1

def test_generate_validation_reports(tmp_path):
    out_dir = tmp_path / "interim"
    summary = generate_validation_reports(tmp_path, out_dir)
    assert "datasets" in summary
    assert (out_dir / "download_validation.json").exists()
    assert (out_dir / "download_validation.md").exists()
