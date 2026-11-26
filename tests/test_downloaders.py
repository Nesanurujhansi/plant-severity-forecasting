"""Unit tests for dataset downloader scripts dry-run execution."""

from ml.data.download_eth_wheat import download_eth_wheat, validate_eth_folder_structure
from ml.data.download_plantseg import download_plantseg
from ml.data.download_coffee_rust import download_coffee_rust
from ml.data.download_leafnet_masks import download_leafnet_masks
from ml.data.download_plantvillage import download_plantvillage

def test_downloader_dry_runs():
    # Dry runs should execute cleanly without making network requests or downloading files
    download_eth_wheat(dry_run=True, raw=False)
    download_plantseg(dry_run=True)
    download_coffee_rust(dry_run=True)
    download_leafnet_masks(dry_run=True)
    download_plantvillage(dry_run=True)
    assert True

def test_eth_validation_missing_folder(tmp_path):
    res = validate_eth_folder_structure(tmp_path)
    assert res["status"] in ["MISSING", "PARTIAL_OR_MISSING"]
    assert res["found_images"] == 0
