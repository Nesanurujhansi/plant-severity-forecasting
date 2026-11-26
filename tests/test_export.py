"""
Phase 09: Unit Tests for Model Bundle Export, Verifier, Manifest, and App Loader
"""

import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
import pytest

from ml.export.manifest import ManifestSchema, build_manifest, calculate_sha256
from ml.export.export_bundle import export_bundle
from ml.export.verify_bundle import check_library_versions, verify_bundle
from app.inference.loader import get_model_container, ModelContainer


def safe_extract_zip(zip_path: Path, target_dir: Path):
    """Safely extract zip archive preventing Zip-Slip path traversal."""
    target_dir_abs = target_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = (target_dir_abs / member.filename).resolve()
            if not str(member_path).startswith(str(target_dir_abs)):
                raise ValueError(f"Zip-Slip path traversal vulnerability detected in zip entry: '{member.filename}'")
        zf.extractall(target_dir_abs)


@pytest.fixture
def temp_export_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_manifest_schema_validation(temp_export_dir):
    test_file = temp_export_dir / "test.txt"
    test_file.write_text("hello world")
    feat_schema = temp_export_dir / "feature_schema.json"
    feat_schema.write_text('{"features": []}')

    files_map = {"test.txt": test_file}
    manifest = build_manifest(
        files_map=files_map,
        feature_schema_path=feat_schema,
        bundle_version="1.0.0"
    )
    assert manifest.bundle_version == "1.0.0"
    assert "test.txt" in manifest.files
    assert manifest.files["test.txt"].sha256 == calculate_sha256(test_file)


def test_export_and_verify_bundle_pipeline(temp_export_dir):
    output_zip = temp_export_dir / "bundle.zip"
    zip_path = export_bundle(
        data_raw_dir="data/raw",
        output_zip_path=output_zip,
        bundle_version="1.0.0"
    )
    assert zip_path.exists()

    unzip_dir = temp_export_dir / "unzipped"
    safe_extract_zip(zip_path, unzip_dir)

    passed, report = verify_bundle(unzip_dir)
    assert passed, f"Verification failed:\n" + "\n".join(report)


def test_hash_mismatch_detection(temp_export_dir):
    output_zip = temp_export_dir / "bundle.zip"
    zip_path = export_bundle(data_raw_dir="data/raw", output_zip_path=output_zip)

    unzip_dir = temp_export_dir / "unzipped"
    safe_extract_zip(zip_path, unzip_dir)

    # Corrupt one file
    corrupt_file = unzip_dir / "severity" / "config.json"
    corrupt_file.write_text('{"corrupted": true}')

    passed, report = verify_bundle(unzip_dir)
    assert not passed
    assert any("hash mismatch" in line.lower() or "size mismatch" in line.lower() for line in report)


def test_missing_file_detection(temp_export_dir):
    output_zip = temp_export_dir / "bundle.zip"
    zip_path = export_bundle(data_raw_dir="data/raw", output_zip_path=output_zip)

    unzip_dir = temp_export_dir / "unzipped"
    safe_extract_zip(zip_path, unzip_dir)

    # Remove a required file
    (unzip_dir / "safety" / "safety_config.json").unlink()

    passed, report = verify_bundle(unzip_dir)
    assert not passed
    assert any("missing required file" in line.lower() for line in report)


def test_library_version_checks():
    manifest_versions = {
        "torch": "99.0.0",  # Mismatch major version
        "numpy": "1.0.0"
    }
    fails, warns = check_library_versions(manifest_versions)
    assert len(fails) > 0
    assert any("Major version mismatch" in f for f in fails)


def test_golden_output_drift_failure(temp_export_dir):
    output_zip = temp_export_dir / "bundle.zip"
    zip_path = export_bundle(data_raw_dir="data/raw", output_zip_path=output_zip)

    unzip_dir = temp_export_dir / "unzipped"
    safe_extract_zip(zip_path, unzip_dir)

    # Tamper with golden expected outputs
    golden_json_p = unzip_dir / "golden" / "expected_outputs.json"
    with open(golden_json_p, "r") as f:
        gdata = json.load(f)
    # Alter expected value dramatically
    gdata["test_cases"]["golden_leaf_01"]["expected"]["severity_pct"] = 999.9
    with open(golden_json_p, "w") as f:
        json.dump(gdata, f, indent=2)

    passed, report = verify_bundle(unzip_dir)
    assert not passed
    assert any("drift" in line.lower() for line in report)


def test_app_loader_stub_mode():
    container = get_model_container(dev_stub=True, force_reload=True)
    assert container.is_stub
    assert container.stub_banner is not None
    assert "DEV_STUB_MODELS ACTIVE" in container.stub_banner

    res = container.severity_predictor.predict(None)
    assert res["stub"] is True
    assert res["severity_pct"] == 12.5


def test_app_loader_verified_mode(temp_export_dir):
    output_zip = temp_export_dir / "bundle.zip"
    zip_path = export_bundle(data_raw_dir="data/raw", output_zip_path=output_zip)

    unzip_dir = temp_export_dir / "unzipped"
    safe_extract_zip(zip_path, unzip_dir)

    container = get_model_container(bundle_dir=unzip_dir, dev_stub=False, force_reload=True)
    assert not container.is_stub
    info = container.get_bundle_info()
    assert info["version"] == "1.0.0"
    assert "disclaimer" in info


def test_zip_slip_prevention(temp_export_dir):
    malicious_zip = temp_export_dir / "malicious.zip"
    with zipfile.ZipFile(malicious_zip, "w") as zf:
        zf.writestr("../evil.txt", "evil payload")

    extract_target = temp_export_dir / "target"
    extract_target.mkdir()

    with pytest.raises(ValueError, match="Zip-Slip"):
        safe_extract_zip(malicious_zip, extract_target)
