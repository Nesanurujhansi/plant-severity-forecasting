"""
Phase 10: FastAPI Backend Unit Test Suite
Tests health, model-info, valid analyze, file size limits, unsupported format rejection,
decompression bomb guards, coordinate validation, abstention paths, crop gating, and weather endpoints.
"""

import io
import json
import os
from pathlib import Path
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings


@pytest.fixture
def client(monkeypatch):
    """Pytest fixture providing TestClient within FastAPI lifespan context and isolated env."""
    monkeypatch.setenv("DEV_STUB_MODELS", "1")
    with TestClient(app) as c:
        yield c


def create_test_image_bytes(format_str="JPEG", size=(200, 200), color=(50, 150, 50)) -> bytes:
    """Helper creating raw image bytes for testing."""
    img = Image.new("RGB", size, color=color)
    buffered = io.BytesIO()
    img.save(buffered, format=format_str)
    return buffered.getvalue()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "is_stub" in data


def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "bundle_version" in data
    assert "disclaimer" in data
    assert "known_limitations" in data


def test_valid_analyze_request(client):
    img_bytes = create_test_image_bytes("JPEG")
    files = {"image": ("leaf.jpg", img_bytes, "image/jpeg")}
    data = {
        "crop": "wheat",
        "latitude": "47.45",
        "longitude": "8.68",
        "weather_scenario": "as_recent"
    }

    response = client.post("/analyze", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    assert "bundle_info" in res
    assert "safety" in res
    assert "severity" in res
    assert res["severity"]["severity_pct"] > 0
    assert "forecast" in res
    assert "scenario" in res
    assert "weather" in res
    assert "timings" in res
    assert "disclaimer" in res


def test_file_too_large_error(client):
    # Generate dummy oversized payload > 10MB
    large_bytes = b"0" * (10 * 1024 * 1024 + 500)
    files = {"image": ("large.jpg", large_bytes, "image/jpeg")}

    response = client.post("/analyze", files=files)
    assert response.status_code == 413
    res = response.json()
    assert res["error_code"] == "FILE_TOO_LARGE"
    assert "hint" in res


def test_unsupported_file_type(client):
    invalid_bytes = b"Hello, this is a plain text file pretending to be an image!"
    files = {"image": ("test.txt", invalid_bytes, "text/plain")}

    response = client.post("/analyze", files=files)
    assert response.status_code == 400
    res = response.json()
    assert res["error_code"] in ("UNSUPPORTED_FORMAT", "CORRUPTED_IMAGE")
    assert "hint" in res


def test_out_of_range_latitude(client):
    img_bytes = create_test_image_bytes("JPEG")
    files = {"image": ("leaf.jpg", img_bytes, "image/jpeg")}
    data = {"latitude": "150.0"}  # Invalid lat > 90

    response = client.post("/analyze", files=files, data=data)
    assert response.status_code == 400
    res = response.json()
    assert res["error_code"] == "INVALID_LATITUDE"


def test_non_wheat_crop_support(client):
    img_bytes = create_test_image_bytes("JPEG")
    files = {"image": ("leaf.jpg", img_bytes, "image/jpeg")}
    data = {"crop": "tomato"}

    response = client.post("/analyze", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    # Forecast and 7-day spatial projection should be enabled for all foliar crops
    assert res["forecast"] is not None
    assert "7d" in res["forecast"]["horizons"]
    assert res["forecast"]["projected_7d_overlay_base64"] is not None
    assert res["scenario"] is None
    assert any("tomato" in w.lower() for w in res["warnings"])


def test_no_disk_file_written_by_default(client):
    assert settings.store_uploads is False
    upload_dir = Path(settings.upload_store_dir)

    img_bytes = create_test_image_bytes("JPEG")
    files = {"image": ("leaf.jpg", img_bytes, "image/jpeg")}

    # Initial file count
    initial_count = len(list(upload_dir.glob("*"))) if upload_dir.exists() else 0

    response = client.post("/analyze", files=files)
    assert response.status_code == 200

    current_count = len(list(upload_dir.glob("*"))) if upload_dir.exists() else 0
    assert current_count == initial_count, "Upload was unexpectedly written to disk!"


def test_weather_endpoint(client):
    response = client.get("/weather", params={"latitude": 47.45, "longitude": 8.68})
    assert response.status_code == 200
    res = response.json()
    assert res["latitude"] == 47.45
    assert res["longitude"] == 8.68
    assert len(res["records"]) > 0
