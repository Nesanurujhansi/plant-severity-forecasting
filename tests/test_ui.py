import io
import os
from pathlib import Path
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.ui.strings import STRINGS, get_string
from app.ui.plots import blend_mask_overlay, plot_progression_forecast, plot_scenario_comparison
from app.ui.gradio_ui import run_ui_prognosis


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DEV_STUB_MODELS", "1")
    with TestClient(app) as c:
        yield c


def create_test_image() -> Image.Image:
    return Image.new("RGB", (200, 200), color=(50, 150, 50))


def test_strings_completeness():
    assert "en" in STRINGS
    required_keys = ["app_title", "tab_analyze", "tab_results", "global_disclaimer", "privacy_statement"]
    for key in required_keys:
        val = get_string(key, lang="en")
        assert val != f"[{key}]", f"Missing UI string for key '{key}'"


def test_plot_progression_forecast():
    img = plot_progression_forecast(initial_severity=12.5, preset_name="as_recent")
    assert isinstance(img, Image.Image)
    assert img.size[0] > 0 and img.size[1] > 0


def test_plot_scenario_comparison():
    img = plot_scenario_comparison(unprotected_7d=22.5, protected_7d=13.8, relative_reduction_pct=38.7)
    assert isinstance(img, Image.Image)
    assert img.size[0] > 0 and img.size[1] > 0


def test_blend_mask_overlay():
    orig = create_test_image()
    # Dummy base64 png
    overlay_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    blended = blend_mask_overlay(orig, overlay_b64, opacity=0.5)
    assert isinstance(blended, Image.Image)
    assert blended.size == orig.size


def test_ui_callback_formatting():
    img = create_test_image()
    (
        stub_banner, safety_md, orig_out, overlay_out,
        sev_md, fc_chart, sc_chart, w_table, warn_md, timings_md
    ) = run_ui_prognosis(
        image=img,
        crop="wheat",
        weather_scenario="as_recent",
        include_treatment=True
    )
    assert any(c in safety_md for c in ["HIGH CONFIDENCE", "MEDIUM CONFIDENCE", "LOW CONFIDENCE", "ABSTAIN"])
    assert orig_out is not None
    assert sev_md != ""
    assert isinstance(w_table, list)


def test_gradio_ui_mount_smoke(client):
    # Health endpoint check
    res_health = client.get("/health")
    assert res_health.status_code == 200

    # Gradio UI route smoke check
    res_ui = client.get("/ui/", follow_redirects=True)
    assert res_ui.status_code in (200, 307)
