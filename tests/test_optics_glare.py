"""Unit tests for Phase 01: Optical Normalization, Glare Attenuation, and Metric Homography."""

import pytest
import numpy as np
from PIL import Image, ImageDraw
import torch

from ml.safety.quality import check_specular_glare, suppress_specular_glare, evaluate_image_quality
from ml.severity.homography import detect_circular_fiducial, compute_metric_areas, MetricScaleResult
from ml.severity.heads import MultiScaleFPNHead


def test_specular_glare_detection_and_suppression():
    # Base green leaf canvas
    img = np.full((300, 300, 3), [40, 140, 40], dtype=np.uint8)

    # 1. Clean leaf should pass specular glare check
    res_clean = check_specular_glare(img, max_glare_fraction=0.05)
    assert res_clean["passed"] is True
    assert res_clean["glare_fraction"] == 0.0

    # 2. Add blown-out white specular glare patch (V > 245, S < 0.1) occupying ~12% of image
    img_glare = img.copy()
    img_glare[50:160, 50:160] = [252, 253, 254]

    res_glare = check_specular_glare(img_glare, max_glare_fraction=0.05)
    assert res_glare["passed"] is False
    assert res_glare["glare_fraction"] > 0.08
    assert "Specular sunlight glare detected" in res_glare["message"]

    # 3. Suppress glare
    attenuated, mask = suppress_specular_glare(img_glare)
    assert np.any(mask)
    assert attenuated.shape == img_glare.shape
    # Glare pixels should be blended towards background/median color
    assert np.mean(attenuated[mask]) < 240


def test_evaluate_image_quality_includes_glare():
    # Normal leaf photo
    img = np.full((250, 250, 3), [50, 150, 50], dtype=np.uint8)
    res = evaluate_image_quality(img)
    assert "specular_glare" in res["details"]
    assert res["details"]["specular_glare"]["passed"] is True

    # Blown out image with heavy glare
    img_heavy_glare = img.copy()
    img_heavy_glare[20:180, 20:180] = [255, 255, 255]
    res_bad = evaluate_image_quality(img_heavy_glare)
    assert res_bad["details"]["specular_glare"]["passed"] is False


def test_circular_fiducial_detection_and_metric_scaling():
    # Canvas with dark background
    pil_img = Image.new("RGB", (400, 400), color=(30, 30, 30))
    draw = ImageDraw.Draw(pil_img)

    # Draw simulated leaf
    draw.polygon([(100, 50), (350, 200), (300, 350), (100, 350), (50, 200)], fill=(40, 150, 40))

    # Draw circular reference marker: diameter = 60 px (radius = 30 px) at (60, 60)
    # With known diameter = 20 mm, scale is exactly 60 / 20 = 3.0 px/mm
    cx, cy, r = 60, 60, 30
    draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=(240, 240, 240), outline=(20, 20, 20), width=2)

    res = detect_circular_fiducial(pil_img, known_diameter_mm=20.0)
    assert isinstance(res, MetricScaleResult)
    assert res.is_calibrated is True
    assert res.scale_px_per_mm is not None
    assert 2.7 <= res.scale_px_per_mm <= 3.3  # Expected ~3.0 px/mm within contour tolerance

    # Convert pixel areas: 9000 leaf pixels at 3.0 px/mm => 9000 / 9.0 = 1000 mm²
    leaf_mm2, lesion_mm2 = compute_metric_areas(
        leaf_pixel_count=9000,
        lesion_pixel_count=900,
        scale_px_per_mm=3.0
    )
    assert leaf_mm2 == 1000.0
    assert lesion_mm2 == 100.0


def test_uncalibrated_image_fallback():
    # Image without any calibration tag
    clean_img = np.full((300, 300, 3), [50, 120, 50], dtype=np.uint8)
    res = detect_circular_fiducial(clean_img)
    assert res.is_calibrated is False
    assert res.scale_px_per_mm is None

    leaf_mm2, lesion_mm2 = compute_metric_areas(
        leaf_pixel_count=5000,
        lesion_pixel_count=500,
        scale_px_per_mm=None
    )
    assert leaf_mm2 is None
    assert lesion_mm2 is None


def test_multiscale_fpn_head():
    head = MultiScaleFPNHead(deep_dim=384, stem_dim=64, fpn_dim=64)

    # 1. Forward with both deep ViT tokens and shallow stem features
    deep_tokens = torch.randn(2, 32, 32, 384)
    stem_feats = torch.randn(2, 64, 112, 112)

    out = head(deep_tokens, stem_features=stem_feats, target_size=(448, 448))
    assert "lesion_logits" in out
    assert "leaf_logits" in out
    assert out["lesion_logits"].shape == (2, 1, 448, 448)
    assert out["leaf_logits"].shape == (2, 1, 448, 448)

    # 2. Forward with deep ViT tokens only (stem features omitted)
    out_deep_only = head(deep_tokens, stem_features=None, target_size=(448, 448))
    assert out_deep_only["lesion_logits"].shape == (2, 1, 448, 448)
    assert out_deep_only["leaf_logits"].shape == (2, 1, 448, 448)
