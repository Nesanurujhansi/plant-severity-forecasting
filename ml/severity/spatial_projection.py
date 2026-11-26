"""Foliar Spatial Projection & Spore Dispersion Frontier Engine.

Forward-simulates the physical spatial expansion and coalescence of foliar lesions
from Day 0 to Day 7 based on biophysical dispersal decay kernels and exact quantile-matching
calibration against the 7-day epidemiological progression forecast (S_7).
"""

import io
import base64
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import distance_transform_edt
from ml.utils.logging_utils import get_logger

logger = get_logger(__name__)


def compute_spore_dispersion_field(
    leaf_mask: np.ndarray,
    necrotic_mask: np.ndarray,
    chlorotic_mask: np.ndarray,
    rh_mean: float = 75.0,
    precip_total: float = 0.0
) -> np.ndarray:
    """Compute continuous pathogen colonization & dispersion pressure field.

    Args:
        leaf_mask: Binary foliar envelope (H, W).
        necrotic_mask: Binary necrotic core mask (H, W).
        chlorotic_mask: Binary chlorotic margin mask (H, W).
        rh_mean: Mean relative humidity over the forecast interval.
        precip_total: Accumulated precipitation in mm.

    Returns:
        np.ndarray: Normalized dispersion field Phi(x, y) in [0, 1] bounded to leaf.
    """
    h, w = leaf_mask.shape
    leaf_bin = (leaf_mask > 0).astype(np.uint8)
    nec_bin = ((necrotic_mask > 0) & (leaf_bin > 0)).astype(np.uint8)
    chl_bin = ((chlorotic_mask > 0) & (leaf_bin > 0)).astype(np.uint8)
    active_lesions = ((nec_bin > 0) | (chl_bin > 0)).astype(np.uint8)

    if np.sum(active_lesions) == 0:
        return np.zeros((h, w), dtype=np.float32)

    # 1. Microclimate-dependent dispersal radius sigma_eff
    # Higher humidity and rainfall accelerate hyphal advance and splash dispersal
    rh_factor = 1.0 + max(0.0, (rh_mean - 70.0) / 30.0) * 0.75
    precip_factor = 1.0 + min(1.0, precip_total / 5.0) * 0.40
    base_sigma = max(8.0, min(h, w) * 0.025)
    sigma_eff = base_sigma * rh_factor * precip_factor

    # 2. Euclidean distance transform from active lesion boundaries
    # Distances inside lesions are 0; distances outside increase smoothly
    dist_from_lesions = distance_transform_edt(active_lesions == 0)

    # 3. Gaussian exponential decay kernel from lesion foci
    # Weight necrotic cores higher (1.0) and chlorotic margins slightly lower (0.75)
    field = np.exp(-(dist_from_lesions ** 2) / (2.0 * (sigma_eff ** 2)))

    # Force points strictly inside active lesions to peak potential 1.0
    field[nec_bin > 0] = 1.0
    field[chl_bin > 0] = np.maximum(field[chl_bin > 0], 0.85)

    # Restrict strictly to vegetative leaf lamina (0 outside leaf)
    field[leaf_bin == 0] = 0.0

    return field.astype(np.float32)


def calibrate_projected_frontier(
    dispersion_field: np.ndarray,
    leaf_mask: np.ndarray,
    target_severity_pct: float
) -> np.ndarray:
    """Solve for exact scalar threshold theta_7 matching forecasted severity S_7.

    Args:
        dispersion_field: Normalized dispersion potential field (H, W).
        leaf_mask: Binary foliar envelope (H, W).
        target_severity_pct: Forecasted 7-day foliar severity percentage (e.g. 28.5%).

    Returns:
        np.ndarray: Binary mask of projected 7-day lesion area.
    """
    leaf_bin = (leaf_mask > 0).astype(np.uint8)
    leaf_pixels = int(np.sum(leaf_bin))

    if leaf_pixels == 0:
        return np.zeros_like(leaf_bin)

    target_fraction = float(np.clip(target_severity_pct / 100.0, 0.0, 0.95))
    target_pixel_count = int(round(target_fraction * leaf_pixels))

    foliar_vals = dispersion_field[leaf_bin > 0]
    if len(foliar_vals) == 0:
        return np.zeros_like(leaf_bin)

    # Binary search for threshold matching target_pixel_count
    low, high = 0.0, 1.0
    best_thresh = 0.5
    min_diff = float("inf")

    for _ in range(25):
        mid = (low + high) / 2.0
        cnt = int(np.sum(foliar_vals >= mid))
        diff = abs(cnt - target_pixel_count)

        if diff < min_diff:
            min_diff = diff
            best_thresh = mid

        if cnt > target_pixel_count:
            # mid is too low (included too many pixels), increase threshold
            low = mid
        else:
            # mid is too high, decrease threshold
            high = mid

    projected_mask = ((dispersion_field >= best_thresh) & (leaf_bin > 0)).astype(np.uint8)
    return projected_mask


def render_spatial_projection_overlay(
    image: Image.Image,
    leaf_mask: np.ndarray,
    necrotic_mask: np.ndarray,
    chlorotic_mask: np.ndarray,
    target_severity_pct: float,
    rh_mean: float = 75.0,
    precip_total: float = 0.0
) -> str:
    """Generate high-contrast 7-Day Projected Lesion Expansion Frontier RGBA overlay.

    Highlights:
      1. Projected 7-Day Expansion Zone: Translucent Violet/Coral (#9d4edd) with sharp Electric Cyan (#00f0ff) boundary.
      2. Observed Day 0 Chlorotic Margin: Translucent Golden Amber (#e9b44c).
      3. Observed Day 0 Necrotic Core: Translucent Rust Red (#bc4749) with crimson boundary.
      4. Foliar Vegetative Envelope: Deep Forest Green (#386641) boundary.

    Args:
        image: Original foliar PIL Image.
        leaf_mask: Vegetative leaf envelope numpy array.
        necrotic_mask: Observed necrotic core numpy array.
        chlorotic_mask: Observed chlorotic margin numpy array.
        target_severity_pct: Forecasted 7-day severity percentage.
        rh_mean: Mean relative humidity.
        precip_total: Accumulated precipitation.

    Returns:
        Base64-encoded PNG image string of the transparent RGBA overlay.
    """
    w, h = image.size

    # Handle None or empty masks safely
    if leaf_mask is None:
        leaf_arr = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(leaf_arr, (w // 2, h // 2), (int(w * 0.4), int(h * 0.4)), 0, 0, 360, 255, -1)
    else:
        leaf_arr = np.array(leaf_mask, dtype=np.uint8)

    if necrotic_mask is None:
        nec_arr = np.zeros((h, w), dtype=np.uint8)
        if target_severity_pct > 0:
            cv2.circle(nec_arr, (w // 2, h // 2), max(2, int(min(w, h) * 0.05)), 255, -1)
    else:
        nec_arr = np.array(necrotic_mask, dtype=np.uint8)

    if chlorotic_mask is None:
        chl_arr = np.zeros((h, w), dtype=np.uint8)
        if target_severity_pct > 0:
            cv2.circle(chl_arr, (w // 2, h // 2), max(4, int(min(w, h) * 0.08)), 255, -1)
    else:
        chl_arr = np.array(chlorotic_mask, dtype=np.uint8)

    if leaf_arr.shape != (h, w):
        leaf_arr = cv2.resize(leaf_arr, (w, h), interpolation=cv2.INTER_NEAREST)
    if nec_arr.shape != (h, w):
        nec_arr = cv2.resize(nec_arr, (w, h), interpolation=cv2.INTER_NEAREST)
    if chl_arr.shape != (h, w):
        chl_arr = cv2.resize(chl_arr, (w, h), interpolation=cv2.INTER_NEAREST)

    leaf_bin = (leaf_arr > 0).astype(np.uint8)
    nec_bin = ((nec_arr > 0) & (leaf_bin > 0)).astype(np.uint8)
    chl_bin = ((chl_arr > 0) & (leaf_bin > 0) & (nec_bin == 0)).astype(np.uint8)

    rh_val = float(rh_mean) if rh_mean is not None else 75.0
    precip_val = float(precip_total) if precip_total is not None else 0.0

    # 1. Compute biophysical dispersion field & calibrate 7-day projected frontier
    field = compute_spore_dispersion_field(leaf_bin, nec_bin, chl_bin, rh_mean=rh_val, precip_total=precip_val)
    projected_7d_mask = calibrate_projected_frontier(field, leaf_bin, target_severity_pct=target_severity_pct)

    # Expansion zone is newly affected tissue beyond current active lesions
    current_lesions = ((nec_bin > 0) | (chl_bin > 0)).astype(np.uint8)
    expansion_new = ((projected_7d_mask > 0) & (current_lesions == 0) & (leaf_bin > 0)).astype(np.uint8)

    # 2. Build RGBA Overlay
    overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # A. Projected 7-Day Expansion Zone: Translucent Violet (#9d4edd: 157, 78, 221, alpha=130)
    if np.any(expansion_new > 0):
        overlay_rgba[expansion_new > 0] = [157, 78, 221, 130]
        # Draw high-contrast Electric Cyan outer frontier contour (#00f0ff: 0, 240, 255)
        contours_proj, _ = cv2.findContours(projected_7d_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_rgba, contours_proj, -1, (0, 240, 255, 255), 2)

    # B. Observed Day 0 Chlorotic Halo: Translucent Golden Amber (#e9b44c: 233, 180, 76, alpha=200)
    if np.any(chl_bin > 0):
        overlay_rgba[chl_bin > 0] = [233, 180, 76, 200]
        contours_chl, _ = cv2.findContours(chl_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_rgba, contours_chl, -1, (218, 165, 32, 255), 2)

    # C. Observed Day 0 Necrotic Core: Translucent Rust Red (#bc4749: 188, 71, 73, alpha=215)
    if np.any(nec_bin > 0):
        overlay_rgba[nec_bin > 0] = [188, 71, 73, 215]
        contours_nec, _ = cv2.findContours(nec_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_rgba, contours_nec, -1, (220, 50, 50, 255), 2)

    # D. Foliar envelope boundary: Deep Forest Green (#386641: 56, 102, 65)
    if np.any(leaf_bin > 0):
        contours_leaf, _ = cv2.findContours(leaf_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_rgba, contours_leaf, -1, (56, 102, 65, 255), 2)

    overlay_img = Image.fromarray(overlay_rgba, mode="RGBA")
    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def render_treated_reduction_overlay(
    image: Image.Image,
    leaf_mask: np.ndarray,
    necrotic_mask: np.ndarray,
    chlorotic_mask: np.ndarray,
    unprotected_7d_pct: float,
    protected_7d_pct: float,
    rh_mean: float = 75.0,
    precip_total: float = 0.0
) -> str:
    """Generate 7-Day Treated Spread Reduction RGBA overlay showing leaf area saved by spraying under favorable conditions.

    Highlights:
      1. Outer Unchecked 7-Day Spread Frontier (Reference Outline): High-contrast Electric Cyan (#00f0ff)
         contour marking where the disease would have expanded over 7 days if left untreated.
      2. Prevented Spread Zone (Saved Foliar Area): High-contrast Neon Magenta (#ff2e93: 255, 46, 147, alpha=215)
         fill representing foliar lamina rescued from pathogen hyphal expansion.
      3. Arrested Lesion Frontier (Under Protection): Bright Sky Blue (#38bdf8: 56, 189, 248) boundary
         contour showing the contracted lesion extent under fungicide suppression.
      4. Observed Day 0 Chlorotic Margin: Translucent Golden Amber (#e9b44c: 233, 180, 76, alpha=200).
      5. Observed Day 0 Necrotic Core: Translucent Rust Red (#bc4749: 188, 71, 73, alpha=215).
      6. Foliar Vegetative Envelope: Deep Forest Green (#386641: 56, 102, 65) boundary.
    """
    w, h = image.size

    if leaf_mask is None:
        leaf_arr = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(leaf_arr, (w // 2, h // 2), (int(w * 0.4), int(h * 0.4)), 0, 0, 360, 255, -1)
    else:
        leaf_arr = np.array(leaf_mask, dtype=np.uint8)

    if necrotic_mask is None:
        nec_arr = np.zeros((h, w), dtype=np.uint8)
        if unprotected_7d_pct > 0:
            cv2.circle(nec_arr, (w // 2, h // 2), max(2, int(min(w, h) * 0.05)), 255, -1)
    else:
        nec_arr = np.array(necrotic_mask, dtype=np.uint8)

    if chlorotic_mask is None:
        chl_arr = np.zeros((h, w), dtype=np.uint8)
        if unprotected_7d_pct > 0:
            cv2.circle(chl_arr, (w // 2, h // 2), max(4, int(min(w, h) * 0.08)), 255, -1)
    else:
        chl_arr = np.array(chlorotic_mask, dtype=np.uint8)

    if leaf_arr.shape != (h, w):
        leaf_arr = cv2.resize(leaf_arr, (w, h), interpolation=cv2.INTER_NEAREST)
    if nec_arr.shape != (h, w):
        nec_arr = cv2.resize(nec_arr, (w, h), interpolation=cv2.INTER_NEAREST)
    if chl_arr.shape != (h, w):
        chl_arr = cv2.resize(chl_arr, (w, h), interpolation=cv2.INTER_NEAREST)

    leaf_bin = (leaf_arr > 0).astype(np.uint8)
    nec_bin = ((nec_arr > 0) & (leaf_bin > 0)).astype(np.uint8)
    chl_bin = ((chl_arr > 0) & (leaf_bin > 0) & (nec_bin == 0)).astype(np.uint8)

    rh_val = float(rh_mean) if rh_mean is not None else 75.0
    precip_val = float(precip_total) if precip_total is not None else 0.0

    field = compute_spore_dispersion_field(leaf_bin, nec_bin, chl_bin, rh_mean=rh_val, precip_total=precip_val)
    unprot_mask = calibrate_projected_frontier(field, leaf_bin, target_severity_pct=unprotected_7d_pct)
    prot_target = min(float(unprotected_7d_pct), float(protected_7d_pct))
    prot_mask = calibrate_projected_frontier(field, leaf_bin, target_severity_pct=prot_target)

    current_lesions = ((nec_bin > 0) | (chl_bin > 0)).astype(np.uint8)
    prevented_zone = ((unprot_mask > 0) & (prot_mask == 0) & (current_lesions == 0) & (leaf_bin > 0)).astype(np.uint8)
    residual_spread = ((prot_mask > 0) & (current_lesions == 0) & (leaf_bin > 0)).astype(np.uint8)

    # Ensure prevented_zone has clear visual presence and doesn't collapse to a microscopic line
    if np.any(prevented_zone > 0) and np.sum(prevented_zone) < max(80, int(0.015 * np.sum(leaf_bin))):
        kernel_vis = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        prevented_zone = (cv2.dilate(prevented_zone, kernel_vis, iterations=1) & leaf_bin & (current_lesions == 0)).astype(np.uint8)

    overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # 1. Prevented Spread Zone (Saved Foliar Area): High-contrast Neon Magenta fill (#ff2e93: 255, 46, 147, alpha=215)
    if np.any(prevented_zone > 0):
        overlay_rgba[prevented_zone > 0] = [255, 46, 147, 215]
        contours_prev, _ = cv2.findContours(prevented_zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_rgba, contours_prev, -1, (255, 130, 200, 255), 1)

    # 2. Residual Spread Under Protection: Translucent Sky Blue fill (#38bdf8: 56, 189, 248, alpha=150)
    if np.any(residual_spread > 0):
        overlay_rgba[residual_spread > 0] = [56, 189, 248, 150]

    # 3. Arrested Lesion Frontier Contour: Crisp Sky Blue boundary outlining the contracted protected lesions
    if np.any(prot_mask > 0):
        contours_prot, _ = cv2.findContours(prot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_rgba, contours_prot, -1, (56, 189, 248, 255), 2)

    # 4. Outer Unchecked 7-Day Spread Frontier (Reference Outline):
    # Distinct Electric Cyan contour (#00f0ff: 0, 240, 255) matching the 7-Day Untreated view so the user clearly sees where it was stopped!
    if np.any(unprot_mask > 0):
        contours_unprot, _ = cv2.findContours(unprot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_rgba, contours_unprot, -1, (0, 240, 255, 255), 2)

    # 5. Observed Day 0 Chlorotic Halo: Translucent Golden Amber (#e9b44c: 233, 180, 76, alpha=200)
    if np.any(chl_bin > 0):
        overlay_rgba[chl_bin > 0] = [233, 180, 76, 200]
        contours_chl, _ = cv2.findContours(chl_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_rgba, contours_chl, -1, (218, 165, 32, 255), 2)

    # 6. Observed Day 0 Necrotic Core: Translucent Rust Red (#bc4749: 188, 71, 73, alpha=215)
    if np.any(nec_bin > 0):
        overlay_rgba[nec_bin > 0] = [188, 71, 73, 215]
        contours_nec, _ = cv2.findContours(nec_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_rgba, contours_nec, -1, (220, 50, 50, 255), 2)

    # 7. Foliar Vegetative Envelope boundary: Deep Forest Green (#386641: 56, 102, 65)
    if np.any(leaf_bin > 0):
        contours_leaf, _ = cv2.findContours(leaf_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_rgba, contours_leaf, -1, (56, 102, 65, 255), 2)

    overlay_img = Image.fromarray(overlay_rgba, mode="RGBA")
    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
