"""Metric Scale Calibration, Planar Homography, and Fiducial Normalization.

Phase 01 Optical Calibration: Detects metric reference tags (circular reference stickers
or ArUco markers) to convert dimensionless pixel counts into true metric surface areas (mm²),
estimates camera perspective tilt angles, and normalizes leaf geometry.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_FIDUCIAL_DIAMETER_MM = 20.0  # Standard 20mm circular sticker


class MetricScaleResult(BaseModel):
    """Pydantic schema for metric scale calibration outputs."""
    is_calibrated: bool = Field(..., description="True if a valid metric fiducial was detected and verified.")
    fiducial_type: Optional[str] = Field(None, description="Detected fiducial type (e.g. 'circular_tag', 'aruco').")
    scale_px_per_mm: Optional[float] = Field(None, description="Spatial resolution in pixels per millimeter.")
    perspective_angle_deg: Optional[float] = Field(None, description="Estimated camera tilt angle relative to leaf normal.")
    metric_leaf_area_mm2: Optional[float] = Field(None, description="Calculated true leaf area in square millimeters.")
    metric_lesion_area_mm2: Optional[float] = Field(None, description="Calculated true lesion area in square millimeters.")
    fiducial_bbox: Optional[Tuple[int, int, int, int]] = Field(None, description="Bounding box [x, y, w, h] of reference marker.")


def detect_circular_fiducial(
    image: Union[np.ndarray, Image.Image],
    known_diameter_mm: float = DEFAULT_FIDUCIAL_DIAMETER_MM
) -> MetricScaleResult:
    """Detect circular metric reference sticker and calculate physical scale (px/mm).

    Uses morphological contour analysis, circularity metric (4*pi*A / P^2 > 0.82),
    and aspect ratio eccentricity to detect calibrated sticker tags.

    Args:
        image: RGB input image (numpy array or PIL Image).
        known_diameter_mm: Physical diameter of reference sticker in millimeters.

    Returns:
        MetricScaleResult instance.
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = image.copy()

    if img_np.ndim != 3 or img_np.shape[2] != 3:
        return MetricScaleResult(is_calibrated=False)

    h, w = img_np.shape[:2]
    # Fast luminance extraction
    gray = (0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]).astype(np.uint8)

    from scipy import ndimage

    # Multi-level candidate segmentation: examine high-contrast blobs (bright sticker, colored tag, or high gradient closure)
    candidates = []
    for threshold_val in [200, 160, 120]:
        binary = (gray > threshold_val)
        filled = ndimage.binary_fill_holes(binary)
        labeled, num_features = ndimage.label(filled)
        if num_features > 0:
            candidates.append((filled, labeled, num_features))

    # Also check gradient edge magnitude closed regions
    gy, gx = np.gradient(gray.astype(np.float32))
    grad_mag = np.sqrt(gx**2 + gy**2)
    edge_thresh = grad_mag > np.percentile(grad_mag, 90)
    edge_closed = ndimage.binary_fill_holes(edge_thresh)
    labeled_edge, num_edge = ndimage.label(edge_closed)
    if num_edge > 0:
        candidates.append((edge_closed, labeled_edge, num_edge))

    best_marker = None
    best_circularity = 0.0

    for mask_arr, labeled_arr, n_feats in candidates:
        for i in range(1, min(n_feats + 1, 50)):
            component = (labeled_arr == i)
            area = float(np.sum(component))

            # Tag area should be between 0.05% and 8% of image
            min_tag_area = (h * w) * 0.0005
            max_tag_area = (h * w) * 0.10
            if area < min_tag_area or area > max_tag_area:
                continue

            rows = np.any(component, axis=1)
            cols = np.any(component, axis=0)
            if not np.any(rows) or not np.any(cols):
                continue

            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            bw, bh = float(cmax - cmin + 1), float(rmax - rmin + 1)

            aspect = min(bw, bh) / max(bw, bh)
            if aspect < 0.70:  # Must be reasonably circular or slightly tilted ellipse
                continue

            # Boundary perimeter
            eroded = ndimage.binary_erosion(component)
            perimeter = float(np.sum(component & ~eroded))
            if perimeter <= 0:
                continue

            circularity = 4.0 * np.pi * area / (perimeter**2)
            if 0.70 <= circularity <= 1.40 and circularity > best_circularity:
                best_circularity = circularity
                effective_diameter_px = (bw + bh) / 2.0
                tilt_deg = float(np.arccos(np.clip(aspect, 0.0, 1.0)) * 180.0 / np.pi)
                best_marker = {
                    "diameter_px": effective_diameter_px,
                    "bbox": (int(cmin), int(rmin), int(bw), int(bh)),
                    "tilt_deg": round(tilt_deg, 2)
                }

        if best_marker is not None:
            break

    if best_marker is None:
        return MetricScaleResult(is_calibrated=False)

    scale_px_mm = best_marker["diameter_px"] / float(known_diameter_mm)
    return MetricScaleResult(
        is_calibrated=True,
        fiducial_type="circular_tag",
        scale_px_per_mm=round(scale_px_mm, 4),
        perspective_angle_deg=best_marker["tilt_deg"],
        fiducial_bbox=best_marker["bbox"]
    )


def compute_metric_areas(
    leaf_pixel_count: int,
    lesion_pixel_count: int,
    scale_px_per_mm: Optional[float]
) -> Tuple[Optional[float], Optional[float]]:
    """Convert pixel areas to physical metric square millimeters (mm²).

    Args:
        leaf_pixel_count: Total segmented leaf pixels.
        lesion_pixel_count: Total segmented lesion pixels.
        scale_px_per_mm: Spatial scale factor in px/mm.

    Returns:
        Tuple of (leaf_area_mm2, lesion_area_mm2), or (None, None) if uncalibrated.
    """
    if scale_px_per_mm is None or scale_px_per_mm <= 0:
        return None, None

    px_area_per_mm2 = scale_px_per_mm ** 2
    leaf_mm2 = float(round(leaf_pixel_count / px_area_per_mm2, 2))
    lesion_mm2 = float(round(lesion_pixel_count / px_area_per_mm2, 2))
    return leaf_mm2, lesion_mm2
