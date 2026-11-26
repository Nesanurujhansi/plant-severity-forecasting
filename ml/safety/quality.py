"""Image Quality Assessment module for crop-prognosis.

Evaluates photo quality attributes: blur, resolution, brightness, over/under exposure,
leaf coverage, leaf blob count, and lesion visibility.
Returns structured pass/fail results with beginner-friendly user action recommendations.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.logging_utils import get_logger

logger = get_logger(__name__)


def compute_laplacian_variance(gray_image: np.ndarray) -> float:
    """Compute variance of the 2D Laplacian operator as a proxy for image sharpness.

    Uses a discrete 3x3 Laplacian kernel [[0, 1, 0], [1, -4, 1], [0, 1, 0]].
    """
    if gray_image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale array.")

    # 3x3 Laplacian filtering via vectorized numpy slicing
    h, w = gray_image.shape
    if h < 3 or w < 3:
        return 0.0

    img = gray_image.astype(np.float64)
    lap = (
        img[1:-1, 2:] + img[1:-1, :-2] +
        img[2:, 1:-1] + img[:-2, 1:-1] -
        4.0 * img[1:-1, 1:-1]
    )
    return float(np.var(lap))


def compute_resolution_normalized_laplacian(gray_image: np.ndarray) -> float:
    """Compute Laplacian variance normalized by image diagonal length.

    Prevents high-resolution images from artificially inflating sharpness scores.
    """
    h, w = gray_image.shape[:2]
    diag = np.sqrt(h**2 + w**2)
    var = compute_laplacian_variance(gray_image)
    return float((var / max(1.0, diag)) * 10.0)


def check_image_dimensions(
    image: np.ndarray,
    min_width: int = 224,
    min_height: int = 224
) -> Dict[str, Any]:
    """Check if image dimensions meet minimum resolution standards."""
    h, w = image.shape[:2]
    passed = bool(w >= min_width and h >= min_height)
    return {
        "check": "dimensions",
        "passed": passed,
        "width": w,
        "height": h,
        "min_width": min_width,
        "min_height": min_height,
        "message": f"Image resolution {w}x{h} is acceptable." if passed else f"Image resolution {w}x{h} is too low (minimum {min_width}x{min_height}).",
        "action": "" if passed else "Please take a higher-resolution photo of the leaf."
    }


def check_image_blur(
    gray_image: np.ndarray,
    blur_threshold: float = 100.0,
    norm_blur_threshold: float = 0.15
) -> Dict[str, Any]:
    """Check if image sharpness exceeds blur thresholds."""
    lap_var = compute_laplacian_variance(gray_image)
    norm_lap_var = compute_resolution_normalized_laplacian(gray_image)
    passed = bool(lap_var >= blur_threshold or norm_lap_var >= norm_blur_threshold)
    return {
        "check": "blur",
        "passed": passed,
        "laplacian_variance": round(lap_var, 2),
        "norm_laplacian_variance": round(norm_lap_var, 4),
        "threshold": blur_threshold,
        "norm_threshold": norm_blur_threshold,
        "message": "Image sharpness is acceptable." if passed else f"Photo is too blurry (sharpness score {lap_var:.1f} < threshold {blur_threshold:.1f}).",
        "action": "" if passed else "Hold your camera steady and tap to focus on the diseased leaf spots."
    }


def check_exposure_and_brightness(
    gray_image: np.ndarray,
    min_brightness: float = 40.0,
    max_brightness: float = 220.0,
    max_overexposure_frac: float = 0.20,
    max_underexposure_frac: float = 0.25
) -> Dict[str, Any]:
    """Check image mean brightness, contrast, and over/under exposure ratios."""
    mean_bright = float(np.mean(gray_image))
    contrast = float(np.std(gray_image))
    over_frac = float(np.mean(gray_image > 245))
    under_frac = float(np.mean(gray_image < 10))

    passed_bright = bool(min_brightness <= mean_bright <= max_brightness)
    passed_over = bool(over_frac <= max_overexposure_frac)
    passed_under = bool(under_frac <= max_underexposure_frac)
    passed = bool(passed_bright and passed_over and passed_under)

    reasons = []
    actions = []
    if mean_bright < min_brightness:
        reasons.append(f"Image is too dark (mean brightness {mean_bright:.1f} < {min_brightness})")
        actions.append("Take the photo in daylight or bright natural lighting.")
    elif mean_bright > max_brightness:
        reasons.append(f"Image is too washed out (mean brightness {mean_bright:.1f} > {max_brightness})")
        actions.append("Avoid direct flash glare and bright sunlight reflection.")

    if over_frac > max_overexposure_frac:
        reasons.append(f"Overexposure detected ({over_frac*100:.1f}% pixels overexposed)")
        actions.append("Position the leaf out of direct sun glare.")
    if under_frac > max_underexposure_frac:
        reasons.append(f"Heavy shadows detected ({under_frac*100:.1f}% pixels in shadow)")
        actions.append("Avoid heavy shadows across the leaf surface.")

    return {
        "check": "exposure_brightness",
        "passed": passed,
        "mean_brightness": round(mean_bright, 2),
        "contrast": round(contrast, 2),
        "overexposure_fraction": round(over_frac, 4),
        "underexposure_fraction": round(under_frac, 4),
        "message": "Lighting and exposure are acceptable." if passed else "; ".join(reasons),
        "action": " ".join(actions) if not passed else ""
    }


def check_leaf_coverage(
    leaf_mask: Optional[np.ndarray],
    min_coverage_frac: float = 0.10,
    max_blob_count: int = 8
) -> Dict[str, Any]:
    """Check leaf surface area coverage and disconnected component count."""
    if leaf_mask is None:
        return {
            "check": "leaf_coverage",
            "passed": True,
            "coverage_fraction": 1.0,
            "blob_count": 1,
            "message": "Leaf mask not provided; skipping coverage check.",
            "action": ""
        }

    binary_mask = (leaf_mask > 0).astype(np.uint8)
    total_pixels = binary_mask.size
    leaf_pixels = int(np.sum(binary_mask))
    coverage_frac = float(leaf_pixels / max(1, total_pixels))

    # Fast connected components counting via BFS
    visited = np.zeros_like(binary_mask, dtype=bool)
    blob_count = 0
    h, w = binary_mask.shape

    # Simple 2D component count on downsampled mask for speed
    step = max(1, max(h, w) // 100)
    sub_mask = binary_mask[::step, ::step]
    sh, sw = sub_mask.shape

    for r in range(sh):
        for c in range(sw):
            if sub_mask[r, c] == 1 and not visited[r, c]:
                blob_count += 1
                # BFS flood fill
                queue = [(r, c)]
                visited[r, c] = True
                while queue:
                    curr_r, curr_c = queue.pop(0)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < sh and 0 <= nc < sw and sub_mask[nr, nc] == 1 and not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))

    passed_coverage = bool(coverage_frac >= min_coverage_frac)
    passed_blobs = bool(blob_count <= max_blob_count)
    passed = bool(passed_coverage and passed_blobs)

    reasons = []
    actions = []
    if not passed_coverage:
        reasons.append(f"Leaf covers only {coverage_frac*100:.1f}% of image (minimum {min_coverage_frac*100:.0f}%)")
        actions.append("Move camera closer so the crop leaf fills most of the photo frame.")
    if not passed_blobs:
        reasons.append(f"Too many cluttered leaf fragments ({blob_count} blobs detected)")
        actions.append("Focus on a single individual leaf rather than a cluster of leaves.")

    return {
        "check": "leaf_coverage",
        "passed": passed,
        "coverage_fraction": round(coverage_frac, 4),
        "blob_count": blob_count,
        "message": "Leaf coverage is acceptable." if passed else "; ".join(reasons),
        "action": " ".join(actions) if not passed else ""
    }


def check_specular_glare(
    image: np.ndarray,
    max_glare_fraction: float = 0.08
) -> Dict[str, Any]:
    """Check for specular sunlight reflection or flash glare hotspots.

    Specular glare exhibits near-saturation luminance (V > 240) and low saturation (S < 0.15)
    in the HSV color space, washing out foliar lesion textures.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        return {
            "check": "specular_glare",
            "passed": True,
            "glare_fraction": 0.0,
            "threshold": max_glare_fraction,
            "message": "Non-RGB image; skipping specular glare check.",
            "action": ""
        }

    img_float = image.astype(np.float32) / 255.0
    r, g, b = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    s = np.zeros_like(cmax)
    nz = cmax > 1e-5
    s[nz] = delta[nz] / cmax[nz]
    v = cmax

    glare_mask = (v > 0.94) & (s < 0.15)
    glare_fraction = float(np.mean(glare_mask))
    passed = bool(glare_fraction <= max_glare_fraction)

    return {
        "check": "specular_glare",
        "passed": passed,
        "glare_fraction": round(glare_fraction, 4),
        "threshold": max_glare_fraction,
        "message": "No excessive specular glare detected." if passed else f"Specular sunlight glare detected ({glare_fraction*100:.1f}% of leaf area washed out).",
        "action": "" if passed else "Tilt camera angle slightly to eliminate direct sunlight reflection or flash glare on the leaf blade."
    }


def suppress_specular_glare(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Attenuate specular glare hotspots by color blending on identified glare regions.

    Returns:
        Tuple of (attenuated_image, glare_mask).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        return image, np.zeros(image.shape[:2], dtype=bool)

    img_float = image.astype(np.float32) / 255.0
    cmax = np.max(img_float, axis=2)
    cmin = np.min(img_float, axis=2)
    delta = cmax - cmin
    s = np.zeros_like(cmax)
    nz = cmax > 1e-5
    s[nz] = delta[nz] / cmax[nz]
    v = cmax

    glare_mask = (v > 0.94) & (s < 0.15)
    if not np.any(glare_mask):
        return image, glare_mask

    attenuated = image.copy()
    non_glare = ~glare_mask
    if np.any(non_glare):
        med_color = np.median(image[non_glare], axis=0).astype(np.uint8)
        attenuated[glare_mask] = (0.4 * attenuated[glare_mask] + 0.6 * med_color).astype(np.uint8)

    return attenuated, glare_mask


def evaluate_image_quality(
    image: np.ndarray,
    leaf_mask: Optional[np.ndarray] = None,
    lesion_mask: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run all photo quality checks on an image and return structured report."""
    if config is None:
        config = {
            "min_width": 224,
            "min_height": 224,
            "blur_laplacian_min": 100.0,
            "blur_laplacian_norm_min": 0.15,
            "min_brightness": 40.0,
            "max_brightness": 220.0,
            "max_overexposure_fraction": 0.20,
            "max_underexposure_fraction": 0.25,
            "min_leaf_coverage_fraction": 0.10,
            "max_leaf_blob_count": 8,
            "max_glare_fraction": 0.08
        }

    # Convert RGB image to grayscale if 3D
    if image.ndim == 3:
        if image.shape[2] == 3:
            # Standard Rec. 601 Luma formula
            gray = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(np.uint8)
        else:
            gray = image[:, :, 0].astype(np.uint8)
    else:
        gray = image.astype(np.uint8)

    dim_check = check_image_dimensions(image, min_width=config.get("min_width", 224), min_height=config.get("min_height", 224))
    blur_check = check_image_blur(gray, blur_threshold=config.get("blur_laplacian_min", 100.0), norm_blur_threshold=config.get("blur_laplacian_norm_min", 0.15))
    exp_check = check_exposure_and_brightness(
        gray,
        min_brightness=config.get("min_brightness", 40.0),
        max_brightness=config.get("max_brightness", 220.0),
        max_overexposure_frac=config.get("max_overexposure_fraction", 0.20),
        max_underexposure_frac=config.get("max_underexposure_fraction", 0.25)
    )
    cov_check = check_leaf_coverage(leaf_mask, min_coverage_frac=config.get("min_leaf_coverage_fraction", 0.10), max_blob_count=config.get("max_leaf_blob_count", 8))
    glare_check = check_specular_glare(image, max_glare_fraction=config.get("max_glare_fraction", 0.08))

    all_checks = [dim_check, blur_check, exp_check, cov_check, glare_check]
    overall_passed = all(c["passed"] for c in all_checks)

    failed_reasons = [c["message"] for c in all_checks if not c["passed"]]
    recommended_actions = [c["action"] for c in all_checks if not c["passed"] and c["action"]]

    passed_count = sum(1 for c in all_checks if c["passed"])
    quality_score = float(passed_count / len(all_checks))

    return {
        "quality_score": round(quality_score, 4),
        "passed_all_quality_checks": overall_passed,
        "failed_reasons": failed_reasons,
        "recommended_actions": recommended_actions,
        "details": {
            "dimensions": dim_check,
            "blur": blur_check,
            "exposure_brightness": exp_check,
            "leaf_coverage": cov_check,
            "specular_glare": glare_check
        }
    }


def main() -> int:
    dummy_img = (np.random.rand(300, 300, 3) * 255).astype(np.uint8)
    res = evaluate_image_quality(dummy_img)
    print("Quality Assessment Result:", res["passed_all_quality_checks"], "Score:", res["quality_score"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
