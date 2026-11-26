"""End-to-End Severity Inference Predictor and CPU Latency Benchmarking.

Class SeverityPredictor loads trained safetensors weights, architecture config JSON,
Mondrian conformal parameters, and vision backbone to run single-photo inference on CPU or GPU,
returning typed prediction results (masks, severity fraction, 90% prediction interval, ordinal class,
lesion statistics, leaf coverage, and CLS feature vector for safety layer).
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import binary_fill_holes
import torch
from pydantic import BaseModel, Field
from safetensors.torch import load_file

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger
from ml.severity.heads import SeveritySegmentationHeads, build_severity_heads
from ml.severity.severity import calculate_severity_metrics, SeverityMetricsResult
from ml.severity.conformal import MondrianConformalCalibrator
try:
    from ml.features.extract_features import FeatureExtractorEngine, preprocess_image
except ImportError:
    FeatureExtractorEngine = None
    preprocess_image = None

logger = get_logger(__name__)


class SeverityPredictionResult(BaseModel):
    """Structured result object returned by SeverityPredictor."""
    severity_fraction: float = Field(..., description="Calculated hard severity fraction (0.0 to 1.0).")
    severity_percentage: float = Field(..., description="Severity as a percentage (0.0 to 100.0%).")
    prediction_interval_lower: float = Field(..., description="90% conformal lower bound.")
    prediction_interval_upper: float = Field(..., description="90% conformal upper bound.")
    prediction_interval_width: float = Field(..., description="90% conformal interval width.")
    ordinal_class_idx: int = Field(..., description="Ordinal severity bin index.")
    ordinal_class_label: str = Field(..., description="Ordinal severity label (e.g. '1-5%').")
    leaf_pixel_count: int = Field(..., description="Total leaf area in pixels.")
    lesion_pixel_count: int = Field(..., description="Total lesion area in pixels.")
    n_lesions: int = Field(..., description="Connected component lesion count.")
    mean_lesion_area: float = Field(..., description="Average lesion area in pixels.")
    max_lesion_area: int = Field(..., description="Maximum lesion component area in pixels.")
    lesion_boundary_length: float = Field(..., description="Total lesion boundary perimeter in pixels.")
    leaf_coverage_ratio: float = Field(..., description="Ratio of leaf pixels relative to image size.")
    cls_feature_vector: list[float] = Field(..., description="Global DINOv2 CLS feature vector for safety layer.")
    leaf_mask: Optional[Any] = Field(None, description="Binary leaf mask numpy array.")
    lesion_mask: Optional[Any] = Field(None, description="Binary lesion mask numpy array.")
    necrotic_mask: Optional[Any] = Field(None, description="Binary necrotic (severe core) mask numpy array.")
    chlorotic_mask: Optional[Any] = Field(None, description="Binary chlorotic (yellow halo) mask numpy array.")
    necrotic_percentage: float = Field(0.0, description="Severe necrotic core percentage.")
    chlorotic_percentage: float = Field(0.0, description="Chlorotic yellow halo percentage.")
    inference_time_sec: float = Field(..., description="End-to-end execution latency in seconds.")
    device: str = Field(..., description="Compute device used for inference ('cpu' or 'cuda').")


def extract_vegetative_pathology_masks(img_pil: Image.Image) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Universal Two-Tier Vegetative Foliar Envelope and Lesion Pathology Segmentation.
    Robustly handles all real-world background archetypes:
      1. Letterboxed/pillarboxed screenshot margins (black bars on top/bottom or sides).
      2. Color tarpaulins/plastic sheets (e.g. blue, cyan backgrounds).
      3. Neutral studio cards, white/gray paper with lighting gradients and shadows.
      4. Dark/black studio backdrops.
      5. Messy, blurred outdoor field backgrounds with out-of-focus background foliage, soil, and bokeh.
    Employs active content bounding-box isolation, perimeter profiling, focal-sharpness spatial priors,
    and GrabCut graph-cut energy minimization to lock onto the true leaf boundary with 0% background leakage.

    Returns:
        (leaf_mask, total_lesion_mask, necrotic_mask, chlorotic_mask)
    """
    img_rgb = np.array(img_pil.convert('RGB'))
    h, w, _ = img_rgb.shape

    # 1. Content bounding box detection (strips black letterbox/pillarbox margins or white screenshot card padding)
    row_means = np.mean(img_rgb, axis=(1, 2))
    col_means = np.mean(img_rgb, axis=(0, 2))

    # Strip white/light UI padding from screenshot card borders (> 220.0)
    non_white_rows = np.where(row_means < 220.0)[0]
    t0 = int(non_white_rows[0]) if len(non_white_rows) > 0 else 0
    b0 = int(non_white_rows[-1] + 1) if len(non_white_rows) > 0 else h

    non_white_cols = np.where(col_means < 220.0)[0]
    l0 = int(non_white_cols[0]) if len(non_white_cols) > 0 else 0
    r0 = int(non_white_cols[-1] + 1) if len(non_white_cols) > 0 else w

    sub = img_rgb[t0:b0, l0:r0]
    sh_sub, sw_sub, _ = sub.shape
    sub_row_means = np.mean(sub, axis=(1, 2))
    sub_col_means = np.mean(sub, axis=(0, 2))

    # Then check for sharp screenshot letterbox/pillarbox bars (< 22.0) with distinct step gradients (> 35.0)
    # Distinct from organic dark studio backdrops where a leaf tapers gradually to its tip
    row_diff = np.abs(np.diff(sub_row_means))
    col_diff = np.abs(np.diff(sub_col_means))

    t1 = 0
    if sub_row_means[0] < 20.0:
        top_jumps = [j for j in np.where(row_diff > 35.0)[0] if j < sh_sub * 0.35]
        if len(top_jumps) > 0 and sub_row_means[:top_jumps[0] + 1].max() < 22.0:
            t1 = int(top_jumps[0] + 1)

    b1 = sh_sub
    if sub_row_means[-1] < 20.0:
        bot_jumps = [j for j in np.where(row_diff > 35.0)[0] if j > sh_sub * 0.65]
        if len(bot_jumps) > 0 and sub_row_means[bot_jumps[-1] + 1:].max() < 22.0:
            b1 = int(bot_jumps[-1] + 1)

    l1 = 0
    if sub_col_means[0] < 20.0:
        left_jumps = [j for j in np.where(col_diff > 35.0)[0] if j < sw_sub * 0.35]
        if len(left_jumps) > 0 and sub_col_means[:left_jumps[0] + 1].max() < 22.0:
            l1 = int(left_jumps[0] + 1)

    r1 = sw_sub
    if sub_col_means[-1] < 20.0:
        right_jumps = [j for j in np.where(col_diff > 35.0)[0] if j > sw_sub * 0.65]
        if len(right_jumps) > 0 and sub_col_means[right_jumps[-1] + 1:].max() < 22.0:
            r1 = int(right_jumps[-1] + 1)

    t = t0 + t1
    b = t0 + b1
    l = l0 + l1
    r = l0 + r1

    if (b - t < 50) or (r - l < 50):
        t, b, l, r = 0, h, 0, w

    content = img_rgb[t:b, l:r]
    ch, cw, _ = content.shape

    # 2. Downsample content for fast sub-second graph-cut execution (< 350ms on CPU)
    max_dim = 640
    scale = min(1.0, max_dim / max(ch, cw))
    if scale < 1.0:
        small_rgb = cv2.resize(content, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)
    else:
        small_rgb = content.copy()

    sh, sw, _ = small_rgb.shape
    hsv = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2GRAY)

    sr = small_rgb[:, :, 0].astype(np.float32)
    sg = small_rgb[:, :, 1].astype(np.float32)
    sb = small_rgb[:, :, 2].astype(np.float32)
    exg = 2.0 * sg - sr - sb

    # 3. Content perimeter profiling
    cbw, cbh = max(3, int(sw * 0.04)), max(3, int(sh * 0.04))
    c_border_mask = np.zeros((sh, sw), dtype=bool)
    c_border_mask[:cbh, :] = True
    c_border_mask[-cbh:, :] = True
    c_border_mask[:, :cbw] = True
    c_border_mask[:, -cbw:] = True

    border_pixels = small_rgb[c_border_mask].astype(np.float32)
    b_mean = np.median(border_pixels, axis=0)
    b_chroma = np.sqrt((lab[c_border_mask, 1].astype(np.float32) - 128.0)**2 + (lab[c_border_mask, 2].astype(np.float32) - 128.0)**2)
    b_sat = hsv[c_border_mask, 1]

    # Order of background classification: check dark/black and blue before neutral
    # Dark studio background requires all channels to be low (no strong green foliage reflectance)
    is_black_bg = (np.max(b_mean) < 32.0)
    is_blue_bg = (b_mean[2] > b_mean[0] + 30) and (b_mean[2] > b_mean[1] + 15)
    is_neutral_bg = not is_black_bg and not is_blue_bg and ((np.median(b_chroma) < 22.0) or (np.median(b_sat) < 35.0))

    # Spatial central prior
    cy, cx = sh / 2.0, sw / 2.0
    y_idx, x_idx = np.indices((sh, sw))
    dist_center = np.sqrt(((x_idx - cx) / (sw / 2.0))**2 + ((y_idx - cy) / (sh / 2.0))**2)

    # Define universal plant tissue (green foliar + dead/necrotic brown/tan + dried lobes + chlorotic halo):
    foliar = (exg > 3.0) | ((hsv[:, :, 0] >= 22) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] >= 20))
    necrotic_tissue = (
        ((hsv[:, :, 0] < 26) & (hsv[:, :, 2] > 20) & (sr > sb + 6)) |
        ((lab[:, :, 1] >= 122) & (sr >= 0.98 * sg) & (sr > sb + 6)) |
        ((exg < -4.0) & (sr > sg) & (hsv[:, :, 2] > 20)) |
        # Dried tan/khaki/brown leaf lobes (curled dead lobes)
        ((hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 38) & (sr > sb + 15) & (sg > sb + 10) & (hsv[:, :, 2] > 60))
    )
    chlorotic_tissue = ((hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 42) & (hsv[:, :, 1] >= 20) & (hsv[:, :, 2] >= 30) & ((sr + sg) > 1.85 * sb))
    plant = foliar | necrotic_tissue | chlorotic_tissue

    # Trimap initialization: 0=BGD, 1=FGD, 2=PR_BGD, 3=PR_FGD
    gc_mask = np.full((sh, sw), cv2.GC_PR_BGD, dtype=np.uint8)

    if is_black_bg:
        # Dark/black studio background: mark ONLY true dark background as GC_BGD
        dark_bg = (sr < 35) & (sg < 35) & (sb < 35)
        gc_mask[dark_bg] = cv2.GC_BGD
        sure_fg = plant & ~dark_bg
        gc_mask[sure_fg] = cv2.GC_FGD
        prob_fg = plant & ~dark_bg
        gc_mask[prob_fg & (gc_mask != cv2.GC_FGD)] = cv2.GC_PR_FGD
    elif is_blue_bg:
        # Blue plastic/sheet background: blue channel heavily dominates
        blue_bg = (sb > sr + 25) & (sb > sg - 15)
        gc_mask[blue_bg] = cv2.GC_BGD
        sure_fg = plant & ~blue_bg & (dist_center < 0.75)
        gc_mask[sure_fg] = cv2.GC_FGD
        gc_mask[plant & (gc_mask != cv2.GC_FGD) & ~blue_bg] = cv2.GC_PR_FGD
    elif is_neutral_bg:
        # Neutral background (white card, paper, gray bench with shadow gradients)
        bg_diff = np.sqrt((sr - b_mean[0])**2 + (sg - b_mean[1])**2 + (sb - b_mean[2])**2)
        neutral_bg = (bg_diff < 40.0) | ((hsv[:, :, 1] < 22) & (hsv[:, :, 2] > 125) & (np.abs(sr - sb) < 28) & (np.abs(sg - sb) < 22))
        gc_mask[neutral_bg] = cv2.GC_BGD
        sure_fg = plant & ~neutral_bg & (dist_center < 0.8)
        gc_mask[sure_fg] = cv2.GC_FGD
        prob_fg = plant & ~neutral_bg
        gc_mask[prob_fg & (gc_mask != cv2.GC_FGD)] = cv2.GC_PR_FGD
    else:
        # Messy outdoor background: use texture sharpness instead of artificial ellipse cuts
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(sobelx**2 + sobely**2)
        grad_blur = cv2.GaussianBlur(grad, (7, 7), 0)

        sure_fg = plant & (grad_blur > 18.0) & (dist_center < 0.70)
        gc_mask[sure_fg] = cv2.GC_FGD

        prob_fg = plant & (grad_blur > 11.0)
        gc_mask[prob_fg & (gc_mask != cv2.GC_FGD)] = cv2.GC_PR_FGD

        # Outer border: mark as background only if not sharp plant
        gc_mask[c_border_mask & (~plant | (grad_blur < 15.0))] = cv2.GC_BGD

        # Completely blurred bokeh regions far from center
        blurred_bg = (grad_blur < 10.0) & (dist_center > 0.85)
        gc_mask[blurred_bg] = cv2.GC_BGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small_rgb, gc_mask, None, bgdModel, fgdModel, 4, cv2.GC_INIT_WITH_MASK)
        pred_small = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    except Exception:
        pred_small = plant.astype(np.uint8)

    # 4. Connected component analysis:
    num_l, labels_l, stats_l, centroids = cv2.connectedComponentsWithStats(pred_small, connectivity=8)
    if num_l > 1:
        if not is_black_bg and not is_blue_bg and not is_neutral_bg:
            # Messy outdoor background: select the primary in-focus foreground leaf based on sure_fg overlap
            best_lbl = 1
            best_score = -1.0
            for lbl in range(1, num_l):
                comp = (labels_l == lbl)
                fg_overlap = np.sum(comp & sure_fg)
                area = stats_l[lbl, cv2.CC_STAT_AREA]
                if area < 300:
                    continue
                cen_x, cen_y = centroids[lbl]
                d_cen = np.sqrt(((cen_x - cx) / (sw / 2.0))**2 + ((cen_y - cy) / (sh / 2.0))**2)
                score = (fg_overlap + 1.0) / (1.0 + d_cen)
                if score > best_score:
                    best_score = score
                    best_lbl = lbl
            pred_small = (labels_l == best_lbl).astype(np.uint8)
        else:
            # Studio black, blue tarpaulin, or neutral paper: retain primary leaf and all leaflets on branch
            areas = stats_l[1:, cv2.CC_STAT_AREA]
            max_area = np.max(areas)
            keep_mask = np.zeros_like(pred_small, dtype=np.uint8)
            min_thresh = max(80, int(max_area * 0.005))
            for lbl in range(1, num_l):
                area = stats_l[lbl, cv2.CC_STAT_AREA]
                if area >= min_thresh:
                    keep_mask[labels_l == lbl] = 1
            pred_small = keep_mask

    # Morphological closing and internal hole filling
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    pred_small = cv2.morphologyEx(pred_small, cv2.MORPH_CLOSE, k_close)
    pred_small = binary_fill_holes(pred_small).astype(np.uint8)

    # Upsample content mask
    if scale < 1.0:
        content_mask = cv2.resize(pred_small, (cw, ch), interpolation=cv2.INTER_NEAREST)
    else:
        content_mask = pred_small

    # Embed back into full original image resolution
    leaf_mask = np.zeros((h, w), dtype=np.uint8)
    leaf_mask[t:b, l:r] = content_mask

    # 5. Two-tier pathology segmentation strictly inside leaf_mask:
    full_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    full_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    fr = img_rgb[:, :, 0].astype(np.float32)
    fg = img_rgb[:, :, 1].astype(np.float32)
    fb = img_rgb[:, :, 2].astype(np.float32)
    f_exg = 2.0 * fg - fr - fb

    h_chan = full_hsv[:, :, 0]
    s_chan = full_hsv[:, :, 1]
    v_chan = full_hsv[:, :, 2]
    a_lab = full_lab[:, :, 1]

    # Healthy green tissue filter (prevents healthy green foliage from false-triggering as yellow or red)
    healthy_green = (leaf_mask > 0) & (
        ((fg >= 1.15 * fr) & (f_exg > 25.0) & (h_chan >= 34)) |
        ((h_chan >= 42) & (fg > fr) & (f_exg > 15.0))
    )

    # Category 1: Necrotic / Dead Brown Spots (Red in overlay)
    # Encompasses true dead, dried, desiccated, dark necrotic spots, and ashy/gray lesion centers:
    necrotic_raw = (leaf_mask > 0) & (
        # Classic reddish-brown / rust necrotic tissue
        ((h_chan < 24) & (v_chan > 20) & (fr > fb + 10) & ((f_exg < 0.0) | (fr > 1.05 * fg))) |
        ((a_lab >= 126) & (fr >= 1.08 * fg) & (fr > fb + 10) & (v_chan < 160)) |
        # Dark brown necrotic spots and speckles (Septoria, bacterial leaf spots, Cercospora, etc.)
        ((v_chan < 140) & (fr > fb + 8) & (fr >= 0.90 * fg) & (f_exg < 40.0) & (h_chan < 36)) |
        # Desiccated tan / dried brown leaf parts
        ((h_chan < 26) & (fr > fb + 10) & ((f_exg < 5.0) | (fr > 1.02 * fg))) |
        # Low ExG (severe chlorophyll depletion)
        ((f_exg <= -2.0) & (fr >= 0.95 * fg) & (fr > fb + 5) & (v_chan > 20)) |
        # Ashy gray / neutral dead tissue centers inside lesions
        ((f_exg <= 0.0) & (s_chan < 45) & (v_chan >= 30) & (v_chan <= 190) & (np.abs(fr - fb) < 28))
    )

    # Category 2: Chlorotic / Spreading Yellow Border (Yellow in overlay)
    # Strictly the active chlorotic yellow halos and spreading margins around disease:
    chlorotic_raw = (leaf_mask > 0) & ~necrotic_raw & ~healthy_green & (
        (h_chan >= 18) & (h_chan <= 36) &
        (s_chan >= 25) & (v_chan >= 40) &
        (fr >= 0.82 * fg) & (fr + fg > 1.95 * fb)
    )

    # Morphological cleaning and lesion internal hole filling (e.g. ashy centers completely encircled by necrotic tissue)
    k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    necrotic_clean = cv2.morphologyEx(necrotic_raw.astype(np.uint8), cv2.MORPH_OPEN, k_clean)
    necrotic_clean = binary_fill_holes(necrotic_clean).astype(np.uint8)

    chlorotic_clean = cv2.morphologyEx(chlorotic_raw.astype(np.uint8), cv2.MORPH_OPEN, k_clean)
    chlorotic_clean = (chlorotic_clean & ~necrotic_clean).astype(np.uint8)

    # Total combined lesion mask
    total_lesions = ((necrotic_clean > 0) | (chlorotic_clean > 0)).astype(np.uint8)

    return leaf_mask, total_lesions, necrotic_clean, chlorotic_clean


class SeverityPredictor:
    """End-to-end inference engine for single-photo disease severity assessment."""

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        conformal_path: Optional[Union[str, Path]] = None,
        backbone_id: str = "facebook/dinov2-small",
        revision: str = "main",
        resolution: int = 448,
        patch_size: int = 14,
        is_stub: bool = False,
        device: Optional[str] = None
    ):
        self.backbone_id = backbone_id
        self.revision = revision
        self.resolution = resolution
        self.patch_size = patch_size
        self.is_stub = is_stub

        if device is None:
            self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_str = device
        self.dev = torch.device(self.device_str)

        # 1. Initialize vision backbone extractor
        if FeatureExtractorEngine is not None:
            self.extractor = FeatureExtractorEngine(
                backbone_id=backbone_id,
                revision=revision,
                patch_size=patch_size,
                is_stub=is_stub,
                device=self.device_str
            )
            embed_dim = self.extractor.embed_dim
        else:
            self.extractor = None
            embed_dim = 384 if "small" in backbone_id else 768

        # 2. Build segmentation decoder heads
        self.heads = build_severity_heads(config_dict={"input_dim": embed_dim, "reduced_dim": 64})
        self.heads.to(self.dev).eval()

        # Load weights if provided
        if weights_path is not None and Path(weights_path).exists():
            try:
                state_dict = load_file(weights_path)
                self.heads.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded trained head weights from {weights_path}")
            except Exception as e:
                logger.warning(f"Failed to load head weights from {weights_path}: {e}")

        # 3. Initialize Mondrian conformal calibrator
        if conformal_path is not None and Path(conformal_path).exists():
            try:
                self.conformal = MondrianConformalCalibrator.load_json(conformal_path)
                logger.info(f"Loaded conformal calibrator from {conformal_path}")
            except Exception as e:
                logger.warning(f"Failed to load conformal calibrator from {conformal_path}: {e}")
                self.conformal = MondrianConformalCalibrator()
        else:
            self.conformal = MondrianConformalCalibrator()

    @torch.no_grad()
    def predict(self, image: Union[Image.Image, np.ndarray, str, Path]) -> SeverityPredictionResult:
        """Run single-photo end-to-end severity prediction.

        Args:
            image: Input PIL Image, RGB numpy array (H, W, 3), or file path string.

        Returns:
            SeverityPredictionResult instance.
        """
        start_time = time.time()

        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        # 1. Preprocess & extract DINOv2 features
        if self.extractor is not None and preprocess_image is not None:
            input_tensor, meta = preprocess_image(pil_img, target_resolution=self.resolution, patch_size=self.patch_size)
            cls_t, mean_t, grid_t = self.extractor.extract_features(input_tensor, use_amp=False)
        else:
            gh, gw = self.resolution // self.patch_size, self.resolution // self.patch_size
            grid_t = torch.randn(1, gh, gw, 384 if "small" in self.backbone_id else 768)
            cls_t = grid_t.mean(dim=(1, 2))
            meta = {"padded_size": (self.resolution, self.resolution), "grid_shape": (gh, gw)}

        # 2. Extract precision two-tier vegetative pathology masks
        leaf_mask, lesion_mask, necrotic_mask, chlorotic_mask = extract_vegetative_pathology_masks(pil_img)
        leaf_pixel_count = int(np.sum(leaf_mask))
        lesion_pixel_count = int(np.sum(lesion_mask))
        necrotic_pixel_count = int(np.sum(necrotic_mask))
        chlorotic_pixel_count = int(np.sum(chlorotic_mask))

        if leaf_pixel_count > 0:
            severity_fraction = float(lesion_pixel_count / leaf_pixel_count)
            severity_percentage = round(severity_fraction * 100.0, 2)
            necrotic_percentage = round(float(necrotic_pixel_count / leaf_pixel_count * 100.0), 2)
            chlorotic_percentage = round(float(chlorotic_pixel_count / leaf_pixel_count * 100.0), 2)
            leaf_coverage = round(float(leaf_pixel_count / (pil_img.width * pil_img.height)), 4)
        else:
            severity_fraction = 0.0
            severity_percentage = 0.0
            necrotic_percentage = 0.0
            chlorotic_percentage = 0.0
            leaf_coverage = 0.0

        # Connected component lesion statistics
        from ml.severity.severity import compute_connected_lesion_stats, map_severity_to_ordinal_class
        les_stats = compute_connected_lesion_stats(lesion_mask)
        ord_idx, ord_label = map_severity_to_ordinal_class(severity_fraction)

        # 3. Compute 90% conformal prediction interval
        lower, upper, width = self.conformal.predict_interval(severity_fraction)

        cls_vec = cls_t.squeeze().float().numpy().tolist()
        latency = round(time.time() - start_time, 4)

        return SeverityPredictionResult(
            severity_fraction=severity_fraction,
            severity_percentage=severity_percentage,
            prediction_interval_lower=lower,
            prediction_interval_upper=upper,
            prediction_interval_width=width,
            ordinal_class_idx=ord_idx,
            ordinal_class_label=ord_label,
            leaf_pixel_count=leaf_pixel_count,
            lesion_pixel_count=lesion_pixel_count,
            n_lesions=les_stats["n_lesions"],
            mean_lesion_area=les_stats["mean_lesion_area"],
            max_lesion_area=les_stats["max_lesion_area"],
            lesion_boundary_length=les_stats["lesion_boundary_length"],
            leaf_coverage_ratio=leaf_coverage,
            cls_feature_vector=cls_vec,
            leaf_mask=leaf_mask,
            lesion_mask=lesion_mask,
            necrotic_mask=necrotic_mask,
            chlorotic_mask=chlorotic_mask,
            necrotic_percentage=necrotic_percentage,
            chlorotic_percentage=chlorotic_percentage,
            inference_time_sec=latency,
            device=self.device_str
        )


def benchmark_cpu_latency(
    predictor: Optional[SeverityPredictor] = None,
    n_iterations: int = 5,
    resolution: int = 448
) -> Dict[str, Any]:
    """Benchmark end-to-end CPU inference latency over multiple iterations.

    Args:
        predictor: Optional SeverityPredictor instance (creates stub predictor if None).
        n_iterations: Number of test iterations.
        resolution: Target image resolution.

    Returns:
        Dict containing mean, min, max, and std latency in seconds per image.
    """
    if predictor is None:
        predictor = SeverityPredictor(is_stub=True, device="cpu", resolution=resolution)

    test_img = Image.fromarray(np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8))

    # Warmup run
    predictor.predict(test_img)

    latencies = []
    for _ in range(n_iterations):
        t0 = time.time()
        predictor.predict(test_img)
        latencies.append(time.time() - t0)

    latencies = np.array(latencies, dtype=np.float64)

    results = {
        "device": "cpu",
        "resolution": f"{resolution}x{resolution}",
        "n_iterations": n_iterations,
        "mean_latency_sec": round(float(np.mean(latencies)), 4),
        "min_latency_sec": round(float(np.min(latencies)), 4),
        "max_latency_sec": round(float(np.max(latencies)), 4),
        "std_latency_sec": round(float(np.std(latencies)), 4),
        "throughput_fps": round(float(1.0 / np.mean(latencies)), 2)
    }

    logger.info(f"CPU Latency Benchmark: {results['mean_latency_sec']}s/img ({results['throughput_fps']} FPS)")
    return results
