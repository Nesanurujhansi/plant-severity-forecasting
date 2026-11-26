"""DINOv2 Self-Attention Rollout Engine for Visual Foliar Saliency.

Implements recursive Attention Rollout (Abnar & Zuidema, ACL 2020) across all transformer
encoder layers to project self-attention from the global [CLS] token down to individual
14x14 spatial patches. Restricts attention mass strictly to the segmented vegetative leaf
envelope to eliminate non-foliar background artifacts.
"""

import io
import base64
from typing import Optional, Tuple, Union, List
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.features.extract_features import preprocess_image, IMAGENET_MEAN, IMAGENET_STD
from ml.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Global cached backbone with eager attention for XAI
_CACHED_XAI_MODEL: Optional[nn.Module] = None


def get_xai_backbone(backbone_id: str = "facebook/dinov2-small", device: Optional[torch.device] = None) -> Optional[nn.Module]:
    """Load or retrieve singleton DINOv2 model initialized with eager attention implementation."""
    global _CACHED_XAI_MODEL
    if _CACHED_XAI_MODEL is not None:
        return _CACHED_XAI_MODEL

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        from transformers import AutoModel
        logger.info(f"Loading XAI DINOv2 model with attn_implementation='eager': {backbone_id}...")
        try:
            model = AutoModel.from_pretrained(backbone_id, attn_implementation="eager", local_files_only=True)
        except Exception:
            model = AutoModel.from_pretrained(backbone_id, attn_implementation="eager")
        model.to(dev)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        _CACHED_XAI_MODEL = model
        return _CACHED_XAI_MODEL
    except Exception as e:
        logger.warning(f"Failed to load eager DINOv2 model for XAI ({e}). Fallback to heuristic saliency.")
        return None


def compute_attention_rollout(
    attentions: List[torch.Tensor],
    discard_ratio: float = 0.05,
    head_fusion: str = "mean"
) -> np.ndarray:
    """Compute recursive Attention Rollout across transformer encoder layers.

    Args:
        attentions: List of L attention tensors, each shape (B, num_heads, seq_len, seq_len).
        discard_ratio: Fraction of smallest attention weights to prune to eliminate noise.
        head_fusion: How to fuse multi-head attention: 'mean', 'max', or 'min'.

    Returns:
        np.ndarray: Cumulative rollout attention from [CLS] to patch tokens, shape (seq_len - 1,).
    """
    # Number of tokens: 1 + num_patches
    # Process for single batch item (B=1)
    rollout = None

    for layer_attn in attentions:
        # layer_attn: (1, num_heads, seq_len, seq_len)
        attn = layer_attn[0]  # (num_heads, seq_len, seq_len)

        if head_fusion == "mean":
            attn_fused = attn.mean(dim=0)  # (seq_len, seq_len)
        elif head_fusion == "max":
            attn_fused = attn.max(dim=0)[0]
        else:
            attn_fused = attn.min(dim=0)[0]

        # Discard lowest percentile weights to sharpen salient focus
        if discard_ratio > 0:
            flat = attn_fused.flatten()
            k = int(len(flat) * discard_ratio)
            if k > 0:
                thresh = torch.kthvalue(flat, k)[0]
                attn_fused = torch.where(attn_fused < thresh, torch.zeros_like(attn_fused), attn_fused)

        # Add identity for residual connection: A_hat = 0.5 * A + 0.5 * I
        eye = torch.eye(attn_fused.size(-1), device=attn_fused.device)
        attn_hat = 0.5 * attn_fused + 0.5 * eye
        # Normalize rows to sum to 1
        attn_hat = attn_hat / (attn_hat.sum(dim=-1, keepdim=True) + 1e-12)

        if rollout is None:
            rollout = attn_hat
        else:
            rollout = torch.matmul(attn_hat, rollout)

    # Rollout from CLS token (index 0) to all spatial patch tokens (indices 1:)
    cls_attention = rollout[0, 1:].detach().cpu().numpy()
    return cls_attention


def generate_attention_saliency_map(
    image: Image.Image,
    leaf_mask: Optional[np.ndarray] = None,
    resolution: int = 448,
    patch_size: int = 14,
    colormap: int = cv2.COLORMAP_INFERNO,
    alpha: float = 0.60
) -> str:
    """Generate high-resolution DINOv2 visual attention saliency map.

    Args:
        image: PIL Image of the foliar subject.
        leaf_mask: Binary vegetative mask (H, W). If None, extracted automatically.
        resolution: Target input resolution for DINOv2.
        patch_size: Patch stride (14 for DINOv2).
        colormap: OpenCV colormap (default COLORMAP_INFERNO).
        alpha: Blending transparency of heatmap over leaf photo.

    Returns:
        Base64-encoded PNG image string of the attention saliency map.
    """
    orig_w, orig_h = image.size
    img_rgb = np.array(image.convert("RGB"))

    # 1. Vegetative leaf mask
    if leaf_mask is None:
        from ml.severity.predict import extract_vegetative_pathology_masks
        l_m, _, _, _ = extract_vegetative_pathology_masks(image)
        leaf_mask = l_m
    leaf_arr = np.array(leaf_mask, dtype=np.uint8)
    if leaf_arr.shape != (orig_h, orig_w):
        leaf_arr = cv2.resize(leaf_arr, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    leaf_bin = (leaf_arr > 0).astype(np.uint8)

    # 2. Extract DINOv2 attention rollout
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_xai_backbone(device=dev)

    saliency_2d = None
    if model is not None:
        try:
            tensor, meta = preprocess_image(image, target_resolution=resolution, patch_size=patch_size)
            tensor = tensor.unsqueeze(0).to(dev)
            with torch.no_grad():
                outputs = model(tensor, output_attentions=True)
                attentions = outputs.attentions
                if attentions is not None and len(attentions) > 0:
                    cls_attn = compute_attention_rollout(attentions, discard_ratio=0.1)
                    gh, gw = meta["grid_shape"]
                    if len(cls_attn) == gh * gw:
                        saliency_2d = cls_attn.reshape(gh, gw)
        except Exception as e:
            logger.warning(f"Error computing attention rollout: {e}")
            saliency_2d = None

    # Fallback to high-contrast pathology gradient saliency if model is unavailable
    if saliency_2d is None:
        # Heuristic spectral saliency: contrast of green vs red/brown inside leaf
        r = img_rgb[:, :, 0].astype(np.float32)
        g = img_rgb[:, :, 1].astype(np.float32)
        b = img_rgb[:, :, 2].astype(np.float32)
        diff = np.abs(r - g) + np.abs(b - g)
        raw_heat = cv2.GaussianBlur(diff, (31, 31), 0)
    else:
        # Upsample patch grid to original image size
        raw_heat = cv2.resize(saliency_2d, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # 3. Mask strictly to the vegetative leaf envelope (zero background false positives)
    raw_heat = raw_heat.astype(np.float32)
    raw_heat[leaf_bin == 0] = 0.0

    # 4. Normalize inside the foliar envelope to [0, 255]
    leaf_pixels = raw_heat[leaf_bin > 0]
    if len(leaf_pixels) > 0 and (np.max(leaf_pixels) - np.min(leaf_pixels)) > 1e-6:
        p_min = np.percentile(leaf_pixels, 5)
        p_max = np.percentile(leaf_pixels, 98)
        norm_heat = np.clip((raw_heat - p_min) / max(1e-6, p_max - p_min), 0.0, 1.0)
    else:
        norm_heat = np.zeros_like(raw_heat)

    heat_u8 = (norm_heat * 255.0).astype(np.uint8)

    # 5. Render colored heatmap
    heat_color = cv2.applyColorMap(heat_u8, colormap)  # BGR
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    # 6. Composite with original image:
    # Within leaf: blend original RGB with heatmap
    # Outside leaf: muted neutral background
    comp = img_rgb.copy()
    comp[leaf_bin > 0] = (
        (1.0 - alpha) * img_rgb[leaf_bin > 0] + alpha * heat_color[leaf_bin > 0]
    ).astype(np.uint8)

    # Draw sharp foliar boundary
    contours_leaf, _ = cv2.findContours(leaf_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(comp, contours_leaf, -1, (56, 102, 65), 2)  # Forest Green

    # Convert to base64 PNG
    res_img = Image.fromarray(comp, mode="RGB")
    buf = io.BytesIO()
    res_img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
