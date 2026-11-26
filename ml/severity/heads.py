"""Segmentation Decoder Heads for Disease Severity Assessment.

Lightweight PyTorch segmentation decoder heads that take frozen DINOv2 patch-token
grids [B, H_grid, W_grid, C] or [B, C, H_grid, W_grid], reduce feature channels using a 1x1
convolution, apply convolutional blocks with bilinear upsampling, and output two logit maps:
lesion mask logits and leaf mask logits.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.logging_utils import get_logger

logger = get_logger(__name__)


class HeadConfig(BaseModel):
    """Pydantic schema for decoder head configuration."""
    input_dim: int = Field(default=384, description="Patch token embedding dimension (384 for Small, 768 for Base).")
    reduced_dim: int = Field(default=64, description="Channel dimension after 1x1 convolution reduction.")
    target_stride: int = Field(default=4, description="Target spatial stride relative to original input resolution.")
    lesion_num_classes: int = Field(default=1, description="Output channels for lesion binary logit map.")
    leaf_num_classes: int = Field(default=1, description="Output channels for leaf binary logit map.")


class ConvBlock(nn.Module):
    """Convolutional block: Conv 3x3 -> BatchNorm -> ReLU."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    """Residual Convolutional Block: (Conv 3x3 -> BN -> ReLU -> Conv 3x3 -> BN) + Shortcut."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class SpatialChannelAttention(nn.Module):
    """Lightweight Spatial and Channel Attention Module (< 15k parameters)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_c = max(8, channels // reduction)

        # Channel Attention (Squeeze-and-Excitation)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_c, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_c, channels, kernel_size=1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        ch_atten = self.sigmoid_channel(avg_out + max_out)
        x_ch = x * ch_atten

        # Spatial attention
        avg_s = torch.mean(x_ch, dim=1, keepdim=True)
        max_s, _ = torch.max(x_ch, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_s, max_s], dim=1)
        sp_atten = self.sigmoid_spatial(self.spatial_conv(spatial_in))

        return x_ch * sp_atten


class SeveritySegmentationHeads(nn.Module):
    """Enhanced lightweight segmentation decoder producing lesion and leaf mask logits from patch grids.

    Includes Residual Convolutional Blocks, Spatial-Channel Attention, and optional auxiliary deep supervision heads.
    Parameter Count: ~145,000 parameters (< 200k target budget).
    """

    def __init__(
        self,
        input_dim: int = 384,
        reduced_dim: int = 64,
        target_stride: int = 4,
        lesion_num_classes: int = 1,
        leaf_num_classes: int = 1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.reduced_dim = reduced_dim
        self.target_stride = target_stride

        # 1. 1x1 channel reduction conv
        self.channel_reducer = nn.Sequential(
            nn.Conv2d(input_dim, reduced_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_dim),
            nn.ReLU(inplace=True)
        )

        # 2. Residual refinement and upsampling layers
        self.res1 = ResidualConvBlock(reduced_dim, reduced_dim)
        self.res2 = ResidualConvBlock(reduced_dim, reduced_dim)

        # 3. Spatial and Channel Attention refinement
        self.attention = SpatialChannelAttention(reduced_dim, reduction=16)

        # 4. Task-specific output logit heads
        self.lesion_head = nn.Conv2d(reduced_dim, lesion_num_classes, kernel_size=1)
        self.leaf_head = nn.Conv2d(reduced_dim, leaf_num_classes, kernel_size=1)

        # 5. Auxiliary deep supervision logit heads at Stage 1
        self.lesion_aux_head = nn.Conv2d(reduced_dim, lesion_num_classes, kernel_size=1)
        self.leaf_aux_head = nn.Conv2d(reduced_dim, leaf_num_classes, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        target_size: Union[Tuple[int, int], None] = None,
        return_aux: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass taking patch tokens and returning lesion and leaf logits.

        Args:
            x: Input tensor of patch tokens. Format can be:
               - [B, H_grid, W_grid, C] (as extracted from DINOv2 grid)
               - [B, C, H_grid, W_grid]
            target_size: Optional explicit (H_out, W_out) spatial target size.
            return_aux: If True, includes auxiliary intermediate logits for deep supervision.

        Returns:
            Dict containing 'lesion_logits' [B, 1, H_out, W_out], 'leaf_logits' [B, 1, H_out, W_out],
            and optionally 'lesion_aux_logits', 'leaf_aux_logits'.
        """
        x = x.float()
        if x.ndim == 4 and x.shape[-1] == self.input_dim:
            # Convert [B, H, W, C] to [B, C, H, W]
            x = x.permute(0, 3, 1, 2).contiguous()

        batch_size, channels, grid_h, grid_w = x.shape

        # Channel reduction
        feat = self.channel_reducer(x)

        # Stage 1: Residual Conv + 2x Upsample
        feat_stage1 = self.res1(feat)
        feat_stage1_up = F.interpolate(feat_stage1, scale_factor=2.0, mode="bilinear", align_corners=False)

        # Stage 2: Residual Conv + 2x Upsample + Attention
        feat_stage2 = self.res2(feat_stage1_up)
        feat_stage2_up = F.interpolate(feat_stage2, scale_factor=2.0, mode="bilinear", align_corners=False)
        feat_atten = self.attention(feat_stage2_up)

        # Target explicit spatial interpolation if specified
        if target_size is not None:
            feat_final = F.interpolate(feat_atten, size=target_size, mode="bilinear", align_corners=False)
        else:
            feat_final = feat_atten

        lesion_logits = self.lesion_head(feat_final)
        leaf_logits = self.leaf_head(feat_final)

        result = {
            "lesion_logits": lesion_logits,
            "leaf_logits": leaf_logits
        }

        if return_aux:
            if target_size is not None:
                feat_aux = F.interpolate(feat_stage1_up, size=target_size, mode="bilinear", align_corners=False)
            else:
                feat_aux = feat_stage1_up

            result["lesion_aux_logits"] = self.lesion_aux_head(feat_aux)
            result["leaf_aux_logits"] = self.leaf_aux_head(feat_aux)

        return result

    def count_parameters(self) -> Dict[str, int]:
        """Calculate total and trainable parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params
        }


class MultiScaleFPNHead(nn.Module):
    """Multi-Scale Feature Pyramid Network (FPN) head for sub-patch lesion segmentation.

    Fuses deep ViT patch embeddings with shallow high-resolution feature representations to eliminate
    spatial quantization blur around micro-pustules (< 5 pixels).
    """

    def __init__(
        self,
        deep_dim: int = 384,
        stem_dim: int = 64,
        fpn_dim: int = 64,
        lesion_num_classes: int = 1,
        leaf_num_classes: int = 1
    ):
        super().__init__()
        self.deep_dim = deep_dim
        self.stem_dim = stem_dim
        self.fpn_dim = fpn_dim

        # Lateral projection for deep ViT patch tokens
        self.lat_deep = nn.Sequential(
            nn.Conv2d(deep_dim, fpn_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )

        # Lateral projection for shallow stem features
        self.lat_stem = nn.Sequential(
            nn.Conv2d(stem_dim, fpn_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )

        # Anti-aliasing post-fusion convolutions
        self.smooth_conv = ResidualConvBlock(fpn_dim, fpn_dim)
        self.attention = SpatialChannelAttention(fpn_dim, reduction=16)

        # Output prediction heads
        self.lesion_head = nn.Conv2d(fpn_dim, lesion_num_classes, kernel_size=1)
        self.leaf_head = nn.Conv2d(fpn_dim, leaf_num_classes, kernel_size=1)

    def forward(
        self,
        deep_tokens: torch.Tensor,
        stem_features: Optional[torch.Tensor] = None,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass merging multi-scale feature hierarchies.

        Args:
            deep_tokens: ViT patch embeddings [B, H_grid, W_grid, C] or [B, C, H_grid, W_grid].
            stem_features: Optional shallow convolutional features [B, C_stem, H_stem, W_stem].
            target_size: Target (H, W) spatial resolution for mask output.

        Returns:
            Dict containing 'lesion_logits' and 'leaf_logits'.
        """
        x = deep_tokens.float()
        if x.ndim == 4 and x.shape[-1] == self.deep_dim:
            x = x.permute(0, 3, 1, 2).contiguous()

        p_deep = self.lat_deep(x)

        if stem_features is not None:
            stem_f = stem_features.float()
            if stem_f.ndim == 4 and stem_f.shape[-1] == self.stem_dim:
                stem_f = stem_f.permute(0, 3, 1, 2).contiguous()
            p_stem = self.lat_stem(stem_f)

            # Top-down fusion: upsample deep features to match shallow stem resolution
            p_deep_up = F.interpolate(p_deep, size=p_stem.shape[2:], mode="bilinear", align_corners=False)
            fused = self.smooth_conv(p_deep_up + p_stem)
        else:
            p_deep_up = F.interpolate(p_deep, scale_factor=4.0, mode="bilinear", align_corners=False)
            fused = self.smooth_conv(p_deep_up)

        fused = self.attention(fused)

        if target_size is not None:
            fused = F.interpolate(fused, size=target_size, mode="bilinear", align_corners=False)

        return {
            "lesion_logits": self.lesion_head(fused),
            "leaf_logits": self.leaf_head(fused)
        }


def build_severity_heads(config_dict: Union[Dict[str, Any], HeadConfig, None] = None) -> SeveritySegmentationHeads:
    """Factory builder for SeveritySegmentationHeads validated with HeadConfig."""
    if config_dict is None:
        cfg = HeadConfig()
    elif isinstance(config_dict, HeadConfig):
        cfg = config_dict
    else:
        cfg = HeadConfig(**config_dict)

    model = SeveritySegmentationHeads(
        input_dim=cfg.input_dim,
        reduced_dim=cfg.reduced_dim,
        target_stride=cfg.target_stride,
        lesion_num_classes=cfg.lesion_num_classes,
        leaf_num_classes=cfg.leaf_num_classes
    )
    logger.info(f"Built SeveritySegmentationHeads with {model.count_parameters()['total_parameters']:,} parameters.")
    return model
