"""
Phase 10: Upload Safety & Image Validation Module
Validates uploaded image byte sizes, image format types (JPEG, PNG, WebP), decompression bomb limits,
and strips EXIF metadata in memory.
"""

import io
from typing import Tuple
from PIL import Image, ImageOps

from app.config import settings


class UploadSafetyError(ValueError):
    """Exception raised when an uploaded image fails security or format validation."""
    def __init__(self, error_code: str, message: str, hint: str):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.hint = hint


def validate_and_decode_upload(
    image_bytes: bytes,
    filename: str = "upload.jpg"
) -> Tuple[Image.Image, Tuple[int, int], str]:
    """
    Validate raw upload bytes and decode safely in memory.
    
    Args:
        image_bytes: Raw binary image bytes from HTTP request payload.
        filename: Original filename provided in request header.
        
    Returns:
        (PIL Image object in RGB mode with EXIF stripped, (width, height), format_str)
        
    Raises:
        UploadSafetyError: If file exceeds size limit, is corrupted, or uses disallowed format.
    """
    # 1. File Size Limit Check
    byte_len = len(image_bytes)
    if byte_len == 0:
        raise UploadSafetyError(
            error_code="EMPTY_FILE",
            message="Uploaded file is empty (0 bytes).",
            hint="Please select a valid image photo file."
        )
    if byte_len > settings.max_upload_size_bytes:
        max_mb = settings.max_upload_size_bytes / (1024 * 1024)
        actual_mb = byte_len / (1024 * 1024)
        raise UploadSafetyError(
            error_code="FILE_TOO_LARGE",
            message=f"Uploaded image size ({actual_mb:.2f} MB) exceeds maximum allowed limit ({max_mb:.1f} MB).",
            hint="Please compress or resize your image photo before uploading."
        )

    # 2. Decode Image Format & Validate Format Type
    try:
        img_raw = Image.open(io.BytesIO(image_bytes))
        fmt = (img_raw.format or "").upper()
    except Exception as e:
        raise UploadSafetyError(
            error_code="CORRUPTED_IMAGE",
            message=f"Failed to decode image file: {e}",
            hint="Ensure the uploaded file is a valid, uncorrupted image."
        )

    allowed_formats = {"JPEG", "MPO", "PNG", "WEBP"}
    if fmt not in allowed_formats:
        raise UploadSafetyError(
            error_code="UNSUPPORTED_FORMAT",
            message=f"Unsupported image format '{fmt}'. Allowed formats are JPEG, PNG, WebP.",
            hint="Please convert your photo to JPEG, PNG, or WebP format."
        )

    # 3. Decompression Bomb Guard (Pixel Count Limit)
    width, height = img_raw.size
    total_pixels = width * height
    if total_pixels > settings.max_image_pixels:
        max_mp = settings.max_image_pixels / 1_000_000
        actual_mp = total_pixels / 1_000_000
        raise UploadSafetyError(
            error_code="DECOMPRESSION_BOMB",
            message=f"Image resolution ({actual_mp:.1f} MP) exceeds decompression limit ({max_mp:.1f} MP).",
            hint="Please downscale photo resolution before submitting."
        )

    # 4. Strip EXIF Metadata & Normalize Orientation
    try:
        img_oriented = ImageOps.exif_transpose(img_raw)
    except Exception:
        img_oriented = img_raw

    # Convert to standard RGB without EXIF dict attached
    img_rgb = Image.new("RGB", img_oriented.size)
    img_rgb.paste(img_oriented)

    return img_rgb, (width, height), fmt
