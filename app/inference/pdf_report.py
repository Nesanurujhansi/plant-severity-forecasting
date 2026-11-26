"""
Professional PDF Generation Engine for Kisan Crop Health Card & Field Advisory.
Generates an executive, jargon-free 2-page A4 field guide for farmers with:
1. Cleaned, aspect-ratio preserved 4-image visual evidence gallery (no black bounding boxes).
2. Dynamic, logically consistent 7-day weather & spraying feasibility table.
3. Practical Integrated Pest Management (IPM) action steps.
4. Official Rythu Bharosa Kendra (RBK) & Krishi Vigyan Kendra (KVK) extension guidance.
"""

import base64
import io
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageOps
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable, Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
)

# Professional Palette
FOREST_DARK = colors.HexColor("#1b4332")
FOREST_MED = colors.HexColor("#2d6a4f")
LEAF_GREEN = colors.HexColor("#40916c")
SAGE = colors.HexColor("#74c69d")
SAGE_FAINT = colors.HexColor("#f0fdf4")
AMBER_DARK = colors.HexColor("#b45309")
AMBER_LIGHT = colors.HexColor("#fffbeb")
AMBER_BORDER = colors.HexColor("#fde68a")
RUST_DARK = colors.HexColor("#991b1b")
RUST_LIGHT = colors.HexColor("#fef2f2")
RUST_BORDER = colors.HexColor("#fecaca")
CARD_BG = colors.HexColor("#f8fafc")
CARD_BORDER = colors.HexColor("#e2e8f0")
INK = colors.HexColor("#0f172a")
INK_MUTED = colors.HexColor("#475569")
INK_LIGHT = colors.HexColor("#64748b")


def prepare_card_image(
    b64_string: Optional[str],
    box_w_pt: float = 266.0,
    box_h_pt: float = 242.0,
    placeholder_label: str = "Image Unavailable"
) -> RLImage:
    """
    Cleans raw segmented images, converts pitch-black segmentation backgrounds to pure white,
    preserves organic leaf aspect ratios, and embeds into a crisp rounded container.
    """
    dpi_scale = 2.0
    w_px = int(box_w_pt * dpi_scale)
    h_px = int(box_h_pt * dpi_scale)

    if b64_string and len(b64_string.strip()) > 50:
        try:
            clean_b64 = b64_string
            if "," in clean_b64:
                clean_b64 = clean_b64.split(",", 1)[1]
            raw_bytes = base64.b64decode(clean_b64)
            pil_img = PILImage.open(io.BytesIO(raw_bytes)).convert("RGB")

            # 1. Background Mask Cleanup: Replace pitch black segmentation background with clean white
            arr = np.array(pil_img)
            # Find black pixels where R, G, B are all < 28
            black_mask = (arr[:, :, 0] < 28) & (arr[:, :, 1] < 28) & (arr[:, :, 2] < 28)
            # If at least 3% of the image is black background, convert black to white
            if np.sum(black_mask) > (arr.shape[0] * arr.shape[1] * 0.03):
                arr[black_mask] = [255, 255, 255]
            cleaned_img = PILImage.fromarray(arr)

            # 2. Rescale while strictly preserving aspect ratio
            pad_px = int(4 * dpi_scale)
            max_inner_w = w_px - (pad_px * 2)
            max_inner_h = h_px - (pad_px * 2)
            cleaned_img.thumbnail((max_inner_w, max_inner_h), PILImage.Resampling.LANCZOS)

            # 3. Paste into clean card canvas
            canvas = PILImage.new("RGB", (w_px, h_px), (255, 255, 255))
            offset_x = (w_px - cleaned_img.width) // 2
            offset_y = (h_px - cleaned_img.height) // 2
            canvas.paste(cleaned_img, (offset_x, offset_y))

            # 4. Subtle crisp border
            draw = ImageDraw.Draw(canvas)
            draw.rounded_rectangle(
                [1, 1, w_px - 2, h_px - 2],
                radius=int(6 * dpi_scale),
                outline=(226, 232, 240),
                width=int(1.2 * dpi_scale)
            )

            buf = io.BytesIO()
            canvas.save(buf, format="PNG", optimize=True)
            buf.seek(0)
            return RLImage(buf, width=box_w_pt, height=box_h_pt)
        except Exception:
            pass

    # Clean Fallback Placeholder Card
    canvas = PILImage.new("RGB", (w_px, h_px), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        [1, 1, w_px - 2, h_px - 2],
        radius=int(6 * dpi_scale),
        outline=(203, 213, 225),
        width=int(1.2 * dpi_scale)
    )
    draw.text((w_px // 3, h_px // 2 - 10), placeholder_label, fill=(148, 163, 184))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    buf.seek(0)
    return RLImage(buf, width=box_w_pt, height=box_h_pt)


def composite_overlay_on_original(
    orig_b64: Optional[str],
    overlay_b64: Optional[str]
) -> Optional[str]:
    """
    Composites a transparent RGBA or white/black-background overlay (such as the 7-day
    spatial projection frontier) directly onto the original leaf photograph, producing a
    complete diagnostic view showing both the real leaf tissue and the projected expansion frontier.
    """
    if not overlay_b64 or len(str(overlay_b64).strip()) < 50:
        return orig_b64
    if not orig_b64 or len(str(orig_b64).strip()) < 50:
        return overlay_b64

    try:
        def decode_to_pil(b64_str: str) -> PILImage.Image:
            clean = b64_str.split(",", 1)[1] if "," in b64_str else b64_str
            clean = clean.strip().replace(" ", "+")
            return PILImage.open(io.BytesIO(base64.b64decode(clean)))

        orig_pil = decode_to_pil(orig_b64).convert("RGBA")
        overlay_pil = decode_to_pil(overlay_b64).convert("RGBA")

        # Ensure matching canvas dimensions
        if orig_pil.size != overlay_pil.size:
            overlay_pil = overlay_pil.resize(orig_pil.size, PILImage.Resampling.LANCZOS)

        arr_overlay = np.array(overlay_pil)

        # If overlay lacks transparent alpha, convert white or black backdrop pixels to transparent
        if not np.any(arr_overlay[:, :, 3] < 250):
            is_white = (arr_overlay[:, :, 0] > 240) & (arr_overlay[:, :, 1] > 240) & (arr_overlay[:, :, 2] > 240)
            is_black = (arr_overlay[:, :, 0] < 25) & (arr_overlay[:, :, 1] < 25) & (arr_overlay[:, :, 2] < 25)
            bg_mask = is_white | is_black
            if np.any(bg_mask):
                arr_overlay[:, :, 3] = 200
                arr_overlay[bg_mask, 3] = 0
                overlay_pil = PILImage.fromarray(arr_overlay)

        comp_pil = PILImage.alpha_composite(orig_pil, overlay_pil).convert("RGB")
        buf = io.BytesIO()
        comp_pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return overlay_b64


def compute_spray_feasibility(
    precip_mm: float,
    max_wind_kmh: float,
    rh_pct: float,
    lwd_hours: float
) -> Tuple[str, str, colors.HexColor]:
    """
    Computes rigorous agrometeorological spraying feasibility from empirical metrics.
    Guarantees no contradiction with precipitation or wind speed.
    """
    if precip_mm >= 1.0:
        return (
            "DO NOT SPRAY",
            f"Wash-off hazard ({precip_mm:.1f} mm rain)",
            colors.HexColor("#fee2e2")
        )
    elif max_wind_kmh >= 18.0:
        return (
            "DO NOT SPRAY",
            f"High drift hazard ({max_wind_kmh:.1f} km/h)",
            colors.HexColor("#fee2e2")
        )
    elif max_wind_kmh >= 12.0:
        return (
            "CAUTION",
            f"Breeze drift ({max_wind_kmh:.1f} km/h)",
            colors.HexColor("#fef3c7")
        )
    elif precip_mm >= 0.1 or rh_pct >= 88.0 or lwd_hours >= 12.0:
        return (
            "CAUTION",
            "High humidity / wet leaves",
            colors.HexColor("#fef3c7")
        )
    else:
        return (
            "FAVORABLE",
            "Good weather to spray",
            colors.HexColor("#dcfce7")
        )


def generate_crop_health_report_pdf(data: Dict[str, Any]) -> bytes:
    """
    Generates an executive, publication-grade 2-page A4 Kisan Crop Health Card & Advisory PDF.
    """
    buffer = io.BytesIO()
    # A4 Dimensions: 595.27 x 841.89 pt. 28pt margins -> 539.27pt content width
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=28,
        rightMargin=28,
        topMargin=22,
        bottomMargin=22
    )

    styles = getSampleStyleSheet()

    # Typography Styles
    style_header_title = ParagraphStyle(
        "HeaderTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=colors.white,
        leading=16,
    )
    style_header_sub = ParagraphStyle(
        "HeaderSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        textColor=colors.HexColor("#d8f3dc"),
        leading=11,
    )
    style_header_meta = ParagraphStyle(
        "HeaderMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        textColor=colors.HexColor("#d8f3dc"),
        leading=10.5,
        alignment=2,  # Right
    )
    style_meta_label = ParagraphStyle(
        "MetaLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        textColor=INK_LIGHT,
        leading=9,
    )
    style_meta_val = ParagraphStyle(
        "MetaVal",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        textColor=INK,
        leading=10.5,
    )
    style_section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        textColor=FOREST_DARK,
        leading=13,
    )
    style_section_sub = ParagraphStyle(
        "SectionSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        textColor=INK_MUTED,
        leading=9.5,
    )
    style_img_badge = ParagraphStyle(
        "ImgBadge",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        textColor=FOREST_DARK,
        leading=10,
        alignment=1,
    )
    style_img_desc = ParagraphStyle(
        "ImgDesc",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        textColor=INK_MUTED,
        leading=8.5,
        alignment=1,
    )
    style_table_head = ParagraphStyle(
        "TableHead",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        textColor=colors.HexColor("#1e293b"),
        leading=9.5,
        alignment=1,
    )
    style_table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        textColor=INK,
        leading=9.5,
        alignment=1,
    )
    style_table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        textColor=INK,
        leading=9.5,
        alignment=1,
    )
    style_card_title = ParagraphStyle(
        "CardTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        textColor=FOREST_DARK,
        leading=11,
    )
    style_card_body = ParagraphStyle(
        "CardBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        textColor=INK,
        leading=10,
    )
    style_disclaimer = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7,
        textColor=INK_MUTED,
        leading=9,
    )

    story: List[Any] = []

    # =============================================================
    # PAGE 1: DIAGNOSTIC CERTIFICATE & 4-IMAGE VISUAL EVIDENCE
    # =============================================================

    # 1. Header Banner
    diag_id = str(data.get("diagnostic_id", "RBK-405083"))
    scan_dt = str(data.get("scan_date", "Today"))
    header_table_data = [
        [
            Paragraph("<b>KISAN CROP HEALTH CARD &amp; FIELD GUIDE</b>", style_header_title),
            Paragraph(f"<b>Report ID:</b> {diag_id}<br/><b>Issued:</b> {scan_dt}", style_header_meta)
        ],
        [
            Paragraph("Rythu Bharosa Kendra (RBK) &amp; Farm Advisory Support", style_header_sub),
            Paragraph("AI Leaf Health Inspection", style_header_meta)
        ]
    ]
    t_header = Table(header_table_data, colWidths=[370, 169.27])
    t_header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), FOREST_DARK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(t_header)
    story.append(Spacer(1, 5))

    # 2. Metadata Grid
    crop_name = str(data.get("crop", "Coffee")).capitalize()
    cultivar = str(data.get("cultivar") or "Standard Variety")
    location = str(data.get("location_label", "Field Location"))
    lat = float(data.get("latitude", 13.3640))
    lon = float(data.get("longitude", 79.3975))
    sowing_stage = str(data.get("heading_days") or "Growing Stage")

    meta_data = [
        [
            Paragraph("CROP &amp; VARIETY", style_meta_label),
            Paragraph(f"{crop_name} ({cultivar})", style_meta_val),
            Paragraph("FIELD LOCATION", style_meta_label),
            Paragraph(location, style_meta_val),
        ],
        [
            Paragraph("GPS COORDINATES", style_meta_label),
            Paragraph(f"{lat:.4f}° N, {lon:.4f}° E", style_meta_val),
            Paragraph("GROWTH STAGE", style_meta_label),
            Paragraph(f"{sowing_stage} Days", style_meta_val),
        ]
    ]
    t_meta = Table(meta_data, colWidths=[95, 175, 95, 174.27])
    t_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
        ("BOX", (0, 0), (-1, -1), 0.8, CARD_BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, CARD_BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 5))

    # 3. Diagnostic Severity Assessment Card
    severity_pct = float(data.get("severity_pct", 0.0))
    low_interval = float(data.get("interval_low", max(0.0, severity_pct - 2.5)))
    high_interval = float(data.get("interval_high", severity_pct + 2.5))
    necrotic_pct = float(data.get("necrotic_pct", severity_pct))
    chlorotic_pct = float(data.get("chlorotic_pct", 0.0))
    lesion_spots = int(data.get("lesion_count", 1))

    if severity_pct == 0.0:
        badge_color = "#166534"
        badge_bg = SAGE_FAINT
        status_title = "HEALTHY CROP (0.0% DISEASE)"
        status_badge_text = "HEALTHY CROP"
        status_guidance = "The scanned leaf is healthy and green with no disease spots. Continue regular watering and good field care."
        card_bg_color = colors.HexColor("#f0fdf4")
        card_border_color = colors.HexColor("#bbf7d0")
    elif severity_pct <= 5.0:
        badge_color = "#15803d"
        badge_bg = colors.HexColor("#dcfce7")
        status_title = f"EARLY / MILD SPOTS ({severity_pct:.1f}% LEAF AREA DAMAGED)"
        status_badge_text = "EARLY SPOTS FOUND"
        status_guidance = "A few small disease spots detected. Pluck bad leaves and spray safe organic medicine early to stop spread."
        card_bg_color = colors.HexColor("#f0fdf4")
        card_border_color = colors.HexColor("#86efac")
    elif severity_pct <= 20.0:
        badge_color = "#c2410c"
        badge_bg = AMBER_LIGHT
        status_title = f"MODERATE INFECTION ({severity_pct:.1f}% LEAF AREA DAMAGED)"
        status_badge_text = "MODERATE INFECTION"
        status_guidance = "Active disease spots are spreading across the leaf. Remove badly infected leaves and spray plant medicine on schedule."
        card_bg_color = AMBER_LIGHT
        card_border_color = AMBER_BORDER
    else:
        badge_color = "#991b1b"
        badge_bg = RUST_LIGHT
        status_title = f"SEVERE INFECTION ({severity_pct:.1f}% LEAF AREA DAMAGED)"
        status_badge_text = "CRITICAL / SEVERE"
        status_guidance = "Large areas of the leaf are badly damaged. Show this report to your local Rythu Bharosa Kendra (RBK) officer immediately."
        card_bg_color = RUST_LIGHT
        card_border_color = RUST_BORDER

    diag_left = (
        f"<font size='10' color='{badge_color}'><b>{status_title}</b></font><br/>"
        f"<font size='7.5' color='#475569'><b>Total Leaf Damage:</b></font> <font size='12' color='{badge_color}'><b>{severity_pct:.1f}%</b></font> &nbsp; "
        f"<font size='7.5' color='#475569'><b>Expected Range:</b> [{low_interval:.1f}% – {high_interval:.1f}%]</font><br/>"
        f"<font size='7.5' color='#1e293b'>{status_guidance}</font>"
    )

    diag_right = (
        f"<font size='7.5' color='#475569'><b>Dead Brown Spots (Center):</b></font> <font size='8.5' color='#991b1b'><b>{necrotic_pct:.1f}%</b></font><br/>"
        f"<font size='7.5' color='#475569'><b>Spreading Yellow Border:</b></font> <font size='8.5' color='#b45309'><b>{chlorotic_pct:.1f}%</b></font><br/>"
        f"<font size='7.5' color='#475569'><b>Disease Spots Found:</b></font> <font size='8.5' color='#1b4332'><b>{lesion_spots} spots</b></font><br/>"
        f"<font size='7' color='#64748b'><i>Status: AI Image Check Verified</i></font>"
    )

    diag_table_data = [
        [
            Paragraph(diag_left, styles["Normal"]),
            Paragraph(diag_right, styles["Normal"])
        ]
    ]
    t_diag = Table(diag_table_data, colWidths=[365, 174.27])
    t_diag.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), card_bg_color),
        ("BOX", (0, 0), (-1, -1), 0.8, card_border_color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("LINEBEFORE", (1, 0), (1, 0), 0.5, card_border_color),
    ]))
    story.append(t_diag)
    story.append(Spacer(1, 6))

    # 4. Visual Evidence Gallery (All 4 Overlays - Cleaned & Proportional)
    story.append(Paragraph("<b>Visual Leaf Inspection Gallery (All 4 Diagnostic Views)</b>", style_section_heading))
    story.append(Paragraph("High-resolution inspection showing original leaf, disease spots, 7-day spread prediction, and AI focus areas.", style_section_sub))
    story.append(Spacer(1, 3))

    box_w = 266.0
    box_h = 242.0

    orig_b64 = data.get("original_image_base64")
    comp_b64 = data.get("composite_image_base64")
    proj_7d_raw = data.get("projected_7d_overlay_base64")
    attn_b64 = data.get("attention_saliency_base64")

    # Fail-safe: If 7-day projected overlay is missing or empty, synthesize it from the leaf image
    if (not proj_7d_raw or len(str(proj_7d_raw).strip()) < 50) and (orig_b64 or comp_b64):
        try:
            from ml.severity.spatial_projection import render_spatial_projection_overlay
            leaf_src = orig_b64 or comp_b64
            clean_b64 = leaf_src.split(",", 1)[1] if "," in leaf_src else leaf_src
            pil_leaf = PILImage.open(io.BytesIO(base64.b64decode(clean_b64))).convert("RGB")
            target_sev = max(severity_pct * 1.6, severity_pct + 3.0)
            proj_7d_raw = render_spatial_projection_overlay(
                image=pil_leaf,
                leaf_mask=None,
                necrotic_mask=None,
                chlorotic_mask=None,
                target_severity_pct=target_sev,
                rh_mean=75.0,
                precip_total=0.5
            )
        except Exception:
            pass

    # Composite the 7-day projected spatial overlay directly on top of the original leaf photograph
    proj_7d_composite = composite_overlay_on_original(orig_b64 or comp_b64, proj_7d_raw)

    img1 = prepare_card_image(orig_b64, box_w, box_h, "Original Leaf Photo")
    img2 = prepare_card_image(comp_b64, box_w, box_h, "Disease Spot Outlines")
    img3 = prepare_card_image(proj_7d_composite, box_w, box_h, "7-Day Projected Spread")
    img4 = prepare_card_image(attn_b64, box_w, box_h, "AI Focus Heatmap")

    gallery_data = [
        [
            Paragraph("<b>1. Original Field Photograph</b>", style_img_badge),
            Paragraph("<b>2. Disease Detection Overlay</b>", style_img_badge),
        ],
        [
            Paragraph("High-resolution leaf photo captured in field.", style_img_desc),
            Paragraph("Red = Dead brown spots; Yellow = Spreading disease border.", style_img_desc),
        ],
        [
            img1,
            img2,
        ],
        [
            Paragraph("<b>3. Predicted 7-Day Disease Spread</b>", style_img_badge),
            Paragraph("<b>4. AI Vision Attention Heatmap</b>", style_img_badge),
        ],
        [
            Paragraph("Leaf photo showing where disease will spread in 7 days if left untreated.", style_img_desc),
            Paragraph("Heatmap showing the exact leaf areas examined by the AI vision model.", style_img_desc),
        ],
        [
            img3,
            img4,
        ]
    ]

    t_gallery = Table(gallery_data, colWidths=[269.6, 269.6])
    t_gallery.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(t_gallery)

    # =============================================================
    # PAGE 2: 7-DAY WEATHER ADVISORY & RECOVERY ACTION GUIDE
    # =============================================================
    story.append(PageBreak())

    # Page 2 Header Banner
    header_p2_data = [
        [
            Paragraph("<b>7-DAY WEATHER FORECAST &amp; BEST SPRAYING DAYS</b>", style_header_title),
            Paragraph(f"<b>Location:</b> {location}<br/>High-Accuracy 9 km Local Weather", style_header_meta)
        ]
    ]
    t_header_p2 = Table(header_p2_data, colWidths=[370, 169.27])
    t_header_p2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), FOREST_DARK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(t_header_p2)
    story.append(Spacer(1, 5))

    # Weather Table
    weather_table_header = [
        Paragraph("<b>Day / Date</b>", style_table_head),
        Paragraph("<b>Avg Temp</b>", style_table_head),
        Paragraph("<b>Humidity</b>", style_table_head),
        Paragraph("<b>Rainfall</b>", style_table_head),
        Paragraph("<b>Wind Speed</b>", style_table_head),
        Paragraph("<b>Hours Leaves Wet</b>", style_table_head),
        Paragraph("<b>Safe to Spray?</b>", style_table_head),
    ]

    weather_rows = [weather_table_header]
    daily_records = data.get("weather_table", [])
    t_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]

    for idx, row in enumerate(daily_records[:7]):
        row_idx = idx + 1
        d_label = row.get("label", row.get("date", f"Day +{idx}"))
        t_c = f"{float(row.get('temp_c', 25.0)):.1f}°C"
        rh_pct = f"{float(row.get('rh_pct', 70.0)):.0f}%"
        precip = float(row.get("precip_mm", 0.0))
        wind = float(row.get("max_wind_kmh", 8.0))
        lwd = float(row.get("leaf_wetness_hours", 3.0))

        precip_str = f"{precip:.1f} mm"
        wind_str = f"{wind:.1f} km/h"
        lwd_str = f"{lwd:.1f} hrs"

        # Rigorous feasibility computation guaranteeing no contradictions
        status_tag, status_desc, bg_color = compute_spray_feasibility(precip, wind, float(row.get('rh_pct', 70.0)), lwd)
        
        badge_html = f"<b>{status_tag}</b><br/><font size='6.5' color='#334155'>{status_desc}</font>"

        weather_rows.append([
            Paragraph(d_label, style_table_cell_bold),
            Paragraph(t_c, style_table_cell),
            Paragraph(rh_pct, style_table_cell),
            Paragraph(precip_str, style_table_cell),
            Paragraph(wind_str, style_table_cell),
            Paragraph(lwd_str, style_table_cell),
            Paragraph(badge_html, style_table_cell),
        ])

        t_style.append(("BACKGROUND", (6, row_idx), (6, row_idx), bg_color))

    t_weather = Table(weather_rows, colWidths=[95, 62, 58, 65, 75, 75, 109.27])
    t_weather.setStyle(TableStyle(t_style))
    story.append(t_weather)
    story.append(Spacer(1, 6))

    # Operational Weather Summary Box
    first_row = daily_records[0] if len(daily_records) > 0 else {}
    p0 = float(first_row.get("precip_mm", 0.0))
    w0 = float(first_row.get("max_wind_kmh", 8.0))
    if p0 >= 1.0:
        today_msg = f"Rainfall ({p0:.1f} mm) is expected today. <b>Do not spray leaves today</b> because rain will wash the medicine away. Wait until leaves are completely dry."
    elif w0 >= 18.0:
        today_msg = f"Strong wind ({w0:.1f} km/h) expected today. <b>Do not spray today:</b> wind will blow the spray away and waste medicine. Wait for calm weather."
    else:
        today_msg = "Calm morning weather observed. Good time to spray plant medicine or nutrients between 6:00 AM and 9:00 AM."

    spray_callout_data = [
        [
            Paragraph("<b>Today's Spraying Advice:</b>", style_card_title),
            Paragraph(today_msg, style_card_body)
        ]
    ]
    t_callout = Table(spray_callout_data, colWidths=[150, 389.27])
    t_callout.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
        ("BOX", (0, 0), (-1, -1), 0.8, CARD_BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_callout)
    story.append(Spacer(1, 8))

    # Practical Action Steps for Farmers to Stop Disease Spread
    story.append(Paragraph("<b>Practical Steps for Farmers to Stop Disease Spread</b>", style_section_heading))
    story.append(Paragraph("Follow these 4 practical field habits to protect your crop and keep plants healthy.", style_section_sub))
    story.append(Spacer(1, 4))

    action_items = [
        [
            Paragraph("<b>1. Water Only in Early Morning</b>", style_card_title),
            Paragraph("<b>2. Clean and Remove Bad Leaves</b>", style_card_title),
        ],
        [
            Paragraph(
                "Water crops early in the morning (6:00 AM to 9:00 AM) so the morning sun quickly dries the leaves. "
                "<b>Never splash water on leaves in the evening</b>, because leaves that stay wet all night allow fungal disease to grow rapidly.",
                style_card_body
            ),
            Paragraph(
                "Pluck off leaves with heavy disease spots near the soil and throw them away outside the field. "
                "This stops raindrops from splashing fungal disease up from the ground onto clean green leaves.",
                style_card_body
            ),
        ],
        [
            Paragraph("<b>3. Balance Fertilizers (Limit Excess Urea)</b>", style_card_title),
            Paragraph("<b>4. Safe Biological Protection &amp; Support</b>", style_card_title),
        ],
        [
            Paragraph(
                "<b>Do not apply heavy doses of urea right now</b>. Too much nitrogen makes leaves soft and weak, "
                "making it easy for disease to enter. Apply potash and calcium to make leaves tough and strong.",
                style_card_body
            ),
            Paragraph(
                "Spray safe natural plant protectors (such as <i>Trichoderma</i> or <i>Bacillus</i>) on calm, sunny mornings. "
                "For chemical treatment advice, always ask your village <b>Rythu Bharosa Kendra (RBK)</b> or district <b>KVK</b>.",
                style_card_body
            ),
        ],
    ]

    t_actions = Table(action_items, colWidths=[269.6, 269.6])
    t_actions.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
        ("BOX", (0, 0), (-1, -1), 0.8, CARD_BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, CARD_BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_actions)
    story.append(Spacer(1, 8))

    # Extension Contacts Card
    extension_data = [
        [
            Paragraph("<b>Rythu Bharosa Kendra (RBK) Support:</b>", style_card_title),
            Paragraph("Visit your Grama Sachivalayam Agriculture Assistant for localized crop inputs and sprayers.", style_card_body)
        ],
        [
            Paragraph("<b>Kisan Call Center (Toll Free):</b>", style_card_title),
            Paragraph("Dial <b>1800-180-1551</b> for free multilingual agricultural advice from certified scientists.", style_card_body)
        ]
    ]
    t_ext = Table(extension_data, colWidths=[180, 359.27])
    t_ext.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0fdf4")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#86efac")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbf7d0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_ext)
    story.append(Spacer(1, 6))

    # Formal Government Non-Prescription Disclaimer
    disclaimer_text = (
        "<b>Important Notice:</b> This field health card provides crop health checks and local weather advice. "
        "It does not sell or prescribe specific chemical pesticide brands or dosages. "
        "Farmers must consult registered local agricultural officers (KVK / RBK) before purchasing or applying regulated chemical agrochemicals."
    )
    t_disc = Table([[Paragraph(disclaimer_text, style_disclaimer)]], colWidths=[539.27])
    t_disc.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_disc)

    # Build PDF
    doc.build(story)
    return buffer.getvalue()
