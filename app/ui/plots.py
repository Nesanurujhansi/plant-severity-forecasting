"""
Phase 11: Plotting and Visualization Engine
Generates publication-quality Matplotlib figures for 3 & 7-day disease progression forecasts
and counterfactual treatment scenario comparisons.
"""

import base64
import io
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive CPU backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_progression_forecast(
    initial_severity: float,
    forecast_data: Optional[Dict[str, Any]] = None,
    preset_name: str = "Live NWP Microclimate"
) -> Image.Image:
    """
    Generate a 3-day and 7-day disease progression forecast curve with shaded 90% conformal prediction intervals.
    
    Returns:
        PIL Image object containing rendered plot.
    """
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=120)

    days = [0, 3, 7]
    severities = [initial_severity]
    lowers = [initial_severity * 0.9]
    uppers = [initial_severity * 1.1]

    if forecast_data and "horizons" in forecast_data:
        hz = forecast_data["horizons"]
        h3 = hz.get("3d", {}) if isinstance(hz, dict) else getattr(hz, "3d", {})
        h7 = hz.get("7d", {}) if isinstance(hz, dict) else getattr(hz, "7d", {})

        def get_field(obj, key, default):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        sev3 = get_field(h3, "severity_pct", initial_severity * 1.3)
        int3 = get_field(h3, "interval", [sev3 * 0.85, sev3 * 1.25])
        severities.append(sev3)
        lowers.append(int3[0])
        uppers.append(int3[1])

        sev7 = get_field(h7, "severity_pct", initial_severity * 1.8)
        int7 = get_field(h7, "interval", [sev7 * 0.85, sev7 * 1.25])
        severities.append(sev7)
        lowers.append(int7[0])
        uppers.append(int7[1])
    else:
        # Fallback values
        sev3 = initial_severity * 1.3
        severities.append(sev3)
        lowers.append(sev3 * 0.85)
        uppers.append(sev3 * 1.25)

        sev7 = initial_severity * 1.8
        severities.append(sev7)
        lowers.append(sev7 * 0.85)
        uppers.append(sev7 * 1.25)

    # Plot forecast trajectory line and points
    # Strict project palette: #bc4749 (rust) for disease line, #bc4749 with alpha for conformal band
    ax.plot(days, severities, marker="o", color="#bc4749", linewidth=2.5, markersize=8, label="Predicted Severity (%)")
    ax.fill_between(days, lowers, uppers, color="#bc4749", alpha=0.18, label="90% Conformal Interval Band")

    # Styling following Apple design principles
    title_suffix = " (Healthy Baseline)" if initial_severity == 0.0 else ""
    title_text = "7-Day Epidemic Progression Forecast (Open-Meteo 9km NWP)"
    ax.set_title(f"{title_text}{title_suffix}", fontsize=11, fontweight="bold", pad=12, color="#1b1b1b")
    ax.set_xlabel("Timeline (Days)", fontsize=10, color="#4a4a4a", labelpad=8)
    ax.set_ylabel("Leaf Area Affected (%)", fontsize=10, color="#4a4a4a", labelpad=8)
    ax.set_xticks(days)
    ax.set_xticklabels(["Today (Day 0)", "+3 Days", "+7 Days"], fontsize=9)
    max_y = max(10.0, min(100.0, max(uppers) * 1.25))
    ax.set_ylim(0, max_y)
    ax.grid(True, linestyle="--", alpha=0.4, color="#dcdcdc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, edgecolor="#e0e0e0")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def plot_scenario_comparison(
    unprotected_7d: float = 22.5,
    protected_7d: float = 13.8,
    relative_reduction_pct: float = 38.7
) -> Image.Image:
    """
    Generate a bar chart comparing 7-day predicted disease severity under untreated vs fungicide-protected regimes.
    Styled according to Apple design principles with project palette (#bc4749 rust, #386641 forest green).
    
    Returns:
        PIL Image object containing rendered chart.
    """
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=120)

    categories = ["Untreated (7 Days)", "Fungicide-Protected (7 Days)"]
    values = [unprotected_7d, protected_7d]
    # Project palette: #bc4749 (rust / untreated), #386641 (forest green / fungicide protected)
    colors = ["#bc4749", "#386641"]

    bars = ax.bar(categories, values, color=colors, width=0.42, edgecolor="none", zorder=3)

    # Add numeric labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#1b1b1b"
        )

    title_str = (
        f"7-Day Treatment Scenario Comparison (-{relative_reduction_pct:.1f}% Reduction)"
        if relative_reduction_pct > 0
        else "7-Day Treatment Scenario Comparison (Healthy Leaf / 0% Baseline)"
    )
    ax.set_title(title_str, fontsize=12, fontweight="bold", pad=12, color="#1b1b1b")
    ax.set_ylabel("Predicted 7-Day Severity (%)", fontsize=10, color="#4a4a4a", labelpad=8)
    max_y = max(10.0, min(100.0, max(values) * 1.35))
    ax.set_ylim(0, max_y)
    ax.grid(axis="y", linestyle="--", alpha=0.4, color="#dcdcdc", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="x", labelsize=9.5)
    ax.tick_params(axis="y", labelsize=9)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def blend_mask_overlay(
    original_img: Image.Image,
    overlay_base64: Optional[str],
    opacity: float = 0.5
) -> Image.Image:
    """
    Blend base64 PNG overlay onto original PIL image at specified opacity.
    """
    if not overlay_base64 or not isinstance(overlay_base64, str):
        return original_img

    try:
        overlay_bytes = base64.b64decode(overlay_base64)
        overlay_img = Image.open(io.BytesIO(overlay_bytes)).convert("RGBA")
        overlay_resized = overlay_img.resize(original_img.size, Image.Resampling.NEAREST)

        base_rgba = original_img.convert("RGBA")

        # Adjust alpha channel opacity
        r, g, b, a = overlay_resized.split()
        a = a.point(lambda p: int(p * opacity))
        overlay_adjusted = Image.merge("RGBA", (r, g, b, a))

        blended = Image.alpha_composite(base_rgba, overlay_adjusted)
        return blended.convert("RGB")
    except Exception:
        return original_img


def plot_omitted_notice(title: str, subtitle: str) -> Image.Image:
    """Generate a clean informational banner figure when module predictions are omitted."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=120)
    ax.set_facecolor("#22272e")
    fig.patch.set_facecolor("#22272e")
    ax.axis("off")

    ax.text(0.5, 0.58, title, fontsize=13, fontweight="bold", color="#f0883e", ha="center", va="center")
    ax.text(0.5, 0.38, subtitle, fontsize=10, color="#adbac7", ha="center", va="center", multialignment="center")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

