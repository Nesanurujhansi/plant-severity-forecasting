"""
Phase 11: Gradio Web User Interface Module
Provides an accessible, interactive Gradio Blocks interface mounted inside FastAPI at /ui,
reusing the shared InferencePipeline without HTTP overhead.
"""

import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import gradio as gr
from PIL import Image

from app.inference.loader import get_model_container
from app.inference.pipeline import InferencePipeline
from app.ui.plots import blend_mask_overlay, plot_omitted_notice, plot_progression_forecast, plot_scenario_comparison
from app.ui.strings import get_string

logger = logging.getLogger("app.ui.gradio_ui")

# Global pipeline instance initialized on demand
_ui_pipeline: Optional[InferencePipeline] = None


def get_ui_pipeline() -> InferencePipeline:
    """Get or create singleton InferencePipeline for UI callbacks."""
    global _ui_pipeline
    if _ui_pipeline is None:
        container = get_model_container()
        _ui_pipeline = InferencePipeline(model_container=container)
    return _ui_pipeline


def run_ui_prognosis(
    image: Optional[Image.Image],
    crop: str = "wheat",
    cultivar: Optional[str] = None,
    days_since_heading: Optional[float] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    weather_scenario: str = "as_recent",
    weather_override_json: Optional[str] = None,
    include_treatment: bool = True,
    overlay_opacity: float = 0.5
) -> Tuple[
    str,            # Stub Banner
    str,            # Safety / Confidence Status Markdown
    Optional[Image.Image], # Original Image
    Optional[Image.Image], # Mask Overlay Image
    str,            # Severity Markdown Result
    Optional[Image.Image], # Forecast Chart Image
    Optional[Image.Image], # Scenario Chart Image
    List[List[Any]],# Weather Table
    str,            # System Warnings Markdown
    str             # Execution Timings Markdown
]:
    """
    Main callback executing disease prognosis pipeline and formatting display outputs for Gradio.
    """
    pipeline = get_ui_pipeline()
    bundle_info = pipeline.container.get_bundle_info()

    # Stub Banner Check
    stub_banner = bundle_info.get("banner", "") if bundle_info.get("is_stub") else ""

    if image is None:
        empty_safety = "### ⚠️ No Image Provided\nPlease upload or capture a photo of a crop leaf to analyze."
        return stub_banner, empty_safety, None, None, "N/A", None, None, [], "", ""

    # Convert PIL Image to Bytes for pipeline ingestion
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    # Parse weather override JSON if present
    parsed_override: Optional[Dict[str, Any]] = None
    if weather_override_json and weather_override_json.strip():
        try:
            parsed_override = json.loads(weather_override_json)
        except Exception:
            pass

    # Execute pipeline
    response = pipeline.process(
        image_bytes=image_bytes,
        filename="gradio_upload.png",
        crop=crop,
        cultivar=cultivar if cultivar and cultivar.strip() else None,
        days_since_heading=float(days_since_heading) if days_since_heading is not None else None,
        latitude=float(latitude) if latitude is not None else None,
        longitude=float(longitude) if longitude is not None else None,
        weather_override=parsed_override,
        weather_scenario=weather_scenario,
        horizons=[3, 7],
        include_treatment_scenario=include_treatment
    )

    # 1. Format Safety / Confidence Banner
    safety = response.safety
    tier = safety.confidence_tier
    if safety.should_abstain:
        badge_color = "red"
        status_title = "🚨 SAFETY ABSTENTION (Retake Photo Required)"
    elif tier == "HIGH":
        badge_color = "green"
        status_title = "✅ HIGH CONFIDENCE (Photo & Range Validated)"
    elif tier == "MEDIUM":
        badge_color = "orange"
        status_title = "⚠️ MEDIUM CONFIDENCE (Slight Quality Degradation)"
    else:
        badge_color = "darkred"
        status_title = "🛑 LOW CONFIDENCE (Unreliable Input)"

    safety_md = (
        f"### {status_title}\n"
        f"**User Guidance**: {safety.user_message}\n\n"
        f"- **Photo Quality Passed**: `{safety.quality_passed}`\n"
        f"- **Distribution Status**: `{safety.ood_status}`\n"
    )
    if safety.failure_reasons:
        safety_md += f"- **Issues Detected**: {', '.join(safety.failure_reasons)}\n"

    # If abstained, return early
    if safety.should_abstain or response.severity is None:
        return stub_banner, safety_md, image, None, "### Prediction Withheld Due to Safety Policy", None, None, [], "\n".join(response.warnings), f"Total Time: {response.timings.total_ms} ms"

    # 2. Format Severity Outputs & Overlay
    sev = response.severity
    overlay_img = blend_mask_overlay(image, sev.overlay_base64, opacity=overlay_opacity)

    healthy_badge = " *(Healthy Foliage - No Active Lesions)*" if sev.severity_pct == 0.0 else ""
    sev_md = (
        f"### Current Severity: `{sev.severity_pct:.1f}%`{healthy_badge}\n"
        f"**90% Conformal Interval**: `[{sev.severity_interval[0]:.1f}%, {sev.severity_interval[1]:.1f}%]`\n\n"
        f"- **Severity Class**: `{sev.severity_class}`\n"
        f"- **Connected Lesion Spots**: `{sev.lesion_count}` spots\n"
        f"- **Average Spot Size**: `{sev.mean_lesion_size_px:.1f} px`\n\n"
        f"*{get_string('conformal_explanation')}*"
    )

    # 3. Format Forecast Chart
    if response.forecast:
        fc_dict = response.forecast.model_dump()
        fc_chart = plot_progression_forecast(
            initial_severity=sev.severity_pct,
            forecast_data=fc_dict,
            preset_name=response.weather.preset
        )
    else:
        fc_chart = plot_omitted_notice(
            "Progression Forecast Omitted",
            f"Epidemic progression forecasting is calibrated exclusively for Wheat.\nSelected Crop: '{crop.capitalize()}'"
        )

    # 4. Format Scenario Chart
    if response.scenario:
        sc = response.scenario
        sc_chart = plot_scenario_comparison(
            unprotected_7d=sc.unprotected_7d_pct,
            protected_7d=sc.protected_7d_pct,
            relative_reduction_pct=sc.relative_reduction_pct
        )
    else:
        sc_chart = plot_omitted_notice(
            "Treatment Scenario Omitted",
            f"Treatment scenario simulations are calibrated exclusively for Wheat.\nSelected Crop: '{crop.capitalize()}'"
        )

    # 5. Format Weather Table
    w_table = []
    if response.weather and response.weather.table_summary:
        for row in response.weather.table_summary:
            w_table.append([
                row.get("label", f"Day +{row.get('day_offset', 0)}"),
                f"{row.get('temp_c', 0.0)} °C",
                f"{row.get('rh_pct', 0.0)} %",
                f"{row.get('vpd_kpa', 0.0)} kPa",
                f"{row.get('precip_mm', 0.0)} mm"
            ])

    # 6. Format Warnings & Timings
    warn_md = "\n".join([f"- ⚠️ {w}" for w in response.warnings]) if response.warnings else "No system warnings."
    timings_md = f"**Total Pipeline Duration**: `{response.timings.total_ms} ms`\n\n"
    for stage_name, stage_ms in response.timings.stages_ms.items():
        timings_md += f"- `{stage_name}`: `{stage_ms} ms`\n"

    return stub_banner, safety_md, image, overlay_img, sev_md, fc_chart, sc_chart, w_table, warn_md, timings_md


def create_gradio_ui() -> gr.Blocks:
    """
    Build Gradio Blocks UI interface.
    """
    examples_dir = Path("app/ui/examples")
    example_list = []
    if (examples_dir / "sample_leaf_01.png").exists():
        example_list.append([str(examples_dir / "sample_leaf_01.png"), "wheat", "as_recent"])
    if (examples_dir / "sample_leaf_02.png").exists():
        example_list.append([str(examples_dir / "sample_leaf_02.png"), "wheat", "warm_humid"])

    with gr.Blocks(title="crop-prognosis UI", analytics_enabled=False) as demo:
        gr.Markdown(f"# {get_string('app_title')}\n{get_string('app_subtitle')}")
        
        stub_banner_out = gr.Markdown("", visible=True)

        with gr.Tabs():
            # TAB 1: ANALYZE & RESULTS
            with gr.TabItem(get_string("tab_analyze")):
                with gr.Row():
                    # Left Column: Inputs
                    with gr.Column(scale=1):
                        gr.Markdown(f"### {get_string('input_header')}")
                        image_input = gr.Image(type="pil", label=get_string("image_label"), sources=["upload", "clipboard"])
                        
                        crop_input = gr.Dropdown(
                            choices=["wheat", "rice", "tomato", "corn", "sugarcane", "cotton"],
                            value="wheat",
                            label=get_string("crop_label"),
                            info=get_string("crop_wheat_notice")
                        )
                        cultivar_input = gr.Textbox(label=get_string("cultivar_label"), placeholder="e.g. ArinaLrFor")
                        heading_days_input = gr.Number(label=get_string("heading_days_label"), value=None)

                        with gr.Accordion(get_string("geo_header"), open=False):
                            gr.Markdown(get_string("geo_notice"))
                            lat_input = gr.Number(label=get_string("lat_label"), value=None)
                            lon_input = gr.Number(label=get_string("lon_label"), value=None)

                        weather_scenario_input = gr.Dropdown(
                            choices=["as_recent", "live_nwp", "climatology", "warm_humid", "cool_dry", "user_entered"],
                            value="as_recent",
                            label=get_string("weather_scenario_label")
                        )
                        weather_override_input = gr.Textbox(
                            label=get_string("weather_table_override_label"),
                            placeholder='{"temp_c": 20.0, "rh_pct": 85.0}',
                            lines=2
                        )
                        treatment_input = gr.Checkbox(value=True, label=get_string("treatment_checkbox_label"))
                        
                        analyze_btn = gr.Button(get_string("btn_analyze"), variant="primary", size="lg")

                        if example_list:
                            gr.Examples(examples=example_list, inputs=[image_input, crop_input, weather_scenario_input])

                    # Right Column: Results
                    with gr.Column(scale=1):
                        gr.Markdown(f"### {get_string('results_header')}")
                        safety_out = gr.Markdown("### Upload a photo and click Analyze to view prognosis.")
                        
                        with gr.Row():
                            orig_img_out = gr.Image(label="Original Upload", interactive=False)
                            overlay_img_out = gr.Image(label="Lesion Mask Overlay", interactive=False)
                        
                        opacity_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label=get_string("overlay_transparency_label"))

                        severity_out = gr.Markdown()
                        
                        with gr.Row():
                            forecast_plot_out = gr.Image(label="Progression Forecast Chart", interactive=False)
                            scenario_plot_out = gr.Image(label="Treatment Scenario Chart", interactive=False)

                        weather_table_out = gr.Dataframe(
                            headers=["Period", "Temperature", "Relative Humidity", "VPD (kPa)", "Precipitation"],
                            label=get_string("weather_summary_header"),
                            interactive=False
                        )

                        warnings_out = gr.Markdown()
                        
                        with gr.Accordion(get_string("timings_header"), open=False):
                            timings_out = gr.Markdown()

                # Event Wire
                analyze_btn.click(
                    fn=run_ui_prognosis,
                    inputs=[
                        image_input, crop_input, cultivar_input, heading_days_input,
                        lat_input, lon_input, weather_scenario_input, weather_override_input,
                        treatment_input, opacity_slider
                    ],
                    outputs=[
                        stub_banner_out, safety_out, orig_img_out, overlay_img_out,
                        severity_out, forecast_plot_out, scenario_plot_out,
                        weather_table_out, warnings_out, timings_out
                    ]
                )

            # TAB 2: HOW IT WORKS
            with gr.TabItem(get_string("tab_how_it_works")):
                gr.Markdown(
                    "## How crop-prognosis Works\n\n"
                    "### The Four Core Deep Learning Modules\n"
                    "1. **Vision Severity Module**: Uses Meta AI's frozen DINOv2 Vision Transformer (`facebook/dinov2-small`) with SegFormer FPN attention heads to calculate leaf and lesion pixel areas. Outputs a guaranteed 90% Mondrian Conformal interval.\n"
                    "2. **Progression Forecasting Module**: A Physics-Informed Neural ODE (PINN-ODE) modeling Yan & Hunt thermal biology and Vapor Pressure Deficit (VPD in kPa) thermodynamics, fused with LightGBM logit residual regression.\n"
                    "3. **Treatment Scenario Module**: Deep Counterfactual Treatment Network (TarNet) estimating trajectories under untreated vs fungicide-protected regimes.\n"
                    "4. **Safety & Quality Guardrails**: Deep Free Energy OOD detector ($E(\\mathbf{x}) = -T \\log \\sum \\exp(f_i/T)$) and Laplacian blur checks ensuring reliable predictions.\n\n"
                    "### How to Take a Good Leaf Photo\n"
                    "- ☀️ Use bright natural daylight (avoid dark indoor lighting or heavy shadows).\n"
                    "- 🍃 Lay leaf flat and fill the camera frame.\n"
                    "- 🎯 Focus sharply on diseased lesion spots.\n"
                )

            # TAB 3: ABOUT & LIMITATIONS
            with gr.TabItem(get_string("tab_about")):
                gr.Markdown(
                    "## About & Scientific Limitations\n\n"
                    "### Dataset Sources & Licences\n"
                    "- **ETH Zurich Wheat Time Series**: Scientific Data 2026 (CC-BY-4.0)\n"
                    "- **PlantSeg Disease Segmentation**: HuggingFace Subset (CC-BY-4.0)\n"
                    "- **Coffee Leaf Rust CV**: HuggingFace Repository (MIT)\n\n"
                    "### Non-Prescription Disclaimer\n"
                    f"{get_string('global_disclaimer')}\n\n"
                    "### Privacy Statement\n"
                    f"{get_string('privacy_statement')}\n"
                )

    return demo
