"""
Phase 11: Centralized UI Text Strings and Localization Dictionary
Provides all user-visible text, labels, tooltips, disclaimers, and screen-reader accessibility descriptions
keyed by language code (default "en").
"""

STRINGS = {
    "en": {
        "app_title": "crop-prognosis: Automated Crop Disease Severity & Forecast Decision Support",
        "app_subtitle": "Upload ONE leaf photo to estimate disease severity, forecast 3 & 7-day progression, and compare treatment scenarios.",
        "stub_banner": "[WARNING] DEV_STUB_MODELS ACTIVE - NOT REAL PREDICTIONS",
        
        # Tabs
        "tab_analyze": "📷 Analyze Leaf Photo",
        "tab_results": "📊 Prognosis Results",
        "tab_how_it_works": "🧠 How It Works",
        "tab_about": "ℹ️ About & Limitations",
        
        # Input Section
        "input_header": "1. Upload Leaf Image & Set Parameters",
        "image_label": "Upload Crop Leaf Photo",
        "crop_label": "Host Crop Category",
        "crop_wheat_notice": "Disease progression and treatment scenarios are calibrated specifically for Wheat host models. Other crops evaluate severity only.",
        "cultivar_label": "Cultivar / Genotype (Optional)",
        "heading_days_label": "Days Since Heading (Optional)",
        "geo_header": "Field Geolocation Coordinates (Optional)",
        "geo_notice": "Coordinates are used strictly to query historical weather observations from NASA POWER API. Location metadata is NEVER stored.",
        "lat_label": "Latitude (-90 to 90)",
        "lon_label": "Longitude (-180 to 180)",
        "weather_scenario_label": "Future Weather Scenario Preset",
        "weather_table_override_label": "Custom Weather Parameters Override (JSON format, Optional)",
        "treatment_checkbox_label": "Include Fungicide-Protected vs Untreated 7-Day Scenario Comparison",
        "btn_analyze": "🔍 Run Disease Prognosis Analysis",
        
        # Results Section
        "results_header": "2. Prognosis & Risk Evaluation Summary",
        "confidence_header": "Safety & Quality Assessment",
        "severity_header": "Current Visual Disease Severity",
        "severity_pct_label": "Leaf Area Affected",
        "severity_interval_label": "90% Mondrian Conformal Prediction Interval",
        "severity_class_label": "Severity Level Class",
        "overlay_transparency_label": "Lesion Mask Overlay Opacity",
        "forecast_header": "3 & 7-Day Weather-Aware Disease Progression Forecast",
        "scenario_header": "7-Day Fungicide-Protected Treatment Scenario Comparison",
        "weather_summary_header": "Microclimate Weather Data Used",
        "warnings_header": "System Warnings & Notifications",
        "timings_header": "Processing Execution Performance Timings",
        
        # Explanations & Disclaimers
        "conformal_explanation": "The prediction interval guarantees that the true disease severity falls within these bounds with 90% statistical probability.",
        "scenario_disclaimer": "Scenario predictions estimate trajectories under Swiss wheat field trial conditions. This tool provides non-prescription decision support ONLY and does NOT give chemical pesticide names, doses, or spray timings.",
        "global_disclaimer": "This tool provides decision support for agricultural students and farmers. It NEVER replaces a certified professional agronomist.",
        "privacy_statement": "Your privacy matters: Uploaded photos are processed in memory and deleted immediately. Uploads are NEVER saved to disk or shared with third parties by default.",
        
        # Accessibility Text
        "forecast_chart_alt": "Line chart showing disease severity today and projected expansion over 3 and 7 days with shaded 90% confidence interval bands.",
        "scenario_chart_alt": "Bar chart comparing 7-day predicted disease severity under untreated conditions versus a fungicide-protected regime.",
    }
}


def get_string(key: str, lang: str = "en") -> str:
    """Retrieve localized string by key."""
    lang_dict = STRINGS.get(lang, STRINGS["en"])
    return lang_dict.get(key, STRINGS["en"].get(key, f"[{key}]"))
