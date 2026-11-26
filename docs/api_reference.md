# API Reference Manual

This document provides a comprehensive specification for the **crop-prognosis** REST API backend endpoints (`app/main.py`), including request payloads, response fields, error codes, and example cURL / Python requests.

---

## 1. Overview & Base URL

* **Base URL**: `http://127.0.0.1:8000`
* **Interactive OpenAPI Docs**: `http://127.0.0.1:8000/docs` (Swagger UI)
* **ReDoc Documentation**: `http://127.0.0.1:8000/redoc`
* **Content-Type**: `multipart/form-data` for analysis uploads, `application/json` for standard responses.

---

## 2. Endpoints Summary

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Server liveness check and bundle version status | No |
| `GET` | `/model-info` | Inspect active model bundle metadata, evaluation metrics, and scientific disclaimers | No |
| `POST` | `/analyze` | Primary disease prognosis endpoint (photo upload + microclimate analysis) | No |
| `GET` | `/weather` | Query daily weather observations for a given geolocation coordinate | No |
| `POST` | `/api/export-health-report-pdf` | Generate publication-grade 2-page A4 field advisory health card PDF | No |

---

## 3. Detailed Endpoint Specifications

### 3.1 Health Check (`GET /health`)

Returns application liveness, version, and whether dev stub models are active.

#### Example Request
```bash
curl -X GET "http://127.0.0.1:8000/health"
```

#### Example Response (HTTP 200)
```json
{
  "status": "ok",
  "version": "1.0.0",
  "is_stub": false,
  "bundle_version": "1.0.0"
}
```

---

### 3.2 Model Bundle Information (`GET /model-info`)

Returns active model bundle version, ISO creation timestamp, git commit, headline metrics summary, dataset licences, and scientific non-prescription disclaimers.

#### Example Request
```bash
curl -X GET "http://127.0.0.1:8000/model-info"
```

#### Example Response (HTTP 200)
```json
{
  "bundle_version": "1.0.0",
  "created_at": "2026-09-22T18:00:00Z",
  "git_commit": "a1b2c3d4",
  "is_stub": false,
  "stub_banner": null,
  "metrics_summary": {
    "severity_mae": 0.038,
    "severity_conformal_coverage": 0.912,
    "forecast_3d_mae": 0.045,
    "forecast_cqr_coverage": 0.895,
    "scenario_atet": -0.142,
    "safety_ood_detection_rate": 1.000
  },
  "dataset_identifiers": [
    {
      "name": "ETH Zurich Wheat Leaf Time Series",
      "doi_or_url": "https://doi.org/10.3929/ethz-b-000739194",
      "licence": "CC-BY-4.0"
    }
  ],
  "disclaimer": "This tool provides non-prescription decision support only. It NEVER provides pesticide names, doses, or spray timings, and does not replace an agronomist.",
  "known_limitations": [
    "Wheat host specific for progression & treatment scenarios",
    "Single field site training (ETH Zurich Lindau-Eschikon)",
    "Coarse weather meteorology resolution (NASA POWER ~50km)"
  ]
}
```

---

### 3.3 Main Disease Prognosis Endpoint (`POST /analyze`)

Accepts a single crop leaf image upload and optional form parameters to return continuous severity %, prediction intervals, 3-day and 7-day weather-aware disease forecasts, counterfactual treatment scenario comparisons, and base64 PNG mask overlays.

#### Form Parameters

| Parameter | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `image` | File | **Yes** | — | Single leaf photo (JPEG, PNG, or WebP; max 10MB) |
| `crop` | string | No | `"wheat"` | Host crop category (e.g., `wheat`, `rice`, `tomato`) |
| `cultivar` | string | No | `null` | Optional genotype / cultivar name |
| `days_since_heading` | float | No | `null` | Growth stage days since heading |
| `latitude` | float | No | `null` | Field latitude (-90.0 to 90.0) |
| `longitude` | float | No | `null` | Field longitude (-180.0 to 180.0) |
| `weather_override` | string (JSON) | No | `null` | Custom JSON string (e.g. `{"temp_c": 20.0, "rh_pct": 80.0}`) |
| `weather_scenario` | string | No | `"as_recent"` | Future preset: `as_recent`, `climatology`, `warm_humid`, `cool_dry`, `user_entered` |
| `horizons` | string | No | `"3,7"` | Comma-separated forecast horizons in days |
| `include_treatment_scenario` | boolean | No | `true` | Include fungicide-protected vs untreated scenario comparison |

#### Example cURL Request
```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "image=@leaf_test.jpg" \
  -F "crop=wheat" \
  -F "latitude=47.45" \
  -F "longitude=8.68" \
  -F "weather_scenario=as_recent"
```

#### Example Python Requests
```python
import requests

url = "http://127.0.0.1:8000/analyze"
files = {"image": ("leaf_test.jpg", open("leaf_test.jpg", "rb"), "image/jpeg")}
data = {
    "crop": "wheat",
    "latitude": 47.45,
    "longitude": 8.68,
    "weather_scenario": "as_recent"
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

#### Example Response (HTTP 200)
```json
{
  "bundle_info": {
    "version": "1.0.0",
    "created_at": "2026-09-22T18:00:00Z",
    "git_commit": "a1b2c3d4",
    "is_stub": false,
    "stub_banner": null
  },
  "safety": {
    "confidence_tier": "HIGH",
    "should_abstain": false,
    "quality_passed": true,
    "ood_status": "IN_DISTRIBUTION",
    "user_message": "Photo accepted with High confidence",
    "module_gating": {
      "allow_severity": true,
      "allow_progression": true,
      "allow_scenario": true
    },
    "failure_reasons": []
  },
  "severity": {
    "severity_pct": 12.5,
    "severity_interval": [9.5, 16.0],
    "severity_class": "10-25%",
    "lesion_count": 3,
    "mean_lesion_size_px": 115.0,
    "leaf_mask_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
    "lesion_mask_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
    "overlay_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
  },
  "forecast": {
    "horizons": {
      "3d": {
        "horizon_days": 3,
        "severity_pct": 16.3,
        "interval": [13.8, 20.3],
        "drivers": ["Temperature (18.5°C)", "VPD (0.47 kPa)"]
      },
      "7d": {
        "horizon_days": 7,
        "severity_pct": 21.3,
        "interval": [18.1, 26.6],
        "drivers": ["Temperature (18.5°C)", "VPD (0.47 kPa)"]
      }
    }
  },
  "scenario": {
    "unprotected_7d_pct": 22.5,
    "protected_7d_pct": 13.8,
    "relative_reduction_pct": 38.7,
    "interval": [33.7, 43.7],
    "label": "Fungicide-Protected vs Untreated 7-Day Comparison",
    "disclaimer": "Estimates expected trajectory under fungicide protection in trial plots. Does NOT prescribe chemical names, doses, or spray timing."
  },
  "weather": {
    "source": "NASA POWER Daily API (Grid Cell Lat: 47.45, Lon: 8.68)",
    "preset": "as_recent",
    "table_summary": [
      {"day_offset": 0, "label": "Observed Today", "temp_c": 18.5, "rh_pct": 78.0, "vpd_kpa": 0.47, "precip_mm": 1.5},
      {"day_offset": 3, "label": "Day +3 (as_recent)", "temp_c": 18.5, "rh_pct": 78.0, "vpd_kpa": 0.47, "precip_mm": 1.5},
      {"day_offset": 7, "label": "Day +7 (as_recent)", "temp_c": 18.5, "rh_pct": 78.0, "vpd_kpa": 0.47, "precip_mm": 1.5}
    ]
  },
  "warnings": [],
  "timings": {
    "total_ms": 14.8,
    "stages_ms": {
      "upload_safety_ms": 1.2,
      "safety_eval_ms": 2.1,
      "severity_pred_ms": 4.5,
      "weather_res_ms": 0.8,
      "progression_ms": 3.2,
      "scenario_ms": 3.0
    }
  },
  "disclaimer": "This tool provides non-prescription decision support only. It NEVER provides pesticide names, doses, or spray timings, and does not replace an agronomist."
}
```

---

### 3.5 Field Health Card PDF Export (`POST /api/export-health-report-pdf` or `/export-health-report-pdf`)

Generates a publication-grade, 2-page A4 field crop health advisory PDF report using ReportLab with 4 inspection cards (Original Photograph, Disease Spot Outlines, 7-Day Disease Spread Composite, and AI Focus Areas), 7-day weather forecast, and actionable field care checklists in plain farmer-friendly language.

#### Example Request
```bash
curl -X POST "http://127.0.0.1:8000/api/export-health-report-pdf" \
  -H "Content-Type: application/json" \
  -d '{
    "diagnostic_id": "RBK-405083",
    "scan_date": "24 Nov 2026",
    "crop": "Coffee",
    "location_label": "Guntur Rural, AP",
    "severity_pct": 14.5,
    "interval_low": 11.2,
    "interval_high": 17.8,
    "necrotic_pct": 8.5,
    "chlorotic_pct": 6.0,
    "lesion_count": 5,
    "original_image_base64": "<base64_png>",
    "composite_image_base64": "<base64_png>",
    "projected_7d_overlay_base64": "<base64_png>",
    "attention_saliency_base64": "<base64_png>",
    "weather_table": [
      {"date": "2026-11-25", "temp_max": 30.5, "temp_min": 21.0, "rh_mean": 75.0, "wind_kmh": 10.0, "rain_prob": 10, "precip_mm": 0.0, "condition": "Clear", "spray_feasibility": "FAVORABLE"}
    ]
  }' --output health_card.pdf
```

#### Example Response Header (HTTP 200)
```http
HTTP/1.1 200 OK
Content-Type: application/pdf
Content-Disposition: attachment; filename="Crop_Health_Guide_coffee_RBK-405083.pdf"
```

---

## 4. Error Responses

All API errors follow a standardized JSON format `{error_code, message, hint}`:

### Common Error Codes

| Error Code | HTTP Status | Description | Action Hint |
| :--- | :--- | :--- | :--- |
| `EMPTY_FILE` | 400 Bad Request | Uploaded file size is 0 bytes | Provide a non-empty image file |
| `FILE_TOO_LARGE` | 413 Payload Too Large | Image exceeds 10 MB limit | Compress image before uploading |
| `UNSUPPORTED_FORMAT` | 400 Bad Request | Image format is not JPEG, PNG, or WebP | Convert photo to JPEG, PNG, or WebP |
| `DECOMPRESSION_BOMB` | 413 Payload Too Large | Image resolution exceeds 25 MP limit | Downscale photo resolution |
| `INVALID_LATITUDE` | 400 Bad Request | Latitude value outside [-90.0, 90.0] | Provide valid latitude coordinates |
| `INVALID_LONGITUDE` | 400 Bad Request | Longitude value outside [-180.0, 180.0] | Provide valid longitude coordinates |
| `RATE_LIMIT_EXCEEDED` | 429 Too Many Requests | Client exceeded 60 requests / minute | Wait before sending additional requests |

#### Example Error Payload (HTTP 413)
```json
{
  "error_code": "FILE_TOO_LARGE",
  "message": "Uploaded image size (14.20 MB) exceeds maximum allowed limit (10.0 MB).",
  "hint": "Please compress or resize your image photo before uploading."
}
```
