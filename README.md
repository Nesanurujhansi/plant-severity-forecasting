# Crop Leaf Health & Disease Guide (crop-prognosis)

> **Decision-Support Web Application for Crop Leaf Disease Severity Assessment, 7-Day Disease Spread Forecasting, and Local Field Care Guidance.**

---

## Safety Disclaimer

> [!IMPORTANT]
> **Educational and Non-Prescription Decision-Support Disclaimer**:
> This platform is designed solely as an educational and advisory decision-support tool for farmers, agricultural students, and field extension officers. It **NEVER** prescribes specific chemical brand names, commercial formulations, or exact chemical dosages, and it does not replace a certified local agricultural extension officer (e.g., Rythu Bharosa Kendra / Krishi Vigyan Kendra).

---

## System Overview

The system takes **one photograph** of a diseased crop leaf and delivers:

1. **Leaf Health & Disease Spot Outlines**:
   - Quantifies the percentage of leaf area damaged with a guaranteed 90% conformal prediction interval.
   - Decomposes damage into dead brown spots (center core) and spreading yellow border (outer margin).
   - Generates high-contrast visual overlays isolating disease spots and leaf boundaries.

2. **7-Day Disease Spread Forecast & Spatial Projection**:
   - Forecasts spot expansion across 3-day and 7-day horizons using semi-mechanistic epidemiological growth modeling fused with quantile residual learning.
   - Generates a 7-day spatial expansion projection composited directly over the original leaf photograph.

3. **High-Accuracy Local Weather (9 km Resolution)**:
   - Queries real-time 7-day forecasts from Open-Meteo (ECMWF IFS / GFS).
   - Computes daily temperature, relative humidity, vapor pressure deficit (VPD in kPa), rainfall, and leaf wetness hours.
   - Identifies safe spraying windows and alerts farmers to disease wash-off or drift hazards.

4. **Explainable AI (XAI) Attribution**:
   - Visualizes vision model attention focus areas over the leaf tissue.
   - Provides TreeSHAP factor breakdowns explaining how temperature, humidity, and leaf spots drive disease spread.

5. **Direct 2-Page Field Health Card (PDF)**:
   - Downloads a formal 2-page A4 crop health advisory report directly to the user's device.
   - Features 4 inspection views, 7-day weather forecast, and actionable field IPM practices in plain, farmer-friendly language.

---

## Architecture & Technology Stack

- **Frontend**: Next.js 14 (App Router), React, TypeScript, Tailwind CSS, Lucide / Phosphor Icons.
- **Backend API**: FastAPI, Uvicorn, Pydantic v2.
- **Machine Learning**: PyTorch, Meta DINOv2 (`facebook/dinov2-small`), SegFormer FPN decoder heads, LightGBM quantile regressors, Mondrian Conformal Prediction.
- **Document Generation**: ReportLab 4.x (A4 2-page publication layout with PIL image processing).
- **Weather Services**: Open-Meteo 9 km Numerical Weather Prediction (NWP).

---

## Quick Start (Local Setup)

### 1. Prerequisites
- Python 3.11 or Python 3.13 installed.
- Node.js 18+ and npm installed.

### 2. Backend Setup & Startup
```powershell
# Create and activate Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install backend dependencies
python -m pip install -r requirements-core.txt -r requirements-app.txt

# Start FastAPI backend daemon on port 8000
python run_server.py
```
Backend API will be running at `http://127.0.0.1:8000` (API docs at `/docs`).

### 3. Frontend Setup & Startup
```powershell
# Navigate to frontend directory
cd frontend

# Install Node.js packages
npm install

# Build optimized production bundle
npm run build

# Start frontend production server on port 3000
npm run start -- -p 3000
```
Open `http://localhost:3000` in your web browser.

---

## Project Documentation Index

- [Master Documentation Index](file:///c:/PlantDiseaseSeverity/docs/00_INDEX.md)
- [Project Context & Dataset Ledger](file:///c:/PlantDiseaseSeverity/docs/PROJECT_CONTEXT.md)
- [User Guide & Interface Manual](file:///c:/PlantDiseaseSeverity/docs/user_guide_draft.md)
- [Project Changelog](file:///c:/PlantDiseaseSeverity/docs/CHANGELOG.md)
- [API Reference Manual](file:///c:/PlantDiseaseSeverity/docs/api_reference.md)
- [Architectural Decision Records (ADR)](file:///c:/PlantDiseaseSeverity/docs/DECISIONS.md)
- [Data Sources & Provenance](file:///c:/PlantDiseaseSeverity/docs/DATA_SOURCES.md)
