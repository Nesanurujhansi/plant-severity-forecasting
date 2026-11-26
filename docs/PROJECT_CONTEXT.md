# Project Context & Architecture Ledger

## Project Summary
**Crop Leaf Health & Disease Guide (crop-prognosis)** is a clinical decision-support web application that analyzes photos of diseased crop leaves. The application provides:
1. **Severity Assessment**: Calculates leaf area damaged (%) with a 90% calibrated conformal prediction interval.
2. **Decomposed Pathology**: Separates dead brown spots (necrotic core) from spreading yellow borders (chlorotic halo) and connected lesion counts.
3. **Progression Forecasting**: Predicts disease severity changes over 3-day and 7-day horizons using local 9 km weather forecasts and semi-mechanistic epidemiological growth modeling.
4. **Spatial 7-Day Spread Overlay**: Renders biophysical spore dispersion projections composited directly onto the original leaf photograph.
5. **Direct 2-Page Health Card (PDF)**: Generates a publication-grade A4 field advisory report with 4 high-resolution inspection cards and spraying advisories.
6. **Safety & Reliability**: Evaluates image quality (blur, lighting, framing), out-of-distribution (OOD) detection in vision backbone feature space, and provides clear abstention guidance when photos are unsuitable.

> [!IMPORTANT]
> Safety Disclaimer: This platform is an educational and decision-support tool. It NEVER prescribes specific chemical trade names, commercial mixtures, or exact chemical dosages. Agrochemical applications must comply with certified local agricultural regulations and approved product labels.

---

## Modular Architecture Overview
- **Module 1: Vision Severity & Segmentation**: Vision backbone (Meta DINOv2 `facebook/dinov2-small`) extracts 384-dimensional visual feature representations. Dual SegFormer FPN decoder heads output leaf and lesion binary masks. Severity is calculated as `lesion_pixels / leaf_pixels`. Mondrian split conformal prediction provides non-parametric 90% prediction intervals grouped by severity class.
- **Module 2: Epidemiological Progression Forecasting**: Semi-mechanistic differential equation model \( \frac{ds}{dt} = r(T, RH) \cdot s \cdot (1 - \frac{s}{K}) \) combined with LightGBM logit-scale residual learning. Quantile forecasting uses Conformalized Quantile Regression (CQR). Feature parity is strictly maintained across CORE (inference-time) and EXTENDED (research) feature sets.
- **Module 3: High-Accuracy Local Weather (Open-Meteo 9 km NWP)**: High-resolution numerical weather prediction engine fetching forward 7-day temperature, relative humidity, rain duration, precipitation accumulation, wind speeds, and boundary layer leaf wetness hours.
- **Module 4: Safety & Quality Layer**: Assesses photo sharpness (Laplacian variance), exposure bounds, leaf coverage fraction, feature distance from training distribution, confidence tiering (`HIGH`, `MEDIUM`, `LOW`), and triggers abstention when safety checks fail.
- **Module 5: ReportLab 2-Page PDF Export**: Generates publication-grade A4 field health cards with 4 visual inspection cards, 7-day weather outlook, and actionable non-prescription IPM guidance.
- **Module 6: Next.js 14 Web Client**: Full-featured multilingual user interface (English, Telugu, Hindi) with plain-language terminology, live location picker, interactive visual overlays, and direct PDF download.

---

## Fact & Dataset Verification Table

| Fact / Source / Claim | Source Document / Reference | Verification Status | Date Verified | Notes / Remarks |
| :--- | :--- | :--- | :--- | :--- |
| ETH Zurich Wheat Leaf Image Time Series (12,520 RGB images, 1,032 leaf time series, 15 cultivars) | Anderegg & McDonald (2026), DOI: 10.1038/s41597-026-06567-y / ETH Data Repository DOI: 10.3929/ethz-b-000739194 | VERIFIED | 2026-09-22 | Verified in sequence processing phase. |
| PlantSeg (11,458 images, 115 disease categories, 34 plant types) | GitHub tqwei05/PlantSeg / Voxel51 Hugging Face subset | VERIFIED | 2026-09-22 | Used for segmentation benchmark evaluation. |
| Coffee Leaf Rust Dataset (1,285 field images, 606 pixel lesion masks) | MaryPazRB/Paper_CLR_CV Hugging Face | VERIFIED | 2026-09-22 | Field evaluation benchmark for multi-crop extension. |
| LeafNet75 Segmented Mask Repositories (apache-2.0 / MIT licenses) | LeafNet75/Leaf-Disease-Segmentation-masks & In_the_Wild_masks600 | VERIFIED | 2026-09-22 | Binary masks verified via SAM 2. |
| PlantVillage Dataset (mohanty/PlantVillage) | Hugging Face / Mohanty et al. | VERIFIED | 2026-09-22 | Multi-crop reference set. |
| Open-Meteo High-Resolution NWP API | https://open-meteo.com/en/docs | VERIFIED | 2026-11-24 | 9 km spatial resolution ECMWF IFS / GFS live weather queries. |
| Vision Backbone Architecture (DINOv2 small) | Hugging Face Hub (`facebook/dinov2-small`) | VERIFIED | 2026-11-24 | Pinned revision, 384-d CLS tokens and patch grids. |
