# Project Changelog

All notable changes to the **crop-prognosis** codebase are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2026-11-24 - Production Web Application & Field Health Card Release

### Added
- **Next.js 14 Production Web Client (`frontend/`)**: Replaced prototype interface with a production Next.js 14 App Router application styled with Tailwind CSS, Lucide, and Phosphor icons.
- **Multilingual Localization (`frontend/app/locales/translations.ts`)**: Built instantaneous client-side language switching for English (`en`), Telugu (`te`), and Hindi (`hi`).
- **Direct 2-Page Field Health Card (PDF) Generation (`app/inference/pdf_report.py`, `/api/export-health-report-pdf`)**:
  - Implemented ReportLab 4.x publication engine generating a standardized 2-page A4 field advisory PDF report.
  - Page 1 includes official header, crop and field metadata bar, severity scorecard, and a 4-card inspection gallery.
  - Page 2 includes 7-day weather forecast table, spraying feasibility windows, practical field IPM checklist, and non-prescription disclaimers.
  - Generates direct browser download attachment with automatic naming (`Crop_Health_Guide_<crop>_<id>.pdf`).
- **Composite 7-Day Disease Spread Spatial Projection (`app/inference/pdf_report.py`)**: Implemented `composite_overlay_on_original` blending the biophysical spore dispersion expansion frontier directly onto the uploaded leaf photograph.
- **High-Accuracy Local Weather Integration (`app/inference/weather.py`)**: Switched to Open-Meteo 9 km Numerical Weather Prediction (NWP), computing hourly temperature, relative humidity, rain duration, precipitation accumulation, wind speeds, and leaf wetness hours.
- **Plain-Language Farmer Terminology Overhaul**: Replaced complex academic and laboratory terminology across both UI and PDF with direct, farmer-friendly terms (e.g. *foliar* to *leaf*, *necrotic core* to *dead brown spots*, *chlorotic halo* to *spreading yellow border*, *two-tier pathology overlay* to *disease spot outlines*, *attention saliency* to *AI focus areas*).

### Fixed
- Resolved download button handler in `frontend/app/page.tsx` with base64 data extraction fallback from `previewUrl`.
- Added proxy rewrite in `frontend/next.config.mjs` routing `/api/export-health-report-pdf` to the FastAPI backend.
- Resolved table height and aspect ratio scaling in `pdf_report.py` ensuring all 4 inspection cards fit Page 1 with zero empty white space.

---

## [0.11.0] - 2026-09-22 - Phase 10 FastAPI Backend Completion

### Added
- **Application Settings & Configuration (`app/config.py`, `.env.example`)**: Configured Pydantic `Settings` managing environment variables, upload limits (10 MB max size, 25 MP pixel limit), bundle directory paths, `store_uploads=False` privacy default, CORS origins, and rate limits.
- **Pydantic v2 Schemas (`app/schemas.py`)**: Built validation and output schemas for `AnalyzeResponse`, `SafetyBlock`, `SeverityBlock`, `ForecastBlock`, `ScenarioBlock`, `WeatherBlock`, `HealthResponse`, `ModelInfoResponse`, `WeatherQueryResponse`, and `ErrorResponse`.
- **Upload Safety & Validation (`app/inference/upload_safety.py`)**: Implemented byte size limit validation, image format decoding (accepts JPEG, PNG, WebP), decompression bomb limits, and in-memory EXIF metadata stripping.
- **Microclimate Weather Engine (`app/inference/weather.py`)**: Implemented weather resolution via user overrides, NASA POWER API lookups, and future weather scenario presets (`as_recent`, `climatology`, `warm_humid`, `cool_dry`, `user_entered`), calculating Vapor Pressure Deficit (VPD in kPa via Magnus formula).
- **Master Inference Pipeline (`app/inference/pipeline.py`)**: Built `InferencePipeline` class linking Safety, Severity, Weather, Progression Forecasting, Treatment Scenario Simulation, and base64 PNG mask overlay generation.
- **FastAPI Web Application Server (`app/main.py`)**: Implemented REST API server with lifespan singleton model loader, request ID middleware (`X-Request-ID`), in-memory rate limiter (60 req/min), CORS, `/health`, `/model-info`, `/analyze`, `/weather`, and marked hook for Phase 11 Gradio mounting (`/ui`).
- **API Reference Manual & Tests (`docs/api_reference.md`, `tests/test_api.py`)**: Written OpenAPI reference guide and 9 unit tests passing 100% on CPU (102 total unit tests passing repository-wide).
- **Documentation (`docs/phase10_fastapi.md`)**: Written 12-section Phase 10 documentation file compliant with all B.Tech beginner guidelines.

---

## [0.10.0] - 2026-09-22 - Phase 09 Model Bundle Export & Verifier Completion

### Added
- **Manifest Schema & Builder (`ml/export/manifest.py`)**: Implemented Pydantic `ManifestSchema` and `build_manifest()` helper tracking bundle version, created timestamp, git commit hash, installed library versions, DINOv2 backbone specs, image preprocessing rules, SHA-256 file hashes, feature schema hash, headline evaluation metrics summary, dataset licences, and non-prescription disclaimers.
- **Model Bundle Exporter (`ml/export/export_bundle.py`)**: Implemented exporter gathering Phase 5–8 artifacts (`severity/`, `forecast/`, `scenario/`, `safety/`), generating synthetic 224x224 leaf images and pre-computed expected outputs in `golden/`, writing `README.txt` and `manifest.json`, packaging into `crop-prognosis-bundle-v1.0.0.zip`, and verifying the zip archive before completion.
- **Model Bundle Verifier CLI & Engine (`ml/export/verify_bundle.py`)**: Implemented verification engine (`python -m ml.export.verify_bundle models/`) validating manifest schema, SHA-256 cryptographic file hashes, library version compatibility (failing on core major version mismatch), feature schema hash consistency, model file loading, and golden reference numerical output tolerance compliance.
- **Application Model Container & Singleton Loader (`app/inference/loader.py`)**: Implemented singleton `ModelContainer` class managing application startup, running bundle verification, exposing model predictors, and providing fallback `DEV_STUB_MODELS=1` development stub mode with a visible warning banner (`"[WARNING] DEV_STUB_MODELS ACTIVE - NOT REAL PREDICTIONS"`).
- **Security Audit & Unit Test Suite (`tests/test_export.py`, `tests/test_export_security.py`)**: Built 10 unit tests verifying manifest schema validation, bundle export & verification, SHA-256 hash tampering detection, missing file handling, version checking, golden drift failures, stub loading, Zip-Slip path traversal protection, and zero-pickle AST compliance (93 total unit tests passing repository-wide).
- **Google Colab Notebook Script (`notebooks/colab_phase09_export.py`)**: Created 5-step executable script for mounting Drive, assembling, verifying, calculating SHA-256 checksums, and downloading the model bundle zip file.
- **Model Transfer Guide (`docs/bundle_transfer_guide.md`)**: Created step-by-step handoff documentation for downloading from Colab, unzipping to `models/`, running the verifier CLI, and testing local application startup.
- **Documentation (`docs/phase09_bundle.md`)**: Written 12-section Phase 09 documentation file compliant with all B.Tech beginner guidelines.

---

## [0.9.0] - 2026-09-22 - Phase 08 Safety Layer Completion

### Added
- **Safety Configuration (`ml/configs/safety.yaml`)**: Configured calibrated thresholds for photo quality checks, OOD detection (Free Energy, Mahalanobis, kNN), weather input bounds, confidence tiers, and per-module gating.
- **Photo Quality Assessment (`ml/safety/quality.py`)**: Implemented image sharpness (variance of 2D Laplacian operator with resolution normalization), resolution checks, brightness/contrast, over/under-exposure ratios, and leaf coverage fraction checks with actionable user recommendations.
- **Deep OOD Detection Ensemble (`ml/safety/ood.py`)**: Implemented Deep Free Energy log-partition score $E(\mathbf{x}) = -T \cdot \log \sum \exp(f_i/T)$, Mahalanobis distance $(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})$, and $k$-NN distance in 384-d DINOv2 feature space.
- **Input Range Validation (`ml/safety/ranges.py`)**: Implemented range validation for microclimate weather variables (temperature, relative humidity, VPD, precipitation), forecast horizons, and multi-crop category support.
- **Safety Policy Aggregator & Module Gating (`ml/safety/policy.py`)**: Implemented aggregate policy engine computing confidence tiers (`HIGH`, `MEDIUM`, `LOW`), abstention flags with beginner-friendly action recommendations, and per-module execution gating.
- **Synthetic Stress Testing Suite (`ml/safety/stress_test.py`)**: Built programmatic stress testing engine evaluating detection rates (%) and severity error degradation across Gaussian blur, glare, dark images, downscaling, JPEG compression, and synthetic OOD non-leaf stand-ins.
- **Zero-Pickle Safety Exporter (`ml/safety/export.py`)**: Exported `safety_config.json`, `ood_stats.safetensors`, and `ood_meta.json` to `data/raw/phase08_safety_models/`.
- **Colab & Standalone Notebook Script (`notebooks/colab_phase08_safety.py`)**: Created 8-step execution script for calibrating thresholds and running stress tests in Cloud GPU or local environments.
- **CPU Test Suite (`tests/test_safety.py`)**: Added 11 new unit tests passing 100% on CPU (83 total unit tests passing repository-wide).
- **Documentation (`docs/phase08_safety.md`)**: Written 12-section Phase 08 documentation file compliant with project standards.

---

## [0.8.0] - 2026-09-21 - Phase 07 Treatment Scenario Completion

### Added
- **Treatment Identifiability Analysis (`ml/scenario/identifiability.py`)**: Analyzed field trial management arms (`FI2`, `0FI1`, `F0I`, `0F0I`) and selected primary causal contrast `FI2_vs_0F0I`.
- **Linear Mixed-Effects Model (`ml/scenario/mixed_effects.py`)**: Implemented Statsmodels MixedLM random-intercept baseline for wheat cultivar and plot ID.
- **Plot-Clustered Bootstrap Resampling (`ml/scenario/bootstrap.py`)**: Implemented spatial cluster bootstrap (50 resamples) computing non-parametric 90% percentile confidence intervals.
- **Deep Counterfactual Treatment Network (`ml/scenario/treatment_net.py`)**: Built TarNet PyTorch neural architecture with shared representation trunk, dual specialized treatment branch heads, and Monte Carlo Dropout epistemic uncertainty estimation.
- **Scenario Simulator Engine (`ml/scenario/simulate.py`)**: Built simulator evaluating untreated vs fungicide-protected 3-day and 7-day trajectories with interval widening and non-prescription disclaimers.
- **Leave-Cultivar-Out Validation (`ml/scenario/validate.py`)**: Implemented Leave-Cultivar-Out cross-validation evaluating treatment multiplier stability.
- **Zero-Pickle Exporter (`ml/scenario/export.py`)**: Exported `treatment_net.safetensors`, `mixed_effects.json`, `effect_bootstrap.json`, and `scenario_config.json` to `data/raw/phase07_scenario_models/`.
- **Colab Script (`notebooks/colab_phase07_treatment.py`)**: Created Colab & standalone GPU execution script.
- **CPU Test Suite (`tests/test_scenario.py`)**: Added 7 unit tests (72 total unit tests passing repository-wide).
- **Documentation (`docs/phase07_treatment.md`)**: Written 12-section Phase 07 documentation file.

---

### Added
- **Progression Module Configuration (`ml/configs/forecast.yaml`)**: Configured CORE (19) / EXTENDED (24) feature sets, PINN-ODE parameter bounds, DQNN 90% quantiles, Mondrian calibration bins, and TarNet counterfactual settings.
- **Microclimate Feature Parity Engine (`ml/forecast/features.py`, `docs/feature_schema.md`)**: Implemented microclimate feature engineering (VPD in kPa via Magnus formula, GDD with $T_{\text{base}}=0^\circ\text{C}$, lesion spatial density) with mandatory offline-online feature parity verification.
- **Physics-Informed Neural ODE (`ml/forecast/growth.py`)**: Implemented PINN-ODE with Yan & Hunt Beta-type thermal response curve $g_T(T)$, Vapor Pressure Deficit (VPD in kPa) response curve $g_{\text{VPD}}(\text{VPD})$, daily step numerical integration, and SciPy least-squares fitting.
- **Monotonic Residual & DQNN Core (`ml/forecast/residual.py`, `ml/forecast/cqr.py`)**: Implemented monotonic LightGBM logit residual regression (`dt_days`, `severity_i`) and Deep Quantile Neural Networks with Mondrian Conformal Calibration binned by initial severity class (MILD, MODERATE, SEVERE).
- **Deep Counterfactual Treatment Network (`ml/scenario/treatment_net.py`)**: Implemented TarNet counterfactual neural architecture with shared representation trunk, dual specialized treatment branch heads (untreated 0F0I vs protected FI2), and Monte Carlo Dropout epistemic uncertainty estimation.
- **Deep Free Energy OOD Safety Evaluator (`ml/safety/deep_ood.py`)**: Implemented Deep Free Energy OOD detector ($E(\mathbf{x}) = -T \cdot \log \sum \exp(f_i/T)$) and Deep Quality Net for blur, exposure, and leaf ratio validation.
- **Benchmark Baselines & Baseline Fix (`ml/forecast/baselines.py`)**: Fixed global feature list mutation bug in `get_feature_matrix()` by returning `list(CORE_FEATURES)` copy.
- **Multi-Horizon Evaluation Engine (`ml/forecast/evaluate.py`)**: Evaluated 3-day (MAE $0.72\%$) and 7-day (MAE $1.40\%$, Skill Score $+0.899$) forecasts across Leave-Cultivar-Out and Leave-Batch-Out folds.
- **Severity Noise-Robustness (`ml/forecast/noise_robustness.py`)**: Demonstrated high robustness ($+0.93\%$ MAE degradation under noise) to Phase 5 measurement noise.
- **Predictor Engine (`ml/forecast/predict.py`)**: Built `Forecaster` class returning typed `ForecastResult` objects with trust tiers (`HIGH`, `MEDIUM`, `LOW`), interval width checks, and VPD driver explanations.
- **Zero-Pickle Exporter (`ml/export/progression_export.py`, `ml/forecast/export.py`)**: Exported all progression artifacts to `models/forecast/` in plain text, PyTorch `.safetensors`, and JSON formats.
- **Colab Notebook Generator & Notebook (`scratch/create_colab_phase06_notebook.py`, `notebooks/colab_phase06_progression.ipynb`)**: Created 10-step Colab notebook.
- **CPU Test Suite (`tests/test_forecast.py`)**: Added 13 unit tests passing on CPU (59 total tests passing repository-wide).
- **Documentation (`docs/phase06_progression.md`)**: Written 12-section Phase 06 Deep Learning documentation file.

---

## [0.6.0] - 2026-09-21 - Phase 05 Completion

### Added
- **Severity Module Configuration (`ml/configs/severity.yaml`)**: Configured head dimensions, loss weights, ordinal severity bin edges (`<1%`, `1-5%`, `5-10%`, `10-25%`, `25-50%`, `>50%`), training hyperparams, and Mondrian conformal settings.
- **Segmentation Decoder Heads (`ml/severity/heads.py`)**: Built `SeveritySegmentationHeads` PyTorch decoder ($1 \times 1$ conv channel reduction, 2-stage conv blocks with bilinear upsampling to target stride 4) producing lesion and leaf mask logits with $< 200\text{k}$ parameters (98,818 parameters).
- **Combined BCE + Dice Loss (`ml/severity/losses.py`)**: Implemented `CombinedBCEDiceLoss` with positive class weighting (`lesion_pos_weight = 5.0`) for rare lesion pixels and selective mask availability filtering.
- **Severity Calculation & Connected Components (`ml/severity/severity.py`)**: Implemented soft/hard severity fraction calculation, structural constraint enforcement ($\text{leaf} := \text{leaf} \cup \text{lesion}$), ordinal bin mapping, and connected-component lesion statistics (lesion count, mean/max size, boundary perimeter).
- **Baselines Module (`ml/severity/baselines.py`)**: Implemented `ClassicalColorThresholdBaseline` (Excess-Green $2G - R - B$ + HSV + Otsu) and `ConstantMeanBaseline` to evaluate model complexity gain.
- **Mondrian Conformal Calibrator (`ml/severity/conformal.py`)**: Built `MondrianConformalCalibrator` computing 90% non-parametric prediction intervals grouped by severity class with sparse bin merging, interval clipping $[0, 1]$, empirical coverage evaluation, and `conformal.json` export.
- **Resumable Training Pipeline (`ml/severity/train.py`)**: Built config-driven training loop with source-balanced dataset sampling, early stopping on validation Dice/severity MAE, CSV metric logging (`train_log.csv`), checkpointing, and `.safetensors` export.
- **Evaluation Engine (`ml/severity/evaluate.py`)**: Built evaluation module calculating Dice, IoU, severity MAE (percentage points), RMSE, Lin's Concordance Correlation Coefficient (CCC), Bland-Altman agreement stats, and Quadratic Weighted Kappa (QWK), exporting reports to `reports/tables/` and `reports/`.
- **Inference Predictor & Latency Benchmark (`ml/severity/predict.py`)**: Built end-to-end `SeverityPredictor` class for CPU/GPU single-photo inference returning typed `SeverityPredictionResult` objects, plus `benchmark_cpu_latency` benchmark utility.
- **Model Exporter (`ml/severity/export.py`)**: Built artifact bundler exporting `severity_heads.safetensors`, `config.json`, and `conformal.json` with embedded header metadata.
- **Colab Notebook (`notebooks/colab_phase05_severity.ipynb`)**: Created 8-step Jupyter notebook for Colab GPU training, conformal calibration, evaluation, and Drive export.
- **CPU Test Suite (`tests/test_severity.py`)**: Added 8 unit tests passing on CPU (46 total tests passing repository-wide).
- **Documentation (`docs/phase05_severity.md`)**: Written 12-section Phase 05 documentation file.

---

## [0.5.0] - 2026-09-20 - Phase 04 Completion

### Added
- **Backbone Verification & Configuration (`ml/configs/features.yaml`)**: Pinned HF model IDs (`facebook/dinov2-small`, `facebook/dinov2-base`), revision `main`, patch size 14, spatial resolutions (`[448, 672, 896]`), and float16 mixed precision.
- **Storage Footprint Estimator (`ml/features/storage_estimate.py`)**: Calculated grid sizes, per-image byte footprints, dataset gigabytes, and Google Drive 15 GB quota safety warnings.
- **Sharded Feature Extractor Engine (`ml/features/extract_features.py`)**: Built DINOv2 feature extractor with aspect-ratio preserving padding to multiples of 14, patch grid token extraction, ETH macro tiling/cropping, SHA-256 `cache_key` mismatch protection, `.safetensors` sharding, and resumable execution.
- **OOD Statistics Module (`ml/features/ood_stats.py`)**: Calculated per-source mean vectors, shrinkage-regularized covariance matrices ($\gamma = 0.01$), precision matrices, and 95th/99th percentile Mahalanobis distance thresholds.
- **Cached Feature Dataset (`ml/features/feature_dataset.py`)**: PyTorch `Dataset` with lazy `.safetensors` shard loading and spatial mask resizing to patch grid dimensions.
- **CPU Test Suite (`tests/test_features.py`)**: Added 9 unit tests passing on CPU using stub backbone (38 total tests passing repository-wide).
- **Colab Notebook (`notebooks/colab_phase04_features.ipynb`)**: Created 8-cell Colab notebook for GPU feature extraction and Drive export.
- **Documentation (`docs/phase04_features.md`)**: Written 12-section Phase 04 documentation file.

---

## [0.4.0] - 2026-09-20 - Phase 03 Completion

### Added
- **Static Dataset Inspection (`docs/static_dataset_notes.md`)**: Documented directory structures, annotation encodings, licenses, and evaluation subsets for PlantSeg, Coffee Rust, LeafNet75, and PlantVillage.
- **Unified Static Manifest (`ml/data/build_static_manifest.py`)**: Built `data/processed/static/manifest.parquet` with mask binarization (0/1), mask integrity validator, and near-duplicate cluster IDs.
- **SAM 2 Leaf Pseudo-Labeler (`ml/data/make_leaf_pseudolabels_sam2.py`)**: Implemented automated leaf mask prediction using pinned SAM 2 backbone (`facebook/sam2.1-hiera-small`) with the leaf union rule ($\text{leaf} = \text{sam2} \cup \text{lesion}$).
- **Human Review QC Tool (`ml/data/review_leaf_masks.py`)**: Created review template CSV generator (`data/interim/leaf_mask_reviews.csv`) and quality flag updater.
- **Static Severity Calculator (`ml/data/static_severity.py`)**: Calculated disease severity fraction $S = \text{lesion} / \text{leaf}$ and assigned 6 ordinal severity bin classes.
- **Cluster-Grouped Splits (`ml/data/static_splits.py`)**: Generated leak-free splits grouped by `cluster_id` and reserved Coffee Rust 100 eval set as `external_test`.
- **Quality Contact Sheets & Reports (`ml/data/contact_sheet.py`, `static_stats.py`)**: Rendered translucent mask overlays and exported figures and summary tables to `reports/`.
- **Colab Notebook (`notebooks/colab_phase03_static.ipynb`)**: Built self-contained Colab notebook for GPU execution.

---

## [0.3.0] - 2026-09-20 - Phase 02 Completion

### Added
- **ETH Zurich Dataset Parser (`ml/data/eth_wheat.py`)**: Parsed experimental design CSVs, weather station logs, and alignment masks; exported `images_long.parquet` and `sequences.parquet`.
- **Pinned Severity Formulation (`docs/severity_definition.md`)**: Pinned severity denominator to Frame 1 leaf area ($S_t = \text{Diseased}_t / A_1$) to prevent camera shrinkage artifacts.
- **Forecasting Pair Generator (`ml/data/make_pairs.py`)**: Generated sequence image pairs $(i, j)$ for elapsed times $dt \in [1, 10]$ days with interval weather stats and reciprocal sequence sample weights ($w = 1 / n_{\text{pairs}}$).
- **Leak-Free GroupKFold Splits (`ml/data/splits.py`)**: Built cultivar, batch, and year GroupKFold splits grouped strictly by `sample_id` with zero sample leakage assertion.
- **Treatment Arm Analysis (`docs/treatment_design_notes.md`)**: Documented factorial treatment arms (`FI2`, `0FI1`, `F0I`, `0F0I`) and identifiable contrasts.

---

## [0.2.0] - 2026-09-20 - Phase 01 Completion

### Added
- **Dataset Cards (`docs/data_cards/`)**: Created data cards for ETH Zurich Wheat, PlantSeg, Coffee Leaf Rust, LeafNet75, PlantVillage, LeafNet, NASA POWER API, and Zenodo Fungicide Trials.
- **Dataset Downloader Suite (`ml/data/`)**:
  - `download_eth_wheat.py`: Resumable downloader and manual download validator checking file counts against author reported numbers.
  - `download_plantseg.py`, `download_coffee_rust.py`, `download_leafnet_masks.py`, `download_plantvillage.py`: Resumable Hugging Face downloaders supporting `--dry-run`.
  - `validate_downloads.py`: Dataset integrity auditor outputting JSON and Markdown reports to `data/interim/download_validation.json` and `.md`.
- **NASA POWER Agroclimatology Client (`ml/data/weather_nasa_power.py`)**: REST client for `T2M`, `T2M_MAX`, `T2M_MIN`, `RH2M`, `PRECTOTCORR` featuring mandatory disk caching, exponential backoff retries, and tidy pandas DataFrame formatting.
- **Weather Strategy & Scenario Presets (`docs/weather_strategy.md`)**: Detailed guide defining 5 forecast scenario presets ("As Recent", "Climatology", "Warm & Humid", "Cool & Dry", "User-Entered") and third-party forecast protocol.
- **Weather Parity Engine (`ml/data/weather_parity.py`)**: Script comparing on-site ETH weather station data with NASA POWER grid estimates, calculating bias, RMSE, and correlation, and generating scatter plots.
- **Data Provenance & Deduplication (`ml/data/provenance.py`)**: Image difference hashing (`dHash`), near-duplicate clustering, and static manifest builder (`data/processed/static_manifest.parquet`).
- **Data Governance Framework (`docs/data_governance.md`)**: Dataset license rules, attribution matrix, user photo privacy guidelines, and BibTeX citations.
- **Google Colab Ingestion Workflow (`notebooks/colab_phase01_data.ipynb`)**: Colab notebook executing Drive mounting, dependency installation, downloader dry-runs, downloads, and validation.
- **Testing Suite**: Added `test_weather_nasa_power.py` (HTTP mocking), `test_downloaders.py`, `test_provenance.py`, and `test_validate_downloads.py` (20 total unit tests passing).

---

## [0.1.0] - 2026-09-20 - Phase 00 Completion

### Added
- **Repository Layout**: Initialized canonical directory structure (`docs/`, `ml/`, `notebooks/`, `app/`, `models/`, `data/`, `reports/`, `tests/`, `scripts/`) with `.gitkeep` placeholders.
- **Dependency Specifications**: Added pinned `requirements-core.txt`, `requirements-train.txt`, and `requirements-app.txt` files.
- **Automation Tooling**: Created `Makefile` for `setup`, `test`, `lint`, `format`, `run-api`, and `docs-check` commands.
- **Core Utilities**: `seed.py`, `io.py`, `hashing.py`, `logging_utils.py`, `paths.py`, `config.py`.
