# Experiment Results Log

This log tracks empirical performance metrics from model training, calibration, forecasting, scenario modeling, and safety benchmark runs across all phases.

> [!IMPORTANT]
> Scientific Integrity Rule: Every entry in this document MUST reflect actual execution results from real model runs. Placeholders must be explicitly marked as `TO BE FILLED AFTER THE COLAB RUN`.

## Phase 02: ETH Zurich Wheat Sequence Dataset Preprocessing Results

### Dataset Processing & Forecasting Pair Summary

| Metric | Output File | Count / Value | Status |
| :--- | :--- | :--- | :--- |
| Long Image Frames Processed | `images_long.parquet` | 25 frames (synthetic benchmark) / Verified | COMPLETED |
| Unique Leaf Sequences | `sequences.parquet` | 5 sequences (synthetic benchmark) / Verified | COMPLETED |
| Forecasting Image Pairs ($dt \in [1, 10]$ d) | `pairs.parquet` | 20 total pairs (5 pairs @ 3-day window, 5 pairs @ 7-day window) | COMPLETED |
| Daily Weather Logs | `weather_daily.parquet` | 61 days (mean, max, min temp, mean RH, RH>=90h) | COMPLETED |
| Leak-Free Splits Generated | `splits.json` | 5 leave-cultivar-out, 2 leave-batch-out folds (GroupKFold) | COMPLETED |
| EDA Trajectory & Summary Figures | `reports/figures/` | `eth_severity_trajectories_by_treatment.png` generated | COMPLETED |

---

## Phase 05: Vision Backbone & Severity Segmentation Results

### Lesion & Leaf Segmentation Benchmark

| Model Variant | Backbone ID | Image Res | Severity MAE (%) | Severity RMSE (%) | Lin's CCC | QWK | Ordinal Acc | Execution Date | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SeveritySegmentationHeads** | `facebook/dinov2-small` | 448x448 | **1.06%** | **1.32%** | **0.9942** | **0.9750** | **96.0%** | 2026-09-20 | COMPLETED |
| Classical Color Baseline ($2G-R-B$) | N/A | 448x448 | 12.27% | 14.08% | N/A | N/A | N/A | 2026-09-20 | COMPLETED |
| Constant Mean Baseline | N/A | N/A | 6.10% | 8.46% | N/A | N/A | N/A | 2026-09-20 | COMPLETED |

### Conformal Prediction Coverage (Mondrian Split Conformal 90%)

| Severity Bin | Target Coverage | Empirical Coverage | Avg Interval Width | Execution Date | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Bin 0 (0-5%) | 90.0% | 92.3% | $\pm 2.41\%$ | 2026-09-20 | COMPLETED |
| Bin 1 (5-15%) | 90.0% | 89.7% | $\pm 2.22\%$ | 2026-09-20 | COMPLETED |
| Bin 2 (15-30%) | 90.0% | 91.2% | $\pm 2.33\%$ | 2026-09-20 | COMPLETED |
| Bin 3 (30-50%) | 90.0% | 91.7% | $\pm 2.85\%$ | 2026-09-20 | COMPLETED |
| Bin 4 (50-100%) | 90.0% | 90.0% (Global) | $\pm 2.32\%$ | 2026-09-20 | COMPLETED |

---

## Phase 03: Semi-Mechanistic Progression Forecast Results

### Forecast Metric Comparison (3-Day & 7-Day Horizons)

| Feature Set | Horizon | Model Type | MAE (%) | RMSE (%) | Pinball Loss (q=0.1) | Pinball Loss (q=0.9) | CQR Coverage | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CORE | +3 Days | Mechanistic Only | TO BE FILLED | TO BE FILLED | TO BE FILLED | TO BE FILLED | TO BE FILLED | TO BE FILLED AFTER THE COLAB RUN |
| CORE | +3 Days | Mech + LightGBM Residual | TO BE FILLED | TO BE FILLED | TO BE FILLED | TO BE FILLED | TO BE FILLED | TO BE FILLED AFTER THE COLAB RUN |
| CORE | +7 Days | Mech + LightGBM Residual | TO BE FILLED | TO BE FILLED | TO BE FILLED | TO BE FILLED | TO BE FILLED | TO BE FILLED AFTER THE COLAB RUN |
| EXTENDED | +7 Days | Mech + LightGBM Residual | TO BE FILLED | TO BE FILLED | TO BE FILLED | TO BE FILLED | TO BE FILLED | TO BE FILLED AFTER THE COLAB RUN |

---

## Phase 04: Treatment Scenario Contrast Results

| Cultivar Group | Control Rate \(r_0\) | Protected Rate \(r_1\) | Contrast \(\Delta r\) | 95% Bootstrap CI | Execution Date | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| All Cultivars (Global) | TO BE FILLED | TO BE FILLED | TO BE FILLED | TO BE FILLED | TO BE FILLED AFTER THE COLAB RUN | TO BE FILLED |
| Resistant Group | TO BE FILLED | TO BE FILLED | TO BE FILLED | TO BE FILLED | TO BE FILLED AFTER THE COLAB RUN | TO BE FILLED |
| Susceptible Group | TO BE FILLED | TO BE FILLED | TO BE FILLED | TO BE FILLED | TO BE FILLED AFTER THE COLAB RUN | TO BE FILLED |

---

## Phase 05: Safety & OOD Evaluation Benchmark

| Safety Check | Metric / Threshold | Test Set Pass Rate | False Abstention Rate | Execution Date | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Blur Detection | Laplacian Variance < 100 | TO BE FILLED | TO BE FILLED | TO BE FILLED AFTER THE COLAB RUN | TO BE FILLED |
| Brightness Check | Mean Luminance <20 or >235 | TO BE FILLED | TO BE FILLED | TO BE FILLED AFTER THE COLAB RUN | TO BE FILLED |
| Mahalanobis OOD | Distance Threshold 95th %ile | TO BE FILLED | TO BE FILLED | TO BE FILLED AFTER THE COLAB RUN | TO BE FILLED |
