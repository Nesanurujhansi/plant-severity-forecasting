# Feature Schema Specification - Progression Forecasting Module

This document defines the exact schema, feature definitions, units, and missing value handling for the epidemiological progression forecasting module (`ml/forecast/`).

## Feature Sets Overview

1. **CORE Feature Set (16 Features)**: Available at inference time from a single crop leaf photo assessment + historical/forecast daily weather series from NASA POWER API or local station.
2. **EXTENDED Feature Set (21 Features)**: Research-only features including detailed cultivar susceptibility traits, hourly humidity variances, and plot-level experimental variables.

---

## CORE Features (Inference & Production)

| Feature Name | Type | Unit | Range / Allowed Values | Description | Missing Value Handling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `severity_i` | float | Fraction | $[0.0, 1.0]$ | Initial disease severity fraction at time $t_i$. | Impute using $0.001$ if zero; error if missing. |
| `dt_days` | float | Days | $[1.0, 14.0]$ | Forecast time gap in days between initial photo $t_i$ and target forecast $t_j$. | Must be $> 0$; default to $3.0$ or $7.0$ if omitted. |
| `n_lesions_i` | int | Count | $\ge 0$ | Connected component lesion count extracted in initial photo. | Impute with $0$ if unavailable. |
| `mean_lesion_area_i` | float | Pixels | $\ge 0.0$ | Average area per lesion in initial photo. | Impute with $0.0$ if $n\_lesions\_i = 0$. |
| `t_mean_past3` | float | ${}^\circ\text{C}$ | $[-10, 50]$ | Mean daily temperature over the 3 days preceding $t_i$. | Fall back to seasonal mean $18.0^\circ\text{C}$. |
| `t_mean_past7` | float | ${}^\circ\text{C}$ | $[-10, 50]$ | Mean daily temperature over the 7 days preceding $t_i$. | Fall back to `t_mean_past3` or $18.0^\circ\text{C}$. |
| `t_mean_interval` | float | ${}^\circ\text{C}$ | $[-10, 50]$ | Mean daily temperature during the forecast interval $[t_i, t_j]$. | Fall back to `t_mean_past7` or $18.0^\circ\text{C}$. |
| `t_max_min_interval` | float | ${}^\circ\text{C}$ | $[-10, 50]$ | Average of $(T_{\text{max}} + T_{\text{min}})/2$ over forecast interval $[t_i, t_j]$. | Fall back to `t_mean_interval`. |
| `rh_mean_past3` | float | $\%$ | $[0, 100]$ | Mean daily relative humidity over the 3 days preceding $t_i$. | Fall back to seasonal mean $75.0\%$. |
| `rh_mean_past7` | float | $\%$ | $[0, 100]$ | Mean daily relative humidity over the 7 days preceding $t_i$. | Fall back to `rh_mean_past3` or $75.0\%$. |
| `rh_mean_interval` | float | $\%$ | $[0, 100]$ | Mean daily relative humidity during forecast interval $[t_i, t_j]$. | Fall back to `rh_mean_past7` or $75.0\%$. |
| `precip_total_past3` | float | mm | $\ge 0.0$ | Total precipitation accumulated over 3 days preceding $t_i$. | Impute with $0.0$ mm. |
| `precip_total_past7` | float | mm | $\ge 0.0$ | Total precipitation accumulated over 7 days preceding $t_i$. | Impute with $0.0$ mm. |
| `precip_total_interval` | float | mm | $\ge 0.0$ | Total precipitation accumulated over forecast interval $[t_i, t_j]$. | Impute with $0.0$ mm. |
| `crop_stage` | float | Code | $[-1, 99]$ | Growth stage (e.g., Zadoks scale or days since heading). | Impute with $-1.0$ (Unknown). |
| `cultivar_group` | float | Code | $[-1, 15]$ | Encoded cultivar susceptibility class. | Impute with $-1.0$ (Unknown). |

---

## EXTENDED Features (Research Only)

In addition to all 16 CORE features:

| Feature Name | Type | Unit | Range / Allowed Values | Description | Missing Value Handling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `cultivar_trait_score` | float | Index | $[0.0, 10.0]$ | Quantified genetic resistance rating for host cultivar. | Impute with $5.0$ (Moderate). |
| `rh_std_interval` | float | $\%$ | $\ge 0.0$ | Standard deviation of hourly/sub-daily humidity. | Impute with $0.0$. |
| `batch_id` | int | Code | $\{1, 2, 3\}$ | Experimental measurement batch ID. | Impute with $-1$. |
| `year_id` | int | Year | $\{2023, 2024\}$ | Field trial season year. | Impute with $-1$. |
| `treatment_code` | int | Code | $\{0, 1, 2, 3\}$ | Plot-level fungicide & inoculation treatment code. | Impute with $-1$. |

---

## Offline-Online Feature Parity Requirement

To guarantee that the model trained on historical sequence pairs (`pairs.parquet`) behaves identically during serving:
1. `build_feature_vector()` in [`ml/forecast/features.py`](file:///C:/PlantDiseaseSeverity/ml/forecast/features.py) enforces identical ordering, dtype casting, and normalization across both offline dataset processing and online REST API requests.
2. An explicit parity test (`test_feature_parity_offline_online`) validates that feature values computed from a dataframe row match those produced from an online `Forecaster` input request down to floating point precision ($10^{-6}$).
