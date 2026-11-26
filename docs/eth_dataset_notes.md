# ETH Zurich Wheat Dataset Notes & Structure Guide

## 1. Real Dataset Folder Structure & File Formats

Based on Anderegg & McDonald (2026) and the `sympathique-wheat` repository, the released dataset is structured as follows:

```
eth_wheat/
├── meta/
│   ├── experimental_design.csv     # Plot metadata: plot_id, cultivar, treatment, batch, season, year, heading_date
│   ├── weather_station_hourly.csv  # Hourly weather: timestamp, temperature (°C), relative_humidity (%)
│   ├── leaf_metadata.csv           # Leaf sequence metadata: sample_id, plot_id, leaf_no, emergence_date
│   └── disease_scores.csv          # Field disease incidence and conditional severity ratings
├── processed/
│   ├── reg/                        # Geometric alignment transformation matrices (.json / .npy) per leaf sequence
│   └── ts/                         # Leaf time series folders named by sample_id (e.g. ESWW001_L1)
│       └── <sample_id>/
│           ├── <image_id>_crop.png # Aligned RGB leaf crop image
│           └── <image_id>_mask.png # Multi-class symptom segmentation mask image
└── markers/                        # Reference ink marker coordinates (.json) used for registration
```

---

## 2. Image & Sample ID Pattern
- **Image ID Pattern**: `YYYYMMDD_HHMMSS_<plot_id>_<leaf_no>` (e.g. `20230515_103000_ESWW001_L1`)
  - Timestamp: `YYYYMMDD_HHMMSS` (Acquisition date and time)
  - Plot ID: `ESWW001` through `ESWW100` (Experimental plot identifier)
  - Leaf Number: `L1`, `L2`, etc. (Flag leaf or top sub-flag leaf)
- **Sample ID Pattern**: `<plot_id>_<leaf_no>` (e.g. `ESWW001_L1`)
  - A `sample_id` uniquely identifies one tracked leaf sequence across all acquisition dates.

---

## 3. Symptom Class Mask Encoding
In the multi-class symptom segmentation mask PNG files (`<image_id>_mask.png`), pixel values correspond to specific biological classes:
- `0`: Background (non-leaf pixels)
- `1`: Healthy green leaf tissue
- `2`: Chlorosis / Senescence (yellowing tissue)
- `3`: Necrosis / Septoria tritici blotch (dead brown tissue)
- `4`: Rust Pustules (yellow rust / brown rust active sporulation)
- `5`: Ink Reference Markers (manual alignment pen marks)

---

## 4. Discrepancy & Adaptation Record
- **Observation**: As noted by the authors, leaf alignment markers can wash off or shift slightly over the 15-day observation window, causing the visible analyzable leaf area to shrink slightly in later frames.
- **Adaptation**: To prevent denominator drift, our severity definition fixes the denominator as the analyzable leaf area of the **first aligned frame** of each sequence.
