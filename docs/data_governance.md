# Data Governance, Licensing & Privacy Framework

---

## 1. Dataset Licensing & Redistribution Policy

Every dataset integrated into **crop-prognosis** is governed by explicit license conditions. The table below outlines attribution requirements, commercial usage rights, and redistribution permissions for each dataset:

| Dataset Name | Primary License | Can Redistribute Data? | Commercial Use? | Attribution Required? | Special Conditions |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ETH Zurich Wheat Time Series** | CC BY 4.0 | Yes (with credit) | Yes | **Mandatory** | Cite Anderegg & McDonald (2026). Do not remove manual reference mark metadata. |
| **PlantSeg** | Per-Image (CC BY-NC / Public) | No (link to Zenodo) | Non-Commercial | **Mandatory** | Preserve individual image license fields in manifest. |
| **Coffee Leaf Rust (`Paper_CLR_CV`)** | CC BY 4.0 | Yes (with credit) | Yes | **Mandatory** | Cite Rodriguez et al. (2024). |
| **LeafNet75 Lesion Masks** | Apache-2.0 | Yes | Yes | **Mandatory** | Include original Apache License notice file. |
| **LeafNet75 Wild Masks** | MIT | Yes | Yes | **Mandatory** | Include original MIT License copyright header. |
| **PlantVillage** | CC0 1.0 (Public Domain) | Yes | Yes | Optional | Restricted to pretraining/sanity check; never for headline claims. |
| **NASA POWER Weather API** | NASA Public Domain | Yes | Yes | **Mandatory** | Acknowledge NASA POWER project in publications. |
| **56 European Wheat Trials** | Restricted | **NO** | Restricted | **Mandatory** | Access request required via Zenodo record 6521152. |

---

## 2. Privacy & Safety Rules for User-Uploaded Photos

When users upload leaf photos to the application for inference:

1. **Default Zero-Storage Rule**: User-uploaded photos are processed in temporary memory during the HTTP request lifetime and are **NEVER** stored on disk or written to permanent databases by default.
2. **No Metadata Leakage**: EXIF metadata (GPS location coordinates, camera device IDs, timestamps) is stripped in memory before passing image arrays to vision backbones.
3. **Explicit Opt-in for Data Contribution**: User photos may only be saved for research dataset expansion if the user explicitly checks an opt-in consent box in the UI.

---

## 3. How to Cite Each Dataset in Research Reports

### ETH Zurich Wheat Leaf Time Series
```bibtex
@article{Anderegg2026ETH,
  author    = {Jonas Anderegg and Andreas McDonald},
  title     = {High-resolution leaf image sequences with geometric alignment for dynamic phenotyping of foliar diseases},
  journal   = {Scientific Data},
  year      = {2026},
  volume    = {13},
  pages     = {6567},
  doi       = {10.1038/s41597-026-06567-y}
}
```

### PlantSeg
```bibtex
@article{Wei2024PlantSeg,
  author    = {Tianqi Wei and others},
  title     = {PlantSeg: A benchmark dataset for in-the-wild plant disease lesion segmentation},
  journal   = {arXiv preprint},
  year      = {2024}
}
```

### NASA POWER Weather API
```bibtex
@misc{NASAPOWER2026,
  author    = {{NASA POWER}},
  title     = {Prediction of Worldwide Energy Resources Daily Agroclimatology API},
  year      = {2026},
  url       = {https://power.larc.nasa.gov/}
}
```
