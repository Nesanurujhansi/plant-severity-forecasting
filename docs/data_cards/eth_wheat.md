# Data Card: ETH Zurich Wheat Leaf Image Time Series

## Dataset Identification
- **Dataset Name**: ETH Zurich Wheat Leaf Image Time Series
- **Authors**: Jonas Anderegg and Andreas McDonald
- **Publication Citation**: Scientific Data (2026), DOI: `10.1038/s41597-026-06567-y`
- **Data Repository DOI**: `10.3929/ethz-b-000739194`
- **Associated Toolkit**: [GitHub - sympathique-wheat](https://github.com/and-jonas/sympathique-wheat)
- **Verification Status**: UNVERIFIED (Date: 2026-09-20)

## Dataset Description & Content
- **Content Summary**: 12,520 high-resolution RGB images (~0.03 mm/pixel, 45 MP camera) tracking 1,032 distinct wheat leaf time series (~12.1 images per sequence, interval 30.6 ± 25.1 hours, average duration 14.9 days).
- **Diseases**: Mixed infections of brown rust (*Puccinia triticina*), yellow rust (*Puccinia striiformis*), and Septoria tritici blotch (*Zymoseptoria tritici*).
- **Metadata**: Genotypes (15 wheat cultivars), treatment groups (`FI2`, `0FI1`, `F0I`, `0F0I`), heading dates, geometric transformation matrices, symptom segmentation masks with lesion tracking, and hourly in-canopy temperature and relative humidity from an on-site weather station.
- **Location**: ETH Research Station Lindau-Eschikon, Switzerland (47.449 N, 8.682 E), 2022–2023 and 2023–2024 seasons.

## Storage Size & Access Method
- **Reported Size**: >100 GB (Raw 45 MP images); Processed masks and metadata are ~2–5 GB.
- **Access Method**: Hosted on ETH Research Collection (`10.3929/ethz-b-000739194`). Programmatic download via script `ml/data/download_eth_wheat.py` or direct manual browser download from the ETH data portal.
- **Project Download Policy**: Download metadata (`meta/`) and processed alignment/mask outputs (`processed/`) first. Raw full-frame images are downloaded only when explicitly requested.

## Project Role
- Primary field dataset for Module 2 (Progression Module) logistic growth parameter fitting and Module 3 (Treatment Scenario Module) mixed-effects fungicide contrast estimation.

## License & Citation
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Citation**:
  > Anderegg, J., & McDonald, A. (2026). High-resolution leaf image sequences with geometric alignment for dynamic phenotyping of foliar diseases. *Scientific Data*, 13, Article 6567. https://doi.org/10.1038/s41597-026-06567-y

## Known Limitations
- Wheat crop only; single field site location in Switzerland; leaves carry manual ink reference markers; limited number of independent growing seasons.
