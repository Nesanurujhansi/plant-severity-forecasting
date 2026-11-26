# Data Card: PlantSeg

## Dataset Identification
- **Dataset Name**: PlantSeg (In-The-Wild Plant Disease Segmentation)
- **Authors**: Tianqi Wei et al.
- **Repository / Code**: [GitHub - tqwei05/PlantSeg](https://github.com/tqwei05/PlantSeg)
- **Hugging Face Subset**: `Voxel51/PlantSeg-Test` (1,200 sample evaluation subset)
- **Full Data Archive**: Zenodo
- **Verification Status**: UNVERIFIED (Date: 2026-09-20)

## Content & Structure
- **Reported Content**: 11,458 in-the-wild field images with pixel-level lesion segmentation masks spanning 115 disease categories across 34 plant species.
- **Image Format**: Variable resolution JPEG images with binary pixel mask annotations.

## Storage Size & Access Method
- **Reported Size**: ~4–8 GB (Full Zenodo archive); ~500 MB for HF evaluation subset.
- **Access Method**: Hugging Face Datasets API (`datasets.load_dataset("Voxel51/PlantSeg-Test")`) or direct Zenodo download via `ml/data/download_plantseg.py`.

## Project Role
- Training and fine-tuning dataset for Module 1 (Severity Module) vision backbone feature extraction and lesion/leaf mask segmentation heads across diverse non-wheat crop species.

## License & Citation
- **License**: Per-image license specified in dataset metadata (mostly CC BY-NC 4.0 or public domain).
- **Citation**:
  > Wei, T., et al. (2024). PlantSeg: A benchmark dataset for in-the-wild plant disease lesion segmentation. *arXiv preprint*.
