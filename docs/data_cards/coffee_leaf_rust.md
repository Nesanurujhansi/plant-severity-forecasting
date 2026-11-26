# Data Card: Coffee Leaf Rust Severity Dataset

## Dataset Identification
- **Dataset Name**: Coffee Leaf Rust Severity Dataset (`MaryPazRB/Paper_CLR_CV`)
- **Authors**: Mary Paz Rodriguez et al.
- **Hugging Face Repository**: `MaryPazRB/Paper_CLR_CV`
- **Verification Status**: UNVERIFIED (Date: 2026-09-20)

## Content & Structure
- **Reported Content**: 1,285 field images of coffee leaves (*Coffea arabica*) with 606 expert-annotated pixel-level rust (*Hemileia vastatrix*) lesion masks and 100 independent evaluation masks.
- **Statistical Metric**: Evaluated using Lin's Concordance Correlation Coefficient (CCC) for annotator agreement.

## Storage Size & Access Method
- **Reported Size**: ~600 MB.
- **Access Method**: Download via Hugging Face Hub API or `ml/data/download_coffee_rust.py`.

## Project Role
- Cross-species severity mask evaluation benchmark for Module 1.

## License & Citation
- **License**: Creative Commons Attribution 4.0 (CC BY 4.0).
- **Citation**:
  > Rodriguez, M. P., et al. (2024). Coffee Leaf Rust Computer Vision Dataset and Concordance Analysis. *Hugging Face Datasets*.
