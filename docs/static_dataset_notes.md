# Static In-The-Wild Dataset Structure & Inspection Notes

This document records the verified directory structures, annotation formats, pixel mask encodings, licences, and evaluation subsets for all static in-the-wild datasets processed in **Phase 03**.

---

## 1. Dataset Overview & Verified Metadata

| Dataset | HuggingFace / Zenodo Path | Images Count | Lesion Masks | Leaf Masks Available | License | Primary Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PlantSeg** | `Voxel51/PlantSeg-Test` (subset) / Zenodo | ~1,200 (subset) | Yes (binary) | No | Custom / Per-image in metadata | In-the-wild disease lesion segmentation across 34 plant types |
| **Coffee Leaf Rust** | `MaryPazRB/Paper_CLR_CV` | 1,285 images | 606 rust masks | No | CC-BY 4.0 | Field coffee rust severity rating & validation against Lin's CCC |
| **LeafNet75 Lesion Masks** | `LeafNet75/Leaf-Disease-Segmentation-masks` | ~1,500 | Yes (binary PNG) | No | Apache-2.0 | Multi-crop in-the-wild lesion segmentation pretraining |
| **LeafNet75 Wild Masks** | `LeafNet75/In_the_Wild_masks600` | 600 | No | Yes (SAM 2 auto) | MIT | Pre-generated SAM 2 leaf pseudo-labels for sanity validation |
| **PlantVillage** | `mohanty/PlantVillage` | ~54,303 (color) | Segmented background | Partial (black background) | CC-BY 4.0 / Open | Controlled lab pretraining & baseline validation |

---

## 2. Directory Layout & Annotation Encodings

### 2.1 PlantSeg (`data/raw/plantseg/`)
- **Format**: Hugging Face `datasets` disk structure (`dataset_dict.json`, `train/`).
- **Columns**: `image` (PIL RGB), `annotation` / `mask` (PIL binary or L mode).
- **Mask Encoding**: `0` = Healthy leaf / background, `255` or `1` = Lesion region.
- **Conversion Strategy**: Re-scale mask values so `0` = Background, `1` = Lesion. Ensure dimensions match the RGB image.

### 2.2 Coffee Leaf Rust (`data/raw/coffee_rust/`)
- **Format**: Standard directory layout (`02_binary_images/base_image/`, `02_binary_images/golden_standard/`).
- **Files**:
  - `02_binary_images/base_image/`: Raw RGB field photographs of coffee leaves.
  - `02_binary_images/golden_standard/`: 606 expert-annotated binary lesion masks.
  - `02_binary_images/golden_standard/100_eval_set/`: 100 independent evaluation masks reserved exclusively for `external_test`.
- **Mask Encoding**: `255` = Rust lesion pustule, `0` = Background/Leaf.
- **Conversion Strategy**: Binarize to `0` and `1`.

### 2.3 LeafNet75 (`data/raw/leafnet_masks/`)
- **Subdirectories**:
  - `lesion_masks/`: Lesion masks across diverse field crops.
  - `wild_masks/`: SAM 2 generated leaf boundaries (`In_the_Wild_masks600`).
- **Mask Encoding**: Single-channel PNG, `0` = Non-lesion, `255` / `1` = Lesion.

### 2.4 PlantVillage (`data/raw/plantvillage/`)
- **Format**: Hugging Face `datasets` disk structure.
- **Configurations**: `color`, `segmented`.
- **Segmented Mask Strategy**: For `segmented` images, black pixels (`(0, 0, 0)`) represent removed background, allowing direct extraction of a binary leaf mask (`leaf = RGB != (0,0,0)`).

---

## 3. Mask Integrity & Standard Specification

All single-channel lesion and leaf masks exported to `data/processed/static/` MUST satisfy the following strict contract:

1. **Format**: Single-channel PNG (8-bit grayscale, mode `L`).
2. **Values**: Strictly binary integer values: `{0, 1}`.
   - `0`: Background / Non-target pixel.
   - `1`: Target pixel (Lesion for lesion masks; Whole Leaf for leaf masks).
3. **Dimensions**: Exact height ($H$) and width ($W$) matching the corresponding RGB source image.
4. **Union Rule Constraint**: For every pixel $(x, y)$, $\text{LesionMask}(x, y) \le \text{LeafMask}(x, y)$. That is, a lesion pixel MUST always be contained inside the leaf mask boundary.

---

## 4. Near-Duplicate Leakage Prevention

In-the-wild datasets often contain near-duplicate image captures (e.g., burst shots of the same leaf from slightly different angles).
To prevent data leakage across splits:
- Perceptual hashes (`dhash`) are calculated for every image using `ml.utils.hashing.compute_dhash`.
- Near-duplicate clusters are formed for images with Hamming distance $\le 10$.
- Cross-validation splits in `static_splits.py` group strictly by `cluster_id` so all near-duplicate images remain within the SAME partition (train, calibration, val, or test).
