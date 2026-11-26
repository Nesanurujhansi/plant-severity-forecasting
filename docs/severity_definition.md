# Disease Severity Mathematical Definition & Formulation

---

## 1. Primary Disease Severity Formulation

In **crop-prognosis**, disease severity ($S_t$) at time $t$ for a given leaf sequence ($s$) is defined as:

$$S_t = \text{clip}\left( \frac{\text{Diseased Pixel Area}_t}{\text{Reference Leaf Area}_1}, 0.0, 1.0 \right)$$

Where:
- $\text{Diseased Pixel Area}_t$: The total count of pixels classified as diseased (union of Necrosis [Class 3] and Rust Pustules [Class 4]) at frame $t$.
- $\text{Reference Leaf Area}_1$: The total count of healthy + diseased leaf pixels ($\text{Class } 1 + 2 + 3 + 4$) in the **FIRST aligned frame ($t=1$)** of the leaf time series sequence.

---

## 2. Why Use the First Frame as the Denominator? (Biological Rationale)

In time-series field phenotyping, estimating disease severity using a per-frame leaf area denominator ($\frac{\text{Diseased Area}_t}{\text{Leaf Area}_t}$) introduces severe artifact errors:
1. **Shrinking Analyzable Region**: As leaves dry or tilt, or as ink reference markers wash off in rain, the visible analyzable leaf region in the camera frame shrinks over time.
2. **Artificial Severity Inflation**: If $\text{Leaf Area}_t$ shrinks faster than lesion expansion, the calculated severity percentage artificially increases even when the lesion size remains unchanged.

By pinning $\text{Reference Leaf Area}_1$ to the leaf area observed in Frame 1, the denominator remains constant across the entire 15-day sequence, ensuring that severity changes reflect true biological pathogen growth.

---

## 3. Symptom Class Inclusions & Exclusions

| Pixel Class ID | Label | Included in Diseased Area? | Reason |
| :--- | :--- | :--- | :--- |
| `0` | Background | ❌ Excluded | Non-plant background pixels |
| `1` | Healthy Green Tissue | ❌ Excluded | Healthy photosynthetic leaf area |
| `2` | Chlorosis / Senescence | ❌ Excluded | Natural leaf aging (yellowing), not direct lesion |
| `3` | Necrosis / Septoria Blotch | **YES (Included)** | Dead brown leaf tissue caused by fungal lesion |
| `4` | Active Rust Pustules | **YES (Included)** | Active sporulating uredinia pustules |
| `5` | Ink Reference Markers | ❌ Excluded | Manual alignment pen marks on leaf surface |

---

## 4. Alternative Metrics & Sensitivity Features

To support sensitivity analysis and detailed lesion morphology evaluation, the pipeline calculates and stores five secondary metrics for every image:

1. **Per-Frame Severity ($S_t^{\text{frame}}$)**:
   $$S_t^{\text{frame}} = \frac{\text{Diseased Area}_t}{\text{Leaf Area}_t}$$
2. **Necrosis Fraction**:
   $$\text{Necrosis Fraction}_t = \frac{\text{Necrotic Pixels}_t}{\text{Reference Leaf Area}_1}$$
3. **Pustule Density**:
   $$\text{Pustule Density}_t = \frac{\text{Rust Pustule Pixels}_t}{\text{Reference Leaf Area}_1}$$
4. **Number of Lesions ($n_{\text{lesions}}$)**: Count of distinct connected components among diseased pixels (8-connectivity).
5. **Mean Lesion Area ($\bar{A}_{\text{lesion}}$)**: Average size in pixels of connected lesion components.

---

## 5. Quality Flags

Each image record includes Boolean quality indicators:
- `missing_alignment`: True if registration transformation matrix failed to converge.
- `washed_off_markers`: True if reference marker detection count drops below 2.
- `missing_masks`: True if symptom segmentation mask file is missing or corrupted.
