# Data Sources & Provenance Registry

This registry tracks candidate datasets evaluated for **crop-prognosis**, recording metadata, download URLs, licenses, reported sizes, and empirical verification status. Detailed information is documented in individual dataset cards in [`docs/data_cards/`](file:///c:/PlantDiseaseSeverity/docs/data_cards/).

---

## Summary of Candidate Datasets

| Dataset Name | Card Link | Primary Source / DOI | Reported Size | License | Verification Status | Date Checked |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ETH Zurich Wheat Time Series** | [eth_wheat.md](file:///c:/PlantDiseaseSeverity/docs/data_cards/eth_wheat.md) | DOI: `10.3929/ethz-b-000739194` | ~100 GB (Raw) / 2-5 GB (Proc) | CC BY 4.0 | UNVERIFIED | 2026-09-20 |
| **PlantSeg** | [plantseg.md](file:///c:/PlantDiseaseSeverity/docs/data_cards/plantseg.md) | Zenodo / HF `Voxel51/PlantSeg-Test` | ~4-8 GB | Metadata per-image | UNVERIFIED | 2026-09-20 |
| **Coffee Leaf Rust Severity** | [coffee_leaf_rust.md](file:///c:/PlantDiseaseSeverity/docs/data_cards/coffee_leaf_rust.md) | HF `MaryPazRB/Paper_CLR_CV` | ~600 MB | CC BY 4.0 | UNVERIFIED | 2026-09-20 |
| **LeafNet75 Masks** | [leafnet75.md](file:///c:/PlantDiseaseSeverity/docs/data_cards/leafnet75.md) | HF `LeafNet75/*` | ~54 MB | Apache-2.0 / MIT | UNVERIFIED | 2026-09-20 |
| **PlantVillage** | [plantvillage.md](file:///c:/PlantDiseaseSeverity/docs/data_cards/plantvillage.md) | HF `mohanty/PlantVillage` | ~800 MB | CC0 1.0 (Public Domain) | UNVERIFIED | 2026-09-20 |
| **LeafNet** | [leafnet.md](file:///c:/PlantDiseaseSeverity/docs/data_cards/leafnet.md) | HF `Project-AgML/LeafNet` | Large | Research Open Access | UNVERIFIED | 2026-09-20 |
| **NASA POWER Weather API** | [nasa_power.md](file:///c:/PlantDiseaseSeverity/docs/data_cards/nasa_power.md) | `power.larc.nasa.gov` API | REST API | Public Domain | VERIFIED | 2026-09-20 |
| **56 European Wheat Trials** | [zenodo_fungicide_trials.md](file:///c:/PlantDiseaseSeverity/docs/data_cards/zenodo_fungicide_trials.md) | Zenodo `6521152` | Tabular | Restricted Access | UNVERIFIED | 2026-09-20 |

> [!IMPORTANT]
> **Data Size Strategy**: Raw ETH Zurich wheat images are 45-megapixel camera files exceeding 100 GB. The project pipeline downloads processed geometric alignment matrices, symptom masks, and metadata files first. Raw full-frame images are only fetched if explicitly requested.
