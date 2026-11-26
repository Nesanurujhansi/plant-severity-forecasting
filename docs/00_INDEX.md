# Documentation Master Index

Welcome to the documentation suite for **Crop Leaf Health & Disease Guide (crop-prognosis)**. Below is the status of each sequential project phase, links to phase-specific guides, and the recommended reading sequence.

---

## Recommended Reading Order
1. [BEGINNERS_ROADMAP.md](file:///c:/PlantDiseaseSeverity/docs/BEGINNERS_ROADMAP.md) - Conceptual story, overview, and architectural flow.
2. [PROJECT_CONTEXT.md](file:///c:/PlantDiseaseSeverity/docs/PROJECT_CONTEXT.md) - Domain context, dataset ledger, and safety disclaimers.
3. [user_guide_draft.md](file:///c:/PlantDiseaseSeverity/docs/user_guide_draft.md) - Step-by-step user interface walkthrough.
4. [CHANGELOG.md](file:///c:/PlantDiseaseSeverity/docs/CHANGELOG.md) - Version history and latest production release notes.
5. [api_reference.md](file:///c:/PlantDiseaseSeverity/docs/api_reference.md) - OpenAPI REST endpoint reference.
6. [DATA_SOURCES.md](file:///c:/PlantDiseaseSeverity/docs/DATA_SOURCES.md) - Dataset provenance, licenses, and verified metadata.
7. [DECISIONS.md](file:///c:/PlantDiseaseSeverity/docs/DECISIONS.md) - Architectural Decision Records (ADR).
8. [GLOSSARY.md](file:///c:/PlantDiseaseSeverity/docs/GLOSSARY.md) - Reference definitions for ML, vision, and epidemiology terms.

---

## Phase Status Summary

| Phase ID | Phase Name | Status | Phase Document Link | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 00** | Repository, Tooling & Docs | **DONE** | [phase00_setup.md](file:///c:/PlantDiseaseSeverity/docs/phase00_setup.md) | Skeleton layout, environment configuration, secure IO, testing, and documentation validator. |
| **Phase 01** | Data Audit, Downloaders & Weather | **DONE** | [phase01_data_audit.md](file:///c:/PlantDiseaseSeverity/docs/phase01_data_audit.md) | Data cards, resumable downloaders, NASA POWER client, weather parity, and dHash deduplication. |
| **Phase 02** | ETH Wheat Sequence Processing | **DONE** | [phase02_eth_sequences.md](file:///c:/PlantDiseaseSeverity/docs/phase02_eth_sequences.md) | Pinned-denominator disease severity, sequence forecasting pairs, leak-free GroupKFold splits, and trajectory EDA. |
| **Phase 04** | Feature Extraction & Caching | **DONE** | [phase04_features.md](file:///c:/PlantDiseaseSeverity/docs/phase04_features.md) | DINOv2 feature extraction, storage footprint estimator, sharded safetensors caching, OOD stats, and PyTorch dataset class. |
| **Phase 05** | Vision Backbone & Mask Heads | **DONE** | [phase05_severity.md](file:///c:/PlantDiseaseSeverity/docs/phase05_severity.md) | Lesion and leaf mask segmentation head training, severity estimation, and Mondrian conformal calibration. |
| **Phase 06** | Epidemiological Progression Forecasting | **DONE** | [phase06_progression.md](file:///c:/PlantDiseaseSeverity/docs/phase06_progression.md) | PINN-ODE Yan & Hunt Beta-type growth curve, Vapor Pressure Deficit (VPD in kPa) physics, Deep Quantile Neural Networks (DQNN), Mondrian Conformal Calibration, and Deep Free Energy OOD safety evaluator. |
| **Phase 07** | Treatment Scenario Modeling | **DONE** | [phase07_treatment.md](file:///c:/PlantDiseaseSeverity/docs/phase07_treatment.md) | Deep Counterfactual Treatment Network (TarNet / DeepNME) with shared feature trunk, dual outcome heads, mixed-effects baseline, and cluster bootstrap bounds. |
| **Phase 08** | Safety, Quality & OOD Layer | **DONE** | [phase08_safety.md](file:///c:/PlantDiseaseSeverity/docs/phase08_safety.md) | Deep Free Energy OOD Detector ($E(\mathbf{x}) = -T \log \sum \exp(f_i/T)$), Mahalanobis/kNN distance, quality evaluation, range checks, confidence tier policy, and synthetic stress testing. |
| **Phase 09** | Model Bundle Export & Loader | **DONE** | [phase09_bundle.md](file:///c:/PlantDiseaseSeverity/docs/phase09_bundle.md) | Zero-pickle model bundle exporter, SHA-256 manifest builder, bundle verifier CLI, app singleton model loader, and golden reference testing. |
| **Phase 10** | FastAPI Backend Application | **DONE** | [phase10_fastapi.md](file:///c:/PlantDiseaseSeverity/docs/phase10_fastapi.md) | High-performance REST API, upload safety validation, microclimate weather resolution, inference pipeline, and OpenAPI reference documentation. |
| **Phase 11** | Next.js 14 Production Web Client | **DONE** | [frontend/README.md](file:///c:/PlantDiseaseSeverity/frontend/README.md) | Modern Next.js 14 multilingual web interface (EN, TE, HI) with interactive layer overlays and field location picker. |
| **Phase 12** | Direct 2-Page PDF Export & Plain Language | **DONE** | [user_guide_draft.md](file:///c:/PlantDiseaseSeverity/docs/user_guide_draft.md) | Direct ReportLab publication-grade 2-page A4 field advisory PDF report generation with plain-language farmer terminology. |
