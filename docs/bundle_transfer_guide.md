# Model Bundle Transfer & Verification Guide

This document provides step-by-step instructions for transferring trained model weights, configuration parameters, calibration metrics, and reference tests from **Google Colab** to your **local machine** using the secure zero-pickle zip bundle architecture.

---

## Big-Picture Transfer Flow

```
+------------------------------------+
|       Google Colab GPU Session     |
| - Runs notebooks/colab_phase09...  |
| - Assembles model bundle zip       |
| - Generates manifest.json          |
| - Computes SHA-256 Checksum        |
+-----------------+------------------+
                  |
                  | Download crop-prognosis-bundle-v1.0.0.zip
                  v
+------------------------------------+
|          Local Machine             |
| 1. Extract to crop-prognosis/models|
| 2. Run local verifier CLI:         |
|    python -m ml.export.verify_bundle|
| 3. Application loads verified      |
|    ModelContainer at startup       |
+------------------------------------+
```

---

## Target Local Directory Tree

After extracting `crop-prognosis-bundle-v1.0.0.zip`, your local `models/` directory MUST look exactly like this:

```
crop-prognosis/
└── models/
    ├── manifest.json
    ├── README.txt
    ├── severity/
    │   ├── config.json
    │   ├── conformal.json
    │   └── severity_heads.safetensors
    ├── forecast/
    │   ├── cqr_calibration.json
    │   ├── feature_schema.json
    │   ├── forecast_config.json
    │   ├── growth_params.json
    │   ├── lgbm_q_high.txt
    │   ├── lgbm_q_low.txt
    │   ├── lgbm_q_med.txt
    │   ├── lgbm_residual.txt
    │   └── pinn_residual.safetensors
    ├── scenario/
    │   ├── effect_bootstrap.json
    │   ├── mixed_effects.json
    │   ├── scenario_config.json
    │   └── treatment_net.safetensors
    ├── safety/
    │   ├── ood_meta.json
    │   ├── ood_stats.safetensors
    │   └── safety_config.json
    └── golden/
        ├── inputs/
        │   └── golden_leaf_01.png
        └── expected_outputs.json
```

---

## Step-by-Step Instructions

### Step 1: Run Export Notebook in Google Colab
1. Open [`notebooks/colab_phase09_export.py`](file:///c:/PlantDiseaseSeverity/notebooks/colab_phase09_export.py) in Google Colab.
2. Execute all cells. The script will assemble all model files from Drive, generate synthetic golden test cases, write `manifest.json`, and create `data/raw/crop-prognosis-bundle-v1.0.0.zip`.
3. Take note of the printed **SHA-256 Checksum** and **File Size**.

### Step 2: Download Archive File
1. Allow Colab to download `crop-prognosis-bundle-v1.0.0.zip` automatically, or download it manually from `data/raw/` in Drive.
2. (Optional) Verify the SHA-256 hash on your local terminal:
   ```bash
   certutil -hashfile crop-prognosis-bundle-v1.0.0.zip SHA256
   ```

### Step 3: Extract Archive into `models/`
1. Unzip the contents of `crop-prognosis-bundle-v1.0.0.zip` directly into the `models/` root folder of this repository.

### Step 4: Run Local Verification CLI
Execute the verifier module to confirm checksum integrity, library compatibility, feature schema hash matching, model file loadability, and golden output tolerance compliance:

```bash
python -m ml.export.verify_bundle models/
```

**Expected Console Output**:
```text
===========================================================================
CROP-PROGNOSIS MODEL BUNDLE VERIFICATION REPORT
Bundle Directory: C:\PlantDiseaseSeverity\models
===========================================================================
[PASS] manifest.json schema validated (Bundle Version: v1.0.0)
[PASS] All 21 registered bundle files passed SHA-256 hash checks.
[PASS] Feature schema hash is 100% consistent with forecast schema.
[PASS] Golden reference numerical output test passed within tolerances.
---------------------------------------------------------------------------
SUMMARY: VERIFICATION PASSED 100% SUCCESSFUL!
The model bundle is authentic, uncorrupted, and ready for app inference.
===========================================================================
```

---

## Development Stub Mode (`DEV_STUB_MODELS`)

If you want to run or test the FastAPI/Gradio application before unzipping a real model bundle, activate stub mode by setting the environment variable:

```bash
set DEV_STUB_MODELS=1
```

Or in PowerShell:
```powershell
$env:DEV_STUB_MODELS="1"
```

When active, the application will initialize instantly using lightweight stub predictors and present a clear warning banner:
`"[WARNING] DEV_STUB_MODELS ACTIVE - NOT REAL PREDICTIONS"`.
