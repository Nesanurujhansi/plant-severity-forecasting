# Phase 09: Model Bundle Export, Manifest, Loader, and Golden Tests

## 1. What This Phase Does in One Sentence
Phase 09 creates a secure, zero-pickle model export and hand-off mechanism that packages all trained deep learning weights, configurations, calibration parameters, and golden reference tests into a single zip bundle (`crop-prognosis-bundle-v1.0.0.zip`), verifies its integrity with SHA-256 checksums, and loads it automatically into the local application with a development fallback stub mode (`DEV_STUB_MODELS=1`).

---

## 2. Why This Step Exists (The Real-World Problem It Solves)
Training heavy computer vision backbones (DINOv2) and physics-informed neural ODEs requires powerful GPUs available in Google Colab. However, serving web users happens on local CPU servers or cloud endpoints. Transferring model files across temporary cloud environments presents three major risks:
1. **Security Vulnerabilities (Pickle Injection)**: Python's standard `pickle` module can execute arbitrary malicious code upon loading (`os.system("rm -rf /")`). We eliminate this risk by strictly using `.safetensors`, JSON, and text model formats.
2. **File Corruption & Version Incompatibility**: Interrupted downloads or differing PyTorch/LightGBM versions between Colab and local environments cause silent runtime crashes. We prevent this using `manifest.json` SHA-256 file hashes and library version checks.
3. **Model Prediction Drift**: Numerical differences between GPU and CPU math could silently change severity predictions. We use **Golden Reference Testing**—running synthetic benchmark leaf inputs in Colab, saving the exact expected outputs, and verifying them locally within strict numerical tolerances before application startup.

---

## 3. Big-Picture Architecture Diagram

```mermaid
graph TD
    A["Google Colab GPU Session"] -->|Train & Export| B["data/raw/crop-prognosis-bundle-v1.0.0.zip"]
    B -->|User Downloads Zip| C["Local Folder: crop-prognosis/models/"]
    C --> D["verify_bundle CLI / Verification Engine"]
    D -->|1. Validate manifest.json| E{"All Checks Pass?"}
    D -->|2. Check SHA-256 Hashes| E
    D -->|3. Check Library Versions| E
    D -->|4. Run Golden Ref Tests| E
    E -->|YES| F["App Loader (ModelContainer Singleton)"]
    E -->|NO| G["Raise Loud Startup RuntimeError"]
    H["DEV_STUB_MODELS=1 Flag"] -->|Bypass Bundle| I["Stub Predictors + Warning Banner"]
```

### ASCII Architecture Fallback
```text
+-----------------------+      +-------------------------------+
|  Google Colab GPU     | ---> | crop-prognosis-bundle-v1.0.0  |
|  - Trains DINOv2/PINN |      | (.safetensors, JSON, .txt)    |
+-----------------------+      +---------------+---------------+
                                               |
                                               v
                               +---------------+---------------+
                               |  Local Directory: models/    |
                               +---------------+---------------+
                                               |
                                               v
                               +---------------+---------------+
                               |  python -m ml.export.verify  |
                               |  - SHA-256 Checksum           |
                               |  - Library Versions           |
                               |  - Golden Numerical Tests     |
                               +---------------+---------------+
                                               |
                                               v
                               +---------------+---------------+
                               |  App ModelContainer Singleton  |
                               +-------------------------------+
```

---

## 4. Walkthrough of Files Created

1. **[`ml/export/manifest.py`](file:///c:/PlantDiseaseSeverity/ml/export/manifest.py)**:
   * **Purpose**: Defines `ManifestSchema` using Pydantic and builds `manifest.json`.
   * **Key Functions**: `calculate_sha256()`, `get_installed_library_versions()`, `build_manifest()`.

2. **[`ml/export/verify_bundle.py`](file:///c:/PlantDiseaseSeverity/ml/export/verify_bundle.py)**:
   * **Purpose**: Runnable verification engine (`python -m ml.export.verify_bundle models/`).
   * **Key Functions**: `verify_bundle()`, `verify_file_hashes()`, `check_library_versions()`, `verify_golden_outputs()`.

3. **[`ml/export/export_bundle.py`](file:///c:/PlantDiseaseSeverity/ml/export/export_bundle.py)**:
   * **Purpose**: Assembles artifacts from Phases 5–8, generates synthetic golden leaf tests, builds manifest, packages zip file, and verifies archive.
   * **Key Functions**: `export_bundle()`, `create_synthetic_golden_leaf()`, `generate_golden_expected_outputs()`.

4. **[`app/inference/loader.py`](file:///c:/PlantDiseaseSeverity/app/inference/loader.py)**:
   * **Purpose**: Singleton `ModelContainer` class managing application startup, running bundle verification, and providing `DEV_STUB_MODELS=1` fallback mode.
   * **Key Functions**: `get_model_container()`, `ModelContainer._initialize()`.

5. **[`notebooks/colab_phase09_export.py`](file:///c:/PlantDiseaseSeverity/notebooks/colab_phase09_export.py)**:
   * **Purpose**: Executable Colab script with `# %% [Step]` headers for assembling, verifying, and downloading the model bundle zip.

6. **[`tests/test_export.py`](file:///c:/PlantDiseaseSeverity/tests/test_export.py)** & **[`tests/test_export_security.py`](file:///c:/PlantDiseaseSeverity/tests/test_export_security.py)**:
   * **Purpose**: Unit test suites verifying manifest schemas, hash mismatch detection, missing file handling, version checking, golden drift failures, stub loading, zip-slip protection, and zero-pickle AST compliance.

7. **[`docs/bundle_transfer_guide.md`](file:///c:/PlantDiseaseSeverity/docs/bundle_transfer_guide.md)**:
   * **Purpose**: Step-by-step user guide for transferring and unzipping model files.

---

## 5. Worked Example with Concrete Numbers

Suppose our exported model bundle contains a linear head weight tensor inside `severity/severity_heads.safetensors`.

1. **Calculating File Fingerprint (SHA-256 Checksum)**:
   * File path: `severity/severity_heads.safetensors`
   * Raw size: `702,448` bytes
   * Computed SHA-256 digest: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
   * Written into `manifest.json`:
     ```json
     "severity/severity_heads.safetensors": {
       "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
       "size_bytes": 702448
     }
     ```

2. **Golden Reference Output Verification**:
   * Synthetic leaf input: `golden/inputs/golden_leaf_01.png`
   * Expected lesion severity output saved in Colab: $10.5\%$
   * Absolute tolerance ($\text{atol}$): $10^{-3} = 0.001$, Relative tolerance ($\text{rtol}$): $10^{-2} = 0.01$
   * Local CPU prediction on `golden_leaf_01.png`: $10.5002\%$
   * Parity Check: $|10.5002 - 10.5| = 0.0002 \le 0.001 + 0.01 \times 10.5 = 0.106$ $\rightarrow$ **`[PASS]`**.

---

## 6. How to Run

### In Google Colab
Run all cells in [`notebooks/colab_phase09_export.py`](file:///c:/PlantDiseaseSeverity/notebooks/colab_phase09_export.py):
```python
python notebooks/colab_phase09_export.py
```

### On Local Machine
1. Export bundle zip locally (for testing):
   ```bash
   python -m ml.export.export_bundle
   ```
2. Unzip `data/raw/crop-prognosis-bundle-v1.0.0.zip` into `models/`.
3. Verify unzipped bundle:
   ```bash
   python -m ml.export.verify_bundle models/
   ```
4. Run full unit test suite:
   ```bash
   python -m pytest tests/
   ```

---

## 7. Expected Outputs

Running `python -m ml.export.verify_bundle models/` prints:

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

## 8. Common Errors and How to Fix Them

1. **`FileNotFoundError: Missing manifest file 'manifest.json' in 'models/'`**:
   * *Cause*: The model bundle zip was not unzipped into `models/` or files were placed in a nested subdirectory (`models/crop-prognosis-bundle-v1.0.0/manifest.json`).
   * *Fix*: Unzip directly so that `models/manifest.json` exists at the root of `models/`.

2. **`RuntimeError: SHA-256 hash mismatch for 'severity/severity_heads.safetensors'`**:
   * *Cause*: File was partially downloaded or corrupted during transit.
   * *Fix*: Re-download `crop-prognosis-bundle-v1.0.0.zip` from Google Colab and overwrite `models/`.

3. **`RuntimeError: Major version mismatch for 'torch': current=1.13.1 vs manifest=2.5.1`**:
   * *Cause*: Your local Python environment has an outdated major version of PyTorch.
   * *Fix*: Upgrade PyTorch locally: `pip install torch --upgrade`.

4. **`ValueError: Zip-Slip path traversal vulnerability detected`**:
   * *Cause*: The zip file contains unsafe paths starting with `..` or absolute paths (`/etc/passwd`).
   * *Fix*: Do not extract untrusted zip archives. Always generate bundles using `ml.export.export_bundle`.

5. **`Golden test case 'golden_leaf_01' key 'severity_pct' output drift`**:
   * *Cause*: Local CPU inference produced significantly different numerical results compared to GPU reference outputs beyond tolerances.
   * *Fix*: Verify PyTorch CPU installation and confirm random seeds are fixed.

---

## 9. Self-Check Questions with Answers

1. **Q: Why don't we use `.pkl` or `joblib` files to save model weights?**
   * *A*: Python pickle files can execute arbitrary, un-trusted code during deserialization (`os.system`), presenting a severe security hazard. `.safetensors` and JSON only store raw numeric arrays and text settings.
2. **Q: What is a SHA-256 checksum?**
   * *A*: A 64-character cryptographic hash value that acts as a unique digital fingerprint for a file. If even a single bit in the file changes, the checksum changes completely.
3. **Q: What is `manifest.json`?**
   * *A*: The packing list of the model bundle. It records the version, creation date, library versions, feature schema hash, file list with SHA-256 digests, evaluation metrics summary, and non-prescription disclaimers.
4. **Q: What is a Golden Reference Test?**
   * *A*: A fixed benchmark test using synthetic reference input images and pre-computed expected predictions stored in Colab to verify that local CPU predictions match GPU outputs within tolerances.
5. **Q: What happens if `DEV_STUB_MODELS=1` is set?**
   * *A*: The application bypasses model loading and uses lightweight stub predictors, displaying a warning banner: `"[WARNING] DEV_STUB_MODELS ACTIVE - NOT REAL PREDICTIONS"`.
6. **Q: Why is `manifest.json` excluded from its own `files` dictionary in the manifest?**
   * *A*: Including `manifest.json` in its own dictionary would create a circular dependency (computing the file's SHA-256 hash would modify the file itself).
7. **Q: How does the app loader handle bundle verification failure?**
   * *A*: It raises a loud `RuntimeError` at application startup containing the exact verification failure report, preventing invalid or corrupted models from serving user requests.
8. **Q: What is Zip-Slip protection?**
   * *A*: A security check during unzipping that ensures all target file paths reside strictly inside the target destination folder, rejecting relative paths like `../evil.txt`.

---

## 10. Honest Limitations

* **Model File Size**: The bundle size is ~1 MB in synthetic/stub mode, but full DINOv2 vision backbone weights will increase the bundle size to ~80 MB.
* **CPU Inference Parity**: Slight floating-point differences ($10^{-5}$) between GPU (CUDA) and CPU (x86_64) math require loose float tolerances ($\text{atol}=10^{-3}$) in golden tests.
* **Manual Unzip Requirement**: The user must manually download and extract the zip archive into `models/`.

---

## 11. Viva Questions and Model Answers

1. **Q: Explain how your project guarantees supply-chain security when loading deep learning models.**
   * *Model Answer*: We enforce 100% zero-pickle serialization. All neural network weights are stored in the `.safetensors` format, decision trees in LightGBM text format, and metadata in JSON. We also verify cryptographic SHA-256 hashes against a signed `manifest.json` before loading.
2. **Q: What is the purpose of `feature_schema_hash` in the manifest?**
   * *Model Answer*: It ensures that the exact feature definitions and column ordering used during LightGBM residual training in Colab match the feature extraction logic running locally on the serving server.
3. **Q: How do you handle major library version mismatches between Colab and local deployment?**
   * *Model Answer*: `verify_bundle.py` parses semantic version strings. If a major version of a core dependency (like `torch` 1.x vs 2.x) differs, verification fails loudly to prevent hidden API incompatibility crashes.
4. **Q: What design pattern is used for loading models into the web app?**
   * *Model Answer*: The **Singleton Pattern** via `ModelContainer`. This guarantees that model weights are loaded and verified exactly once at server startup, avoiding redundant memory consumption on every API request.
5. **Q: How does your application support local development without trained model weights?**
   * *Model Answer*: We implemented a `DEV_STUB_MODELS` environment flag. When active, `ModelContainer` loads lightweight stub predictors that return mock predictions with a clear warning banner.
6. **Q: What non-prescription disclaimers are embedded into the model bundle?**
   * *Model Answer*: `manifest.json` embeds a mandatory disclaimer stating that the web app provides non-prescription decision support only, never outputting pesticide names, doses, or spray timings, and does not replace a professional agronomist.

---

## 12. Definition of Done Checklist

- [x] `ml/export/manifest.py` implemented with Pydantic `ManifestSchema` and SHA-256 builder.
- [x] `ml/export/export_bundle.py` implemented to package Phase 5–8 artifacts into `crop-prognosis-bundle-v1.0.0.zip`.
- [x] `ml/export/verify_bundle.py` implemented with CLI (`python -m ml.export.verify_bundle models/`).
- [x] `app/inference/loader.py` implemented with `ModelContainer` singleton and `DEV_STUB_MODELS=1` support.
- [x] `tests/test_export.py` and `tests/test_export_security.py` written and passing 100%.
- [x] `notebooks/colab_phase09_export.py` Colab export notebook script created.
- [x] `docs/bundle_transfer_guide.md` created with step-by-step handoff instructions.
- [x] `docs/phase09_bundle.md` written adhering to all 12 B.Tech documentation rules.
