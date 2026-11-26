# Plant Disease Severity — Sample Test Dataset

This directory contains standardized benchmark images designed to evaluate every pipeline capability in **Crop Disease Prognosis & Progression Forecasting**:

---

## 📁 Available Test Samples

| File Name | Crop Type | Expected Behavior | Tested Pipeline Stage |
|---|---|---|---|
| `01_healthy_wheat_leaf.png` | Wheat | Clean foliage, 0% baseline, no active lesions | Healthy Leaf baseline & 0% progression safety |
| `02_mild_septoria_leaf.png` | Wheat | Early-stage isolated necrotic spots (~5–15% severity) | Vision Segmentation & early exponential growth |
| `03_moderate_septoria_leaf.png` | Wheat | Established lesion patches (~30–45% severity) | Full 3d/7d Progression & Treatment Scenario reduction |
| `04_severe_rust_leaf.png` | Wheat | Confluent necrotic rust lesions (>50% severity) | High severity binning & logistic saturation curve |
| `05_test_blurry_photo.png` | Wheat | Out-of-focus capture (Laplacian variance < 100) | **Safety Quality Gate** (Blur detection & guidance) |
| `06_test_glare_photo.png` | Wheat | Specular sunlight reflection / flash hotspot | **Safety Quality Gate** (Glare detection filter) |
| `real_leaves/set_12_10_obj2.png` | Coffee | Real field photograph of infected coffee leaf | **Multi-crop segmentation** (Coffee Rust model) |

---

## 🚀 How to Test

### Option 1: In the Web UI (Gradio)
1. Open your browser and navigate to: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**
2. Drag and drop any image from `sample_data/` into the upload box.
3. Select the corresponding crop (`wheat` or `coffee`).
4. Choose a weather scenario preset (e.g. `warm_humid` or `cool_dry`).
5. Click **"Run Foliar Severity & Prognosis"**.

### Option 2: Automated Benchmark Evaluation Script
Run the automated batch test script against the running server:
```powershell
python sample_data/run_eval.py
```

### Option 3: Via REST API (`curl`)
```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "image=@sample_data/03_moderate_septoria_leaf.png" \
  -F "crop=wheat" \
  -F "weather_scenario=warm_humid"
```
