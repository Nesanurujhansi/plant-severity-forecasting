# Phase 11: Gradio Web UI Integration & Single Process Mounting

## 1. What This Phase Does in One Sentence
Phase 11 mounts an interactive, accessible Gradio web user interface (`app/ui/gradio_ui.py`) directly onto the existing FastAPI application at path `/ui`, giving students and farmers an intuitive tabbed dashboard for leaf image upload, disease severity measurement, 3-day and 7-day progression forecasting, treatment scenario comparison, and educational safety guidance without requiring API technical knowledge.

---

## 2. Why This Step Exists (The Real-World Problem It Solves)
While Phase 10 established a robust FastAPI REST backend (`/analyze`, `/weather`, `/health`), raw JSON REST APIs are inaccessible to agricultural end-users, students, and farmers who need visual decision support. Mounting a Gradio web interface directly inside the FastAPI server solves four major operational challenges:
1. **Single-Process Deployment**: Eliminates the operational complexity and network latency of running separate frontend and backend servers. A single `uvicorn app.main:app` command serves both the REST API and the web UI seamlessly.
2. **Direct Pipeline Execution**: The Gradio interface interacts directly with `InferencePipeline` in memory, bypassing HTTP serialization overhead while retaining all safety policy checks, conformal prediction intervals, and OOD detection.
3. **Publication-Quality Visualizations**: Renders 90% CQR conformal progression forecast bands and counterfactual treatment scenario bar charts via custom Matplotlib rendering functions (`app/ui/plots.py`) alongside lesion segmentation overlays.
4. **Non-Technical Accessibility & Localization**: Offers a simple 4-tab workflow ("Analyze", "Results", "How It Works", "About & Limitations"), built-in sample leaf images, localized screen-reader accessibility labels, clear non-prescription disclaimers, and user privacy guarantees (`store_uploads=False`).

---

## 3. Big-Picture Diagram & Architecture Flow

```mermaid
sequenceDiagram
    autonumber
    actor User as Web User / Browser
    participant Fast as FastAPI App (app/main.py)
    participant Grad as Gradio App (app/ui/gradio_ui.py)
    participant Pipe as InferencePipeline (app/inference/pipeline.py)
    participant Plot as Plotting Engine (app/ui/plots.py)

    User->>Fast: GET / or GET /ui
    Fast->>Grad: Route request to Gradio Mount
    Grad-->>User: Render 4-Tab Web Dashboard HTML/CSS/JS

    User->>Grad: Upload leaf image & click "Analyze Leaf & Forecast"
    Grad->>Pipe: process(image_bytes, crop_type, weather_preset, lat, lon)
    
    alt Safety Abstention Triggered (Blurry / OOD / Non-Leaf)
        Pipe-->>Grad: SafetyBlock (should_abstain=True, user_message)
        Grad-->>User: Display Safety Warning Box & Retake Recommendation
    else Safety Check Passed
        Pipe-->>Grad: AnalyzeResponse (Severity, Weather, Forecast, Scenario)
        Grad->>Plot: plot_progression_forecast(ForecastBlock)
        Plot-->>Grad: Forecast Matplotlib Figure
        Grad->>Plot: plot_scenario_comparison(ScenarioBlock)
        Plot-->>Grad: Scenario Matplotlib Figure
        Grad-->>User: Populate Results Tab (KPIs, Segmentation Overlay, Plots)
    end
```

### ASCII Architecture Overview
```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FastAPI Application (app/main.py)                     │
│                                                                             │
│   ├── REST Endpoints: GET /health, GET /model-info, POST /analyze           │
│   └── Mounted Web UI: GET /ui (gradio.mount_gradio_app)                     │
│                                                                             │
│       ┌─────────────────────────────────────────────────────────────┐       │
│       │                 Gradio UI (app/ui/gradio_ui.py)             │       │
│       │                                                             │       │
│       │   Tab 1: 🌿 Analyze        Tab 2: 📊 Results                 │       │
│       │   ├── Image Upload         ├── Leaf & Mask Overlay          │       │
│       │   ├── Crop Selection       ├── Severity & Conformal KPIs    │       │
│       │   └── Weather Settings     ├── Progression Plot (3d/7d)     │       │
│       │                            └── Treatment Scenario Plot      │       │
│       │   Tab 3: ❓ How It Works   Tab 4: ℹ️ About & Limitations     │       │
│       └──────────────────────────────┬──────────────────────────────┘       │
│                                      │                                      │
│                                      ▼                                      │
│                      InferencePipeline (Direct Memory)                      │
│        ├── Upload Safety Validation & EXIF Stripping                        │
│        ├── Deep Free Energy Safety & OOD Evaluation                         │
│        ├── DINOv2 SegFormer Vision Severity & Conformal Bounds              │
│        ├── PINN-ODE 3d/7d Epidemic Progression Forecasting                  │
│        └── TarNet Counterfactual Treatment Scenario Simulation              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Walkthrough of Files Created

1. **[`app/ui/strings.py`](file:///c:/PlantDiseaseSeverity/app/ui/strings.py)**:
   * **Purpose**: Centralized dictionary (`STRINGS["en"]`) containing localized UI text, titles, tooltips, form labels, confidence tier explanations, screen-reader aria labels, and non-prescription disclaimers.
   * **Key Component**: Prevents hardcoded strings across UI elements and supports clean future multi-language localization.

2. **[`app/ui/plots.py`](file:///c:/PlantDiseaseSeverity/app/ui/plots.py)**:
   * **Purpose**: Matplotlib visualization module generating high-resolution forecast curves and scenario bar charts for Gradio output blocks.
   * **Key Functions**:
     * `plot_progression_forecast()`: Draws 3-day and 7-day severity trajectories with 90% CQR conformal uncertainty bands.
     * `plot_scenario_comparison()`: Draws grouped bar charts comparing 7-day untreated vs. fungicide-protected severity predictions.
     * `blend_mask_overlay()`: Blends binary lesion segmentation masks onto RGB leaf images with configurable opacity and color mapping.

3. **[`app/ui/examples/`](file:///c:/PlantDiseaseSeverity/app/ui/examples)**:
   * **Purpose**: Directory containing sample leaf images (`sample_leaf_01.png`, `sample_leaf_02.png`) and `README.md` documenting CC-BY-4.0 sample provenance for one-click testing in the UI.

4. **[`app/ui/gradio_ui.py`](file:///c:/PlantDiseaseSeverity/app/ui/gradio_ui.py)**:
   * **Purpose**: Main Gradio Blocks UI definition creating the 4-tab interface, linking input forms to pipeline callbacks, and handling STUB mode warning banners.
   * **Key Functions**:
     * `create_gradio_ui()`: Constructs the `gr.Blocks` interface.
     * `run_ui_analysis()`: Asynchronous callback invoking `InferencePipeline.process()` and rendering results or safety warnings.

5. **[`app/main.py`](file:///c:/PlantDiseaseSeverity/app/main.py)** *(Updated)*:
   * **Purpose**: Mounts the Gradio app to path `/ui` via `gradio.mount_gradio_app(app, gradio_ui, path="/ui")` and adds a root endpoint `GET /` redirecting to `/ui`.

6. **[`tests/test_ui.py`](file:///c:/PlantDiseaseSeverity/tests/test_ui.py)**:
   * **Purpose**: Unit test suite covering UI string completeness, Matplotlib plot generation, mask blending algorithms, analysis callback formatting, and `/ui` endpoint mounting response codes.

7. **[`docs/user_guide_draft.md`](file:///c:/PlantDiseaseSeverity/docs/user_guide_draft.md)**:
   * **Purpose**: User manual with step-by-step instructions, input parameter descriptions, output interpretations, and ASCII screen layout mockups for non-technical users.

---

## 5. Worked Example with Concrete Numbers

Suppose an agricultural student opens `http://127.0.0.1:8000/ui` in their browser and runs an analysis:

1. **User Action**:
   * Clicks example image `sample_leaf_01.png` (Wheat septoria leaf).
   * Selects Crop Type: `Wheat (Triticum aestivum)`.
   * Selects Weather Preset: `Warm & Humid (High Risk)`.
   * Clicks **"Analyze Leaf & Forecast"**.

2. **Pipeline Execution & UI Output**:
   * **Safety Block**: Laplacian Blur Var $= 912.4$ (Sharp), Free Energy $= -19.4$ $\rightarrow$ Tier = `HIGH`, Abstain = `False`.
   * **Severity Block**: Affected Area $= 14.2\%$, 90% Conformal Interval $= [10.8\%, 17.6\%]$, Class = `10-25%`.
   * **Weather Block**: Temp $= 22.0^\circ\text{C}$, RH $= 85.0\%$, VPD $= 0.40\text{ kPa}$ (Favorable for fungal sporulation).
   * **Progression Forecast Plot**:
     * Current (Day 0): $14.2\%$
     * Day 3 Forecast: $18.5\%$ ($[15.1\%, 21.9\%]$)
     * Day 7 Forecast: $24.8\%$ ($[20.4\%, 29.2\%]$)
     * Renders shaded green 90% CQR confidence interval band across 7 days.
   * **Scenario Plot**:
     * Untreated 7-Day Severity: $24.8\%$
     * Fungicide-Protected 7-Day Severity: $14.9\%$
     * Expected Relative Severity Reduction: $39.9\%$ ($[34.5\%, 45.3\%]$)
   * **Execution Latency**: $18.2\text{ ms}$ on CPU.

---

## 6. How to Run It

### Launching Single-Process Application
Start the unified FastAPI + Gradio server on port 8000:

```bash
python -m uvicorn app.main:app --reload
```

### Accessing the Interface
Open your web browser and navigate to:
* **Gradio Web UI**: `http://127.0.0.1:8000/ui` (or root `http://127.0.0.1:8000/` which automatically redirects to `/ui`)
* **OpenAPI REST Docs**: `http://127.0.0.1:8000/docs`

### Running the Phase 11 Test Suite
Execute pytest on the UI test module:

```bash
python -m pytest tests/test_ui.py -v
```

To run the entire 108-test project suite:

```bash
python -m pytest
```

---

## 7. What You Should See

1. **Root Redirect & Mounting**:
   Navigating to `http://127.0.0.1:8000/` immediately redirects to `http://127.0.0.1:8000/ui`, displaying the header title **"Wheat Leaf Disease Prognosis System"**.

2. **Tabbed Layout**:
   * **Tab 1 (🌿 Analyze)**: Displays image uploader, preset selector dropdowns, latitude/longitude input fields, and example thumbnails.
   * **Tab 2 (📊 Results)**: Displays key metrics (Severity %, Confidence Tier, Lesion Count), base64 lesion overlay image, Matplotlib 7-day progression curve, and treatment scenario comparison chart.
   * **Tab 3 (❓ How It Works)**: Explains the underlying 4-module deep learning architecture (DINOv2, PINN-ODE, TarNet, Energy OOD) in beginner-friendly terms.
   * **Tab 4 (ℹ️ About & Limitations)**: Highlights decision-support nature, non-prescription compliance, privacy guarantees, and known boundary conditions.

3. **Console Output**:
   ```text
   INFO:     Started server process [12345]
   INFO:     Waiting for application startup.
   INFO:     InferencePipeline container initialized successfully (STUB_MODE=True).
   INFO:     Gradio UI successfully mounted at /ui
   INFO:     Application startup complete.
   INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
   ```

---

## 8. Common Errors and How to Fix Them

| Error | Root Cause | Solution |
| :--- | :--- | :--- |
| **HTTP 404 Not Found at `/ui`** | Gradio app was not mounted to FastAPI via `gradio.mount_gradio_app`. | Verify `app/main.py` calls `mount_gradio_app(app, create_gradio_ui(), path="/ui")`. |
| **`ImportError: cannot import name 'mount_gradio_app' from 'gradio'`** | Incompatible or outdated Gradio package version. | Ensure `gradio>=4.0.0` is installed (`pip install "gradio>=4.0.0"`). |
| **Matplotlib GUI Thread Warning / Crash** | Non-GUI backend not specified before rendering plots. | `app/ui/plots.py` sets `matplotlib.use('Agg')` at top-level before plot creation. |
| **Pytest Env Var Contamination across tests** | Direct `os.environ["DEV_STUB_MODELS"] = "1"` setting in test module root. | Use `monkeypatch.setenv("DEV_STUB_MODELS", "1")` inside pytest fixtures instead. |
| **HTTP 422 Unprocessable Entity on `/weather`** | Endpoint used `Form(...)` instead of `Query(...)` for `GET` query parameters. | Change parameter definitions in `app/main.py` to `Query(...)`. |

---

## 9. Self-Check Questions

1. **Q**: Why is Gradio mounted on FastAPI at `/ui` instead of running as a separate script via `demo.launch()`?  
   **A**: Mounting on FastAPI runs both REST API and web UI in a single process, sharing model memory and avoiding extra server setup or network hops.

2. **Q**: How does the UI handle low-quality or out-of-distribution (OOD) leaf images?  
   **A**: The UI captures `SafetyBlock.should_abstain = True`, displays an alert banner explaining why (e.g. image too blurry or non-leaf detected), and asks the user to retake the photo.

3. **Q**: Are uploaded images saved to the server's disk when submitted through the Gradio UI?  
   **A**: No. Uploads are processed in-memory and discarded immediately unless `store_uploads=True` is explicitly enabled in configuration.

4. **Q**: How are uncertainty ranges displayed to non-technical users in the UI?  
   **A**: Forecast plots show shaded 90% CQR prediction bands, and KPI cards state the calibrated confidence range (e.g. `14.2% [10.8% - 17.6%]`).

---

## 10. Honest Limitations & What Could Go Wrong

1. **CPU Matplotlib Rendering Overhead**: Generating Matplotlib figures per request adds ~10–15 ms latency. While minimal for single users, high concurrency may benefit from pre-rendered client-side JavaScript charting (e.g. Plotly or Chart.js).
2. **Single-Process Global Interpreter Lock (GIL)**: Under high simultaneous UI request loads, Python's GIL can bottleneck CPU inference and plot generation. Production scaling requires Uvicorn multi-worker processes (`uvicorn --workers 4`).
3. **Web Browser Static Caching**: Browser caching of static assets can occasionally require a hard refresh (`Ctrl+F5`) after updating Gradio UI components.

---

## 11. Viva Questions & Model Answers

**Q1: What architectural pattern was used to integrate the Gradio web interface with the FastAPI backend?**  
*Answer*: We used single-process WSGI/ASGI mounting via `gradio.mount_gradio_app(app, gradio_ui, path="/ui")`. This embeds Gradio's Starlette route handler directly inside our FastAPI ASGI application tree, allowing both REST endpoints and UI components to share the same event loop and in-memory model singletons.

**Q2: How does the UI ensure decision-support safety and regulatory compliance?**  
*Answer*: The UI strictly enforces decision support by presenting disease severity as a percentage rather than a chemical prescription, providing calibrated 90% prediction intervals, displaying prominent non-prescription compliance disclaimers on every tab, and triggering automatic abstention banners when OOD or degraded inputs are detected.

**Q3: How are progression forecasts and treatment scenarios visualized for non-technical users?**  
*Answer*: We implemented custom Matplotlib plotting utilities (`app/ui/plots.py`) that render publication-quality 7-day disease progression curves with shaded 90% CQR conformal bands alongside grouped bar charts comparing untreated vs. fungicide-protected scenarios.

---

## 12. Definition of Done

To consider Phase 11 complete, all of the following requirements must be met and verified:

- [x] **Gradio UI Module (`app/ui/gradio_ui.py`)**: Built 4-tab interface (Analyze, Results, How It Works, About) connected directly to `InferencePipeline`.
- [x] **Localized Strings (`app/ui/strings.py`)**: Created centralized text dictionary for UI titles, form labels, tooltips, screen-reader accessibility labels, and disclaimers.
- [x] **Plotting Utilities (`app/ui/plots.py`)**: Built Matplotlib engines rendering 90% CQR conformal forecast bands and scenario comparison bar charts.
- [x] **Example Assets (`app/ui/examples/`)**: Created sample leaf images with CC-BY-4.0 provenance documentation.
- [x] **FastAPI Mount (`app/main.py`)**: Mounted Gradio UI at `/ui` and added `GET /` redirect to `/ui`.
- [x] **Unit Testing (`tests/test_ui.py`)**: All unit tests pass, verifying strings, plot rendering, mask blending, callbacks, and `/ui` endpoint mounting.
- [x] **Documentation (`docs/user_guide_draft.md` & `docs/phase11_gradio_ui.md`)**: User guide manual and B.Tech documentation created and validated by `scripts/check_docs.py`.
- [x] **Full Suite Pass**: All 108 pytest unit tests pass cleanly across 22 test files.
