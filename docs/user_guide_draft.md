# User Guide & Interface Manual

Welcome to **Crop Leaf Health & Disease Guide (crop-prognosis)**, a clinical decision-support web application for farmers, agricultural students, and field extension officers.

This guide provides a walkthrough of the Next.js Web Interface hosted at `http://localhost:3000`.

---

## Screen Layout Tour

```text
+-----------------------------------------------------------------------------------+
|               Crop Leaf Health & Disease Guide (crop-prognosis)                   |
|       Leaf Disease Check, 7-Day Local Weather & Field Care Guide                  |
+-----------------------------------------------------------------------------------+
|  [ Single Leaf Check ]      [ How It Works ]      [ Safety & Information ]        |
+-----------------------------------------------------------------------------------+
| LEFT COLUMN (Inputs & Field Setup)      | RIGHT COLUMN (Diagnostic Results)       |
|                                         |                                         |
| 1. Upload Leaf Photo:                   | 1. Photo Quality & Safety Check:        |
|    - Click or drag-and-drop leaf photo  |    - High Confidence (Quality verified) |
|    - Quick-select benchmark samples     |                                         |
|                                         | 2. Disease Damage Card:                 |
| 2. Crop Details:                        |    - Total Leaf Damage (% with range)   |
|    - Crop Selector (Wheat, Rice, Coffee)|    - Dead Brown Spots (%)               |
|    - Crop Variety (Optional)            |    - Spreading Yellow Border (%)        |
|    - Crop Age in Days (Optional)        |    - Spots Found Count                  |
|                                         |                                         |
| 3. Field Location & Local Weather:      | 3. Visual Leaf Inspection Gallery:      |
|    - Andhra Pradesh Hub Presets         |    - Original Leaf Photo                |
|    - Live Mandal / Town Search          |    - Disease Spot Outlines              |
|    - Google Maps Share Link Resolver    |    - 7-Day Disease Spread Forecast      |
|    - Device GPS Detection               |    - AI Vision Focus Areas              |
|                                         |                                         |
| [ Check Leaf Health & Weather Forecast] | 4. 7-Day Weather & Spraying Days:       |
|                                         |    - Temperature, Humidity, VPD, Rain   |
|                                         |    - Safe Spray Window Feasibility      |
|                                         |                                         |
|                                         | [ Download Crop Health Report (PDF) ]   |
+-----------------------------------------------------------------------------------+
```

---

## Detailed Step-by-Step User Workflow

### Step 1: Upload a Clear Leaf Photograph
- Take a clear photo of a single diseased leaf in natural daylight.
- Drag and drop the image into the upload box, or click to choose from your device.
- Alternatively, click any of the benchmark test samples (e.g. Coffee Rust, Rice Blast, Wheat Blotch) to test the system immediately.

### Step 2: Select Crop & Farm Location
- Choose your crop from the dropdown list (e.g., Wheat, Rice, Coffee, Tomato, Cotton).
- Select your field location using one of four methods:
  1. **Region Hub**: Click any instant preset hub.
  2. **Mandal Search**: Type your town or mandal name and click Search.
  3. **Google Maps Link**: Paste a Google Maps share link (`maps.app.goo.gl/...`) or latitude and longitude coordinates.
  4. **Device GPS**: Click "Detect Current Field GPS" to query device coordinates.

### Step 3: Run the Health Check
- Click **Check Leaf Health & Weather Forecast**.
- The vision model processes the leaf in memory, segmenting damaged tissue and fetching local 9 km weather.

### Step 4: Review Diagnostic Results
- **Current Leaf Damage**: Shows total percentage of damaged leaf area with a guaranteed 90% expected range.
- **Spot Breakdown**: Displays the percentage of dead brown spots versus active spreading yellow margins.
- **Inspection View Modes**:
  - *Disease Overlay*: Outlines the exact borders of active disease spots.
  - *7-Day Spread*: Displays forecasted spot expansion frontier over the next 7 days.
  - *Side-by-Side*: Compares the original photograph with the segmented overlay.
  - *AI Focus Areas*: Heatmap showing the exact leaf areas examined by the AI vision model.

### Step 5: Check 7-Day Weather & Spray Windows
- View the 7-day forecast table for your field.
- Review daily spraying feasibility ratings (`FAVORABLE`, `MARGINAL`, `UNFAVORABLE`) to avoid rain wash-off or chemical drift.

### Step 6: Download the Official PDF Health Report
- Click **Download Crop Health Report (PDF)**.
- A publication-grade 2-page A4 PDF report (`Crop_Health_Guide_<crop>_<id>.pdf`) downloads directly to your device.
- Hand or show this report to your local agricultural officer (KVK / RBK) for registered product recommendations.
