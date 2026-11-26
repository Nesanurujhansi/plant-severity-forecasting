# Frontend Client: Crop Leaf Health & Disease Guide

Modern, responsive decision-support web interface built with **Next.js 14**, **React**, **TypeScript**, and **Tailwind CSS**.

---

## Key Capabilities

1. **Clean Farmer-First Interface**:
   - Plain-language terminology (no complex academic or medical jargon).
   - Multi-language localization with instant switching:
     - English (`en`)
     - Telugu (`te`)
     - Hindi (`hi`)
   - Adaptive responsive design for mobile phones, tablets, and desktop displays.

2. **Interactive Visual Diagnostics**:
   - **Disease Overlay**: High-contrast outline isolating dead brown spots and spreading yellow borders over the uploaded leaf.
   - **7-Day Spread Forecast**: Visual projection showing forecasted disease spot expansion frontier.
   - **AI Focus Areas**: Visual heatmap highlighting the exact leaf areas examined by the DINOv2 vision model.
   - **Side-by-Side Comparison**: Original photograph beside segmented disease mask.

3. **Field Location & Weather Grid**:
   - Pinpoints field coordinates via Andhra Pradesh agricultural presets, live Mandal search, Google Maps URL resolver, or device GPS.
   - Live 7-day weather table (temperature, humidity, vapor pressure deficit, rainfall, and leaf wetness hours).

4. **Direct PDF Health Guide Export**:
   - Downloads a formal 2-page A4 advisory report directly to the device without opening printer dialogues.

---

## Project Structure

```text
frontend/
├── app/
│   ├── api/
│   │   └── resolve-maps/    # Server route resolving Google Maps shortlinks to lat/long
│   ├── locales/
│   │   └── translations.ts  # Multi-language dictionary (EN, TE, HI)
│   ├── layout.tsx           # Root HTML layout and font optimization
│   └── page.tsx             # Main interactive application dashboard & modal
├── public/
│   └── samples/             # Bundled leaf inspection benchmark images
├── next.config.mjs          # Proxy rewrites routing /api/* to FastAPI backend (port 8000)
├── tailwind.config.ts       # Agronomic palette (forest, sage, parchment, rust, amber)
└── package.json
```

---

## Local Development & Production

```powershell
# Install dependencies
npm install

# Run local development server
npm run dev

# Build production bundle
npm run build

# Start production server
npm run start -- -p 3000
```

The web application runs at `http://localhost:3000` and proxies API requests to `http://127.0.0.1:8000`.
