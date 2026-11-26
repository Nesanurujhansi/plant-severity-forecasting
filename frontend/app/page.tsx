"use client";

import React, { useState, useRef } from "react";
import {
  Plant,
  UploadSimple,
  ArrowsClockwise,
  ShieldCheck,
  WarningCircle,
  TrendUp,
  Thermometer,
  Drop,
  Sparkle,
  CheckCircle,
  SlidersHorizontal,
  CloudSun,
  FirstAid,
  Leaf,
  X,
  Info,
  Clock,
  MapPin,
  Calendar,
  Faders,
  ChartBar,
  ChartLineUp,
  Table,
  Sliders,
  Article,
  FileText,
  Compass,
  Crosshair,
  MagnifyingGlass,
  LinkSimple,
  Check,
  Printer,
  DownloadSimple,
} from "@phosphor-icons/react";
import { translations, Language } from "./locales/translations";

interface WeatherSummaryRow {
  label: string;
  day_offset?: number;
  date?: string;
  temp_c: number;
  rh_pct: number;
  vpd_kpa: number;
  precip_mm: number;
  leaf_wetness_hours?: number;
  infection_risk?: number;
  max_wind_kmh?: number;
  spray_feasibility?: string;
  spray_advisory?: string;
}

interface AnalysisResult {
  bundle_info: {
    version: string;
    is_stub: boolean;
    banner?: string;
  };
  safety: {
    confidence_tier: string;
    should_abstain: boolean;
    quality_passed: boolean;
    ood_status: string;
    user_message: string;
    failure_reasons: string[];
  };
  severity: {
    severity_pct: number;
    severity_interval: [number, number];
    severity_class: string;
    lesion_count: number;
    mean_lesion_size_px: number;
    necrotic_pct?: number;
    chlorotic_pct?: number;
    senescence_pct?: number;
    leaf_mask_base64?: string;
    lesion_mask_base64?: string;
    overlay_base64?: string;
    composite_image_base64?: string;
    projected_7d_overlay_base64?: string;
    treated_7d_overlay_base64?: string;
    treated_7d_severity_pct?: number;
    spread_reduction_pct?: number;
  } | null;
  forecast?: {
    horizons: {
      [key: string]: {
        horizon_days: number;
        severity_pct: number;
        interval: [number, number];
        drivers: string[];
      };
    };
    chart_base64?: string;
    projected_7d_overlay_base64?: string;
    treated_7d_overlay_base64?: string;
    treated_7d_severity_pct?: number;
    spread_reduction_pct?: number;
  } | null;
  scenario?: {
    unprotected_7d_pct: number;
    protected_7d_pct: number;
    relative_reduction_pct: number;
    interval?: [number, number];
    label?: string;
    disclaimer?: string;
    chart_base64?: string;
  } | null;
  weather?: {
    source: string;
    preset: string;
    latitude?: number;
    longitude?: number;
    current_temp_c?: number;
    current_rh_pct?: number;
    current_vpd_kpa?: number;
    is_default_location?: boolean;
    table_summary: WeatherSummaryRow[];
  };
  xai?: {
    attention_saliency_base64?: string;
    baseline_severity_pct: number;
    forecasted_severity_pct: number;
    net_change_pct: number;
    factors: Array<{
      factor_id: string;
      factor_name: string;
      raw_value: string;
      delta_pct: number;
      impact_direction: string;
      color: string;
      description: string;
    }>;
    narrative: string;
  } | null;
  warnings?: string[];
  timings?: {
    total_ms: number;
    stages_ms?: { [key: string]: number };
  };
  disclaimer?: string;
}

// Dedicated Agricultural Hubs for Andhra Pradesh (with primary crop species)
const AP_AG_REGIONS = [
  { name: "Guntur", note: "Chilli & Cotton Belt", lat: "16.3067", lon: "80.4365", crop: "cotton" },
  { name: "Vijayawada", note: "Krishna Basin Rice", lat: "16.5062", lon: "80.6480", crop: "rice" },
  { name: "Rajahmundry", note: "Godavari Delta Paddy", lat: "17.0005", lon: "81.8040", crop: "rice" },
  { name: "Anantapur", note: "Groundnut & Millets", lat: "14.6819", lon: "77.6006", crop: "wheat" },
  { name: "Tirupati", note: "Chittoor Tomato & Mango", lat: "13.6288", lon: "79.4192", crop: "tomato" },
  { name: "Visakhapatnam", note: "North Coast Horticulture", lat: "17.6868", lon: "83.2185", crop: "sugarcane" },
  { name: "Kurnool", note: "Cotton & Sunflower", lat: "15.8281", lon: "78.0373", crop: "cotton" },
  { name: "Nellore", note: "Delta Rice & Aqua", lat: "14.4426", lon: "79.9865", crop: "rice" },
];

export default function Home() {
  const [lang, setLang] = useState<Language>("en");
  const [isHealthCardOpen, setIsHealthCardOpen] = useState<boolean>(false);
  const [isDownloadingPdf, setIsDownloadingPdf] = useState<boolean>(false);
  const [pdfError, setPdfError] = useState<string | null>(null);
  const [diagnosticId, setDiagnosticId] = useState<string>("RBK-814920");
  const [scanTimestamp, setScanTimestamp] = useState<string>("");
  const t = translations[lang];

  const [activeTab, setActiveTab] = useState<"analyze" | "how_it_works" | "about">("analyze");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [crop, setCrop] = useState<string>("cotton");
  const [cultivar, setCultivar] = useState<string>("");
  const [headingDays, setHeadingDays] = useState<string>("");
  
  // Geolocation & Weather States (Defaulting to Guntur, Andhra Pradesh)
  const [latitude, setLatitude] = useState<string>("16.3067");
  const [longitude, setLongitude] = useState<string>("80.4365");
  const [locationLabel, setLocationLabel] = useState<string>("Guntur, Andhra Pradesh");
  const [locationTab, setLocationTab] = useState<"ap_presets" | "search" | "google_maps">("ap_presets");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [searchResults, setSearchResults] = useState<Array<{
    id: number;
    name: string;
    latitude: number;
    longitude: number;
    admin1?: string;
    country?: string;
  }>>([]);
  const [isSearchingLocation, setIsSearchingLocation] = useState<boolean>(false);
  const [googleMapsInput, setGoogleMapsInput] = useState<string>("");
  const [googleMapsStatus, setGoogleMapsStatus] = useState<string | null>(null);
  const [isResolvingMaps, setIsResolvingMaps] = useState<boolean>(false);
  const [isDetectingGps, setIsDetectingGps] = useState<boolean>(false);
  
  const [weatherOverride, setWeatherOverride] = useState<string>("");
  const [overlayOpacity, setOverlayOpacity] = useState<number>(0.65);
  const [viewMode, setViewMode] = useState<"overlay" | "projected_7d" | "treated_7d" | "side_by_side" | "attention">("overlay");
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [showXaiDetails, setShowXaiDetails] = useState<boolean>(true);

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = (file: File) => {
    setSelectedFile(file);
    setErrorMsg(null);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setResult(null);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const detectCurrentGpsLocation = () => {
    if (typeof window === "undefined" || !navigator.geolocation) {
      setErrorMsg("Geolocation is not supported by your browser.");
      return;
    }
    setIsDetectingGps(true);
    setErrorMsg(null);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const lat = pos.coords.latitude.toFixed(4);
        const lon = pos.coords.longitude.toFixed(4);
        setLatitude(lat);
        setLongitude(lon);
        setLocationLabel(`Device Hardware GPS (${lat}°, ${lon}°)`);
        setIsDetectingGps(false);
      },
      (err) => {
        setIsDetectingGps(false);
        setErrorMsg(`Geolocation note: ${err.message}. (Note: Desktops and Wi-Fi networks route through ISP Hyderabad gateways; use town search or Google Maps pin for exact farm location).`);
      },
      { enableHighAccuracy: true, timeout: 8000 }
    );
  };

  const handleLocationSearch = async (val: string) => {
    setSearchQuery(val);
    if (val.trim().length < 2) {
      setSearchResults([]);
      return;
    }
    setIsSearchingLocation(true);
    try {
      const res = await fetch(
        `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(val)}&count=8&language=en&format=json`
      );
      if (res.ok) {
        const data = await res.json();
        const raw = (data.results || []) as Array<{
          id: number;
          name: string;
          latitude: number;
          longitude: number;
          admin1?: string;
          country?: string;
        }>;
        // Prioritize Andhra Pradesh & India results
        const sorted = [...raw].sort((a, b) => {
          const aAP = (a.admin1 || "").toLowerCase().includes("andhra") ? 2 : (a.country === "India" ? 1 : 0);
          const bAP = (b.admin1 || "").toLowerCase().includes("andhra") ? 2 : (b.country === "India" ? 1 : 0);
          return bAP - aAP;
        });
        setSearchResults(sorted);
      }
    } catch {
      // Offline fallback
    } finally {
      setIsSearchingLocation(false);
    }
  };

  const selectSearchResult = (item: {
    name: string;
    latitude: number;
    longitude: number;
    admin1?: string;
    country?: string;
  }) => {
    const lat = item.latitude.toFixed(4);
    const lon = item.longitude.toFixed(4);
    setLatitude(lat);
    setLongitude(lon);
    const label = `${item.name}${item.admin1 ? `, ${item.admin1}` : ""}${item.country ? ` (${item.country})` : ""}`;
    setLocationLabel(label);
    setSearchQuery(label);
    setSearchResults([]);
  };

  const parseGoogleMapsPin = async (val: string) => {
    setGoogleMapsInput(val);
    setGoogleMapsStatus(null);
    const trimmed = val.trim();
    if (!trimmed) return;

    // 1. Direct comma or space separated decimal coordinates: "16.3067, 80.4365" or "16.3067 80.4365"
    const decMatch = trimmed.match(/^(-?\d+\.\d+)[,\s]+(-?\d+\.\d+)$/);
    if (decMatch) {
      const lat = parseFloat(decMatch[1]).toFixed(4);
      const lon = parseFloat(decMatch[2]).toFixed(4);
      setLatitude(lat);
      setLongitude(lon);
      setLocationLabel(`Field Pin (${lat}°, ${lon}°)`);
      setGoogleMapsStatus(`Coordinates applied: ${lat}° N, ${lon}° E`);
      return;
    }

    // 2. Direct DMS format: 16°18'24"N 80°26'11"E or 17°23'36.2"N 78°28'14.2"E
    const dmsRegex = /(\d+)°\s*(\d+)'\s*([\d.]+)"?\s*([NSEW])[\s,]+(\d+)°\s*(\d+)'\s*([\d.]+)"?\s*([NSEW])/i;
    const dmsMatch = trimmed.match(dmsRegex);
    if (dmsMatch) {
      let lat = parseInt(dmsMatch[1], 10) + parseInt(dmsMatch[2], 10) / 60 + parseFloat(dmsMatch[3]) / 3600;
      if (dmsMatch[4].toUpperCase() === "S") lat = -lat;
      let lon = parseInt(dmsMatch[5], 10) + parseInt(dmsMatch[6], 10) / 60 + parseFloat(dmsMatch[7]) / 3600;
      if (dmsMatch[8].toUpperCase() === "W") lon = -lon;

      const latStr = lat.toFixed(4);
      const lonStr = lon.toFixed(4);
      setLatitude(latStr);
      setLongitude(lonStr);
      setLocationLabel(`Field Coordinates (${latStr}°, ${lonStr}°)`);
      setGoogleMapsStatus(`DMS converted & applied: ${latStr}° N, ${lonStr}° E`);
      return;
    }

    // 3. Inline coordinates in standard full Google Maps URLs: /@16.3067,80.4365... or q=16.3067,80.4365...
    const urlDirectMatch = trimmed.match(/(?:@|q=|place\/|ll=|search\/)(-?\d+\.\d+)[,\+]+\s*(-?\d+\.\d+)/i);
    if (urlDirectMatch) {
      const lat = parseFloat(urlDirectMatch[1]).toFixed(4);
      const lon = parseFloat(urlDirectMatch[2]).toFixed(4);
      setLatitude(lat);
      setLongitude(lon);
      setLocationLabel(`Google Maps Pin (${lat}°, ${lon}°)`);
      setGoogleMapsStatus(`Coordinates extracted: ${lat}° N, ${lon}° E`);
      return;
    }

    // 4. Shortened Google Maps links (e.g. https://maps.app.goo.gl/... or goo.gl redirects)
    if (
      trimmed.startsWith("http://") ||
      trimmed.startsWith("https://") ||
      trimmed.includes("goo.gl") ||
      trimmed.includes("google.com/maps")
    ) {
      setIsResolvingMaps(true);
      setGoogleMapsStatus("Following shortened Google Maps link to extract exact coordinates...");

      try {
        let res = await fetch(`/api/resolve-maps?url=${encodeURIComponent(trimmed)}`);
        if (!res.ok) {
          // Fallback to FastAPI backend port 8000
          res = await fetch(`http://127.0.0.1:8000/api/resolve-maps-url?url=${encodeURIComponent(trimmed)}`);
        }

        if (res.ok) {
          const data = await res.json();
          if (data.status === "ok" && data.latitude !== undefined && data.longitude !== undefined) {
            const latStr = Number(data.latitude).toFixed(4);
            const lonStr = Number(data.longitude).toFixed(4);
            setLatitude(latStr);
            setLongitude(lonStr);
            setLocationLabel(`Google Maps Pin (${latStr}°, ${lonStr}°)`);
            setGoogleMapsStatus(`Google Maps Link Resolved: ${latStr}° N, ${lonStr}° E`);
            return;
          }
        }
        setGoogleMapsStatus("Could not extract coordinates from link. Try copying decimal coordinates directly from Google Maps.");
      } catch (err: unknown) {
        setGoogleMapsStatus(`Failed to resolve link: ${err instanceof Error ? err.message : "Network error"}`);
      } finally {
        setIsResolvingMaps(false);
      }
      return;
    }

    setGoogleMapsStatus("Could not parse coordinates. Please paste a Google Maps link or enter numbers like '16.3067, 80.4365'.");
  };

  const loadSample = async (
    samplePath: string,
    sampleCrop: string,
    sampleLat = "16.3067",
    sampleLon = "80.4365",
    sampleLoc = "Guntur, Andhra Pradesh"
  ) => {
    try {
      setLoading(true);
      setErrorMsg(null);
      setCrop(sampleCrop);
      setLatitude(sampleLat);
      setLongitude(sampleLon);
      setLocationLabel(sampleLoc);

      const res = await fetch(samplePath);
      const blob = await res.blob();
      const file = new File([blob], samplePath.split("/").pop() || "sample.png", {
        type: blob.type || "image/png",
      });
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
    } catch {
      setErrorMsg("Failed to load sample image.");
    } finally {
      setLoading(false);
    }
  };

  const runAnalysis = async () => {
    if (!selectedFile) {
      setErrorMsg("Please select or drop a leaf photograph first.");
      return;
    }

    setLoading(true);
    setErrorMsg(null);

    try {
      const formData = new FormData();
      formData.append("image", selectedFile);
      formData.append("crop", crop);
      if (cultivar.trim()) formData.append("cultivar", cultivar.trim());
      if (headingDays.trim()) formData.append("days_since_heading", headingDays.trim());
      if (latitude.trim()) formData.append("latitude", latitude.trim());
      if (longitude.trim()) formData.append("longitude", longitude.trim());
      formData.append("weather_scenario", "live_nwp");
      if (weatherOverride.trim()) formData.append("weather_override", weatherOverride.trim());
      formData.append("include_treatment_scenario", "false");

      let response: Response;
      try {
        response = await fetch("/api/analyze", {
          method: "POST",
          body: formData,
        });
        if (!response.ok && response.status >= 500) {
          response = await fetch("http://127.0.0.1:8000/analyze", {
            method: "POST",
            body: formData,
          });
        }
      } catch {
        response = await fetch("http://127.0.0.1:8000/analyze", {
          method: "POST",
          body: formData,
        });
      }

      if (!response.ok) {
        const errData = await response.json().catch(() => null);
        throw new Error(errData?.detail || `Inference error (Status: ${response.status})`);
      }

      const data: AnalysisResult = await response.json();
      setResult(data);
      const randId = Math.floor(100000 + Math.random() * 900000);
      setDiagnosticId(`RBK-${randId}`);
      const now = new Date();
      setScanTimestamp(
        `${now.toLocaleDateString()} ${now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
      );
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to complete leaf analysis.";
      setErrorMsg(msg);
    } finally {
      setLoading(false);
    }
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setErrorMsg(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const getOptimizedBase64 = async (file: File | null, url: string | null): Promise<string> => {
    return new Promise<string>((resolve) => {
      try {
        let src = "";
        let isBlobCreated = false;
        if (file) {
          src = URL.createObjectURL(file);
          isBlobCreated = true;
        } else if (url) {
          src = url;
        } else {
          return resolve("");
        }

        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
          try {
            const maxDim = 1000;
            let w = img.naturalWidth || img.width;
            let h = img.naturalHeight || img.height;
            if (w > maxDim || h > maxDim) {
              if (w > h) {
                h = Math.round((h * maxDim) / w);
                w = maxDim;
              } else {
                w = Math.round((w * maxDim) / h);
                h = maxDim;
              }
            }
            const canvas = document.createElement("canvas");
            canvas.width = w;
            canvas.height = h;
            const ctx = canvas.getContext("2d");
            if (ctx) {
              ctx.drawImage(img, 0, 0, w, h);
              const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
              if (isBlobCreated) URL.revokeObjectURL(src);
              resolve(dataUrl);
            } else {
              if (isBlobCreated) URL.revokeObjectURL(src);
              resolve("");
            }
          } catch {
            if (isBlobCreated) URL.revokeObjectURL(src);
            resolve("");
          }
        };
        img.onerror = () => {
          if (isBlobCreated) URL.revokeObjectURL(src);
          resolve("");
        };
        img.src = src;
      } catch {
        resolve("");
      }
    });
  };

  const downloadDirectPdf = async () => {
    if (!result || !result.severity) return;
    setIsDownloadingPdf(true);
    setPdfError(null);
    try {
      const origB64 = await getOptimizedBase64(selectedFile, previewUrl);

      const payload = {
        diagnostic_id: diagnosticId,
        scan_date: scanTimestamp || new Date().toLocaleString(),
        crop: crop,
        cultivar: cultivar,
        location_label: locationLabel,
        latitude: parseFloat(latitude) || 16.3067,
        longitude: parseFloat(longitude) || 80.4365,
        heading_days: headingDays,
        severity_pct: result.severity.severity_pct,
        interval_low: result.severity.severity_interval[0],
        interval_high: result.severity.severity_interval[1],
        necrotic_pct: result.severity.necrotic_pct ?? result.severity.severity_pct,
        chlorotic_pct: result.severity.chlorotic_pct ?? 0.0,
        lesion_count: result.severity.lesion_count,
        original_image_base64: origB64,
        composite_image_base64: result.severity.composite_image_base64 || "",
        projected_7d_overlay_base64: result.forecast?.projected_7d_overlay_base64 || result.severity.projected_7d_overlay_base64 || "",
        attention_saliency_base64: result.xai?.attention_saliency_base64 || "",
        weather_table: result.weather?.table_summary || []
      };

      let resp: Response;
      try {
        resp = await fetch("/api/export-health-report-pdf", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        if (!resp.ok) {
          throw new Error(`API returned ${resp.status}`);
        }
      } catch {
        // Fallback directly to FastAPI port
        resp = await fetch("http://127.0.0.1:8000/api/export-health-report-pdf", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
      }

      if (!resp.ok) {
        const errDetail = await resp.text().catch(() => resp.statusText);
        throw new Error(`Failed to generate PDF (${resp.status}): ${errDetail}`);
      }

      const blob = await resp.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      const cleanCrop = crop.toLowerCase().replace(/[^a-z0-9]/g, "_");
      link.download = `Crop_Health_Guide_${cleanCrop}_${diagnosticId}.pdf`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      console.error("PDF download error:", err);
      const msg = err?.message || "Could not download PDF report directly. Please try again or use Paper Print.";
      setPdfError(msg);
      setErrorMsg(msg);
    } finally {
      setIsDownloadingPdf(false);
    }
  };

  const isHealthy = result?.severity ? result.severity.severity_pct === 0 : false;

  return (
    <div className="min-h-screen bg-parchment-light text-ink antialiased">
      {/* Top Navigation Bar following Apple design principles */}
      <header className="sticky top-0 z-50 apple-glass border-b border-parchment-dark/50 px-6 py-3.5">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-forest/10 flex items-center justify-center text-forest">
              <Plant size={24} weight="duotone" />
            </div>
            <div>
              <h1 className="text-sm font-semibold tracking-tight text-ink">
                {t.app_title}
              </h1>
              <p className="text-[11px] text-ink-muted">
                {t.app_subtitle}
              </p>
            </div>
          </div>

          {/* Center: Navigation Tabs and Language Switcher */}
          <div className="flex items-center gap-3">
            <nav className="flex items-center p-1 bg-parchment rounded-xl border border-parchment-dark/60 text-xs font-medium">
              <button
                onClick={() => setActiveTab("analyze")}
                className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
                  activeTab === "analyze"
                    ? "bg-forest text-parchment-light shadow-sm"
                    : "text-ink hover:text-forest"
                }`}
              >
                <ChartLineUp size={15} />
                <span>{t.tabs.analyze}</span>
              </button>
              <button
                onClick={() => setActiveTab("how_it_works")}
                className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
                  activeTab === "how_it_works"
                    ? "bg-forest text-parchment-light shadow-sm"
                    : "text-ink hover:text-forest"
                }`}
              >
                <Article size={15} />
                <span>{t.tabs.how_it_works}</span>
              </button>
              <button
                onClick={() => setActiveTab("about")}
                className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
                  activeTab === "about"
                    ? "bg-forest text-parchment-light shadow-sm"
                    : "text-ink hover:text-forest"
                }`}
              >
                <FileText size={15} />
                <span>{t.tabs.about}</span>
              </button>
            </nav>

            {/* Regional Language Toggle */}
            <div className="flex items-center p-0.5 bg-parchment-dark/40 rounded-xl border border-parchment-dark/70 text-xs font-medium">
              <button
                type="button"
                onClick={() => setLang("en")}
                className={`px-2.5 py-1 rounded-lg transition-all ${
                  lang === "en" ? "bg-forest text-parchment-light shadow-sm font-semibold" : "text-ink hover:text-forest"
                }`}
              >
                English
              </button>
              <button
                type="button"
                onClick={() => setLang("te")}
                className={`px-2.5 py-1 rounded-lg transition-all ${
                  lang === "te" ? "bg-forest text-parchment-light shadow-sm font-semibold" : "text-ink hover:text-forest"
                }`}
              >
                తెలుగు
              </button>
              <button
                type="button"
                onClick={() => setLang("hi")}
                className={`px-2.5 py-1 rounded-lg transition-all ${
                  lang === "hi" ? "bg-forest text-parchment-light shadow-sm font-semibold" : "text-ink hover:text-forest"
                }`}
              >
                हिन्दी
              </button>
            </div>
          </div>

          {/* Engine Status Badge */}
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-sage-faint border border-sage/40 text-xs font-medium text-forest">
            <CheckCircle size={14} weight="fill" className="text-leaf" />
            <span>Neural Engine Ready</span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* TAB 1: ANALYZE & RESULTS */}
        {activeTab === "analyze" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            {/* Left Column: Upload & Controls (5 Cols) */}
            <section className="lg:col-span-5 space-y-6">
              {/* Photo Upload Card */}
              <div className="apple-glass-card rounded-3xl p-6 shadow-glass">
                <div className="flex items-center justify-between mb-3.5">
                  <span className="text-xs uppercase tracking-wider font-semibold text-ink-muted">
                    Leaf Photograph Input
                  </span>
                  {previewUrl && (
                    <button
                      onClick={clearSelection}
                      className="text-xs text-rust hover:text-rust-hover flex items-center gap-1 font-medium transition-colors"
                    >
                      <X size={14} />
                      <span>Remove</span>
                    </button>
                  )}
                </div>

                {!previewUrl ? (
                  <div
                    onDragOver={(e) => {
                      e.preventDefault();
                      setDragActive(true);
                    }}
                    onDragLeave={() => setDragActive(false)}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                    className={`border-2 border-dashed rounded-2xl p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 ${
                      dragActive
                        ? "border-forest bg-forest/5 scale-[0.99]"
                        : "border-parchment-dark/70 hover:border-leaf bg-parchment/30 hover:bg-parchment/60"
                    }`}
                  >
                    <div className="w-12 h-12 rounded-2xl bg-forest/10 text-forest flex items-center justify-center mb-3">
                      <UploadSimple size={24} weight="regular" />
                    </div>
                    <p className="text-sm font-medium text-ink mb-1">
                      Drag leaf photo here, or click to browse
                    </p>
                    <p className="text-xs text-ink-muted">
                      JPG, PNG, WebP up to 10 MB
                    </p>
                  </div>
                ) : (
                  <div className="relative rounded-2xl overflow-hidden bg-parchment-card/80 border border-parchment-dark/60 aspect-square flex items-center justify-center">
                    <img
                      src={previewUrl}
                      alt="Uploaded leaf"
                      className="w-full h-full object-contain"
                    />
                  </div>
                )}

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      handleFile(e.target.files[0]);
                    }
                  }}
                />

                {/* 1-Touch Benchmark Samples */}
                <div className="mt-4 pt-4 border-t border-parchment-dark/40">
                  <p className="text-xs text-ink-muted font-medium mb-2.5">
                    Preloaded benchmark test leaves:
                  </p>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      type="button"
                      onClick={() => loadSample("/samples/real_leaf_septoria.png", "wheat", "16.3067", "80.4365", "Guntur, Andhra Pradesh (Real Field)")}
                      className="col-span-2 text-left text-xs px-3 py-2 rounded-xl bg-forest/10 hover:bg-forest/15 border border-forest/30 font-semibold text-forest transition-colors flex items-center justify-between"
                    >
                      <div className="flex items-center gap-1.5">
                        <Sparkle size={14} className="text-forest" weight="fill" />
                        <span>Real Field Leaf (set_12_31_obj0)</span>
                      </div>
                      <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-forest/20 text-forest font-bold">Real Field</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => loadSample("/samples/healthy.png", "wheat", "16.5062", "80.6480", "Vijayawada, Andhra Pradesh (Healthy)")}
                      className="text-left text-xs px-3 py-2 rounded-xl bg-parchment hover:bg-parchment-dark/50 border border-parchment-dark/50 font-medium text-ink transition-colors flex items-center justify-between"
                    >
                      <span>Healthy Foliage</span>
                      <Leaf size={14} className="text-leaf" />
                    </button>
                    <button
                      type="button"
                      onClick={() => loadSample("/samples/mild.png", "wheat", "17.0005", "81.8040", "Rajahmundry, Andhra Pradesh (Mild)")}
                      className="text-left text-xs px-3 py-2 rounded-xl bg-parchment hover:bg-parchment-dark/50 border border-parchment-dark/50 font-medium text-ink transition-colors flex items-center justify-between"
                    >
                      <span>Mild Septoria</span>
                      <Sparkle size={14} className="text-sage" />
                    </button>
                    <button
                      type="button"
                      onClick={() => loadSample("/samples/moderate.png", "wheat", "14.6819", "77.6006", "Anantapur, Andhra Pradesh (Moderate)")}
                      className="text-left text-xs px-3 py-2 rounded-xl bg-parchment hover:bg-parchment-dark/50 border border-parchment-dark/50 font-medium text-ink transition-colors flex items-center justify-between"
                    >
                      <span>Moderate Infection</span>
                      <TrendUp size={14} className="text-rust" />
                    </button>
                    <button
                      type="button"
                      onClick={() => loadSample("/samples/severe.png", "wheat", "16.3067", "80.4365", "Guntur, Andhra Pradesh (Severe)")}
                      className="text-left text-xs px-3 py-2 rounded-xl bg-parchment hover:bg-parchment-dark/50 border border-parchment-dark/50 font-medium text-ink transition-colors flex items-center justify-between"
                    >
                      <span>Severe Rust</span>
                      <WarningCircle size={14} className="text-rust" />
                    </button>
                  </div>
                </div>
              </div>

              {/* Crop & Weather Controls */}
              <div className="apple-glass-card rounded-3xl p-6 shadow-glass space-y-5">
                {/* Crop Species Selector */}
                <div>
                  <label className="block text-xs uppercase tracking-wider font-semibold text-ink-muted mb-2">
                    Crop Species
                  </label>
                  <select
                    value={crop}
                    onChange={(e) => setCrop(e.target.value)}
                    className="w-full text-xs font-medium px-3.5 py-2.5 rounded-xl bg-parchment border border-parchment-dark/60 text-ink focus:outline-none focus:border-forest transition-colors"
                  >
                    <option value="wheat">Wheat (Triticum aestivum)</option>
                    <option value="coffee">Coffee (Coffea arabica)</option>
                    <option value="rice">Rice (Oryza sativa)</option>
                    <option value="corn">Corn / Maize (Zea mays)</option>
                    <option value="tomato">Tomato (Solanum lycopersicum)</option>
                    <option value="cotton">Cotton (Gossypium)</option>
                    <option value="sugarcane">Sugarcane (Saccharum)</option>
                  </select>
                </div>

                {/* Field Geolocation & Numerical Weather Prediction */}
                <div className="space-y-3.5 pt-1">
                  <div className="flex items-center justify-between">
                    <label className="text-xs uppercase tracking-wider font-semibold text-ink-muted flex items-center gap-1.5">
                      <MapPin size={15} className="text-forest" weight="fill" />
                      <span>Field Location &amp; Live Weather</span>
                    </label>
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-leaf/20 text-leaf font-bold">
                      9 km NWP
                    </span>
                  </div>

                  {/* 3-Mode Location Mode Switcher */}
                  <div className="grid grid-cols-3 gap-1 p-1 bg-parchment rounded-xl border border-parchment-dark/60 text-xs">
                    <button
                      type="button"
                      onClick={() => setLocationTab("ap_presets")}
                      className={`py-1.5 px-2 rounded-lg font-medium transition-all text-center text-[11px] truncate ${
                        locationTab === "ap_presets"
                          ? "bg-forest text-parchment-light shadow-sm font-semibold"
                          : "text-ink hover:text-forest"
                      }`}
                    >
                      AP Districts
                    </button>
                    <button
                      type="button"
                      onClick={() => setLocationTab("search")}
                      className={`py-1.5 px-2 rounded-lg font-medium transition-all text-center text-[11px] truncate ${
                        locationTab === "search"
                          ? "bg-forest text-parchment-light shadow-sm font-semibold"
                          : "text-ink hover:text-forest"
                      }`}
                    >
                      Town Search
                    </button>
                    <button
                      type="button"
                      onClick={() => setLocationTab("google_maps")}
                      className={`py-1.5 px-2 rounded-lg font-medium transition-all text-center text-[11px] truncate ${
                        locationTab === "google_maps"
                          ? "bg-forest text-parchment-light shadow-sm font-semibold"
                          : "text-ink hover:text-forest"
                      }`}
                    >
                      Maps Pin
                    </button>
                  </div>

                  {/* TAB 1: Andhra Pradesh District Agricultural Presets */}
                  {locationTab === "ap_presets" && (
                    <div className="space-y-2">
                      <div className="text-[11px] text-ink-muted font-medium flex items-center justify-between">
                        <span>Andhra Pradesh Farming Hubs:</span>
                        <span className="text-[10px] font-mono text-forest font-semibold truncate max-w-[170px]">{locationLabel}</span>
                      </div>
                      <div className="grid grid-cols-2 gap-1.5 max-h-[190px] overflow-y-auto pr-0.5">
                        {AP_AG_REGIONS.map((reg) => {
                          const isSelected = latitude === reg.lat && longitude === reg.lon;
                          return (
                            <button
                              key={reg.name}
                              type="button"
                              onClick={() => {
                                setLatitude(reg.lat);
                                setLongitude(reg.lon);
                                setLocationLabel(`${reg.name}, Andhra Pradesh`);
                                setCrop(reg.crop);
                              }}
                              className={`text-left p-2 rounded-xl border transition-all text-xs ${
                                isSelected
                                  ? "border-forest bg-forest/15 text-forest font-semibold shadow-xs"
                                  : "border-parchment-dark/50 bg-parchment/60 text-ink hover:border-parchment-dark"
                              }`}
                            >
                              <div className="flex items-center justify-between">
                                <span className="font-semibold text-ink text-xs">{reg.name}</span>
                                {isSelected && <Check size={12} weight="bold" className="text-forest" />}
                              </div>
                              <div className="text-[10px] text-ink-muted mt-0.5">{reg.note}</div>
                              <div className="text-[9px] text-forest font-mono mt-0.5">{reg.lat}°, {reg.lon}°</div>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* TAB 2: Live Town / Mandal Geocoding Search Bar */}
                  {locationTab === "search" && (
                    <div className="space-y-2 relative">
                      <div className="text-[11px] text-ink-muted font-medium">
                        Search any Town, Mandal or Village in Andhra Pradesh:
                      </div>
                      <div className="relative">
                        <input
                          type="text"
                          value={searchQuery}
                          onChange={(e) => handleLocationSearch(e.target.value)}
                          placeholder="e.g. Guntur, Tenali, Eluru, Kadapa, Nellore..."
                          className="w-full text-xs pl-8 pr-8 py-2 rounded-xl bg-parchment-light border border-parchment-dark/60 text-ink focus:outline-none focus:border-forest"
                        />
                        <MagnifyingGlass size={14} className="absolute left-2.5 top-2.5 text-ink-muted" />
                        {isSearchingLocation && (
                          <ArrowsClockwise size={14} className="absolute right-2.5 top-2.5 animate-spin text-forest" />
                        )}
                      </div>

                      {/* Dropdown Results List */}
                      {searchResults.length > 0 && (
                        <div className="p-1.5 rounded-xl bg-parchment-light border border-forest/30 shadow-lg space-y-1 max-h-[180px] overflow-y-auto">
                          {searchResults.map((item) => (
                            <button
                              key={item.id}
                              type="button"
                              onClick={() => selectSearchResult(item)}
                              className="w-full text-left p-2 rounded-lg hover:bg-forest/10 transition-colors flex items-center justify-between text-xs"
                            >
                              <div>
                                <div className="font-semibold text-ink">{item.name}</div>
                                <div className="text-[10px] text-ink-muted">
                                  {item.admin1 ? `${item.admin1}, ` : ""}{item.country || "India"}
                                </div>
                              </div>
                              <div className="text-right font-mono text-[10px] text-forest">
                                {item.latitude.toFixed(2)}°, {item.longitude.toFixed(2)}°
                              </div>
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* TAB 3: Google Maps Pin / Coordinates Input */}
                  {locationTab === "google_maps" && (
                    <div className="space-y-2">
                      <div className="text-[11px] text-ink-muted font-medium">
                        Paste Google Maps Link, Pin Coordinates, or DMS:
                      </div>
                      <div className="flex gap-1.5">
                        <input
                          type="text"
                          value={googleMapsInput}
                          onChange={(e) => setGoogleMapsInput(e.target.value)}
                          placeholder="e.g. 16.3067, 80.4365 or maps.app.goo.gl link"
                          className="flex-1 text-xs px-3 py-2 rounded-xl bg-parchment-light border border-parchment-dark/60 text-ink font-mono text-[11px] focus:outline-none focus:border-forest"
                        />
                        <button
                          type="button"
                          onClick={() => parseGoogleMapsPin(googleMapsInput)}
                          disabled={isResolvingMaps}
                          className="px-3.5 py-2 rounded-xl bg-forest text-parchment-light text-xs font-semibold hover:bg-forest-hover transition-colors whitespace-nowrap flex items-center gap-1.5"
                        >
                          {isResolvingMaps ? (
                            <>
                              <ArrowsClockwise size={13} className="animate-spin text-parchment-light" />
                              <span>Resolving Link...</span>
                            </>
                          ) : (
                            <span>Apply Pin</span>
                          )}
                        </button>
                      </div>

                      {googleMapsStatus && (
                        <div className="text-[10px] font-medium text-forest bg-forest/10 p-2 rounded-lg border border-forest/20">
                          {googleMapsStatus}
                        </div>
                      )}

                      <p className="text-[10px] text-ink-muted italic leading-relaxed">
                        Tip: In Google Maps on your phone or computer, tap on your agricultural plot and copy the coordinates (e.g. 16.3067, 80.4365).
                      </p>
                    </div>
                  )}

                  {/* Active Coordinate Badges & Manual Inputs */}
                  <div className="p-3 rounded-2xl bg-parchment/70 border border-parchment-dark/50 space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-medium text-ink-muted text-[11px]">Active Field Target:</span>
                      <strong className="text-forest text-[11px] truncate max-w-[190px]">{locationLabel}</strong>
                    </div>

                    <div className="grid grid-cols-2 gap-2 pt-0.5">
                      <div>
                        <label className="block text-[10px] font-semibold text-ink-muted mb-0.5">
                          Latitude (°N)
                        </label>
                        <input
                          type="number"
                          step="0.0001"
                          value={latitude}
                          onChange={(e) => {
                            setLatitude(e.target.value);
                            setLocationLabel(`Custom (${e.target.value}°, ${longitude}°)`);
                          }}
                          placeholder="16.3067"
                          className="w-full text-xs font-mono px-2.5 py-1.5 rounded-lg bg-parchment-light border border-parchment-dark/60 text-ink focus:outline-none focus:border-forest"
                        />
                      </div>
                      <div>
                        <label className="block text-[10px] font-semibold text-ink-muted mb-0.5">
                          Longitude (°E)
                        </label>
                        <input
                          type="number"
                          step="0.0001"
                          value={longitude}
                          onChange={(e) => {
                            setLongitude(e.target.value);
                            setLocationLabel(`Custom (${latitude}°, ${e.target.value}°)`);
                          }}
                          placeholder="80.4365"
                          className="w-full text-xs font-mono px-2.5 py-1.5 rounded-lg bg-parchment-light border border-parchment-dark/60 text-ink focus:outline-none focus:border-forest"
                        />
                      </div>
                    </div>

                    {/* Hardware GPS Fallback Button with Explanatory Hint */}
                    <div className="pt-1 border-t border-parchment-dark/30 flex items-center justify-between">
                      <button
                        type="button"
                        onClick={detectCurrentGpsLocation}
                        disabled={isDetectingGps}
                        className="text-[11px] font-semibold text-forest hover:text-forest-hover flex items-center gap-1.5 transition-colors"
                      >
                        <Crosshair size={13} weight="bold" />
                        <span>{isDetectingGps ? "Acquiring GPS..." : "Query Device GPS"}</span>
                      </button>
                      <span className="text-[9px] text-ink-muted italic">Broadband IPs default to Hyderabad</span>
                    </div>
                  </div>

                  {/* Live Weather Source Notice */}
                  <div className="p-2.5 rounded-xl bg-parchment/60 border border-parchment-dark/40 text-[11px] text-ink-muted flex items-start gap-2">
                    <CloudSun size={16} className="text-forest shrink-0 mt-0.5" />
                    <div className="leading-tight">
                      Live local weather via <strong className="text-ink">Open-Meteo High-Accuracy Weather (9 km)</strong>. Real-time temperature, humidity, rainfall, and leaf wetness are checked for your exact field.
                    </div>
                  </div>
                </div>

                {/* Collapsible Advanced Parameters */}
                <div>
                  <button
                    type="button"
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    className="text-xs font-medium text-forest hover:text-forest-hover flex items-center gap-1 transition-colors"
                  >
                    <Faders size={14} />
                    <span>{showAdvanced ? "Hide Advanced Options" : "Advanced Options (Crop Variety, Age)"}</span>
                  </button>

                  {showAdvanced && (
                    <div className="mt-3 p-4 rounded-2xl bg-parchment/50 border border-parchment-dark/40 space-y-3">
                      <div>
                        <label className="block text-[11px] font-medium text-ink-muted mb-1">
                          Crop Variety (Optional)
                        </label>
                        <input
                          type="text"
                          value={cultivar}
                          onChange={(e) => setCultivar(e.target.value)}
                          placeholder="e.g. BPT 5204"
                          className="w-full text-xs px-3 py-1.5 rounded-lg bg-parchment-light border border-parchment-dark/60 text-ink"
                        />
                      </div>

                      <div>
                        <label className="block text-[11px] font-medium text-ink-muted mb-1">
                          Crop Age in Days (Optional)
                        </label>
                        <input
                          type="number"
                          value={headingDays}
                          onChange={(e) => setHeadingDays(e.target.value)}
                          placeholder="e.g. 45"
                          className="w-full text-xs px-3 py-1.5 rounded-lg bg-parchment-light border border-parchment-dark/60 text-ink"
                        />
                      </div>

                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <label className="block text-[11px] font-medium text-ink-muted mb-1">
                            Latitude
                          </label>
                          <input
                            type="number"
                            step="0.01"
                            value={latitude}
                            onChange={(e) => setLatitude(e.target.value)}
                            placeholder="47.37"
                            className="w-full text-xs px-3 py-1.5 rounded-lg bg-parchment-light border border-parchment-dark/60 text-ink"
                          />
                        </div>
                        <div>
                          <label className="block text-[11px] font-medium text-ink-muted mb-1">
                            Longitude
                          </label>
                          <input
                            type="number"
                            step="0.01"
                            value={longitude}
                            onChange={(e) => setLongitude(e.target.value)}
                            placeholder="8.54"
                            className="w-full text-xs px-3 py-1.5 rounded-lg bg-parchment-light border border-parchment-dark/60 text-ink"
                          />
                        </div>
                      </div>

                      <div>
                        <label className="block text-[11px] font-medium text-ink-muted mb-1">
                          Weather JSON Override
                        </label>
                        <input
                          type="text"
                          value={weatherOverride}
                          onChange={(e) => setWeatherOverride(e.target.value)}
                          placeholder='{"temp_c": 19.5, "rh_pct": 80.0}'
                          className="w-full text-xs px-3 py-1.5 rounded-lg bg-parchment-light border border-parchment-dark/60 text-ink font-mono text-[11px]"
                        />
                      </div>
                    </div>
                  )}
                </div>

                {/* Primary Action Button */}
                <button
                  type="button"
                  onClick={runAnalysis}
                  disabled={loading || !selectedFile}
                  className={`w-full py-3.5 px-4 rounded-2xl font-medium text-sm flex items-center justify-center gap-2 transition-all duration-200 shadow-card ${
                    loading || !selectedFile
                      ? "bg-parchment-dark/60 text-ink-muted cursor-not-allowed"
                      : "bg-forest hover:bg-forest-hover active:scale-[0.99] text-parchment-light"
                  }`}
                >
                  {loading ? (
                    <>
                      <ArrowsClockwise size={18} className="animate-spin text-parchment-light" />
                      <span>Checking Leaf Health...</span>
                    </>
                  ) : (
                    <>
                      <Sparkle size={18} weight="fill" className="text-sage" />
                      <span>Check Leaf Disease &amp; Spread</span>
                    </>
                  )}
                </button>

                {errorMsg && (
                  <div className="p-3.5 rounded-xl bg-rust-faint border border-rust/30 text-rust text-xs flex items-start gap-2">
                    <WarningCircle size={16} className="shrink-0 mt-0.5" />
                    <span>{errorMsg}</span>
                  </div>
                )}
              </div>
            </section>

            {/* Right Column: High-Signal Visualizations & Prognosis (7 Cols) */}
            <section className="lg:col-span-7 space-y-6">
              {!result ? (
                <div className="apple-glass-card rounded-3xl p-12 text-center shadow-glass flex flex-col items-center justify-center min-h-[480px]">
                  <div className="w-16 h-16 rounded-3xl bg-parchment text-forest/70 flex items-center justify-center mb-4 border border-parchment-dark/50">
                    <SlidersHorizontal size={28} weight="duotone" />
                  </div>
                  <h3 className="text-base font-semibold text-ink mb-1.5">No Leaf Scanned Yet</h3>
                  <p className="text-xs text-ink-muted max-w-sm">
                    Upload a photo of a crop leaf or select an example leaf to check disease spots and see forecasted spread over the next 7 days.
                  </p>
                </div>
              ) : (
                <div className="space-y-6">
                  {/* Safety / Reliability Banner */}
                  <div className={`apple-glass rounded-2xl p-4 border flex items-center justify-between ${
                    result.safety.quality_passed
                      ? "border-leaf/40 bg-leaf/5"
                      : "border-rust/40 bg-rust/5"
                  }`}>
                    <div className="flex items-center gap-3">
                      {result.safety.quality_passed ? (
                        <ShieldCheck size={26} weight="fill" className="text-leaf" />
                      ) : (
                        <WarningCircle size={26} weight="fill" className="text-rust" />
                      )}
                      <div>
                        <div className="text-xs font-semibold text-ink">
                          {result.safety.quality_passed
                            ? "High Confidence — Photo Quality Validated"
                            : "Input Warning Detected"}
                        </div>
                        <div className="text-[11px] text-ink-muted">
                          {result.safety.user_message || (result.safety.quality_passed ? "Clear leaf image verified." : result.safety.failure_reasons.join(", "))}
                        </div>
                      </div>
                    </div>

                    <div className="text-right text-[11px] text-ink-muted">
                      <div>Accuracy: <strong className="text-ink">{result.safety.confidence_tier}</strong></div>
                      <div>Status: <span className="font-mono text-[10px]">{result.safety.ood_status}</span></div>
                    </div>
                  </div>

                  {/* 1. PRIMARY FOLIAR SEVERITY & DIAGNOSIS SUMMARY (PROMINENT AT TOP) */}
                  {result.severity && (
                    <div className="apple-glass-card rounded-3xl p-6 shadow-glass border border-forest/20 bg-parchment/90 space-y-4">
                      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-parchment-dark/50">
                        <div>
                          <div className="text-xs uppercase tracking-wider font-semibold text-ink-muted flex items-center gap-1.5">
                            <Leaf size={15} className="text-forest" weight="fill" />
                            <span>Current Leaf Disease Damage</span>
                          </div>
                          <div className="flex items-baseline gap-3 mt-1.5 flex-wrap">
                            <span className="text-4xl font-extrabold tracking-tight text-forest">
                              {result.severity.severity_pct.toFixed(1)}%
                            </span>
                            {isHealthy ? (
                              <span className="text-xs font-semibold px-3 py-1 rounded-full bg-leaf/20 text-leaf border border-leaf/30">
                                Healthy Leaf (0.0% Damage)
                              </span>
                            ) : (
                              <div className="flex items-center gap-2 flex-wrap">
                                <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-rust-faint text-rust border border-rust/30 flex items-center gap-1.5">
                                  <span className="w-2 h-2 rounded-full bg-rust inline-block shrink-0" />
                                  <span>Dead Brown Spots: {result.severity.necrotic_pct !== undefined ? result.severity.necrotic_pct.toFixed(1) : "0.0"}%</span>
                                </span>
                                <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-amber-500/10 text-amber-800 border border-amber-500/30 flex items-center gap-1.5">
                                  <span className="w-2 h-2 rounded-full bg-amber-500 inline-block shrink-0" />
                                  <span>Spreading Yellow Border: {result.severity.chlorotic_pct !== undefined ? result.severity.chlorotic_pct.toFixed(1) : "0.0"}%</span>
                                </span>
                              </div>
                            )}
                          </div>
                        </div>

                        <div className="flex flex-col sm:items-end gap-2">
                          <div className="flex items-center gap-2">
                            <button
                              type="button"
                              onClick={downloadDirectPdf}
                              disabled={isDownloadingPdf}
                              className="px-3.5 py-1.5 rounded-xl bg-forest text-parchment-light hover:bg-forest/90 text-xs font-semibold flex items-center gap-1.5 transition-all shadow-sm disabled:opacity-50"
                            >
                              {isDownloadingPdf ? (
                                <ArrowsClockwise size={15} className="animate-spin" />
                              ) : (
                                <DownloadSimple size={15} weight="bold" />
                              )}
                              <span>{isDownloadingPdf ? "Downloading PDF..." : t.health_card.download_slip}</span>
                            </button>
                            <button
                              type="button"
                              onClick={() => setIsHealthCardOpen(true)}
                              className="px-3 py-1.5 rounded-xl border border-forest/30 text-forest hover:bg-forest/10 text-xs font-medium flex items-center gap-1.5 transition-colors"
                              title={t.health_card.preview_action}
                            >
                              <FileText size={15} />
                              <span>{t.health_card.preview_action}</span>
                            </button>
                          </div>
                          <div className="sm:text-right">
                            <div className="text-xs text-ink-muted font-medium">
                              {t.results.conformal_interval}
                            </div>
                            <div className="text-base font-bold text-ink mt-0.5 font-mono">
                              [{result.severity.severity_interval[0].toFixed(1)}%, {result.severity.severity_interval[1].toFixed(1)}%]
                            </div>
                            <div className="text-[11px] text-ink-muted">
                              Guaranteed coverage band
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Diagnostic Summary Grid */}
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                        <div className="p-3 rounded-xl bg-parchment/70 border border-parchment-dark/40">
                          <div className="text-ink-muted text-[11px] mb-0.5">Disease Spots Found</div>
                          <div className="text-base font-bold text-ink">
                            {result.severity.lesion_count} <span className="text-xs font-normal text-ink-muted">spots</span>
                          </div>
                        </div>
                        <div className="p-3 rounded-xl bg-parchment/70 border border-parchment-dark/40">
                          <div className="text-ink-muted text-[11px] mb-0.5">Average Spot Size</div>
                          <div className="text-base font-bold text-ink">
                            {result.severity.mean_lesion_size_px.toFixed(1)} <span className="text-xs font-normal text-ink-muted">px</span>
                          </div>
                        </div>
                        <div className="p-3 rounded-xl bg-parchment/70 border border-parchment-dark/40">
                          <div className="text-ink-muted text-[11px] mb-0.5">Confidence Tier</div>
                          <div className="text-base font-bold text-forest">
                            {result.safety.confidence_tier}
                          </div>
                        </div>
                        <div className="p-3 rounded-xl bg-parchment/70 border border-parchment-dark/40">
                          <div className="text-ink-muted text-[11px] mb-0.5">Photo Status</div>
                          <div className="text-xs font-bold text-ink font-mono mt-0.5">
                            {result.safety.ood_status}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* 2. VISUAL SEGMENTATION & LESION MASK OVERLAY (Side-by-Side + Slider) */}
                  {result.severity && (
                    <div className="apple-glass-card rounded-3xl p-6 shadow-glass">
                      <div className="flex items-center justify-between mb-4">
                        <div>
                          <h2 className="text-sm font-semibold text-ink tracking-tight">
                            Disease Spot Outlines &amp; Leaf Analysis
                          </h2>
                          <div className="flex items-center gap-3 mt-1 text-[11px] text-ink-muted flex-wrap">
                            {viewMode === "projected_7d" ? (
                              <>
                                <span className="flex items-center gap-1">
                                  <span className="w-2 h-2 rounded-full bg-rust inline-block" />
                                  <strong className="text-rust">Dead Brown Spots ({result.severity.necrotic_pct?.toFixed(1) || "0.0"}%)</strong>
                                </span>
                                <span className="flex items-center gap-1">
                                  <span className="w-2 h-2 rounded-full bg-amber-500 inline-block" />
                                  <strong className="text-amber-800">Spreading Yellow Border ({result.severity.chlorotic_pct?.toFixed(1) || "0.0"}%)</strong>
                                </span>
                                <span className="flex items-center gap-1">
                                  <span className="w-2 h-2 rounded-full bg-[#00f0ff] inline-block border border-cyan-700" />
                                  <strong className="text-cyan-700 font-bold">
                                    Forecasted 7-Day Spread ({result.forecast?.horizons["7d"]?.severity_pct.toFixed(1) || (result.severity.severity_pct * 1.5).toFixed(1)}%)
                                  </strong>
                                </span>
                                <span className="flex items-center gap-1">
                                  <span className="w-2 h-2 rounded-full bg-forest inline-block" />
                                  <strong className="text-forest">Leaf Outline</strong>
                                </span>
                              </>
                            ) : viewMode === "treated_7d" ? (
                              <>
                                <span className="flex items-center gap-1">
                                  <span className="w-2 h-2 rounded-full bg-[#ff2e93] inline-block border border-pink-800" />
                                  <strong className="text-pink-700 font-bold">
                                    Spread Prevented ({(result.forecast?.spread_reduction_pct ?? result.severity.spread_reduction_pct ?? 0).toFixed(1)}% Saved)
                                  </strong>
                                </span>
                                <span className="flex items-center gap-1">
                                  <span className="w-2 h-2 rounded-full bg-sky-400 inline-block border border-sky-700" />
                                  <strong className="text-sky-700">
                                    Treated Spots Limit ({(result.forecast?.treated_7d_severity_pct ?? result.severity.treated_7d_severity_pct ?? 0).toFixed(1)}%)
                                  </strong>
                                </span>
                                <span className="flex items-center gap-1">
                                  <span className="w-2 h-2 rounded-full bg-cyan-400 inline-block border border-cyan-700" />
                                  <strong className="text-cyan-700">
                                    Without Spray Boundary ({result.forecast?.horizons["7d"]?.severity_pct.toFixed(1) || (result.severity.severity_pct * 1.5).toFixed(1)}%)
                                  </strong>
                                </span>
                                <span className="flex items-center gap-1">
                                  <span className="w-2 h-2 rounded-full bg-rust inline-block" />
                                  <strong className="text-rust">Day 0 Spots</strong>
                                </span>
                              </>
                            ) : (
                              <>
                                <span className="flex items-center gap-1">
                                  <span className="w-2 h-2 rounded-full bg-rust inline-block" />
                                  <strong className="text-rust">Dead Brown Spots (Red)</strong>
                                </span>
                                <span className="flex items-center gap-1">
                                  <span className="w-2 h-2 rounded-full bg-amber-500 inline-block" />
                                  <strong className="text-amber-800">Spreading Yellow Border (Yellow)</strong>
                                </span>
                                <span className="flex items-center gap-1">
                                  <span className="w-2 h-2 rounded-full bg-forest inline-block" />
                                  <strong className="text-forest">Healthy Leaf Area (Green)</strong>
                                </span>
                              </>
                            )}
                          </div>
                        </div>

                        {/* View Mode Toggle (Overlay vs 7-Day Projected vs Side-by-Side vs Attention Saliency) */}
                        <div className="flex items-center p-1 bg-parchment rounded-xl border border-parchment-dark/60 text-xs flex-wrap gap-1">
                          <button
                            type="button"
                            onClick={() => setViewMode("overlay")}
                            className={`px-3 py-1 rounded-lg font-medium transition-all ${
                              viewMode === "overlay"
                                ? "bg-forest text-parchment-light shadow-sm"
                                : "text-ink hover:text-forest"
                            }`}
                          >
                            Disease Overlay
                          </button>
                          <button
                            type="button"
                            onClick={() => setViewMode("projected_7d")}
                            className={`px-3 py-1 rounded-lg font-medium transition-all ${
                              viewMode === "projected_7d"
                                ? "bg-forest text-parchment-light shadow-sm"
                                : "text-ink hover:text-forest"
                            }`}
                          >
                            7-Day Spread
                          </button>
                          <button
                            type="button"
                            onClick={() => setViewMode("treated_7d")}
                            className={`px-3 py-1 rounded-lg font-medium transition-all ${
                              viewMode === "treated_7d"
                                ? "bg-pink-600 text-parchment-light shadow-sm"
                                : "text-pink-700 hover:text-pink-900"
                            }`}
                          >
                            Spray Reduction (7d)
                          </button>
                          <button
                            type="button"
                            onClick={() => setViewMode("side_by_side")}
                            className={`px-3 py-1 rounded-lg font-medium transition-all ${
                              viewMode === "side_by_side"
                                ? "bg-forest text-parchment-light shadow-sm"
                                : "text-ink hover:text-forest"
                            }`}
                          >
                            Side-by-Side
                          </button>
                          <button
                            type="button"
                            onClick={() => setViewMode("attention")}
                            className={`px-3 py-1 rounded-lg font-medium transition-all ${
                              viewMode === "attention"
                                ? "bg-forest text-parchment-light shadow-sm"
                                : "text-ink hover:text-forest"
                            }`}
                          >
                            AI Focus Areas
                          </button>
                        </div>
                      </div>

                      {/* Display Canvas with Priority High-Scale Bounding */}
                      {viewMode === "projected_7d" ? (
                        <div className="space-y-3">
                          <div className="relative rounded-3xl overflow-hidden bg-[#151614] h-[540px] sm:h-[600px] w-full flex items-center justify-center border border-parchment-dark/60 shadow-inner">
                            {/* CSS Grid Stack: Exactly 100% pixel-perfect coincidence */}
                            <div className="relative inline-grid grid-cols-1 grid-rows-1 items-center justify-center max-h-full max-w-full">
                              {previewUrl && (
                                <img
                                  src={previewUrl}
                                  alt="Original leaf"
                                  className="col-start-1 row-start-1 max-h-[540px] sm:max-h-[600px] max-w-full w-auto h-auto object-contain block select-none pointer-events-none"
                                />
                              )}
                              {(result.forecast?.projected_7d_overlay_base64 || result.severity.projected_7d_overlay_base64 || result.severity.overlay_base64) && (
                                <img
                                  src={`data:image/png;base64,${result.forecast?.projected_7d_overlay_base64 || result.severity.projected_7d_overlay_base64 || result.severity.overlay_base64}`}
                                  alt="7-Day Projected Lesion Frontier"
                                  className="col-start-1 row-start-1 max-h-[540px] sm:max-h-[600px] max-w-full w-auto h-auto object-contain block select-none pointer-events-none transition-opacity duration-150"
                                  style={{ opacity: overlayOpacity }}
                                />
                              )}
                            </div>
                          </div>
                          <div className="flex items-center justify-between text-xs text-ink-muted px-2 flex-wrap gap-2 pt-1">
                            <div className="flex items-center gap-2">
                              <span className="font-semibold text-ink">Spread Boundary:</span>
                              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-cyan-950/20 text-cyan-700 border border-cyan-500/30 font-bold">
                                Electric Cyan Line = Projected 7-Day Spread Boundary
                              </span>
                            </div>
                            <span className="italic text-[11px] text-ink-muted">
                              7-Day spread projection based on live local weather forecast ({result.forecast?.horizons["7d"]?.severity_pct.toFixed(1) || (result.severity.severity_pct * 1.5).toFixed(1)}%)
                            </span>
                          </div>
                        </div>
                      ) : viewMode === "treated_7d" ? (
                        <div className="space-y-3">
                          <div className="relative rounded-3xl overflow-hidden bg-[#151614] h-[540px] sm:h-[600px] w-full flex items-center justify-center border border-parchment-dark/60 shadow-inner">
                            <div className="relative inline-grid grid-cols-1 grid-rows-1 items-center justify-center max-h-full max-w-full">
                              {previewUrl && (
                                <img
                                  src={previewUrl}
                                  alt="Original leaf"
                                  className="col-start-1 row-start-1 max-h-[540px] sm:max-h-[600px] max-w-full w-auto h-auto object-contain block select-none pointer-events-none"
                                />
                              )}
                              {(result.forecast?.treated_7d_overlay_base64 || result.severity.treated_7d_overlay_base64) && (
                                <img
                                  src={`data:image/png;base64,${result.forecast?.treated_7d_overlay_base64 || result.severity.treated_7d_overlay_base64}`}
                                  alt="7-Day Treated Spread Reduction"
                                  className="col-start-1 row-start-1 max-h-[540px] sm:max-h-[600px] max-w-full w-auto h-auto object-contain block select-none pointer-events-none transition-opacity duration-150"
                                  style={{ opacity: overlayOpacity }}
                                />
                              )}
                            </div>
                          </div>
                          <div className="flex items-center justify-between text-xs text-ink-muted px-2 flex-wrap gap-2 pt-1">
                            <div className="flex items-center gap-2 flex-wrap">
                              <span className="font-semibold text-ink">Treatment Impact:</span>
                              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-pink-950/20 text-pink-700 border border-pink-500/40 font-bold">
                                Neon Magenta = Spread Prevented ({(result.forecast?.spread_reduction_pct ?? result.severity.spread_reduction_pct ?? 0).toFixed(1)}% Saved)
                              </span>
                              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-sky-950/20 text-sky-700 border border-sky-500/30 font-semibold">
                                Sky Blue = Treated Spots Limit ({(result.forecast?.treated_7d_severity_pct ?? result.severity.treated_7d_severity_pct ?? 0).toFixed(1)}%)
                              </span>
                              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-cyan-950/20 text-cyan-700 border border-cyan-500/30 font-semibold">
                                Electric Cyan = Without Spray Boundary ({result.forecast?.horizons["7d"]?.severity_pct.toFixed(1) || (result.severity.severity_pct * 1.5).toFixed(1)}%)
                              </span>
                            </div>
                            <span className="italic text-[11px] text-ink-muted">
                              Estimated disease spread reduction under favorable spraying weather (non-prescription advisory)
                            </span>
                          </div>
                        </div>
                      ) : viewMode === "attention" ? (
                        <div className="space-y-3">
                          <div className="relative rounded-3xl overflow-hidden bg-[#151614] h-[540px] sm:h-[600px] w-full flex items-center justify-center border border-parchment-dark/60 shadow-inner">
                            {result.xai?.attention_saliency_base64 ? (
                              <img
                                src={`data:image/png;base64,${result.xai.attention_saliency_base64}`}
                                alt="AI Focus Areas Map"
                                className="max-h-[540px] sm:max-h-[600px] max-w-full w-auto h-auto object-contain block select-none"
                              />
                            ) : previewUrl ? (
                              <img
                                src={previewUrl}
                                alt="Original leaf"
                                className="max-h-[540px] sm:max-h-[600px] max-w-full w-auto h-auto object-contain block select-none"
                              />
                            ) : null}
                          </div>
                          <div className="flex items-center justify-between text-xs text-ink-muted px-2 flex-wrap gap-2 pt-1">
                            <div className="flex items-center gap-2">
                              <span className="font-medium text-ink">AI Focus Areas:</span>
                              <div className="h-3 w-40 rounded-full bg-gradient-to-r from-[#000004] via-[#721f81] via-[#cd4071] via-[#fd9668] to-[#fcfdbf] border border-parchment-dark/40 shadow-sm" />
                              <span className="text-[11px] font-mono text-ink font-semibold">Low → Peak Attention</span>
                            </div>
                            <span className="italic text-[11px] text-ink-muted">Areas on the leaf examined by AI model to diagnose spots</span>
                          </div>
                        </div>
                      ) : viewMode === "overlay" ? (
                        <div className="relative rounded-3xl overflow-hidden bg-[#151614] h-[540px] sm:h-[600px] w-full flex items-center justify-center border border-parchment-dark/60 shadow-inner">
                          {/* Zero-offset pixel fidelity: when opacity is 100% and pre-composited image exists, use it directly */}
                          {overlayOpacity >= 0.98 && result.severity.composite_image_base64 ? (
                            <img
                              src={`data:image/png;base64,${result.severity.composite_image_base64}`}
                              alt="Disease Spot Outlines"
                              className="max-h-[540px] sm:max-h-[600px] max-w-full w-auto h-auto object-contain block select-none"
                            />
                          ) : (
                            <div className="relative inline-grid grid-cols-1 grid-rows-1 items-center justify-center max-h-full max-w-full">
                              {previewUrl && (
                                <img
                                  src={previewUrl}
                                  alt="Original leaf"
                                  className="col-start-1 row-start-1 max-h-[540px] sm:max-h-[600px] max-w-full w-auto h-auto object-contain block select-none pointer-events-none"
                                />
                              )}
                              {result.severity.overlay_base64 && (
                                <img
                                  src={`data:image/png;base64,${result.severity.overlay_base64}`}
                                  alt="Lesion overlay"
                                  className="col-start-1 row-start-1 max-h-[540px] sm:max-h-[600px] max-w-full w-auto h-auto object-contain block select-none pointer-events-none transition-opacity duration-150"
                                  style={{ opacity: overlayOpacity }}
                                />
                              )}
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <span className="text-xs font-semibold text-ink flex items-center gap-1.5">
                              <span className="w-2 h-2 rounded-full bg-forest" />
                              Original Upload
                            </span>
                            <div className="rounded-3xl overflow-hidden bg-[#151614] h-[460px] sm:h-[520px] w-full flex items-center justify-center border border-parchment-dark/60 shadow-inner">
                              {previewUrl && (
                                <img
                                  src={previewUrl}
                                  alt="Original leaf"
                                  className="max-h-[460px] sm:max-h-[520px] max-w-full w-auto h-auto object-contain block select-none"
                                />
                              )}
                            </div>
                          </div>

                          <div className="space-y-2">
                            <span className="text-xs font-semibold text-ink flex items-center gap-1.5">
                              <span className="w-2 h-2 rounded-full bg-rust" />
                              Disease Spot Outlines
                            </span>
                            <div className="relative rounded-3xl overflow-hidden bg-[#151614] h-[460px] sm:h-[520px] w-full flex items-center justify-center border border-parchment-dark/60 shadow-inner">
                              {result.severity.composite_image_base64 ? (
                                <img
                                  src={`data:image/png;base64,${result.severity.composite_image_base64}`}
                                  alt="Disease Spot Outlines"
                                  className="max-h-[460px] sm:max-h-[520px] max-w-full w-auto h-auto object-contain block select-none"
                                />
                              ) : (
                                <div className="relative inline-grid grid-cols-1 grid-rows-1 items-center justify-center max-h-full max-w-full">
                                  {previewUrl && (
                                    <img
                                      src={previewUrl}
                                      alt="Original leaf"
                                      className="col-start-1 row-start-1 max-h-[460px] sm:max-h-[520px] max-w-full w-auto h-auto object-contain block select-none pointer-events-none"
                                    />
                                  )}
                                  {result.severity.overlay_base64 && (
                                    <img
                                      src={`data:image/png;base64,${result.severity.overlay_base64}`}
                                      alt="Mask overlay"
                                      className="col-start-1 row-start-1 max-h-[460px] sm:max-h-[520px] max-w-full w-auto h-auto object-contain block select-none pointer-events-none"
                                    />
                                  )}
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Transparency / Opacity Slider (exact functionality from Gradio) */}
                      {(viewMode === "overlay" || viewMode === "projected_7d" || viewMode === "treated_7d") && (
                        <div className="mt-4 pt-3 border-t border-parchment-dark/40 flex items-center justify-between gap-4">
                          <span className="text-xs text-ink-muted font-medium whitespace-nowrap">
                            Mask Opacity: <strong>{(overlayOpacity * 100).toFixed(0)}%</strong>
                          </span>
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={overlayOpacity}
                            onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                            className="w-full accent-forest cursor-pointer"
                          />
                        </div>
                      )}
                    </div>
                  )}

                  {/* 3. EPIDEMIC PROGRESSION FORECAST (Matplotlib Visual Chart) */}
                  {result.forecast && (
                    <div className="apple-glass-card rounded-3xl p-6 shadow-glass space-y-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <h2 className="text-sm font-semibold text-ink tracking-tight">
                            Disease Progression Forecast Chart
                          </h2>
                          <p className="text-xs text-ink-muted">
                            Predicted disease spread over 3 and 7 days with expected range based on local weather
                          </p>
                        </div>
                        <ChartLineUp size={20} className="text-forest" />
                      </div>

                      {/* Matplotlib Rendered Plot from Backend */}
                      {result.forecast.chart_base64 ? (
                        <div className="rounded-2xl overflow-hidden bg-white/90 border border-parchment-dark/60 shadow-sm">
                          <img
                            src={`data:image/png;base64,${result.forecast.chart_base64}`}
                            alt="Disease Progression Forecast Chart"
                            className="w-full h-auto object-contain"
                          />
                        </div>
                      ) : (
                        <div className="grid grid-cols-3 gap-3">
                          <div className="p-3.5 rounded-xl bg-parchment/70 border border-parchment-dark/40 text-center">
                            <div className="text-[11px] font-medium text-ink-muted mb-1">Today (Day 0)</div>
                            <div className="text-lg font-bold text-ink">
                              {result.severity ? result.severity.severity_pct.toFixed(1) : "0.0"}%
                            </div>
                          </div>
                          <div className="p-3.5 rounded-xl bg-parchment/70 border border-parchment-dark/40 text-center">
                            <div className="text-[11px] font-medium text-ink-muted mb-1">+3 Days</div>
                            <div className="text-lg font-bold text-forest">
                              {result.forecast.horizons["3d"]?.severity_pct.toFixed(1) || "0.0"}%
                            </div>
                          </div>
                          <div className="p-3.5 rounded-xl bg-parchment/70 border border-parchment-dark/40 text-center">
                            <div className="text-[11px] font-medium text-ink-muted mb-1">+7 Days</div>
                            <div className="text-lg font-bold text-rust">
                              {result.forecast.horizons["7d"]?.severity_pct.toFixed(1) || "0.0"}%
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* 4. EXPLAINABLE AI: 7-DAY PROGRESSION DRIVER BREAKDOWN (TreeSHAP) */}
                  {result.xai && (
                    <div className="apple-glass-card rounded-3xl p-6 shadow-glass space-y-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="flex items-center gap-2">
                            <h2 className="text-sm font-semibold text-ink tracking-tight">
                              Why Disease Spreads: 7-Day Growth Guide
                            </h2>
                            <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-forest/10 text-forest border border-forest/20">
                              Field Inspection Guide
                            </span>
                          </div>
                          <p className="text-xs text-ink-muted mt-0.5">
                            Simple breakdown showing what speeds up or slows down disease spread on your crop
                          </p>
                        </div>
                        <button
                          type="button"
                          onClick={() => setShowXaiDetails(!showXaiDetails)}
                          className="px-3 py-1 rounded-xl bg-parchment text-forest text-xs font-medium border border-parchment-dark/60 hover:bg-parchment-dark/30 transition-all"
                        >
                          {showXaiDetails ? "Hide Contributing Factors" : "View Contributing Factors"}
                        </button>
                      </div>

                      {/* Prominent Plain-Language Agronomic Synthesis (FIRST THING) */}
                      <div className="p-4 rounded-2xl bg-parchment-light/95 border border-forest/25 text-xs space-y-2 shadow-sm">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2 font-semibold text-forest text-xs">
                            <Info size={17} weight="bold" />
                            <span>Key Takeaway: What Drives Disease Spread over 7 Days</span>
                          </div>
                          <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-forest/10 text-forest border border-forest/20">
                            Summary
                          </span>
                        </div>
                        <p className="text-ink text-xs leading-relaxed font-medium">
                          {result.xai.narrative}
                        </p>
                      </div>

                      {/* 3-Metric Progression Breakdown */}
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                        <div className="p-3.5 rounded-2xl bg-parchment/70 border border-parchment-dark/50 text-center">
                          <div className="text-[11px] text-ink-muted mb-0.5">Current Leaf Damage</div>
                          <div className="text-xl font-bold text-ink">{result.xai.baseline_severity_pct.toFixed(1)}%</div>
                          <div className="text-[10px] text-ink-muted">Measured today</div>
                        </div>
                        <div className="p-3.5 rounded-2xl bg-parchment/70 border border-parchment-dark/50 text-center">
                          <div className="text-[11px] text-ink-muted mb-0.5">Expected Damage in 7 Days</div>
                          <div className="text-xl font-bold text-forest">{result.xai.forecasted_severity_pct.toFixed(1)}%</div>
                          <div className="text-[10px] text-ink-muted">If untreated</div>
                        </div>
                        <div className="p-3.5 rounded-2xl bg-parchment/70 border border-parchment-dark/50 text-center">
                          <div className="text-[11px] text-ink-muted mb-0.5">Expected New Spread</div>
                          <div className={`text-xl font-bold ${result.xai.net_change_pct >= 0 ? "text-rust" : "text-forest"}`}>
                            {result.xai.net_change_pct >= 0 ? "+" : ""}{result.xai.net_change_pct.toFixed(1)}%
                          </div>
                          <div className="text-[10px] text-ink-muted">Additional damage area</div>
                        </div>
                      </div>

                      {/* Diverging Horizontal Bar Chart */}
                      {showXaiDetails && (
                        <div className="space-y-3 pt-2">
                          <div className="flex items-center justify-between text-[11px] text-ink-muted px-1 pb-1 border-b border-parchment-dark/40">
                            <span className="w-1/2 text-left">
                              <strong className="text-leaf">← Factors Slowing Disease Down (Protective)</strong>
                            </span>
                            <span className="w-1/2 text-right">
                              <strong className="text-rust">Factors Speeding Up Spread (Risk) →</strong>
                            </span>
                          </div>

                          <div className="space-y-3">
                            {result.xai.factors.map((factor) => {
                              const isPositive = factor.delta_pct >= 0;
                              const maxMag = 15.0;
                              const barWidthPct = Math.min(100, (Math.abs(factor.delta_pct) / maxMag) * 100);

                              return (
                                <div key={factor.factor_id} className="p-3 rounded-2xl bg-parchment/50 border border-parchment-dark/40 space-y-1.5">
                                  <div className="flex items-center justify-between text-xs">
                                    <div className="flex items-center gap-2">
                                      <span className="font-semibold text-ink">{factor.factor_name}</span>
                                      <span className="text-[11px] font-mono text-ink-muted px-2 py-0.5 rounded bg-parchment border border-parchment-dark/40">
                                        {factor.raw_value}
                                      </span>
                                    </div>
                                    <div className="flex items-center gap-1.5 font-mono font-bold text-xs">
                                      <span className={isPositive ? "text-rust" : "text-forest"}>
                                        {isPositive ? `+${factor.delta_pct.toFixed(1)}%` : `${factor.delta_pct.toFixed(1)}%`}
                                      </span>
                                    </div>
                                  </div>

                                  {/* Center-anchored Diverging Bar Track */}
                                  <div className="h-3 w-full bg-parchment-dark/40 rounded-full relative overflow-hidden flex">
                                    {/* Left half (Negative / Retarding) */}
                                    <div className="w-1/2 h-full flex justify-end">
                                      {!isPositive && (
                                        <div
                                          className="h-full bg-forest rounded-l-full transition-all duration-300"
                                          style={{ width: `${barWidthPct}%` }}
                                        />
                                      )}
                                    </div>
                                    {/* Center Divider */}
                                    <div className="w-0.5 h-full bg-ink/30 z-10" />
                                    {/* Right half (Positive / Accelerating) */}
                                    <div className="w-1/2 h-full flex justify-start">
                                      {isPositive && (
                                        <div
                                          className="h-full bg-rust rounded-r-full transition-all duration-300"
                                          style={{ width: `${barWidthPct}%` }}
                                        />
                                      )}
                                    </div>
                                  </div>

                                  <div className="text-[11px] text-ink-muted leading-tight">
                                    {factor.description}
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* 5. LIVE PERIODICAL MICROCLIMATE WEATHER TABLE (Open-Meteo 9km NWP) */}
                  {result.weather && result.weather.table_summary && (
                    <div className="apple-glass-card rounded-3xl p-6 shadow-glass space-y-4">
                      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-parchment-dark/50">
                        <div>
                          <div className="flex items-center gap-2">
                            <h2 className="text-sm font-semibold text-ink tracking-tight">
                              {t.weather_table.heading}
                            </h2>
                            <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-forest/10 text-forest border border-forest/20">
                              Live Local Weather
                            </span>
                          </div>
                          <p className="text-xs text-ink-muted mt-0.5">
                            {result.weather.source}
                          </p>
                        </div>
                        <div className="flex items-center gap-2 text-xs">
                          <span className="font-mono text-[11px] px-2.5 py-1 rounded-xl bg-forest/10 text-forest border border-forest/20 font-bold">
                            Resolution: 0.1° (~9 km)
                          </span>
                        </div>
                      </div>

                      <div className="overflow-x-auto rounded-xl border border-parchment-dark/60">
                        <table className="w-full text-xs text-left">
                          <thead className="bg-parchment/80 text-ink-muted font-medium border-b border-parchment-dark/50">
                            <tr>
                              <th className="py-2.5 px-3">{t.weather_table.day}</th>
                              <th className="py-2.5 px-3">{t.weather_table.temperature}</th>
                              <th className="py-2.5 px-3">{t.weather_table.relative_humidity}</th>
                              <th className="py-2.5 px-3">{t.weather_table.vpd}</th>
                              <th className="py-2.5 px-3">{t.weather_table.rain_duration}</th>
                              <th className="py-2.5 px-3">{t.weather_table.leaf_wetness}</th>
                              <th className="py-2.5 px-3">{t.weather_table.wind_speed}</th>
                              <th className="py-2.5 px-3">{t.weather_table.spray_feasibility}</th>
                              <th className="py-2.5 px-3">{t.weather_table.infection_risk}</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-parchment-dark/30">
                            {result.weather.table_summary.map((row, idx) => (
                              <tr key={idx} className="hover:bg-parchment/40 transition-colors">
                                <td className="py-2.5 px-3 font-medium text-ink flex items-center gap-1.5 whitespace-nowrap">
                                  {idx === 0 ? (
                                    <span className="w-2 h-2 rounded-full bg-forest inline-block shrink-0" />
                                  ) : (
                                    <span className="w-2 h-2 rounded-full bg-parchment-dark inline-block shrink-0" />
                                  )}
                                  <span>{row.label}</span>
                                </td>
                                <td className="py-2.5 px-3 text-forest font-mono font-semibold whitespace-nowrap">{row.temp_c.toFixed(1)} °C</td>
                                <td className="py-2.5 px-3 text-ink font-mono whitespace-nowrap">{row.rh_pct.toFixed(1)} %</td>
                                <td className="py-2.5 px-3 text-ink font-mono whitespace-nowrap">{row.vpd_kpa.toFixed(2)} kPa</td>
                                <td className="py-2.5 px-3 text-ink font-mono whitespace-nowrap">{row.precip_mm.toFixed(1)} mm</td>
                                <td className="py-2.5 px-3 text-ink font-mono whitespace-nowrap">
                                  {row.leaf_wetness_hours !== undefined ? `${row.leaf_wetness_hours.toFixed(1)} h` : "—"}
                                </td>
                                <td className="py-2.5 px-3 text-ink font-mono whitespace-nowrap">
                                  {row.max_wind_kmh !== undefined ? `${row.max_wind_kmh.toFixed(1)} km/h` : "8.0 km/h"}
                                </td>
                                <td className="py-2.5 px-3">
                                  {row.spray_feasibility === "FAVORABLE" ? (
                                    <div className="space-y-0.5">
                                      <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-forest/15 text-forest border border-forest/30 inline-block">
                                        {t.spray_status.favorable}
                                      </span>
                                      {row.spray_advisory && (
                                        <p className="text-[10px] text-ink-muted leading-tight max-w-[170px] truncate" title={row.spray_advisory}>
                                          {row.spray_advisory}
                                        </p>
                                      )}
                                    </div>
                                  ) : row.spray_feasibility === "MARGINAL" ? (
                                    <div className="space-y-0.5">
                                      <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-800 border border-amber-500/30 inline-block">
                                        {t.spray_status.marginal}
                                      </span>
                                      {row.spray_advisory && (
                                        <p className="text-[10px] text-ink-muted leading-tight max-w-[170px] truncate" title={row.spray_advisory}>
                                          {row.spray_advisory}
                                        </p>
                                      )}
                                    </div>
                                  ) : (
                                    <div className="space-y-0.5">
                                      <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-rust/15 text-rust border border-rust/30 inline-block">
                                        {t.spray_status.unfavorable}
                                      </span>
                                      {row.spray_advisory && (
                                        <p className="text-[10px] text-ink-muted leading-tight max-w-[170px] truncate" title={row.spray_advisory}>
                                          {row.spray_advisory}
                                        </p>
                                      )}
                                    </div>
                                  )}
                                </td>
                                <td className="py-2.5 px-3">
                                  {row.infection_risk !== undefined ? (
                                    <span className={`text-[10px] font-mono px-2 py-0.5 rounded font-bold ${
                                      row.infection_risk > 0.6
                                        ? "bg-rust/15 text-rust border border-rust/30"
                                        : row.infection_risk > 0.3
                                        ? "bg-amber-500/15 text-amber-800 border border-amber-500/30"
                                        : "bg-leaf/15 text-leaf border border-leaf/30"
                                    }`}>
                                      {(row.infection_risk * 100).toFixed(0)}%
                                    </span>
                                  ) : (
                                    <span className="text-ink-muted text-[10px] font-mono">Normal</span>
                                  )}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* 6. AGRONOMIC IPM ADVISORY & MITIGATION ACTIONS */}
                  <div className="apple-glass-card rounded-3xl p-6 shadow-glass space-y-4">
                    <div className="pb-2 border-b border-parchment-dark/50">
                      <div className="flex items-center gap-2">
                        <FirstAid size={18} className="text-forest" />
                        <h2 className="text-sm font-semibold text-ink tracking-tight">
                          {t.ipm_advisory.heading}
                        </h2>
                      </div>
                      <p className="text-xs text-ink-muted mt-0.5">
                        {t.ipm_advisory.subtitle}
                      </p>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                      <div className="p-3.5 rounded-2xl bg-parchment/60 border border-parchment-dark/50 space-y-1">
                        <strong className="text-forest font-semibold block">{t.ipm_advisory.pillar_canopy}</strong>
                        <p className="text-[11px] text-ink-muted leading-relaxed">{t.ipm_advisory.pillar_canopy_desc}</p>
                      </div>
                      <div className="p-3.5 rounded-2xl bg-parchment/60 border border-parchment-dark/50 space-y-1">
                        <strong className="text-forest font-semibold block">{t.ipm_advisory.pillar_sanitation}</strong>
                        <p className="text-[11px] text-ink-muted leading-relaxed">{t.ipm_advisory.pillar_sanitation_desc}</p>
                      </div>
                      <div className="p-3.5 rounded-2xl bg-parchment/60 border border-parchment-dark/50 space-y-1">
                        <strong className="text-forest font-semibold block">{t.ipm_advisory.pillar_nutrition}</strong>
                        <p className="text-[11px] text-ink-muted leading-relaxed">{t.ipm_advisory.pillar_nutrition_desc}</p>
                      </div>
                      <div className="p-3.5 rounded-2xl bg-parchment/60 border border-parchment-dark/50 space-y-1">
                        <strong className="text-forest font-semibold block">{t.ipm_advisory.pillar_biological}</strong>
                        <p className="text-[11px] text-ink-muted leading-relaxed">{t.ipm_advisory.pillar_biological_desc}</p>
                      </div>
                    </div>
                  </div>

                  {/* 7. PIPELINE TIMINGS ACCORDION */}
                  {result.timings && (
                    <div className="apple-glass rounded-2xl p-4 border border-parchment-dark/50 text-xs">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2 text-ink font-medium">
                          <Clock size={16} className="text-forest" />
                          <span>Total Pipeline Latency: <strong>{result.timings.total_ms.toFixed(0)} ms</strong></span>
                        </div>
                        <span className="text-[11px] text-ink-muted">CPU Optimized</span>
                      </div>

                      {result.timings.stages_ms && (
                        <div className="mt-3 pt-2.5 border-t border-parchment-dark/30 grid grid-cols-2 sm:grid-cols-3 gap-2 text-[11px] text-ink-muted">
                          {Object.entries(result.timings.stages_ms).map(([stage, ms]) => (
                            <div key={stage} className="flex justify-between px-2 py-1 rounded bg-parchment/60">
                              <span className="truncate">{stage.replace("_ms", "")}:</span>
                              <strong className="text-ink font-mono ml-1">{ms.toFixed(0)} ms</strong>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Non-Prescription Disclaimer */}
                  <div className="text-[11px] text-ink-muted/80 text-center leading-relaxed px-4">
                    Non-prescription decision support only. Calibrated on agronomic research field trials. Does NOT prescribe chemical brand names, dosages, or spray schedules.
                  </div>
                </div>
              )}
            </section>
          </div>
        )}

        {/* TAB 2: HOW IT WORKS */}
        {activeTab === "how_it_works" && (
          <div className="max-w-4xl mx-auto space-y-6">
            <div className="apple-glass-card rounded-3xl p-8 shadow-glass space-y-6">
              <div>
                <h2 className="text-lg font-bold text-ink tracking-tight mb-1">
                  How Crop Health Check Works
                </h2>
                <p className="text-xs text-ink-muted">
                  Simple overview of how AI scans leaves, predicts disease spread, and checks weather
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                <div className="p-5 rounded-2xl bg-parchment/60 border border-parchment-dark/50 space-y-2">
                  <div className="flex items-center gap-2 text-forest font-semibold text-sm">
                    <Leaf size={18} weight="duotone" />
                    <span>1. Leaf Health &amp; Spot Detection</span>
                  </div>
                  <p className="text-xs text-ink-muted leading-relaxed">
                    Uses smart vision AI to scan leaf photos, cleanly separate the leaf from the background, and pinpoint both dead brown spots and spreading yellow borders.
                  </p>
                </div>

                <div className="p-5 rounded-2xl bg-parchment/60 border border-parchment-dark/50 space-y-2">
                  <div className="flex items-center gap-2 text-forest font-semibold text-sm">
                    <TrendUp size={18} weight="duotone" />
                    <span>2. 7-Day Disease Spread Forecast</span>
                  </div>
                  <p className="text-xs text-ink-muted leading-relaxed">
                    Predicts how quickly spots will expand over the next 3 to 7 days by combining crop biology with real local temperature and humidity.
                  </p>
                </div>

                <div className="p-5 rounded-2xl bg-parchment/60 border border-parchment-dark/50 space-y-2">
                  <div className="flex items-center gap-2 text-forest font-semibold text-sm">
                    <CloudSun size={18} weight="duotone" />
                    <span>3. High-Accuracy Local Weather</span>
                  </div>
                  <p className="text-xs text-ink-muted leading-relaxed">
                    Checks real 7-day weather forecast (9 km local area). Uses temperature, air moisture, and rain forecast to determine how long leaves stay wet and alert you to disease danger.
                  </p>
                </div>

                <div className="p-5 rounded-2xl bg-parchment/60 border border-parchment-dark/50 space-y-2">
                  <div className="flex items-center gap-2 text-forest font-semibold text-sm">
                    <ShieldCheck size={18} weight="duotone" />
                    <span>4. Photo Quality Safety Check</span>
                  </div>
                  <p className="text-xs text-ink-muted leading-relaxed">
                    Checks each photo for clarity. Automatically warns you if the photo is too dark, blurry, or washed out so you get trustworthy results every time.
                  </p>
                </div>
              </div>

              <div className="p-5 rounded-2xl bg-sage-faint/60 border border-sage/40 space-y-2">
                <h3 className="text-xs font-semibold text-forest uppercase tracking-wider">
                  Tips for Taking Good Leaf Photos
                </h3>
                <ul className="text-xs text-ink-muted space-y-1.5 list-disc list-inside">
                  <li>Take photos in bright natural daylight (avoid dark shadows or night flash).</li>
                  <li>Hold the camera steady and focus clearly on the disease spots.</li>
                  <li>Fill the picture with the leaf so the spots can be examined in detail.</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: ABOUT & SCIENTIFIC LIMITATIONS */}
        {activeTab === "about" && (
          <div className="max-w-4xl mx-auto space-y-6">
            <div className="apple-glass-card rounded-3xl p-8 shadow-glass space-y-6">
              <div>
                <h2 className="text-lg font-bold text-ink tracking-tight mb-1">
                  About &amp; Safety Guidelines
                </h2>
                <p className="text-xs text-ink-muted">
                  Crop advisory support, data privacy, and government safety guidance
                </p>
              </div>

              <div className="space-y-4 text-xs text-ink-muted leading-relaxed">
                <div>
                  <h3 className="font-semibold text-ink text-sm mb-1.5">Crop Datasets &amp; Verification</h3>
                  <ul className="list-disc list-inside space-y-1">
                    <li><strong>Wheat Field Benchmark</strong>: Field verified time series research dataset across diverse crop lines.</li>
                    <li><strong>Plant Disease Segmentation Benchmark</strong>: Curated multi-crop leaf spot dataset.</li>
                    <li><strong>Coffee Leaf Benchmark</strong>: Field verified coffee leaf spot dataset.</li>
                  </ul>
                </div>

                <div className="pt-3 border-t border-parchment-dark/40">
                  <h3 className="font-semibold text-ink text-sm mb-1.5">Official Non-Prescription Notice</h3>
                  <p className="p-4 rounded-xl bg-parchment/60 border border-parchment-dark/50 italic">
                    This decision support platform provides crop leaf health checks and disease forecasting for advisory guidance only. It NEVER prescribes or sells chemical pesticide brands, commercial mixtures, or chemical dosages. Always consult your local agricultural extension officer (Rythu Bharosa Kendra / KVK) before buying or applying regulated farm chemicals.
                  </p>
                </div>

                <div className="pt-3 border-t border-parchment-dark/40">
                  <h3 className="font-semibold text-ink text-sm mb-1.5">Privacy &amp; Data Handling</h3>
                  <p>
                    All leaf checks and AI image processing are performed securely in memory on the local server. Your uploaded photographs are never shared or sold to external third parties.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* 8. RYTHU FIELD CROP HEALTH CARD MODAL & PRINT VIEW */}
      {isHealthCardOpen && result && result.severity && (
        <div className="fixed inset-0 z-50 bg-ink/60 backdrop-blur-sm flex items-center justify-center p-4 overflow-y-auto">
          <div
            id="printable-health-card"
            className="bg-white rounded-3xl max-w-2xl w-full p-8 shadow-2xl border border-parchment-dark/60 text-ink space-y-5 my-8"
          >
            {/* Official Government / Extension Header */}
            <div className="print-avoid-break flex items-center justify-between pb-3.5 border-b-2 border-forest">
              <div className="flex items-center gap-3">
                <div className="w-11 h-11 rounded-2xl bg-forest text-parchment-light flex items-center justify-center">
                  <Plant size={26} weight="duotone" />
                </div>
                <div>
                  <h2 className="text-base font-bold text-forest tracking-tight">
                    {t.health_card.card_title}
                  </h2>
                  <p className="text-xs text-ink-muted">
                    {t.health_card.rbk_header}
                  </p>
                </div>
              </div>
              <div className="text-right text-xs font-mono text-ink-muted">
                <div><strong className="text-ink">{t.health_card.diagnostic_id}</strong> {diagnosticId}</div>
                <div>{scanTimestamp || new Date().toLocaleDateString()}</div>
              </div>
            </div>

            {/* Field Geolocation & Crop Metadata Grid */}
            <div className="print-avoid-break grid grid-cols-2 sm:grid-cols-3 gap-3 p-3.5 rounded-2xl bg-parchment/40 border border-parchment-dark/50 text-xs">
              <div>
                <span className="text-ink-muted block text-[11px]">{t.health_card.crop_details}</span>
                <strong className="text-ink text-sm font-semibold capitalize">{crop} {cultivar ? `(${cultivar})` : ""}</strong>
              </div>
              <div>
                <span className="text-ink-muted block text-[11px]">{t.health_card.field_coordinates}</span>
                <strong className="text-ink font-mono">{latitude}° N, {longitude}° E</strong>
              </div>
              <div className="col-span-2 sm:col-span-1">
                <span className="text-ink-muted block text-[11px]">{t.health_card.field_region_label}</span>
                <strong className="text-ink truncate block">{locationLabel}</strong>
              </div>
            </div>

            {/* Primary Diagnosis & Pathology Findings */}
            <div className="print-avoid-break p-4 rounded-2xl bg-forest/5 border border-forest/20 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <span className="text-xs font-semibold text-ink-muted uppercase tracking-wider block">
                  {t.health_card.severity_rating}
                </span>
                <div className="flex items-baseline gap-2 mt-1">
                  <span className="text-3xl sm:text-4xl font-extrabold text-forest font-mono">
                    {result.severity.severity_pct.toFixed(1)}%
                  </span>
                  <span className={`text-xs font-bold px-2.5 py-1 rounded-full border ${
                    result.severity.severity_pct === 0.0
                      ? "bg-leaf/15 text-leaf border-leaf/30"
                      : result.severity.severity_pct > 25
                      ? "bg-rust/15 text-rust border-rust/30"
                      : result.severity.severity_pct > 5
                      ? "bg-amber-500/15 text-amber-800 border-amber-500/30"
                      : "bg-forest/15 text-forest border-forest/30"
                  }`}>
                    {result.severity.severity_pct === 0.0
                      ? t.health_card.severity_status_healthy
                      : result.severity.severity_pct > 25
                      ? t.health_card.severity_status_severe
                      : result.severity.severity_pct > 5
                      ? t.health_card.severity_status_moderate
                      : t.health_card.severity_status_mild}
                  </span>
                </div>
                <div className="text-xs text-ink-muted mt-1">
                  {t.health_card.conformal_range_label} <strong className="font-mono text-ink">[{result.severity.severity_interval[0].toFixed(1)}%, {result.severity.severity_interval[1].toFixed(1)}%]</strong>
                </div>
              </div>

              {/* Pathology Tissue Breakdown */}
              <div className="space-y-1.5 text-xs border-t sm:border-t-0 sm:border-l border-forest/20 pt-2 sm:pt-0 sm:pl-4">
                <div className="flex items-center justify-between gap-4">
                  <span className="text-rust font-medium">{t.health_card.necrotic_core_label}</span>
                  <strong className="font-mono">{result.severity.necrotic_pct !== undefined ? result.severity.necrotic_pct.toFixed(1) : "0.0"}%</strong>
                </div>
                <div className="flex items-center justify-between gap-4">
                  <span className="text-amber-800 font-medium">{t.health_card.chlorotic_halo_label}</span>
                  <strong className="font-mono">{result.severity.chlorotic_pct !== undefined ? result.severity.chlorotic_pct.toFixed(1) : "0.0"}%</strong>
                </div>
                <div className="flex items-center justify-between gap-4">
                  <span className="text-forest font-medium">{t.health_card.active_spots_label}</span>
                  <strong className="font-mono">{result.severity.lesion_count} {t.health_card.spots_unit}</strong>
                </div>
              </div>
            </div>

            {/* Visual Leaf & Pathology Mask Preview */}
            <div className="print-avoid-break grid grid-cols-2 gap-4">
              <div className="space-y-1 text-center">
                <span className="text-[11px] font-semibold text-ink-muted block">{t.health_card.original_photo_title}</span>
                {previewUrl ? (
                  <div className="w-full h-40 rounded-xl overflow-hidden border border-parchment-dark/70 bg-parchment/30 flex items-center justify-center">
                    <img src={previewUrl} alt="Original Leaf" className="w-full h-full object-contain" />
                  </div>
                ) : (
                  <div className="w-full h-40 rounded-xl border border-dashed border-parchment-dark/70 bg-parchment/20 flex items-center justify-center text-[11px] text-ink-muted">
                    Photo unavailable
                  </div>
                )}
              </div>
              <div className="space-y-1 text-center">
                <span className="text-[11px] font-semibold text-ink-muted block">{t.health_card.pathology_contours_title}</span>
                {result.severity.composite_image_base64 ? (
                  <div className="w-full h-40 rounded-xl overflow-hidden border border-parchment-dark/70 bg-parchment/30 flex items-center justify-center">
                    <img src={`data:image/png;base64,${result.severity.composite_image_base64}`} alt="Disease Spot Outlines" className="w-full h-full object-contain" />
                  </div>
                ) : (
                  <div className="w-full h-40 rounded-xl border border-dashed border-parchment-dark/70 bg-parchment/20 flex items-center justify-center text-[11px] text-ink-muted p-2">
                    {t.health_card.no_mask_placeholder}
                  </div>
                )}
              </div>
            </div>

            {/* Today & Tomorrow Spray Window Advisory */}
            {result.weather && result.weather.table_summary && result.weather.table_summary.length > 0 && (
              <div className="print-avoid-break p-3.5 rounded-2xl bg-parchment/40 border border-parchment-dark/50 space-y-2 text-xs">
                <div className="font-semibold text-ink text-xs flex items-center gap-1.5">
                  <CloudSun size={16} className="text-forest" />
                  <span>{t.health_card.spray_window_advisory}</span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
                  <div className="p-2.5 rounded-xl bg-white border border-parchment-dark/40 space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-ink">Today ({result.weather.table_summary[0].date})</span>
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                        result.weather.table_summary[0].spray_feasibility === "FAVORABLE"
                          ? "bg-forest/15 text-forest"
                          : result.weather.table_summary[0].spray_feasibility === "MARGINAL"
                          ? "bg-amber-500/15 text-amber-800"
                          : "bg-rust/15 text-rust"
                      }`}>
                        {result.weather.table_summary[0].spray_feasibility === "FAVORABLE"
                          ? t.spray_status.favorable
                          : result.weather.table_summary[0].spray_feasibility === "MARGINAL"
                          ? t.spray_status.marginal
                          : t.spray_status.unfavorable}
                      </span>
                    </div>
                    <p className="text-[11px] text-ink-muted">{result.weather.table_summary[0].spray_advisory || "Calm conditions, suitable for safe leaf spraying."}</p>
                  </div>
                  {result.weather.table_summary[1] && (
                    <div className="p-2.5 rounded-xl bg-white border border-parchment-dark/40 space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-ink">Tomorrow ({result.weather.table_summary[1].date})</span>
                        <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                          result.weather.table_summary[1].spray_feasibility === "FAVORABLE"
                            ? "bg-forest/15 text-forest"
                            : result.weather.table_summary[1].spray_feasibility === "MARGINAL"
                            ? "bg-amber-500/15 text-amber-800"
                            : "bg-rust/15 text-rust"
                        }`}>
                          {result.weather.table_summary[1].spray_feasibility === "FAVORABLE"
                            ? t.spray_status.favorable
                            : result.weather.table_summary[1].spray_feasibility === "MARGINAL"
                            ? t.spray_status.marginal
                            : t.spray_status.unfavorable}
                        </span>
                      </div>
                      <p className="text-[11px] text-ink-muted">{result.weather.table_summary[1].spray_advisory || "Monitor wind conditions before spraying."}</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Recommended Non-Prescriptive Field IPM Actions */}
            <div className="print-avoid-break space-y-2 text-xs">
              <h3 className="font-bold text-ink text-xs uppercase tracking-wider">
                {t.health_card.ipm_checklist_title}
              </h3>
              <ul className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[11px] text-ink-muted leading-relaxed">
                <li className="p-2.5 rounded-xl bg-parchment/40 border border-parchment-dark/40">
                  <strong className="text-ink block mb-0.5">1. {t.ipm_advisory.pillar_canopy}:</strong>
                  {t.ipm_advisory.pillar_canopy_desc}
                </li>
                <li className="p-2.5 rounded-xl bg-parchment/40 border border-parchment-dark/40">
                  <strong className="text-ink block mb-0.5">2. {t.ipm_advisory.pillar_sanitation}:</strong>
                  {t.ipm_advisory.pillar_sanitation_desc}
                </li>
                <li className="p-2.5 rounded-xl bg-parchment/40 border border-parchment-dark/40">
                  <strong className="text-ink block mb-0.5">3. {t.ipm_advisory.pillar_nutrition}:</strong>
                  {t.ipm_advisory.pillar_nutrition_desc}
                </li>
                <li className="p-2.5 rounded-xl bg-parchment/40 border border-parchment-dark/40">
                  <strong className="text-ink block mb-0.5">4. {t.ipm_advisory.pillar_biological}:</strong>
                  {t.ipm_advisory.pillar_biological_desc}
                </li>
              </ul>
            </div>

            {/* Regulatory Non-Prescription Disclaimer */}
            <div className="print-avoid-break p-3 rounded-xl bg-parchment/30 border border-parchment-dark/40 text-[10px] text-ink-muted italic leading-relaxed">
              {t.ipm_advisory.disclaimer_text}
            </div>

            {/* Action Buttons (Hidden when printing via no-print class) */}
            {pdfError && (
              <div className="no-print p-3 rounded-xl bg-rust-faint border border-rust/30 text-rust text-xs flex items-center justify-between gap-2">
                <span>{pdfError}</span>
                <button
                  type="button"
                  onClick={() => window.print()}
                  className="px-2.5 py-1 rounded-lg bg-rust/10 hover:bg-rust/20 font-semibold text-xs shrink-0 transition-colors"
                >
                  Use Paper Print Instead
                </button>
              </div>
            )}
            <div className="no-print flex items-center justify-end gap-3 pt-3 border-t border-parchment-dark/40 flex-wrap">
              <button
                type="button"
                onClick={() => setIsHealthCardOpen(false)}
                className="px-4 py-2 rounded-xl border border-parchment-dark text-xs font-semibold text-ink hover:bg-parchment/60 transition-colors"
              >
                {t.health_card.close_action}
              </button>
              <button
                type="button"
                onClick={() => window.print()}
                className="px-3.5 py-2 rounded-xl border border-forest/30 text-forest hover:bg-forest/10 text-xs font-semibold flex items-center gap-1.5 transition-colors"
                title="Print directly to paper printer"
              >
                <Printer size={15} />
                <span>Paper Print</span>
              </button>
              <button
                type="button"
                onClick={downloadDirectPdf}
                disabled={isDownloadingPdf}
                className="px-5 py-2 rounded-xl bg-forest text-parchment-light hover:bg-forest/90 text-xs font-semibold flex items-center gap-2 transition-all shadow-md disabled:opacity-50"
              >
                {isDownloadingPdf ? (
                  <ArrowsClockwise size={16} className="animate-spin" />
                ) : (
                  <DownloadSimple size={16} weight="bold" />
                )}
                <span>{isDownloadingPdf ? "Generating PDF..." : t.health_card.print_action}</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
