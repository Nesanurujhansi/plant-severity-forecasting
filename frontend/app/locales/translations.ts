export type Language = 'en' | 'te' | 'hi';

export interface Translations {
  app_title: string;
  app_subtitle: string;
  select_language: string;
  tabs: {
    analyze: string;
    how_it_works: string;
    about: string;
  };
  upload: {
    drag_drop_title: string;
    drag_drop_subtitle: string;
    supported_formats: string;
    change_photo: string;
    crop_selector: string;
    cultivar_label: string;
    cultivar_placeholder: string;
    heading_days_label: string;
    heading_days_placeholder: string;
    analyze_button: string;
    analyzing_progress: string;
  };
  location: {
    heading: string;
    subtitle: string;
    ap_presets: string;
    search_bar: string;
    google_maps_pin: string;
    detect_gps: string;
    detecting_gps: string;
    search_placeholder: string;
    search_button: string;
    apply_pin: string;
    selected_location: string;
    lat_lon_label: string;
    maps_placeholder: string;
  };
  results: {
    heading: string;
    health_status: string;
    severity_measured: string;
    conformal_interval: string;
    lesion_spots: string;
    foliar_overlay: string;
    pathology_breakdown: string;
    necrotic_core: string;
    chlorotic_halo: string;
    healthy_tissue: string;
    spatial_projection_7d: string;
    spatial_projection_desc: string;
    diagnostic_saliency: string;
    saliency_desc: string;
  };
  weather_table: {
    heading: string;
    subtitle: string;
    day: string;
    temperature: string;
    relative_humidity: string;
    vpd: string;
    leaf_wetness: string;
    rain_duration: string;
    wind_speed: string;
    spray_feasibility: string;
    infection_risk: string;
  };
  spray_status: {
    favorable: string;
    marginal: string;
    unfavorable: string;
  };
  health_card: {
    download_slip: string;
    card_title: string;
    rbk_header: string;
    diagnostic_id: string;
    scan_date: string;
    crop_details: string;
    field_coordinates: string;
    field_region_label: string;
    severity_rating: string;
    severity_status_healthy: string;
    severity_status_mild: string;
    severity_status_moderate: string;
    severity_status_severe: string;
    necrotic_core_label: string;
    chlorotic_halo_label: string;
    active_spots_label: string;
    spots_unit: string;
    original_photo_title: string;
    pathology_contours_title: string;
    conformal_range_label: string;
    no_mask_placeholder: string;
    microclimate_outlook: string;
    spray_window_advisory: string;
    ipm_checklist_title: string;
    preview_action: string;
    print_action: string;
    close_action: string;
  };
  ipm_advisory: {
    heading: string;
    subtitle: string;
    pillar_canopy: string;
    pillar_canopy_desc: string;
    pillar_sanitation: string;
    pillar_sanitation_desc: string;
    pillar_nutrition: string;
    pillar_nutrition_desc: string;
    pillar_biological: string;
    pillar_biological_desc: string;
    disclaimer_text: string;
  };
}

export const translations: Record<Language, Translations> = {
  en: {
    app_title: "Crop Leaf Health & Disease Guide",
    app_subtitle: "Leaf Disease Check, 7-Day Local Weather & Field Care Guide",
    select_language: "Language",
    tabs: {
      analyze: "Single Leaf Check",
      how_it_works: "How It Works",
      about: "Safety & Information"
    },
    upload: {
      drag_drop_title: "Click or drag leaf photo here",
      drag_drop_subtitle: "Supports high-resolution single leaf photographs",
      supported_formats: "PNG, JPG, WEBP up to 10MB",
      change_photo: "Change leaf photo",
      crop_selector: "Select Crop:",
      cultivar_label: "Crop Variety (Optional):",
      cultivar_placeholder: "e.g. BPT 5204 (Samba Mahsuri)",
      heading_days_label: "Plant Age in Days (Optional):",
      heading_days_placeholder: "e.g. 45",
      analyze_button: "Check Leaf Health & Weather Forecast",
      analyzing_progress: "Scanning Leaf Image & Fetching Weather Forecast..."
    },
    location: {
      heading: "Field Location & Local Weather",
      subtitle: "Pinpoint your farm field to fetch real-time local weather forecasts for your crops.",
      ap_presets: "Andhra Pradesh Hubs",
      search_bar: "Live Mandal / Town Search",
      google_maps_pin: "Google Maps Pin / Link",
      detect_gps: "Detect Current Field GPS",
      detecting_gps: "Locating via GPS...",
      search_placeholder: "Search town or mandal (e.g. Tenali, Nandyal, Mangalagiri)...",
      search_button: "Search",
      apply_pin: "Apply Pin",
      selected_location: "Active Field Location:",
      lat_lon_label: "GPS Coordinates:",
      maps_placeholder: "Paste Google Maps share link (maps.app.goo.gl/...) or '16.3067, 80.4365'"
    },
    results: {
      heading: "Leaf Health Check & Disease Results",
      health_status: "Plant Condition:",
      severity_measured: "Leaf Area Damaged:",
      conformal_interval: "Expected Damage Range:",
      lesion_spots: "Disease Spots Found:",
      foliar_overlay: "Disease Spot Outlines",
      pathology_breakdown: "Disease Tissue Breakdown:",
      necrotic_core: "Dead Brown Spots (Center):",
      chlorotic_halo: "Spreading Yellow Border:",
      healthy_tissue: "Healthy Green Leaf Area:",
      spatial_projection_7d: "Predicted 7-Day Disease Spread",
      spatial_projection_desc: "Shows where disease spots will spread in 7 days if left untreated, based on upcoming rain and humidity.",
      diagnostic_saliency: "AI Inspection Focus Areas",
      saliency_desc: "Shows the exact areas on the leaf that the AI examined to diagnose the disease."
    },
    weather_table: {
      heading: "7-Day Local Weather Forecast & Spraying Advice",
      subtitle: "Live 7-day weather forecast for your field location to find the safest days for spraying.",
      day: "Day / Date",
      temperature: "Mean Temp (°C)",
      relative_humidity: "Mean RH (%)",
      vpd: "VPD (kPa)",
      leaf_wetness: "Hours Leaves Wet",
      rain_duration: "Rainfall (mm)",
      wind_speed: "Wind (km/h)",
      spray_feasibility: "Safe to Spray?",
      infection_risk: "Infection Risk"
    },
    spray_status: {
      favorable: "Favorable",
      marginal: "Marginal",
      unfavorable: "Unfavorable"
    },
    health_card: {
      download_slip: "Download Crop Health Report (PDF)",
      card_title: "Rythu Field Crop Health Card",
      rbk_header: "Agricultural Extension & Crop Diagnostics Advisory",
      diagnostic_id: "Diagnostic ID:",
      scan_date: "Evaluation Date & Time:",
      crop_details: "Crop & Variety:",
      field_coordinates: "Field Location & Coordinates:",
      field_region_label: "Field Region:",
      severity_rating: "Measured Disease Severity:",
      severity_status_healthy: "Healthy Crop (No Disease Spots)",
      severity_status_mild: "Early / Mild Spots",
      severity_status_moderate: "Moderate Disease Spots",
      severity_status_severe: "Severe Disease Damage",
      necrotic_core_label: "Dead Brown Spots:",
      chlorotic_halo_label: "Spreading Yellow Border:",
      active_spots_label: "Disease Spots:",
      spots_unit: "spots",
      original_photo_title: "Original Leaf Photo",
      pathology_contours_title: "Disease Spot Outlines",
      conformal_range_label: "Expected Damage Range:",
      no_mask_placeholder: "Disease outlines not generated",
      microclimate_outlook: "7-Day Weather Outlook:",
      spray_window_advisory: "Today's Spraying Advice:",
      ipm_checklist_title: "Practical Steps to Stop Disease Spread:",
      preview_action: "Preview On-Screen Card",
      print_action: "Download Crop Health Report (PDF)",
      close_action: "Close"
    },
    ipm_advisory: {
      heading: "Practical Field Care & Disease Management",
      subtitle: "Simple, practical steps farmers can take immediately to stop disease from spreading.",
      pillar_canopy: "Water Only in Early Morning",
      pillar_canopy_desc: "Water your crops early in the morning (6:00 AM to 9:00 AM) so the morning sun quickly dries the leaves. Never splash water on leaves in the evening, because leaves that stay wet all night allow fungal disease to grow rapidly.",
      pillar_sanitation: "Clean Up and Remove Bad Leaves",
      pillar_sanitation_desc: "Pluck off leaves with heavy disease spots near the soil and throw them away outside the field. This prevents rain splashes from bouncing fungal disease up from the ground onto clean green leaves.",
      pillar_nutrition: "Balanced Fertilizers (Limit Excess Urea)",
      pillar_nutrition_desc: "Do not apply heavy doses of urea right now. Too much nitrogen makes leaves soft and weak, making it easy for disease to enter. Apply potash and calcium to make leaves tough and strong.",
      pillar_biological: "Safe Biological Protection & Expert Advice",
      pillar_biological_desc: "Spray safe natural plant protectors (such as Trichoderma or Bacillus) on calm, sunny mornings. For chemical medicines, always consult your local Rythu Bharosa Kendra (RBK) or Krishi Vigyan Kendra (KVK).",
      disclaimer_text: "Notice: This system gives guidance based on plant photographs and local weather. It does not sell or prescribe specific chemical pesticide brands or dosages. Farmers should check with local agriculture officers (RBK / KVK) before buying chemical sprays."
    }
  },

  te: {
    app_title: "పంట తెగుళ్ల తీవ్రత మరియు నివారణ వ్యవస్థ",
    app_subtitle: "పంట ఆకు ఆరోగ్య పరీక్ష, 7 రోజుల స్థానిక వాతావరణం మరియు పొలం సంరక్షణ మార్గదర్శి",
    select_language: "భాష",
    tabs: {
      analyze: "ఆకు ఆరోగ్య పరీక్ష",
      how_it_works: "ఇది ఎలా పనిచేస్తుంది",
      about: "సమాచారం & సూచనలు"
    },
    upload: {
      drag_drop_title: "ఆకు ఫోటోను ఇక్కడ అప్‌లోడ్ చేయండి లేదా క్లిక్ చేయండి",
      drag_drop_subtitle: "స్పష్టమైన సింగిల్ ఆకు ఫోటోను సమర్పించండి",
      supported_formats: "PNG, JPG, WEBP (గరిష్టంగా 10MB)",
      change_photo: "ఫోటో మార్చండి",
      crop_selector: "పంట రకాన్ని ఎంచుకోండి:",
      cultivar_label: "రకం / విత్తన రకం (ఐచ్ఛికం):",
      cultivar_placeholder: "ఉదాహరణ: BPT 5204 (సాంబ మసూరి)",
      heading_days_label: "విత్తిన రోజుల సంఖ్య (ఐచ్ఛికం):",
      heading_days_placeholder: "ఉదాహరణ: 45",
      analyze_button: "ఆకు ఆరోగ్యం & వాతావరణ సూచన చూడండి",
      analyzing_progress: "ఆకు చిత్రాన్ని పరిశీలిస్తోంది & వాతావరణ సమాచారం సేకరిస్తోంది..."
    },
    location: {
      heading: "పొలం స్థానం & స్థానిక వాతావరణం",
      subtitle: "ఖచ్చితమైన 9 కి.మీ స్థానిక వాతావరణ సూచనను పొందడానికి మీ పొలం స్థానాన్ని ఎంచుకోండి.",
      ap_presets: "ఆంధ్రప్రదేశ్ ముఖ్య కేంద్రాలు",
      search_bar: "మండలం / పట్టణం శోధన",
      google_maps_pin: "గూగుల్ మ్యాప్స్ లింక్ / పిన్",
      detect_gps: "ప్రస్తుత పొలం GPS స్థానాన్ని గుర్తించు",
      detecting_gps: "GPS స్థానాన్ని గుర్తిస్తోంది...",
      search_placeholder: "మండలం లేదా పట్టణం పేరు నమోదు చేయండి (ఉదా: తెనాలి, నంద్యాల, మంగళగిరి)...",
      search_button: "శోధించండి",
      apply_pin: "పిన్ జోడించండి",
      selected_location: "ఎంచుకున్న పొలం ప్రాంతం:",
      lat_lon_label: "GPS అక్షాంశ / రేఖాంశాలు:",
      maps_placeholder: "గూగుల్ మ్యాప్స్ లింక్ లేదా '16.3067, 80.4365' నమోదు చేయండి"
    },
    results: {
      heading: "ఆకు ఆరోగ్య ఫలితాలు & తెగులు వివరాలు",
      health_status: "పంట స్థితి:",
      severity_measured: "దెబ్బతిన్న ఆకు విస్తీర్ణం:",
      conformal_interval: "అంచనా నష్టం పరిధి:",
      lesion_spots: "కనిపించిన తెగులు మచ్చలు:",
      foliar_overlay: "తెగులు మచ్చల సరిహద్దులు",
      pathology_breakdown: "తెగులు నష్టం వర్గీకరణ:",
      necrotic_core: "ఎండిన గోధుమ రంగు భాగం:",
      chlorotic_halo: "చుట్టూ వ్యాపిస్తున్న పసుపు అంచు:",
      healthy_tissue: "ఆరోగ్యకరమైన ఆకు విస్తీర్ణం:",
      spatial_projection_7d: "7 రోజుల్లో తెగులు వ్యాప్తి అంచనా",
      spatial_projection_desc: "రాబోయే వాతావరణం ఆధారంగా 7 రోజుల్లో తెగులు ఎంతవరకు వ్యాపిస్తుందో చూపుతుంది.",
      diagnostic_saliency: "ఏఐ పరిశీలించిన ముఖ్య భాగాలు",
      saliency_desc: "రోగ నిర్ధారణ చేయడానికి ఏఐ మోడల్ ఆకులో ప్రత్యేకంగా పరిశీలించిన భాగాలను చూపుతుంది."
    },
    weather_table: {
      heading: "7 రోజుల స్థానిక వాతావరణం & పిచికారీ సలహా",
      subtitle: "మీ పొలానికి రాబోయే 7 రోజుల వాతావరణ సమాచారం ఆధారంగా మందులు ఎప్పుడు చల్లాలో తెలుసుకోండి.",
      day: "రోజు / తేదీ",
      temperature: "ఉష్ణోగ్రత (°C)",
      relative_humidity: "గాలిలో తేమ (%)",
      vpd: "బాష్పీభవన లోటు (VPD)",
      leaf_wetness: "ఆకులు తడిగా ఉండే గంటలు",
      rain_duration: "వర్షపాతం (mm)",
      wind_speed: "గాలి వేగం (km/h)",
      spray_feasibility: "పిచికారీ సురక్షితమేనా?",
      infection_risk: "తెగులు ముప్పు"
    },
    spray_status: {
      favorable: "అనుకూలం",
      marginal: "జాగ్రత్త అవసరం",
      unfavorable: "పిచికారీ వద్దు"
    },
    health_card: {
      download_slip: "పంట ఆరోగ్య నివేదిక డౌన్‌లోడ్ (PDF)",
      card_title: "రైతు పొలం పంట ఆరోగ్య పత్రం (Health Card)",
      rbk_header: "రైతు భరోసా కేంద్రం (RBK) వ్యవసాయ సలహా విభాగం",
      diagnostic_id: "నిర్ధారణ సంఖ్య:",
      scan_date: "పరీక్షించిన తేదీ & సమయం:",
      crop_details: "పంట & రకం వివరాలు:",
      field_coordinates: "పొలం స్థానం & కోఆర్డినేట్స్:",
      field_region_label: "పొలం ప్రాంతం:",
      severity_rating: "కొలిచిన వ్యాధి తీవ్రత శాతం:",
      severity_status_healthy: "ఆరోగ్యకరమైన పంట (తెగులు లేదు)",
      severity_status_mild: "ప్రారంభ / తక్కువ మచ్చలు",
      severity_status_moderate: "మధ్యస్థ తెగులు మచ్చలు",
      severity_status_severe: "తీవ్రమైన తెగులు నష్టం",
      necrotic_core_label: "ఎండిన గోధుమ మచ్చలు:",
      chlorotic_halo_label: "వ్యాపిస్తున్న పసుపు అంచు:",
      active_spots_label: "తెగులు మచ్చల సంఖ్య:",
      spots_unit: "మచ్చలు",
      original_photo_title: "అసలు ఆకు ఫోటో",
      pathology_contours_title: "తెగులు మచ్చల సరిహద్దు పటం",
      conformal_range_label: "అంచనా నష్టం పరిధి:",
      no_mask_placeholder: "తెగులు సరిహద్దు పటం అందుబాటులో లేదు",
      microclimate_outlook: "7 రోజుల వాతావరణ సూచన:",
      spray_window_advisory: "నేటి పిచికారీ సలహా:",
      ipm_checklist_title: "తెగులు ఆగడానికి ఆచరించవలసిన పద్ధతులు:",
      preview_action: "కార్డ్ ప్రివ్యూ చూడండి",
      print_action: "పంట ఆరోగ్య నివేదిక డౌన్‌లోడ్ (PDF)",
      close_action: "మూసివేయి"
    },
    ipm_advisory: {
      heading: "పొలంలో తెగులు వ్యాపించకుండా ఆపే సులభ పద్ధతులు",
      subtitle: "రైతులు వెంటనే పాటించగల ముఖ్యమైన క్షేత్రస్థాయి జాగ్రత్తలు.",
      pillar_canopy: "ఉదయం పూట మాత్రమే నీరు పెట్టండి",
      pillar_canopy_desc: "నీటి తడులను తెల్లవారుజామున (ఉదయం 6 నుండి 9 గంటల లోపు) మాత్రమే ఇవ్వండి. సాయంత్రం వేళల్లో ఆకులపై నీరు చల్లవద్దు; రాత్రంతా ఆకులు తడిగా ఉంటే తెగులు వేగంగా పెరుగుతుంది.",
      pillar_sanitation: "పాడైన ఆకులను తుంచి బయట పారవేయండి",
      pillar_sanitation_desc: "నేల దగ్గర ఎక్కువగా మచ్చలు ఉన్న ఆకులను తుంచి పొలం బయట పారవేయండి. దీనివల్ల వర్షపు తుంపర్ల ద్వారా తెగులు పై ఆకులకు పాకకుండా ఉంటుంది.",
      pillar_nutrition: "సమతుల్య ఎరువులు (యూరియా వాడకం తగ్గించండి)",
      pillar_nutrition_desc: "ప్రస్తుతానికి యూరియా అధికంగా వాడవద్దు. ఎక్కువ నత్రజని ఆకులను మెత్తగా మార్చి తెగులు సులభంగా లోపలికి వెళ్లేలా చేస్తుంది. ఆకులు గట్టిగా ఉండటానికి పొటాష్ మరియు కాల్షియం వాడండి.",
      pillar_biological: "సురక్షిత సహజ మందులు & ఆర్బీకే సలహా",
      pillar_biological_desc: "ఎండగా, ప్రశాంతంగా ఉన్న ఉదయం పూట ట్రైకోడెర్మా లేదా బాసిల్లస్ వంటి సహజ జీవ మందులను పిచికారీ చేయండి. రసాయన మందుల కోసం మీ గ్రామ రైతు భరోసా కేంద్రం (RBK) లేదా KVK అధికారులను సంప్రదించండి.",
      disclaimer_text: "గమనిక: ఈ నివేదిక ఆకు ఫోటో మరియు స్థానిక వాతావరణం ఆధారంగా మార్గదర్శకత్వాన్ని మాత్రమే అందిస్తుంది. ఏ విధమైన రసాయన మందుల బ్రాండ్లను అమ్మదు లేదా సూచించదు. రసాయన మందుల కొనుగోలుకు ముందు స్థానిక వ్యవసాయ అధికారి (RBK / KVK)ని సంప్రదించండి."
    }
  },

  hi: {
    app_title: "फसल रोग गंभीरता एवं पूर्वानुमान प्रणाली",
    app_subtitle: "फसल पत्ती स्वास्थ्य जांच, 7-दिवसीय स्थानीय मौसम एवं खेत देखभाल गाइड",
    select_language: "भाषा",
    tabs: {
      analyze: "पत्ती स्वास्थ्य जांच",
      how_it_works: "यह कैसे काम करता है",
      about: "सुरक्षा एवं जानकारी"
    },
    upload: {
      drag_drop_title: "पत्ती की तस्वीर यहाँ अपलोड करें या क्लिक करें",
      drag_drop_subtitle: "एकल पत्ती की स्पष्ट उच्च-रिज़ॉल्यूशन फोटो सबमिट करें",
      supported_formats: "PNG, JPG, WEBP (अधिकतम 10MB)",
      change_photo: "तस्वीर बदलें",
      crop_selector: "फसल चुनें:",
      cultivar_label: "किस्म / प्रजाति (वैकल्पिक):",
      cultivar_placeholder: "उदा: BPT 5204 (सांबा महसूरी)",
      heading_days_label: "बुवाई के बाद के दिन (वैकल्पिक):",
      heading_days_placeholder: "उदा: 45",
      analyze_button: "पत्ती स्वास्थ्य एवं मौसम का हाल देखें",
      analyzing_progress: "पत्ती की तस्वीर की जांच एवं मौसम का डेटा लिया जा रहा है..."
    },
    location: {
      heading: "खेत स्थान एवं स्थानीय मौसम",
      subtitle: "सटीक 9 किमी स्थानीय मौसम पूर्वानुमान प्राप्त करने के लिए अपना खेत स्थान चुनें।",
      ap_presets: "आंध्र प्रदेश कृषि केंद्र",
      search_bar: "मंडल / शहर खोजें",
      google_maps_pin: "गूगल मैप्स लिंक / पिन",
      detect_gps: "वर्तमान खेत का GPS पहचानें",
      detecting_gps: "GPS लोकेशन प्राप्त हो रही है...",
      search_placeholder: "शहर या मंडल का नाम लिखें (उदा: तेनाली, नांदयाल, गुंटूर)...",
      search_button: "खोजें",
      apply_pin: "पिन लागू करें",
      selected_location: "सक्रिय खेत स्थान:",
      lat_lon_label: "GPS निर्देशांक:",
      maps_placeholder: "गूगल मैप्स शेयर लिंक या '16.3067, 80.4365' दर्ज करें"
    },
    results: {
      heading: "पत्ती स्वास्थ्य जांच एवं रोग परिणाम",
      health_status: "फसल की स्थिति:",
      severity_measured: "प्रभावित पत्ती का हिस्सा:",
      conformal_interval: "अनुमानित नुकसान का दायरा:",
      lesion_spots: "पाए गए रोग के धब्बे:",
      foliar_overlay: "रोग के धब्बों की रूपरेखा",
      pathology_breakdown: "रोग क्षति का विवरण:",
      necrotic_core: "सूखे भूरे धब्बे (केंद्र):",
      chlorotic_halo: "फैलता हुआ पीला घेरा:",
      healthy_tissue: "स्वस्थ हरा पत्ती क्षेत्र:",
      spatial_projection_7d: "7 दिनों में रोग का संभावित फैलाव",
      spatial_projection_desc: "आगामी बारिश और नमी के आधार पर 7 दिनों में रोग कहाँ तक फैलेगा, यह दर्शाता है।",
      diagnostic_saliency: "AI द्वारा जांचे गए मुख्य हिस्से",
      saliency_desc: "रोग की पहचान करने के लिए AI मॉडल ने पत्ती के किन हिस्सों को बारीकी से देखा, यह दर्शाता है।"
    },
    weather_table: {
      heading: "7-दिवसीय स्थानीय मौसम एवं छिड़काव सलाह",
      subtitle: "आपके खेत के मौसम के आधार पर दवा छिड़कने के सबसे सुरक्षित दिन जानें।",
      day: "दिन / तारीख",
      temperature: "औसत तापमान (°C)",
      relative_humidity: "आर्द्रता (%)",
      vpd: "हवा का सूखापन (VPD)",
      leaf_wetness: "पत्तियां गीली रहने के घंटे",
      rain_duration: "वर्षा (mm)",
      wind_speed: "हवा की गति (km/h)",
      spray_feasibility: "क्या आज छिड़काव सुरक्षित है?",
      infection_risk: "संक्रमण जोखिम"
    },
    spray_status: {
      favorable: "अनुकूल (सुरक्षित)",
      marginal: "सावधानी रखें",
      unfavorable: "छिड़काव न करें"
    },
    health_card: {
      download_slip: "फसल स्वास्थ्य रिपोर्ट डाउनलोड करें (PDF)",
      card_title: "किसान फसल स्वास्थ्य पर्ची (Health Card)",
      rbk_header: "कृषि विस्तार एवं पादप स्वास्थ्य परामर्श",
      diagnostic_id: "निदान क्रमांक:",
      scan_date: "परीक्षण तिथि एवं समय:",
      crop_details: "फसल एवं किस्म विवरण:",
      field_coordinates: "खेत स्थान एवं निर्देशांक:",
      field_region_label: "खेत / क्षेत्र:",
      severity_rating: "मापी गई रोग गंभीरता प्रतिशत:",
      severity_status_healthy: "स्वस्थ फसल (कोई रोग नहीं)",
      severity_status_mild: "शुरुआती / हल्के धब्बे",
      severity_status_moderate: "मध्यम रोग के धब्बे",
      severity_status_severe: "गंभीर रोग नुकसान",
      necrotic_core_label: "सूखे भूरे धब्बे:",
      chlorotic_halo_label: "फैलता हुआ पीला घेरा:",
      active_spots_label: "सक्रिय धब्बों की संख्या:",
      spots_unit: "धब्बे",
      original_photo_title: "मूल पत्ती की तस्वीर",
      pathology_contours_title: "रोग के धब्बों की रूपरेखा",
      conformal_range_label: "अनुमानित नुकसान का दायरा:",
      no_mask_placeholder: "रोग का नक्शा उपलब्ध नहीं है",
      microclimate_outlook: "7 दिनों का मौसम पूर्वानुमान:",
      spray_window_advisory: "आज छिड़काव की सलाह:",
      ipm_checklist_title: "रोग का फैलाव रोकने के आसान उपाय:",
      preview_action: "कार्ड पूर्वावलोकन देखें",
      print_action: "फसल स्वास्थ्य रिपोर्ट डाउनलोड करें (PDF)",
      close_action: "बंद करें"
    },
    ipm_advisory: {
      heading: "खेत में रोग रोकने के व्यावहारिक उपाय",
      subtitle: "फसल बचाने और बीमारी रोकने के लिए किसान तुरंत ये 4 आदतें अपनाएं।",
      pillar_canopy: "केवल सुबह के समय पानी दें",
      pillar_canopy_desc: "सिंचाई केवल सुबह (सुबह 6 से 9 बजे के बीच) करें ताकि धूप से पत्तियां जल्दी सूख जाएं। शाम के समय पत्तियों पर पानी न डालें, रात भर गीली पत्तियां रहने से फफूंद तेजी से फैलती है।",
      pillar_sanitation: "खराब पत्तियों को तोड़कर खेत से बाहर फेंकें",
      pillar_sanitation_desc: "जमीन के पास जिन पत्तियों पर ज्यादा धब्बे हों, उन्हें तोड़कर खेत से बाहर नष्ट करें। इससे बारिश की बूंदों से बीमारी ऊपर की हरी पत्तियों पर नहीं उछलती।",
      pillar_nutrition: "संतुलित खाद (यूरिया का ज्यादा इस्तेमाल न करें)",
      pillar_nutrition_desc: "अभी बहुत ज्यादा यूरिया न डालें। ज्यादा नाइट्रोजन से पत्तियां कमजोर और मुलायम हो जाती हैं, जिससे बीमारी आसानी से अंदर घुसती है। पत्ती को मजबूत बनाने के लिए पोटाश और कैल्शियम दें।",
      pillar_biological: "सुरक्षित जैविक दवा एवं विशेषज्ञ सलाह",
      pillar_biological_desc: "शांत और धूप वाली सुबह ट्राइकोडर्मा या बैसिलस जैसी सुरक्षित जैविक दवाइयों का छिड़काव करें। रासायनिक दवाओं के लिए अपने नजदीकी कृषि केंद्र या विशेषज्ञ से संपर्क करें।",
      disclaimer_text: "सूचना: यह प्रणाली केवल पत्ती की तस्वीर और मौसम के आधार पर सलाह देती है। यह किसी भी रासायनिक दवा के ब्रांड या खुराक को नहीं बेचती या निर्धारित करती है। रसायनों के उपयोग से पहले स्थानीय कृषि अधिकारी से सलाह लें।"
    }
  }
};
