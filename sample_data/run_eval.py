import requests, os

test_cases = [
    {
        "file": "01_healthy_wheat_leaf.png",
        "crop": "wheat",
        "scenario": "warm_humid",
        "expected": "Healthy foliage (0% baseline, Quality PASS)"
    },
    {
        "file": "02_mild_septoria_leaf.png",
        "crop": "wheat",
        "scenario": "warm_humid",
        "expected": "Mild early infection (5-15% severity, Quality PASS)"
    },
    {
        "file": "03_moderate_septoria_leaf.png",
        "crop": "wheat",
        "scenario": "cool_dry",
        "expected": "Moderate Septoria (~30-40% severity, full progression curve & treatment reduction)"
    },
    {
        "file": "04_severe_rust_leaf.png",
        "crop": "wheat",
        "scenario": "warm_humid",
        "expected": "Severe confluent lesions (>50% severity, High Severity class)"
    },
    {
        "file": "05_test_blurry_photo.png",
        "crop": "wheat",
        "scenario": "warm_humid",
        "expected": "Safety Quality Gate Rejection (Blur detected)"
    },
    {
        "file": "06_test_glare_photo.png",
        "crop": "wheat",
        "scenario": "warm_humid",
        "expected": "Safety Quality Gate Rejection (Specular glare detected)"
    },
    {
        "file": "real_leaves/set_12_10_obj2.png",
        "crop": "coffee",
        "scenario": "warm_humid",
        "expected": "Multi-crop benchmark (Real Coffee Rust field photo)"
    }
]

print("=" * 80)
print("PLANT DISEASE SEVERITY & PROGNOSIS BENCHMARK TEST")
print("=" * 80)

for tc in test_cases:
    fpath = os.path.join("sample_data", tc["file"])
    if not os.path.exists(fpath):
        continue

    print(f"\n[TEST CASE] {tc['file']}")
    print(f"  Target Goal     : {tc['expected']}")
    print(f"  Crop / Weather  : {tc['crop']} | Scenario: {tc['scenario']}")

    with open(fpath, "rb") as f:
        r = requests.post(
            "http://127.0.0.1:8000/analyze",
            files={"image": f},
            data={"crop": tc["crop"], "weather_scenario": tc["scenario"]}
        )

    if r.status_code != 200:
        print(f"  API Error ({r.status_code}): {r.text}")
        continue

    res = r.json()
    safety = res.get("safety", {})
    sev = res.get("severity")
    fc = res.get("forecast")
    sc = res.get("scenario")

    q_pass = safety.get("quality_passed", False)
    reasons = safety.get("failure_reasons", [])
    print(f"  Quality Passed  : {q_pass} ({'PASS' if q_pass else 'FLAGGED: ' + ', '.join(reasons)})")

    if sev:
        print(f"  Severity Pct    : {sev['severity_pct']:.2f}% (Class: {sev['severity_class']})")
        print(f"  Lesion Spots    : {sev['lesion_count']} spots (Avg size: {sev['mean_lesion_size_px']:.1f} px)")
        print(f"  90% Conformal   : [{sev['severity_interval'][0]:.1f}%, {sev['severity_interval'][1]:.1f}%]")

    if fc and "horizons" in fc:
        d3 = fc["horizons"].get("3d", {}).get("severity_pct", "N/A")
        d7 = fc["horizons"].get("7d", {}).get("severity_pct", "N/A")
        print(f"  Forecast (3d/7d): Day +3: {d3}% | Day +7: {d7}%")

    if sc:
        u7 = sc.get("unprotected_7d_pct", "N/A")
        p7 = sc.get("protected_7d_pct", "N/A")
        red = sc.get("relative_reduction_pct", "N/A")
        print(f"  7d Treatment Sim: Untreated: {u7}% vs Protected: {p7}% (Suppression: {red}%)")

print("\n" + "=" * 80)
print("BENCHMARK COMPLETED SUCCESSFULLY")
print("=" * 80)
