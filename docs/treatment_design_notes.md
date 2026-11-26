# Treatment Design Notes & Identifiable Contrasts

---

## 1. Reported Experimental Treatment Arms

The ETH Zurich Wheat Leaf dataset includes four plot-level experimental treatment arms evaluated across 15 wheat cultivars:

| Treatment Code | Description | Inoculation Timing | Fungicide Applied? | Plot Replicates |
| :--- | :--- | :--- | :--- | :--- |
| **`FI2`** | Fungicide + Late Inoculation | Late season inoculation | **YES** | Multi-replicate |
| **`0FI1`** | Early Inoculation + No Fungicide | Early season inoculation | **NO** | Multi-replicate |
| **`F0I`** | Fungicide + No Inoculation | Natural infection only | **YES** | Multi-replicate |
| **`0F0I`** | Untreated Control | Natural infection only | **NO** | **Single Replicate** |

---

## 2. Experimental Confounding & Limitations

> [!WARNING]
> **Critical Scientific Limitation**: The ETH wheat field experiment is **NOT** a clean two-arm factorial experiment (Fungicide vs. Control).
> 1. **Confounding with Inoculation Timing**: Inoculation timing differs between the untreated disease arm (`0FI1` - early inoculation) and the fungicide-protected arm (`FI2` - late inoculation).
> 2. **Single Replicate Untreated Control**: The untreated, uninoculated control arm (`0F0I`) contains only a single plot replicate, preventing direct variance estimation for uninoculated control plots.

---

## 3. Identifiable Contrasts for Module 3

To estimate the growth rate reduction attributable to fungicide protection in **Module 3 (Treatment-Scenario Module)**, we construct two identifiable contrasts:

### Contrast A: Protected Inoculated vs. Unprotected Inoculated (`FI2` vs. `0FI1`)
- **Formula**: $\Delta r_{\text{inoculated}} = r_{\text{FI2}} - r_{\text{0FI1}}$
- **Interpretation**: Measures the combined effect of fungicide protection and delayed inoculation relative to early untreated infection.

### Contrast B: Protected Natural vs. Untreated Control (`F0I` vs. `0F0I`)
- **Formula**: $\Delta r_{\text{natural}} = r_{\text{F0I}} - r_{\text{0F0I}}$
- **Interpretation**: Measures fungicide protection under natural infection conditions. (Must account for single-replicate control variance using cluster bootstrapping).

---

## 4. Impact on Project Claims

All project documentation, web interface summaries, and research reports MUST explicitly state:
- Treatment scenario predictions reflect wheat field conditions at a single site in Switzerland.
- Growth rate reduction estimates represent specific management regimes combining fungicide application with timing contrasts, not an isolated single chemical effect.
