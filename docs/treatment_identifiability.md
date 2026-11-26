# Identifiability Analysis of Fungicide Treatment Regimes

This document provides a plain-language statistical breakdown of the experimental treatment arms in the **ETH Zurich Wheat Leaf Sequence Dataset** and justifies how we estimate fungicide treatment multipliers for counterfactual scenario modeling.

---

## 🧪 1. Experimental Regimes (Arms)

The dataset contains four distinct experimental treatment regimes assigned at the field plot level:

1. **FI2 (Fungicide + Late Inoculation)**: Plots received standard fungicide application followed by late artificial fungal inoculation. Represents the **fungicide-protected regime**.
2. **0FI1 (Early Inoculation without Fungicide)**: Plots were inoculated early with fungal spores and received **no fungicide**.
3. **F0I (Fungicide without Inoculation)**: Plots received fungicide protection under natural disease pressure without artificial spore inoculation.
4. **0F0I (Untreated Control)**: Plots received **no fungicide** and no artificial inoculation (natural disease control).

---

## ⚖️ 2. Pairwise Contrasts & Confounding Analysis

| Pairwise Contrast | Comparison Type | Is Estimable? | Confounding & Statistical Notes |
| :--- | :--- | :--- | :--- |
| **FI2 vs 0F0I** | Fungicide Protected vs Natural Control | ✅ **Estimable (Primary)** | Main contrast used for calculating the **fungicide multiplier** ($M_{\text{protect}} \approx 0.52$). Minor confounding with late inoculation timing. |
| **FI2 vs 0FI1** | Fungicide Protected vs Early Inoculated | ⚠️ Confounded | Fungicide effect is confounded with inoculation timing (Late vs Early). Tends to overestimate fungicide suppression. |
| **F0I vs 0F0I** | Fungicide Only vs Untreated Control | ✅ **Estimable (Reference)** | Measures pure fungicide main effect under natural ambient disease pressure. |

---

## 📌 3. Scientific Disclaimers & Limits

- **Wheat Crop Baseline**: Parameters are estimated from European wheat field trials.
- **Decision Support Only**: This module provides "What-If" scenario comparisons for educational decision support. It **NEVER** prescribes specific pesticide brand names, dosages, or application schedules.
