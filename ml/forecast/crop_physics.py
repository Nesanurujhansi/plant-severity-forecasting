"""Crop Thermal Physics & Microclimate Species Profile Engine.

Provides species-specific thermal cardinal parameters (T_min, T_opt, T_max),
humidity response parameters, and VPD sensitivity profiles for major crops:
1. Wheat (Septoria tritici blotch, Rust)
2. Rice / Paddy (Rice Blast, Brown Spot)
3. Tomato (Late Blight, Early Blight)
4. Corn / Maize (Northern Corn Leaf Blight, Gray Spot)
5. Sugarcane (Red Rot, Leaf Scald)
6. Cotton (Bacterial Blight, Leaf Curl)
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class CropThermalProfile:
    """Dataclass encapsulating species thermal and humidity microclimate parameters."""

    crop_name: str
    target_diseases: List[str]
    t_min: float  # Minimum growth temperature (°C)
    t_opt: float  # Optimum growth temperature (°C)
    t_max: float  # Maximum growth temperature (°C)
    k_h: float    # Humidity response steepness
    rh_50: float  # RH 50% inflection point (%)
    r0_base: float # Base daily growth rate (1/day)
    max_capacity: float = 0.95

    def yan_hunt_thermal_response(self, T: float) -> float:
        """Compute Yan & Hunt Beta-type asymmetric thermal response g_T(T) in [0, 1]."""
        if T <= self.t_min or T >= self.t_max:
            return 0.0

        num = (T - self.t_min) * (self.t_max - T) ** ((self.t_max - self.t_opt) / (self.t_opt - self.t_min))
        denom = (self.t_opt - self.t_min) * (self.t_max - self.t_opt) ** ((self.t_max - self.t_opt) / (self.t_opt - self.t_min))

        if denom <= 0:
            return 0.0

        val = num / denom
        return float(np.clip(val, 0.0, 1.0))

    def to_dict(self) -> Dict[str, Any]:
        """Convert thermal profile to dictionary."""
        return {
            "crop_name": self.crop_name,
            "target_diseases": self.target_diseases,
            "t_min": self.t_min,
            "t_opt": self.t_opt,
            "t_max": self.t_max,
            "k_h": self.k_h,
            "rh_50": self.rh_50,
            "r0_base": self.r0_base,
            "max_capacity": self.max_capacity,
        }


# Species-specific cardinal thermal profiles
CROP_PHYSICS_REGISTRY: Dict[str, CropThermalProfile] = {
    "wheat": CropThermalProfile(
        crop_name="Wheat",
        target_diseases=["Septoria Tritici Blotch", "Stripe Rust", "Leaf Rust"],
        t_min=4.0,
        t_opt=18.5,
        t_max=30.0,
        k_h=2.2,
        rh_50=60.0,
        r0_base=0.25,
    ),
    "rice": CropThermalProfile(
        crop_name="Rice / Paddy",
        target_diseases=["Rice Leaf Blast", "Brown Spot", "Bacterial Blight", "Sheath Blight"],
        t_min=15.0,
        t_opt=26.0,
        t_max=36.0,
        k_h=2.8,
        rh_50=75.0,
        r0_base=0.32,
    ),
    "tomato": CropThermalProfile(
        crop_name="Tomato",
        target_diseases=["Late Blight", "Early Blight", "Leaf Mold"],
        t_min=10.0,
        t_opt=21.0,
        t_max=30.0,
        k_h=2.5,
        rh_50=70.0,
        r0_base=0.28,
    ),
    "corn": CropThermalProfile(
        crop_name="Corn / Maize",
        target_diseases=["Northern Corn Leaf Blight", "Gray Leaf Spot", "Common Rust"],
        t_min=12.0,
        t_opt=24.0,
        t_max=34.0,
        k_h=2.3,
        rh_50=65.0,
        r0_base=0.27,
    ),
    "sugarcane": CropThermalProfile(
        crop_name="Sugarcane",
        target_diseases=["Red Rot", "Leaf Scald", "Mosaic Virus"],
        t_min=16.0,
        t_opt=28.0,
        t_max=38.0,
        k_h=3.0,
        rh_50=72.0,
        r0_base=0.30,
    ),
    "cotton": CropThermalProfile(
        crop_name="Cotton",
        target_diseases=["Bacterial Blight", "Leaf Curl Virus", "Target Spot"],
        t_min=14.0,
        t_opt=27.0,
        t_max=37.0,
        k_h=2.6,
        rh_50=68.0,
        r0_base=0.29,
    ),
}


def get_crop_thermal_profile(crop_key: str) -> CropThermalProfile:
    """Retrieve crop thermal profile by key (e.g., 'rice', 'wheat', 'tomato')."""
    key_clean = crop_key.lower().strip()
    if key_clean not in CROP_PHYSICS_REGISTRY:
        logger.warning(f"Crop '{crop_key}' not found in registry. Defaulting to 'wheat'.")
        return CROP_PHYSICS_REGISTRY["wheat"]
    return CROP_PHYSICS_REGISTRY[key_clean]
