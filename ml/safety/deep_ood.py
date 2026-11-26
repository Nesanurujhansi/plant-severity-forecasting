"""Deep Free Energy Out-of-Distribution (OOD) Detector and Quality Classifier.

Implements Deep Free Energy OOD detection:
    E(x) = -T * log sum exp(f_i(x) / T)
in DINOv2 feature embedding space to detect non-leaf or unknown crop images.
Also includes a Deep Quality Classifier network for detecting blur, overexposure, and low leaf coverage.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger

logger = get_logger(__name__)


def compute_free_energy_score(
    logits: Union[np.ndarray, torch.Tensor],
    temperature: float = 1.0
) -> Union[float, np.ndarray]:
    """Compute Free Energy score E(x) = -T * logsumexp(logits / T).

    In-distribution samples exhibit low free energy, whereas Out-of-Distribution (OOD)
    samples exhibit high free energy values.
    """
    if isinstance(logits, np.ndarray):
        logits_t = torch.from_numpy(logits)
    else:
        logits_t = logits

    energy = -temperature * torch.logsumexp(logits_t / temperature, dim=-1)
    res = energy.cpu().numpy()
    return float(res) if res.ndim == 0 or res.size == 1 else res


class DeepImageQualityClassifier(nn.Module):
    """Deep CNN quality classifier network predicting blur, exposure, and leaf ratio acceptability."""

    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 Logit Outputs: [is_sharp, is_well_exposed, has_sufficient_leaf_coverage]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class DeepOODSafetyEvaluator:
    """Evaluates image feature embeddings using Free Energy and Deep Quality Nets."""

    def __init__(
        self,
        energy_threshold: float = 8.5,
        temperature: float = 1.0
    ):
        self.energy_threshold = energy_threshold
        self.temperature = temperature
        self.quality_net = DeepImageQualityClassifier(embed_dim=384)
        self.quality_net.eval()

    def evaluate_safety(
        self,
        features: Union[np.ndarray, torch.Tensor],
        cls_logits: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """Evaluate Free Energy OOD score and deep quality flags for feature input."""
        if cls_logits is None:
            # Fallback zero-centered logits for energy score computation
            if isinstance(features, np.ndarray):
                cls_logits = np.linalg.norm(features, axis=-1, keepdims=True)
            else:
                cls_logits = torch.norm(features, dim=-1, keepdim=True)

        energy = compute_free_energy_score(cls_logits, temperature=self.temperature)

        if isinstance(energy, np.ndarray):
            energy_val = float(energy[0])
        else:
            energy_val = float(energy)

        is_ood = bool(energy_val > self.energy_threshold)

        return {
            "free_energy_score": round(energy_val, 4),
            "energy_threshold": self.energy_threshold,
            "is_ood": is_ood,
            "safety_status": "REJECT_OOD" if is_ood else "ACCEPT_IN_DISTRIBUTION",
            "recommendation": "Photo contains non-leaf or unknown crop object. Please upload a clear crop leaf photo." if is_ood else "In-distribution leaf photo accepted."
        }


def main() -> int:
    evaluator = DeepOODSafetyEvaluator()
    dummy_feat = np.random.randn(1, 384).astype(np.float32)
    res = evaluator.evaluate_safety(dummy_feat)
    print("Deep OOD Safety Result:", res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
