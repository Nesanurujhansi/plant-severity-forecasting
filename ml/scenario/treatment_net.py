"""Deep Counterfactual Treatment Network (TarNet / DeepNME).

Implements a Deep Counterfactual Neural Network architecture with a shared representation trunk
and specialized treatment branch heads to predict disease progression under untreated (0F0I)
vs fungicide-protected (FI2/F0I) regimes.
Includes Monte Carlo Dropout for epistemic treatment uncertainty bounds.
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


class SharedRepresentationTrunk(nn.Module):
    """Shared Multi-Layer Perceptron trunk mapping plant features to latent representation space."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, dropout_rate: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TreatmentBranchHead(nn.Module):
    """Specialized branch head predicting differential growth rate delta_r for a specific regime."""

    def __init__(self, hidden_dim: int = 64, dropout_rate: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(32, 1)
        )

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return self.head(phi)


class DeepTreatmentTarNet(nn.Module):
    """Deep Counterfactual Treatment Network (TarNet) with dual outcome heads."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, dropout_rate: float = 0.1):
        super().__init__()
        self.trunk = SharedRepresentationTrunk(input_dim=input_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.head_unprotected = TreatmentBranchHead(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.head_protected = TreatmentBranchHead(hidden_dim=hidden_dim, dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (pred_unprotected_delta_r, pred_protected_delta_r)."""
        phi = self.trunk(x)
        delta_r_unprotected = self.head_unprotected(phi)
        delta_r_protected = self.head_protected(phi)
        return delta_r_unprotected, delta_r_protected

    @torch.no_grad()
    def predict_counterfactuals_mc_dropout(
        self,
        x: torch.Tensor,
        n_mc_samples: int = 30
    ) -> Dict[str, np.ndarray]:
        """Run Monte Carlo Dropout forward passes to compute counterfactual trajectory means and uncertainty bounds."""
        self.train()  # Keep dropout active for MC sampling

        unprot_samples = []
        prot_samples = []

        for _ in range(n_mc_samples):
            u_pred, p_pred = self.forward(x)
            unprot_samples.append(u_pred.cpu().numpy())
            prot_samples.append(p_pred.cpu().numpy())

        unprot_samples = np.concatenate(unprot_samples, axis=1)  # (batch_size, n_mc_samples)
        prot_samples = np.concatenate(prot_samples, axis=1)

        u_mean = np.mean(unprot_samples, axis=1)
        u_std = np.std(unprot_samples, axis=1)

        p_mean = np.mean(prot_samples, axis=1)
        p_std = np.std(prot_samples, axis=1)

        return {
            "unprotected_mean_delta_r": u_mean,
            "unprotected_std": u_std,
            "protected_mean_delta_r": p_mean,
            "protected_std": p_std,
            "net_treatment_reduction": u_mean - p_mean
        }


def main() -> int:
    net = DeepTreatmentTarNet(input_dim=16)
    x_dummy = torch.randn(4, 16)
    u_pred, p_pred = net(x_dummy)
    print("Deep Treatment TarNet Output Shape:", u_pred.shape, p_pred.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
