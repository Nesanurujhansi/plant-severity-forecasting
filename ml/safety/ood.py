"""Out-of-Distribution (OOD) Detection Module for crop-prognosis.

Implements three complementary OOD detection mechanisms in DINOv2 feature space (384-d):
1. Mahalanobis Distance: (x - mu)^T * Sigma^{-1} * (x - mu)
2. kNN Distance: Average Euclidean distance to k-nearest training centroids
3. Deep Free Energy Score: E(x) = -T * log sum exp(f_i(x) / T)

Classifies input photos into:
- IN_DISTRIBUTION: Safe, valid crop leaf photo
- BORDERLINE: Mild distribution shift (between 95th and 99th percentile)
- OUT_OF_DISTRIBUTION: Rejection, non-leaf or unknown object
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MahalanobisDistanceClassifier:
    """Computes Mahalanobis distance in DINOv2 embedding space against target centroid/precision matrix."""

    def __init__(
        self,
        mean_vector: Optional[np.ndarray] = None,
        precision_matrix: Optional[np.ndarray] = None,
        feature_dim: int = 384,
        p95_threshold: float = 18.5,
        p99_threshold: float = 28.2
    ):
        self.feature_dim = feature_dim
        self.p95_threshold = p95_threshold
        self.p99_threshold = p99_threshold

        if mean_vector is None:
            self.mean = np.zeros(feature_dim, dtype=np.float32)
        else:
            self.mean = np.asarray(mean_vector, dtype=np.float32)

        if precision_matrix is None:
            self.precision = np.eye(feature_dim, dtype=np.float32)
        else:
            self.precision = np.asarray(precision_matrix, dtype=np.float32)

    def compute_distance(self, feature: np.ndarray) -> float:
        """Calculate Mahalanobis distance sqrt((x - mu)^T * Sigma^{-1} * (x - mu))."""
        diff = (feature - self.mean).reshape(-1)
        dist_sq = float(np.dot(np.dot(diff, self.precision), diff))
        return float(np.sqrt(max(0.0, dist_sq)))

    def predict(self, feature: np.ndarray) -> Dict[str, Any]:
        dist = self.compute_distance(feature)
        if dist <= self.p95_threshold:
            status = "IN_DISTRIBUTION"
        elif dist <= self.p99_threshold:
            status = "BORDERLINE"
        else:
            status = "OUT_OF_DISTRIBUTION"

        return {
            "mahalanobis_distance": round(dist, 4),
            "p95_threshold": self.p95_threshold,
            "p99_threshold": self.p99_threshold,
            "status": status
        }


class KNNDistanceClassifier:
    """Computes k-Nearest Neighbor Euclidean distance in feature embedding space."""

    def __init__(
        self,
        reference_embeddings: Optional[np.ndarray] = None,
        k: int = 5,
        p95_threshold: float = 1.25,
        p99_threshold: float = 2.10,
        feature_dim: int = 384
    ):
        self.k = k
        self.p95_threshold = p95_threshold
        self.p99_threshold = p99_threshold

        if reference_embeddings is None:
            # Seed 5 synthetic centroids for stub initialization
            np.random.seed(42)
            self.refs = np.random.randn(20, feature_dim).astype(np.float32)
        else:
            self.refs = np.asarray(reference_embeddings, dtype=np.float32)

    def compute_knn_distance(self, feature: np.ndarray) -> float:
        """Compute average distance to k-nearest reference centroids."""
        feat = feature.reshape(1, -1)
        dists = np.linalg.norm(self.refs - feat, axis=1)
        top_k = np.sort(dists)[:self.k]
        return float(np.mean(top_k))

    def predict(self, feature: np.ndarray) -> Dict[str, Any]:
        dist = self.compute_knn_distance(feature)
        if dist <= self.p95_threshold:
            status = "IN_DISTRIBUTION"
        elif dist <= self.p99_threshold:
            status = "BORDERLINE"
        else:
            status = "OUT_OF_DISTRIBUTION"

        return {
            "knn_distance": round(dist, 4),
            "k": self.k,
            "p95_threshold": self.p95_threshold,
            "p99_threshold": self.p99_threshold,
            "status": status
        }


class FreeEnergyClassifier:
    """Computes Deep Free Energy score E(x) = -T * log sum exp(logits / T)."""

    def __init__(
        self,
        temperature: float = 1.0,
        energy_threshold: float = 8.5
    ):
        self.temperature = temperature
        self.energy_threshold = energy_threshold

    def compute_energy(self, logits: np.ndarray) -> float:
        """Compute Free Energy scalar."""
        l = logits.reshape(-1) / self.temperature
        max_l = np.max(l)
        energy = -self.temperature * (max_l + np.log(np.sum(np.exp(l - max_l))))
        return float(energy)

    def predict(self, feature: np.ndarray, logits: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if logits is None:
            # Construct synthetic logit norm proxy
            l = np.array([np.linalg.norm(feature), -np.linalg.norm(feature) * 0.5])
        else:
            l = logits

        energy = self.compute_energy(l)
        is_ood = bool(energy > self.energy_threshold)
        return {
            "free_energy_score": round(energy, 4),
            "energy_threshold": self.energy_threshold,
            "temperature": self.temperature,
            "status": "OUT_OF_DISTRIBUTION" if is_ood else "IN_DISTRIBUTION"
        }


class DeepOODSafetyEvaluator:
    """Ensemble OOD Safety Evaluator combining Mahalanobis, kNN, and Free Energy scores."""

    def __init__(
        self,
        mahalanobis_clf: Optional[MahalanobisDistanceClassifier] = None,
        knn_clf: Optional[KNNDistanceClassifier] = None,
        free_energy_clf: Optional[FreeEnergyClassifier] = None
    ):
        self.mahalanobis = mahalanobis_clf or MahalanobisDistanceClassifier()
        self.knn = knn_clf or KNNDistanceClassifier()
        self.free_energy = free_energy_clf or FreeEnergyClassifier()

    def evaluate_safety(
        self,
        feature_embedding: np.ndarray,
        cls_logits: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Evaluate OOD status across all three classifiers."""
        feat = feature_embedding.astype(np.float32)

        mah_res = self.mahalanobis.predict(feat)
        knn_res = self.knn.predict(feat)
        fe_res = self.free_energy.predict(feat, logits=cls_logits)

        # Conservative consensus policy: OOD if any primary metric flags OOD
        if fe_res["status"] == "OUT_OF_DISTRIBUTION" or mah_res["status"] == "OUT_OF_DISTRIBUTION" or knn_res["status"] == "OUT_OF_DISTRIBUTION":
            overall_status = "OUT_OF_DISTRIBUTION"
            recommendation = "The uploaded photo appears to be a non-leaf object, an unsupported crop, or heavily degraded. Please upload a clear photo of a crop leaf."
        elif mah_res["status"] == "BORDERLINE" or knn_res["status"] == "BORDERLINE":
            overall_status = "BORDERLINE"
            recommendation = "Photo exhibits mild visual distribution shift. Results are provided with a lower confidence tier."
        else:
            overall_status = "IN_DISTRIBUTION"
            recommendation = "Photo verified as an in-distribution crop leaf photo."

        return {
            "overall_ood_status": overall_status,
            "is_ood": bool(overall_status == "OUT_OF_DISTRIBUTION"),
            "is_borderline": bool(overall_status == "BORDERLINE"),
            "recommendation": recommendation,
            "metrics": {
                "mahalanobis": mah_res,
                "knn": knn_res,
                "free_energy": fe_res
            }
        }


def main() -> int:
    evaluator = DeepOODSafetyEvaluator()
    dummy_feat = np.random.randn(384).astype(np.float32)
    res = evaluator.evaluate_safety(dummy_feat)
    print("OOD Evaluation Result:", res["overall_ood_status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
