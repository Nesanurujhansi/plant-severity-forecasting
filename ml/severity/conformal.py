"""Mondrian Split Conformal Prediction for Severity Prediction Intervals.

Computes 90% non-parametric prediction intervals grouped by predicted-severity ordinal bins
using Mondrian split conformal prediction, supporting bin merging for sparse calibration classes,
interval clipping to [0, 1], empirical coverage evaluation, and export to conformal.json.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.logging_utils import get_logger
from ml.severity.severity import DEFAULT_BINS, map_severity_to_ordinal_class

logger = get_logger(__name__)


class ConformalConfig(BaseModel):
    """Pydantic schema for conformal prediction calibration."""
    alpha: float = Field(default=0.10, description="Significance level (alpha=0.10 yields 90% prediction interval).")
    num_bins: int = Field(default=5, description="Number of Mondrian severity calibration bins.")
    min_bin_samples: int = Field(default=10, description="Minimum calibration samples required per bin before merging.")


class MondrianConformalCalibrator:
    """Mondrian Split Conformal Calibrator for continuous severity prediction intervals."""

    def __init__(
        self,
        alpha: float = 0.10,
        bin_edges: Optional[List[float]] = None,
        min_bin_samples: int = 10
    ):
        self.alpha = alpha
        self.bin_edges = bin_edges if bin_edges is not None else [0.0, 0.05, 0.15, 0.30, 0.50, 1.0]
        self.min_bin_samples = min_bin_samples
        self.is_calibrated = False

        self.bin_quantiles: Dict[int, float] = {}
        self.bin_counts: Dict[int, int] = {}
        self.global_quantile: float = 0.05

    def fit(self, y_true_calib: np.ndarray, y_pred_calib: np.ndarray) -> Dict[str, Any]:
        """Calibrate conformal quantiles on calibration split.

        Args:
            y_true_calib: Ground truth continuous severity array [N_calib].
            y_pred_calib: Predicted continuous severity array [N_calib].

        Returns:
            Dict containing bin quantiles, sample counts, and overall calibration summary.
        """
        y_true = np.clip(np.array(y_true_calib, dtype=np.float32), 0.0, 1.0)
        y_pred = np.clip(np.array(y_pred_calib, dtype=np.float32), 0.0, 1.0)
        n_calib = len(y_true)

        if n_calib == 0:
            raise ValueError("Empty calibration split provided to MondrianConformalCalibrator.")

        residuals = np.abs(y_true - y_pred)

        # 1. Compute global fallback quantile with finite-sample adjustment
        q_idx = int(np.ceil((n_calib + 1) * (1.0 - self.alpha))) - 1
        q_idx = max(0, min(q_idx, n_calib - 1))
        sorted_res = np.sort(residuals)
        self.global_quantile = float(sorted_res[q_idx])

        # 2. Assign calibration samples to Mondrian bins based on predicted severity
        n_bins = len(self.bin_edges) - 1
        bin_residuals: Dict[int, List[float]] = {b: [] for b in range(n_bins)}

        for y_t, y_p, res in zip(y_true, y_pred, residuals):
            b_idx = self._get_bin_index(y_p)
            bin_residuals[b_idx].append(float(res))

        # 3. Calculate per-bin quantiles with fallback / merging for sparse bins
        for b in range(n_bins):
            res_b = bin_residuals[b]
            n_b = len(res_b)
            self.bin_counts[b] = n_b

            if n_b >= self.min_bin_samples:
                b_q_idx = int(np.ceil((n_b + 1) * (1.0 - self.alpha))) - 1
                b_q_idx = max(0, min(b_q_idx, n_b - 1))
                sorted_b_res = np.sort(np.array(res_b))
                self.bin_quantiles[b] = float(sorted_b_res[b_q_idx])
            else:
                # Merge / fallback to global quantile for sparse bins
                logger.info(f"Bin {b} has only {n_b} samples (< {self.min_bin_samples}). Using global quantile.")
                self.bin_quantiles[b] = self.global_quantile

        self.is_calibrated = True
        logger.info(f"Conformal calibration completed across {n_calib} samples. Global 90% quantile: {self.global_quantile:.4f}")

        return {
            "alpha": self.alpha,
            "confidence_level": round(1.0 - self.alpha, 2),
            "n_calib_samples": n_calib,
            "global_quantile": round(self.global_quantile, 4),
            "bin_edges": self.bin_edges,
            "bin_quantiles": {b: round(q, 4) for b, q in self.bin_quantiles.items()},
            "bin_counts": self.bin_counts
        }

    def predict_interval(self, y_pred: Union[float, np.ndarray]) -> Union[Tuple[float, float, float], List[Tuple[float, float, float]]]:
        """Compute conformal prediction interval [Lower, Upper] for new prediction(s).

        Args:
            y_pred: Single float severity prediction or 1D array of predictions.

        Returns:
            Tuple of (lower_bound, upper_bound, interval_width) for single float input,
            or list of tuples for array input.
        """
        if not self.is_calibrated:
            logger.warning("Predicting interval before explicit calibration. Using fallback quantile 0.05.")
            q_val = 0.05
        else:
            q_val = self.global_quantile

        if isinstance(y_pred, (float, int, np.floating)):
            yp = float(np.clip(y_pred, 0.0, 1.0))
            if self.is_calibrated:
                b_idx = self._get_bin_index(yp)
                q_val = self.bin_quantiles.get(b_idx, self.global_quantile)

            lower = float(np.clip(yp - q_val, 0.0, 1.0))
            upper = float(np.clip(yp + q_val, 0.0, 1.0))
            width = float(upper - lower)
            return (round(lower, 4), round(upper, 4), round(width, 4))

        y_arr = np.clip(np.array(y_pred, dtype=np.float32), 0.0, 1.0)
        results = []
        for yp in y_arr:
            if self.is_calibrated:
                b_idx = self._get_bin_index(float(yp))
                q_v = self.bin_quantiles.get(b_idx, self.global_quantile)
            else:
                q_v = q_val

            lower = float(np.clip(yp - q_v, 0.0, 1.0))
            upper = float(np.clip(yp + q_v, 0.0, 1.0))
            width = float(upper - lower)
            results.append((round(lower, 4), round(upper, 4), round(width, 4)))

        return results

    def evaluate_coverage(
        self,
        y_true_test: np.ndarray,
        y_pred_test: np.ndarray,
        sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate empirical coverage (% true values in interval) and mean width on test data.

        Args:
            y_true_test: True test ground truth severity array [N_test].
            y_pred_test: Predicted test severity array [N_test].
            sources: Optional source dataset labels for per-source coverage reporting.

        Returns:
            Dict containing overall coverage, mean width, and breakdown per bin / per source.
        """
        y_true = np.clip(np.array(y_true_test, dtype=np.float32), 0.0, 1.0)
        y_pred = np.clip(np.array(y_pred_test, dtype=np.float32), 0.0, 1.0)
        n_test = len(y_true)

        intervals = self.predict_interval(y_pred)
        lowers = np.array([inv[0] for inv in intervals], dtype=np.float32)
        uppers = np.array([inv[1] for inv in intervals], dtype=np.float32)
        widths = np.array([inv[2] for inv in intervals], dtype=np.float32)

        covered = (y_true >= lowers) & (y_true <= uppers)
        overall_coverage = float(np.mean(covered))
        overall_width = float(np.mean(widths))

        # Per-bin breakdown
        bin_metrics = {}
        n_bins = len(self.bin_edges) - 1
        for b in range(n_bins):
            bin_mask = np.array([self._get_bin_index(yp) == b for yp in y_pred])
            if np.sum(bin_mask) > 0:
                bin_cov = float(np.mean(covered[bin_mask]))
                bin_w = float(np.mean(widths[bin_mask]))
                bin_metrics[f"bin_{b}"] = {
                    "count": int(np.sum(bin_mask)),
                    "coverage": round(bin_cov, 4),
                    "coverage_pct": round(bin_cov * 100.0, 2),
                    "mean_width": round(bin_w, 4)
                }

        # Per-source breakdown if provided
        source_metrics = {}
        if sources is not None and len(sources) == n_test:
            unique_sources = set(sources)
            for s in unique_sources:
                s_mask = np.array([src == s for src in sources])
                if np.sum(s_mask) > 0:
                    s_cov = float(np.mean(covered[s_mask]))
                    s_w = float(np.mean(widths[s_mask]))
                    source_metrics[s] = {
                        "count": int(np.sum(s_mask)),
                        "coverage": round(s_cov, 4),
                        "coverage_pct": round(s_cov * 100.0, 2),
                        "mean_width": round(s_w, 4)
                    }

        return {
            "n_test_samples": n_test,
            "target_confidence": round((1.0 - self.alpha) * 100.0, 1),
            "empirical_coverage": round(overall_coverage, 4),
            "empirical_coverage_pct": round(overall_coverage * 100.0, 2),
            "mean_interval_width": round(overall_width, 4),
            "per_bin_breakdown": bin_metrics,
            "per_source_breakdown": source_metrics
        }

    def _get_bin_index(self, val: float) -> int:
        """Find bin index for a given scalar prediction."""
        v = float(np.clip(val, 0.0, 1.0))
        n_bins = len(self.bin_edges) - 1
        for i in range(n_bins):
            low = self.bin_edges[i]
            high = self.bin_edges[i + 1]
            if i == n_bins - 1:
                if low <= v <= high:
                    return i
            else:
                if low <= v < high:
                    return i
        return n_bins - 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert calibrator parameters to serializable JSON dict."""
        return {
            "alpha": self.alpha,
            "confidence_level": round(1.0 - self.alpha, 2),
            "global_quantile": float(self.global_quantile),
            "bin_edges": self.bin_edges,
            "bin_quantiles": {str(b): float(q) for b, q in self.bin_quantiles.items()},
            "bin_counts": {str(b): int(c) for b, c in self.bin_counts.items()},
            "is_calibrated": self.is_calibrated
        }

    def save_json(self, output_path: Union[str, Path]) -> Path:
        """Save conformal calibration parameters to conformal.json."""
        output_p = Path(output_path)
        output_p.parent.mkdir(parents=True, exist_ok=True)
        with open(output_p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved conformal calibration parameters to {output_p}")
        return output_p

    @classmethod
    def load_json(cls, input_path: Union[str, Path]) -> "MondrianConformalCalibrator":
        """Load conformal calibrator from conformal.json."""
        input_p = Path(input_path)
        with open(input_p, "r", encoding="utf-8") as f:
            data = json.load(f)

        calibrator = cls(
            alpha=data.get("alpha", 0.10),
            bin_edges=data.get("bin_edges", [0.0, 0.05, 0.15, 0.30, 0.50, 1.0])
        )
        calibrator.global_quantile = data.get("global_quantile", 0.05)
        calibrator.bin_quantiles = {int(b): float(q) for b, q in data.get("bin_quantiles", {}).items()}
        calibrator.bin_counts = {int(b): int(c) for b, c in data.get("bin_counts", {}).items()}
        calibrator.is_calibrated = data.get("is_calibrated", True)
        return calibrator
