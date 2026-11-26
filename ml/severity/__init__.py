"""Severity Module Package for crop-prognosis.

Contains vision backbone segmentation decoder heads, loss functions, severity metric calculation,
baseline predictors, Mondrian split conformal intervals, evaluation routines, training scripts,
inference predictors, and model export tooling.
"""

from ml.severity.heads import SeveritySegmentationHeads
from ml.severity.losses import CombinedBCEDiceLoss
from ml.severity.severity import calculate_severity_metrics
from ml.severity.conformal import MondrianConformalCalibrator
from ml.severity.predict import SeverityPredictor

__all__ = [
    "SeveritySegmentationHeads",
    "CombinedBCEDiceLoss",
    "calculate_severity_metrics",
    "MondrianConformalCalibrator",
    "SeverityPredictor",
]
