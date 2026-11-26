"""Safety Policy Aggregator and Module Gating Engine for crop-prognosis.

Combines photo quality checks, OOD evaluation, input range validation, and conformal interval width
into:
1. Confidence Tier: HIGH / MEDIUM / LOW
2. Abstention Flag: should_abstain (True/False) with beginner-friendly reasons and user action
3. Per-Module Gating: allow_severity, allow_progression, allow_scenario
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.logging_utils import get_logger
from ml.safety.quality import evaluate_image_quality
from ml.safety.ood import DeepOODSafetyEvaluator
from ml.safety.ranges import evaluate_input_ranges

logger = get_logger(__name__)


def evaluate_safety_policy(
    quality_result: Dict[str, Any],
    ood_result: Dict[str, Any],
    range_result: Dict[str, Any],
    conformal_interval_width: Optional[float] = None,
    max_interval_width: float = 35.0
) -> Dict[str, Any]:
    """Aggregate all safety evaluations and compute confidence tier, abstention status, and module gating."""

    reasons = []
    actions = []

    # 1. Check OOD Status
    is_ood = ood_result.get("is_ood", False)
    is_borderline = ood_result.get("is_borderline", False)

    if is_ood:
        reasons.append("Uploaded photo appears to be a non-leaf or unknown crop object.")
        actions.append("Please upload a clear photo of a crop leaf.")

    # 2. Check Photo Quality
    passed_quality = quality_result.get("passed_all_quality_checks", True)
    q_score = quality_result.get("quality_score", 1.0)

    if not passed_quality:
        reasons.extend(quality_result.get("failed_reasons", []))
        actions.extend(quality_result.get("recommended_actions", []))

    # 3. Check Input Ranges
    ranges_valid = range_result.get("all_ranges_valid", True)
    range_warnings = range_result.get("warnings", [])
    crop_details = range_result.get("details", {}).get("crop", {})

    if range_warnings:
        reasons.extend(range_warnings)

    # 4. Check Conformal Interval Width
    interval_too_wide = False
    if conformal_interval_width is not None:
        if conformal_interval_width > max_interval_width:
            interval_too_wide = True
            reasons.append(f"Forecast prediction interval width ({conformal_interval_width:.1f}%) is too wide for reliable decision support.")
            actions.append("Consider providing localized weather observations or retaking a clearer leaf image.")

    # Determine Abstention Flag
    # Abstain completely if photo is OOD, or quality score < 0.50 (multiple critical quality failures)
    should_abstain = bool(is_ood or q_score < 0.50)

    # Determine Per-Module Gating
    allow_severity = bool(not should_abstain and crop_details.get("severity_allowed", True))

    allow_progression = bool(
        allow_severity and
        not interval_too_wide and
        ranges_valid and
        crop_details.get("progression_allowed", True)
    )

    allow_scenario = bool(
        allow_progression and
        crop_details.get("treatment_allowed", True)
    )

    # Determine Confidence Tier
    if should_abstain:
        confidence_tier = "LOW"
    elif is_borderline or not passed_quality or range_warnings or interval_too_wide or q_score < 0.90:
        confidence_tier = "MEDIUM"
    else:
        confidence_tier = "HIGH"

    # Deduplicate actions and reasons
    unique_reasons = list(dict.fromkeys(reasons))
    unique_actions = list(dict.fromkeys(actions))

    if not unique_actions and should_abstain:
        unique_actions.append("Please retake the photo closer, in steady daylight, and tap to focus on the leaf.")

    user_message = (
        "Photo rejected. " + " ".join(unique_reasons) + " " + " ".join(unique_actions)
        if should_abstain
        else ("In-distribution photo accepted with High confidence." if confidence_tier == "HIGH" else "Photo accepted with Medium confidence. " + " ".join(unique_reasons))
    )

    return {
        "confidence_tier": confidence_tier,
        "should_abstain": should_abstain,
        "reasons": unique_reasons,
        "user_actions": unique_actions,
        "user_message": user_message,
        "module_gating": {
            "allow_severity": allow_severity,
            "allow_progression": allow_progression,
            "allow_scenario": allow_scenario
        },
        "summary": {
            "is_ood": is_ood,
            "is_borderline": is_borderline,
            "quality_passed": passed_quality,
            "quality_score": q_score,
            "ranges_valid": ranges_valid,
            "interval_too_wide": interval_too_wide
        }
    }


def evaluate_full_safety(
    image: np.ndarray,
    feature_embedding: Optional[np.ndarray] = None,
    cls_logits: Optional[np.ndarray] = None,
    leaf_mask: Optional[np.ndarray] = None,
    lesion_mask: Optional[np.ndarray] = None,
    temp_c: float = 20.0,
    rh_pct: float = 70.0,
    horizon_days: float = 7.0,
    crop_name: str = "Wheat",
    conformal_interval_width: Optional[float] = None
) -> Dict[str, Any]:
    """Run end-to-end safety evaluation pipeline."""

    # 1. Quality Checks
    q_res = evaluate_image_quality(image, leaf_mask=leaf_mask, lesion_mask=lesion_mask)

    # 2. OOD Evaluation
    ood_evaluator = DeepOODSafetyEvaluator()
    if feature_embedding is None:
        # Construct stub 384-d feature vector if omitted
        feature_embedding = np.random.randn(384).astype(np.float32)
    ood_res = ood_evaluator.evaluate_safety(feature_embedding, cls_logits=cls_logits)

    # 3. Input Range Checks
    range_res = evaluate_input_ranges(temp_c=temp_c, rh_pct=rh_pct, horizon_days=horizon_days, crop_name=crop_name)

    # 4. Aggregate Policy
    policy_res = evaluate_safety_policy(q_res, ood_res, range_res, conformal_interval_width=conformal_interval_width)

    return {
        "confidence_tier": policy_res["confidence_tier"],
        "should_abstain": policy_res["should_abstain"],
        "user_message": policy_res["user_message"],
        "module_gating": policy_res["module_gating"],
        "quality": q_res,
        "ood": ood_res,
        "ranges": range_res
    }


def main() -> int:
    dummy_img = (np.random.rand(300, 300, 3) * 255).astype(np.uint8)
    dummy_feat = np.random.randn(384).astype(np.float32)
    res = evaluate_full_safety(dummy_img, feature_embedding=dummy_feat)
    print("Full Safety Policy Result:", res["confidence_tier"], "Abstain:", res["should_abstain"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
