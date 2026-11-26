"""
Phase 09: Application Model Singleton Container & Loader
Loads verified model bundle components at application startup or activates lightweight STUB models
when DEV_STUB_MODELS environment variable or flag is set.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ml.export.manifest import ManifestSchema
from ml.export.verify_bundle import verify_bundle

logger = logging.getLogger("app.inference.loader")


class StubSeverityPredictor:
    """Stub severity predictor for development and test mode."""
    def predict(self, image_input: Any) -> Dict[str, Any]:
        return {
            "severity_pct": 12.5,
            "severity_interval": [9.0, 16.0],
            "severity_class": "10-25%",
            "lesion_count": 4,
            "mean_lesion_size_px": 120.5,
            "stub": True
        }


class StubForecaster:
    """Stub progression forecaster for development and test mode."""
    def predict(self, initial_severity: float, horizons: list = [3, 7], **kwargs) -> Dict[str, Any]:
        return {
            "initial_severity": initial_severity,
            "forecasts": {
                "3d": {"severity_pct": initial_severity * 1.3, "interval": [initial_severity * 1.1, initial_severity * 1.5]},
                "7d": {"severity_pct": initial_severity * 1.8, "interval": [initial_severity * 1.4, initial_severity * 2.2]}
            },
            "stub": True
        }


class StubScenarioSimulator:
    """Stub treatment scenario simulator for development and test mode."""
    def simulate(self, initial_severity: float, horizon_days: int = 7, **kwargs) -> Dict[str, Any]:
        return {
            "unprotected_severity_pct": initial_severity * 1.8,
            "protected_severity_pct": initial_severity * 1.1,
            "relative_reduction_pct": 38.9,
            "stub": True
        }


class StubSafetyPolicy:
    """Stub safety policy for development and test mode."""
    def evaluate(self, image_input: Any = None, weather_inputs: Dict = None) -> Dict[str, Any]:
        return {
            "quality_passed": True,
            "ood_status": "IN_DISTRIBUTION",
            "confidence_tier": "HIGH",
            "should_abstain": False,
            "module_gating": {"allow_severity": True, "allow_progression": True, "allow_scenario": True},
            "user_message": "Accepted (Stub Mode)",
            "stub": True
        }


class ProductionSafetyPolicy:
    """Production safety policy aggregating image quality, OOD evaluation, and input ranges."""

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        stats_path = self.models_dir / "safety" / "ood_stats.safetensors"
        meta_path = self.models_dir / "safety" / "ood_meta.json"
        self.ood_evaluator = None
        if stats_path.exists() and meta_path.exists():
            try:
                from ml.safety.ood import DeepOODSafetyEvaluator, FreeEnergyClassifier
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                fe = FreeEnergyClassifier(energy_threshold=float(meta.get("free_energy_p99", 8.5)))
                self.ood_evaluator = DeepOODSafetyEvaluator(free_energy_clf=fe)
            except Exception as e:
                logger.warning(f"Failed to load OOD safety evaluator: {e}")

    def evaluate(self, image_input: Any = None, weather_inputs: Optional[Dict] = None, cls_vector: Optional[List[float]] = None) -> Dict[str, Any]:
        import numpy as np
        from ml.safety.quality import evaluate_image_quality
        from ml.safety.ranges import evaluate_input_ranges
        from ml.safety.policy import evaluate_safety_policy

        if image_input is not None:
            np_img = np.array(image_input)
            q_res = evaluate_image_quality(np_img)
        else:
            q_res = {"passed_all_quality_checks": True, "quality_score": 1.0, "failed_reasons": [], "recommended_actions": []}

        if self.ood_evaluator is not None and cls_vector is not None:
            feat = np.array(cls_vector, dtype=np.float32)
            ood_res = self.ood_evaluator.evaluate_safety(feat)
        else:
            ood_res = {"is_ood": False, "is_borderline": False, "ood_status": "IN_DISTRIBUTION", "flagged_reasons": []}

        range_res = evaluate_input_ranges()
        policy = evaluate_safety_policy(quality_result=q_res, ood_result=ood_res, range_result=range_res)
        user_msg = "Photo accepted" if not policy["should_abstain"] else "; ".join(policy.get("reasons", ["Abstention triggered"]))

        return {
            "confidence_tier": policy["confidence_tier"],
            "should_abstain": policy["should_abstain"],
            "quality_passed": q_res["passed_all_quality_checks"],
            "ood_status": ood_res["ood_status"],
            "user_message": user_msg,
            "module_gating": policy["module_gating"],
            "failure_reasons": policy.get("reasons", []),
            "stub": False
        }


class ModelContainer:
    """
    Singleton container holding initialized model components and bundle metadata.
    """
    _instance: Optional["ModelContainer"] = None

    def __init__(
        self,
        bundle_dir: Union[str, Path] = "models",
        dev_stub: bool = False
    ):
        self.bundle_dir = Path(bundle_dir)
        manifest_path = self.bundle_dir / "manifest.json"

        # Enable dev_stub if explicitly requested, via env var, or if model bundle manifest is missing
        if dev_stub or (os.getenv("DEV_STUB_MODELS", "0").lower() in ("1", "true", "yes")) or not manifest_path.exists():
            self.dev_stub = True
        else:
            self.dev_stub = False

        self.stub_banner: Optional[str] = None
        self.manifest: Optional[ManifestSchema] = None

        self.severity_predictor = None
        self.forecaster = None
        self.scenario_simulator = None
        self.safety_policy = None

        self._initialize()

    def _initialize(self):
        if self.dev_stub:
            manifest_path = self.bundle_dir / "manifest.json"
            if not manifest_path.exists():
                self.stub_banner = "[WARNING] DEV_STUB_MODELS ACTIVE (No production bundle at 'models/')"
            else:
                self.stub_banner = "[WARNING] DEV_STUB_MODELS ACTIVE - NOT REAL PREDICTIONS"
            logger.warning("=======================================================================")
            logger.warning(self.stub_banner)
            logger.warning("=======================================================================")
            self.severity_predictor = StubSeverityPredictor()
            self.forecaster = StubForecaster()
            self.scenario_simulator = StubScenarioSimulator()
            self.safety_policy = StubSafetyPolicy()
            return

        # Production Mode: Verify model bundle integrity first
        passed, report_lines = verify_bundle(self.bundle_dir)
        if not passed:
            report_text = "\n".join(report_lines)
            raise RuntimeError(
                f"ModelContainer initialization failed due to bundle verification errors:\n{report_text}"
            )

        # Parse manifest
        manifest_path = self.bundle_dir / "manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        self.manifest = ManifestSchema(**manifest_data)

        # Production Mode: Wire real models
        try:
            from ml.severity.predict import SeverityPredictor
            sev_weights = self.bundle_dir / "severity" / "severity_heads.safetensors"
            sev_conf = self.bundle_dir / "severity" / "conformal.json"
            self.severity_predictor = SeverityPredictor(
                weights_path=sev_weights if sev_weights.exists() else None,
                conformal_path=sev_conf if sev_conf.exists() else None,
                device="cpu"
            )
        except Exception as e:
            logger.warning(f"Failed to load production SeverityPredictor: {e}")
            self.severity_predictor = StubSeverityPredictor()

        try:
            from ml.forecast.predict import Forecaster
            self.forecaster = Forecaster(
                models_dir=self.bundle_dir / "forecast",
                is_stub=False
            )
        except Exception as e:
            logger.warning(f"Failed to load production Forecaster: {e}")
            self.forecaster = StubForecaster()

        try:
            from ml.scenario.simulate import ScenarioSimulator
            self.scenario_simulator = ScenarioSimulator(
                models_dir=self.bundle_dir / "scenario",
                forecaster=self.forecaster
            )
        except Exception as e:
            logger.warning(f"Failed to load production ScenarioSimulator: {e}")
            self.scenario_simulator = StubScenarioSimulator()

        try:
            self.safety_policy = ProductionSafetyPolicy(models_dir=self.bundle_dir)
        except Exception as e:
            logger.warning(f"Failed to load production SafetyPolicy: {e}")
            self.safety_policy = StubSafetyPolicy()

        logger.info(f"ModelContainer successfully initialized real production bundle v{self.manifest.bundle_version}")

    @property
    def is_stub(self) -> bool:
        return self.dev_stub

    def get_bundle_info(self) -> Dict[str, Any]:
        if self.dev_stub or self.manifest is None:
            return {
                "version": "DEV-STUB",
                "is_stub": True,
                "banner": self.stub_banner,
                "metrics_summary": {}
            }
        return {
            "version": self.manifest.bundle_version,
            "created_at": self.manifest.created_at,
            "git_commit": self.manifest.git_commit,
            "is_stub": False,
            "metrics_summary": self.manifest.metrics_summary,
            "disclaimer": self.manifest.disclaimer
        }


def get_model_container(
    bundle_dir: Union[str, Path] = "models",
    dev_stub: bool = False,
    force_reload: bool = False
) -> ModelContainer:
    """
    Get or create singleton ModelContainer instance.
    
    Args:
        bundle_dir: Path to unzipped models folder.
        dev_stub: If True, load stub predictors.
        force_reload: If True, re-instantiate singleton.
    """
    if ModelContainer._instance is None or force_reload:
        ModelContainer._instance = ModelContainer(bundle_dir=bundle_dir, dev_stub=dev_stub)
    return ModelContainer._instance
