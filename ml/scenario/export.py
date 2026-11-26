"""Zero-Pickle Export Engine for Treatment Scenario Artifacts.

Exports:
1. data/raw/phase07_scenario_models/mixed_effects.json
2. data/raw/phase07_scenario_models/effect_bootstrap.json
3. data/raw/phase07_scenario_models/scenario_config.json
4. data/raw/phase07_scenario_models/treatment_net.safetensors
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
import torch

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger
from ml.scenario.identifiability import analyze_treatment_identifiability
from ml.scenario.mixed_effects import fit_mixed_effects_treatment_model
from ml.scenario.bootstrap import run_cluster_bootstrap
from ml.scenario.treatment_net import DeepTreatmentTarNet

logger = get_logger(__name__)


def export_scenario_artifacts(output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Export zero-pickle treatment scenario JSON & safetensors artifacts."""
    paths = get_paths()
    if output_dir is None:
        output_dir = paths.raw_data / "phase07_scenario_models"

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Mixed Effects Model JSON
    mixed_res = fit_mixed_effects_treatment_model()
    mixed_path = output_dir / "mixed_effects.json"
    with open(mixed_path, "w") as f:
        json.dump(mixed_res, f, indent=2)

    # 2. Bootstrap Percentiles JSON
    boot_res = run_cluster_bootstrap(n_resamples=50)
    boot_path = output_dir / "effect_bootstrap.json"
    with open(boot_path, "w") as f:
        json.dump(boot_res, f, indent=2)

    # 3. Scenario Config JSON
    ident_res = analyze_treatment_identifiability()
    config_dict = {
        "module_name": "Treatment Scenario Modeling",
        "primary_contrast": ident_res["recommended_fungicide_multiplier_contrast"],
        "disclaimer": (
            "Estimated under the conditions of a Swiss wheat field trial or multi-crop research baseline; "
            "not a spray recommendation. Consult a certified local agronomist before taking crop protection actions."
        ),
        "limitations": ident_res["limitations"],
    }
    config_path = output_dir / "scenario_config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # 4. Deep TarNet Weights .safetensors
    tarnet_model = DeepTreatmentTarNet(input_dim=16, hidden_dim=64)
    safetensors_path = output_dir / "treatment_net.safetensors"
    from safetensors.torch import save_file
    save_file(tarnet_model.state_dict(), str(safetensors_path))

    exported_files = {
        "mixed_effects": mixed_path,
        "effect_bootstrap": boot_path,
        "scenario_config": config_path,
        "treatment_net": safetensors_path,
    }

    logger.info(f"Successfully exported scenario artifacts to {output_dir}")
    return exported_files


def main() -> int:
    files = export_scenario_artifacts()
    for name, path in files.items():
        print(f"Exported {name} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
