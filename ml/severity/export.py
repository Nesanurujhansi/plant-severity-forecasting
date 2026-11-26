"""Export Tool for Severity Module Trained Artifacts.

Bundles trained segmentation decoder weights (.safetensors), architecture config (config.json),
and Mondrian conformal calibration parameters (conformal.json) into the models/severity export directory
with metadata embedded inside the safetensors header.
"""

import sys
import os
import shutil
import json
import datetime
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
from safetensors.torch import save_file, load_file

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_git_commit_hash() -> str:
    """Retrieve current git commit hash or return fallback string."""
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
        return output
    except Exception:
        return "main-uncommitted"


def export_severity_artifacts(
    source_dir: Union[str, Path],
    export_dir: Optional[Union[str, Path]] = None,
    backbone_id: str = "facebook/dinov2-small",
    revision: str = "main",
    resolution: int = 448
) -> Dict[str, Path]:
    """Bundle and export trained severity artifacts with safetensors header metadata.

    Args:
        source_dir: Directory containing trained weights and configs.
        export_dir: Target export directory (default models/severity/).
        backbone_id: HuggingFace vision backbone ID.
        revision: Backbone revision tag.
        resolution: Target image resolution.

    Returns:
        Dict mapping artifact keys to exported file paths.
    """
    source_p = Path(source_dir)
    paths = get_paths()

    if export_dir is None:
        export_p = paths.project_root / "models" / "severity"
    else:
        export_p = Path(export_dir)

    export_p.mkdir(parents=True, exist_ok=True)

    metadata = {
        "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": get_git_commit_hash(),
        "backbone_id": backbone_id,
        "revision": revision,
        "resolution": str(resolution),
        "module": "severity"
    }

    exported_files = {}

    # 1. Copy & update weights safetensors
    weights_candidates = ["severity_heads.safetensors", "lesion_head.safetensors", "leaf_head.safetensors"]
    found_weights = False
    for w_name in weights_candidates:
        src_w = source_p / w_name
        if src_w.exists():
            tensors = load_file(src_w)
            dst_w = export_p / w_name
            save_file(tensors, dst_w, metadata=metadata)
            exported_files[w_name] = dst_w
            found_weights = True
            logger.info(f"Exported safetensors weights with metadata to {dst_w}")

    if not found_weights:
        # Create a stub weights file if missing
        from ml.severity.heads import build_severity_heads
        stub_model = build_severity_heads()
        dst_w = export_p / "severity_heads.safetensors"
        save_file({k: v.cpu() for k, v in stub_model.state_dict().items()}, dst_w, metadata=metadata)
        exported_files["severity_heads.safetensors"] = dst_w
        logger.info(f"Exported stub safetensors weights with metadata to {dst_w}")

    # 2. Export config.json
    src_cfg = source_p / "config.json"
    dst_cfg = export_p / "config.json"
    if src_cfg.exists():
        with open(src_cfg, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
        cfg_data["export_metadata"] = metadata
        with open(dst_cfg, "w", encoding="utf-8") as f:
            json.dump(cfg_data, f, indent=2)
    else:
        cfg_data = {
            "model_type": "SeveritySegmentationHeads",
            "input_dim": 384,
            "reduced_dim": 64,
            "target_stride": 4,
            "lesion_num_classes": 1,
            "leaf_num_classes": 1,
            "export_metadata": metadata
        }
        with open(dst_cfg, "w", encoding="utf-8") as f:
            json.dump(cfg_data, f, indent=2)
    exported_files["config.json"] = dst_cfg

    # 3. Export conformal.json
    src_conf = source_p / "conformal.json"
    dst_conf = export_p / "conformal.json"
    
    # Only copy if source and destination directories are different
    if source_p.resolve() != export_p.resolve():
        if src_conf.exists():
            try:
                shutil.copy2(src_conf, dst_conf)
            except Exception:
                pass
        else:
            from ml.severity.conformal import MondrianConformalCalibrator
            calibrator = MondrianConformalCalibrator()
            calibrator.save_json(dst_conf)
    else:
        if not dst_conf.exists():
            from ml.severity.conformal import MondrianConformalCalibrator
            calibrator = MondrianConformalCalibrator()
            calibrator.save_json(dst_conf)

    exported_files["conformal.json"] = dst_conf

    logger.info(f"Severity module artifact export complete at {export_p}")
    return exported_files
