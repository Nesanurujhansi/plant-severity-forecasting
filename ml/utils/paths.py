"""Centralized project path management.

Resolves paths relative to the repository root locally or using the environment variable DATA_ROOT
for seamless execution in Google Colab with Google Drive mounts.
"""

import os
from pathlib import Path
from typing import NamedTuple

class ProjectPaths(NamedTuple):
    project_root: Path
    data_root: Path
    raw_data: Path
    interim_data: Path
    processed_data: Path
    models_dir: Path
    reports_dir: Path
    figures_dir: Path
    tables_dir: Path
    configs_dir: Path
    docs_dir: Path

def get_paths() -> ProjectPaths:
    """Resolve and return all project directory paths.
    
    Reads DATA_ROOT from environment if present (e.g., in Google Colab),
    otherwise defaults to data/ inside the repository root.
    """
    # ml/utils/paths.py is 2 levels below repository root
    project_root = Path(__file__).resolve().parent.parent.parent
    
    env_data_root = os.getenv("DATA_ROOT")
    if env_data_root:
        data_root = Path(env_data_root).resolve()
    else:
        data_root = (project_root / "data").resolve()
        
    raw_data = data_root / "raw"
    interim_data = data_root / "interim"
    processed_data = data_root / "processed"
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"
    tables_dir = reports_dir / "tables"
    configs_dir = project_root / "ml" / "configs"
    docs_dir = project_root / "docs"
    
    # Ensure directories exist
    for path in [data_root, raw_data, interim_data, processed_data, models_dir, reports_dir, figures_dir, tables_dir]:
        path.mkdir(parents=True, exist_ok=True)
        
    return ProjectPaths(
        project_root=project_root,
        data_root=data_root,
        raw_data=raw_data,
        interim_data=interim_data,
        processed_data=processed_data,
        models_dir=models_dir,
        reports_dir=reports_dir,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        configs_dir=configs_dir,
        docs_dir=docs_dir,
    )
