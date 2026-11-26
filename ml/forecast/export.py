"""Module entrypoint for zero-pickle progression artifact export."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.export.progression_export import export_progression_artifacts

if __name__ == "__main__":
    export_progression_artifacts()
