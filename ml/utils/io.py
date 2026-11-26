"""Safe file input/output utilities enforcing secure serialization.

Supports JSON, YAML, and Parquet. Explicitly blocks and refuses any pickle operations
to protect against unsafe deserialization vulnerabilities.
"""

import json
from pathlib import Path
from typing import Any, Union
import yaml
import pandas as pd

class PickleRefusedError(ValueError):
    """Exception raised when an unsafe pickle operation is attempted."""
    pass

def _check_not_pickle(filepath: Union[str, Path]) -> None:
    """Refuse files with pickle extensions."""
    path = Path(filepath)
    ext = path.suffix.lower()
    if ext in [".pkl", ".pickle", ".pck", ".pd"]:
        raise PickleRefusedError(
            f"Security policy error: Refusing to interact with pickle file '{path}'. "
            "Use JSON, YAML, Safetensors, or Parquet for model storage and data persistence."
        )

def read_json(filepath: Union[str, Path]) -> Any:
    """Safely read and parse a JSON file."""
    _check_not_pickle(filepath)
    path = Path(filepath)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """Safely write data to a JSON file."""
    _check_not_pickle(filepath)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def read_yaml(filepath: Union[str, Path]) -> dict[str, Any]:
    """Safely read and parse a YAML file using yaml.safe_load."""
    _check_not_pickle(filepath)
    path = Path(filepath)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def write_yaml(data: dict[str, Any], filepath: Union[str, Path]) -> None:
    """Safely write data to a YAML file using yaml.safe_dump."""
    _check_not_pickle(filepath)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def read_parquet(filepath: Union[str, Path]) -> pd.DataFrame:
    """Safely read a Parquet file into a pandas DataFrame."""
    _check_not_pickle(filepath)
    path = Path(filepath)
    return pd.read_parquet(path)

def write_parquet(df: pd.DataFrame, filepath: Union[str, Path]) -> None:
    """Safely write a pandas DataFrame to a Parquet file."""
    _check_not_pickle(filepath)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def load_pickle(*args: Any, **kwargs: Any) -> None:
    """Explicitly blocked function raising PickleRefusedError."""
    raise PickleRefusedError(
        "Security policy error: Pickle loading is strictly forbidden in crop-prognosis. "
        "Use read_json, read_yaml, read_parquet, or safetensors."
    )

def dump_pickle(*args: Any, **kwargs: Any) -> None:
    """Explicitly blocked function raising PickleRefusedError."""
    raise PickleRefusedError(
        "Security policy error: Pickle saving is strictly forbidden in crop-prognosis. "
        "Use write_json, write_yaml, write_parquet, or safetensors."
    )
