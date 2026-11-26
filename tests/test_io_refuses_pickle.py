"""Unit tests verifying safe IO operations and explicit pickle rejection."""

import tempfile
from pathlib import Path
import pytest
import pandas as pd
from ml.utils.io import (
    read_json, write_json,
    read_yaml, write_yaml,
    read_parquet, write_parquet,
    load_pickle, dump_pickle,
    PickleRefusedError
)

def test_safe_json_yaml_parquet_io():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # JSON
        json_file = tmp_path / "data.json"
        data = {"key": "value", "numbers": [1, 2, 3]}
        write_json(data, json_file)
        assert read_json(json_file) == data
        
        # YAML
        yaml_file = tmp_path / "data.yaml"
        write_yaml(data, yaml_file)
        assert read_yaml(yaml_file) == data
        
        # Parquet
        parquet_file = tmp_path / "data.parquet"
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_parquet(df, parquet_file)
        df_read = read_parquet(parquet_file)
        pd.testing.assert_frame_equal(df, df_read)

def test_pickle_refused():
    with pytest.raises(PickleRefusedError):
        load_pickle("model.pkl")
        
    with pytest.raises(PickleRefusedError):
        dump_pickle({}, "model.pkl")

def test_pickle_extension_refused():
    with tempfile.TemporaryDirectory() as tmpdir:
        pickle_path = Path(tmpdir) / "test.pkl"
        
        with pytest.raises(PickleRefusedError):
            write_json({"a": 1}, pickle_path)
            
        with pytest.raises(PickleRefusedError):
            read_json(pickle_path)
