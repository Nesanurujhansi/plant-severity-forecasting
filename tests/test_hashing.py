"""Unit tests for SHA-256 file and directory hashing."""

import tempfile
from pathlib import Path
from ml.utils.hashing import hash_file, hash_dir

def test_hash_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "sample.txt"
        test_file.write_text("crop-prognosis-test-content", encoding="utf-8")
        
        digest1 = hash_file(test_file)
        digest2 = hash_file(test_file)
        
        assert isinstance(digest1, str)
        assert len(digest1) == 64
        assert digest1 == digest2

def test_hash_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir)
        (dir_path / "sub").mkdir(parents=True)
        (dir_path / "file1.txt").write_text("content1", encoding="utf-8")
        (dir_path / "sub" / "file2.txt").write_text("content2", encoding="utf-8")
        
        h1 = hash_dir(dir_path)
        h2 = hash_dir(dir_path)
        
        assert h1 == h2
        assert len(h1) == 64
        
        # Modify a file and verify hash changes
        (dir_path / "file1.txt").write_text("content1_modified", encoding="utf-8")
        h3 = hash_dir(dir_path)
        assert h1 != h3
