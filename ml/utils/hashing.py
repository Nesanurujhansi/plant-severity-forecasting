"""Hashing utilities for cryptographic integrity verification of files and directories."""

import hashlib
from pathlib import Path
from typing import Union

def hash_file(filepath: Union[str, Path], chunk_size: int = 65536) -> str:
    """Calculate the SHA-256 hash of a single file.
    
    Args:
        filepath: Path to the target file.
        chunk_size: Read buffer size in bytes.
        
    Returns:
        Hexadecimal SHA-256 string.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for hashing: {path}")
        
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def hash_dir(dirpath: Union[str, Path]) -> str:
    """Calculate a deterministic combined SHA-256 hash for a directory.
    
    Includes relative paths and file content in lexicographical order.
    
    Args:
        dirpath: Path to the target directory.
        
    Returns:
        Hexadecimal SHA-256 string representing the directory contents state.
    """
    path = Path(dirpath)
    if not path.is_dir():
        raise NotADirectoryError(f"Directory not found for hashing: {path}")
        
    hasher = hashlib.sha256()
    # Walk directory in deterministic sorted order
    for file in sorted(path.rglob("*")):
        if file.is_file() and file.name != ".gitkeep":
            rel_path = file.relative_to(path).as_posix().encode("utf-8")
            hasher.update(rel_path)
            file_hash = hash_file(file).encode("utf-8")
            hasher.update(file_hash)
            
    return hasher.hexdigest()
