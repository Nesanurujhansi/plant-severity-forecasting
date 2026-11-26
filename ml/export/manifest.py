"""
Phase 09: Manifest Schema and Builder
Provides Pydantic schemas and utility functions for generating and validating manifest.json
in the crop-prognosis model bundle.
"""

import hashlib
import importlib.metadata
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class BackboneSpec(BaseModel):
    id: str = Field(default="facebook/dinov2-small", description="HuggingFace model ID")
    pinned_revision: str = Field(default="main", description="Pinned git revision or commit SHA")


class PreprocessingSpec(BaseModel):
    resolution: List[int] = Field(default_factory=lambda: [224, 224], description="Target image resolution [H, W]")
    mean: List[float] = Field(default_factory=lambda: [0.485, 0.456, 0.406], description="Normalization channel means")
    std: List[float] = Field(default_factory=lambda: [0.229, 0.224, 0.225], description="Normalization channel standard deviations")
    padding_rule: str = Field(default="aspect_ratio_preserve", description="Image padding strategy")


class FileEntry(BaseModel):
    sha256: str = Field(..., description="SHA-256 hash hex string")
    size_bytes: int = Field(..., description="File size in bytes")


class DatasetIdentifier(BaseModel):
    name: str = Field(..., description="Dataset title or identification name")
    doi_or_url: str = Field(..., description="DOI link or web repository URL")
    licence: str = Field(..., description="Licence description (e.g., CC-BY-4.0, Apache-2.0)")


class ManifestSchema(BaseModel):
    bundle_version: str = Field(default="1.0.0", description="Semantic bundle version string")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    git_commit: str = Field(default="uncommitted", description="Git commit hash at export time")
    library_versions: Dict[str, str] = Field(..., description="Installed versions of critical runtime libraries")
    backbone: BackboneSpec = Field(default_factory=BackboneSpec)
    preprocessing: PreprocessingSpec = Field(default_factory=PreprocessingSpec)
    severity_bins: List[str] = Field(
        default_factory=lambda: ["<1%", "1-5%", "5-10%", "10-25%", "25-50%", ">50%"]
    )
    conformal_alpha: float = Field(default=0.1, description="Mondrian conformal prediction error rate (90% coverage)")
    forecast_horizons: List[int] = Field(default_factory=lambda: [3, 7], description="Prediction horizons in days")
    feature_schema_hash: str = Field(..., description="SHA-256 hash of forecast feature_schema.json")
    files: Dict[str, FileEntry] = Field(..., description="Mapping of relative file path to SHA-256 & size")
    metrics_summary: Dict[str, Any] = Field(default_factory=dict, description="Headline evaluation metrics from all phases")
    dataset_identifiers: List[DatasetIdentifier] = Field(default_factory=list, description="Dataset sources and licences")
    disclaimer: str = Field(
        default="This tool provides non-prescription decision support only. It NEVER provides pesticide names, doses, or spray timings, and does not replace an agronomist.",
        description="Mandatory scientific non-prescription disclaimer"
    )
    known_limitations: List[str] = Field(
        default_factory=lambda: [
            "Wheat host specific for progression & treatment scenarios",
            "Single field site training (ETH Zurich Lindau-Eschikon)",
            "Coarse weather meteorology resolution (NASA POWER ~50km)",
            "Leaf-level disease severity observation only"
        ],
        description="Known scientific limitations of the model bundle"
    )

    @field_validator("bundle_version")
    @classmethod
    def validate_semver(cls, v: str) -> str:
        parts = v.split(".")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError(f"bundle_version must be a valid semantic version (e.g. '1.0.0'), got '{v}'")
        return v


def calculate_sha256(filepath: Union[str, Path]) -> str:
    """Calculate SHA-256 hex digest for a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_installed_library_versions() -> Dict[str, str]:
    """Retrieve installed versions of core project dependencies."""
    packages = [
        "torch",
        "transformers",
        "lightgbm",
        "safetensors",
        "numpy",
        "scipy",
        "statsmodels",
        "pillow",
        "pydantic"
    ]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            versions[pkg] = "not-installed"
    versions["python"] = sys.version.split()[0]
    return versions


def build_manifest(
    files_map: Dict[str, Union[str, Path]],
    feature_schema_path: Union[str, Path],
    metrics_summary: Optional[Dict[str, Any]] = None,
    bundle_version: str = "1.0.0",
    git_commit: str = "uncommitted",
    dataset_identifiers: Optional[List[DatasetIdentifier]] = None
) -> ManifestSchema:
    """
    Build a complete ManifestSchema instance given relative file paths and feature schema file.
    
    Args:
        files_map: Dict mapping relative file path inside bundle to actual filesystem path.
        feature_schema_path: Path to forecast feature_schema.json file.
        metrics_summary: Optional headline evaluation metrics dictionary.
        bundle_version: Semantic version string.
        git_commit: Current git commit hash string.
        dataset_identifiers: List of dataset sources and licences.
    """
    file_entries: Dict[str, FileEntry] = {}
    for rel_path, abs_path in files_map.items():
        if rel_path == "manifest.json":
            continue
        abs_p = Path(abs_path)
        if not abs_p.exists():
            raise FileNotFoundError(f"Cannot build manifest: file '{abs_p}' does not exist.")
        file_entries[rel_path] = FileEntry(
            sha256=calculate_sha256(abs_p),
            size_bytes=abs_p.stat().st_size
        )

    feat_schema_p = Path(feature_schema_path)
    if not feat_schema_p.exists():
        raise FileNotFoundError(f"Feature schema file '{feat_schema_p}' does not exist.")
    feature_schema_hash = calculate_sha256(feat_schema_p)

    created_at = datetime.now(timezone.utc).isoformat()
    lib_versions = get_installed_library_versions()

    if dataset_identifiers is None:
        dataset_identifiers = [
            DatasetIdentifier(
                name="ETH Zurich Wheat Leaf Time Series",
                doi_or_url="https://doi.org/10.3929/ethz-b-000739194",
                licence="CC-BY-4.0"
            ),
            DatasetIdentifier(
                name="PlantSeg Leaf Mask Subset",
                doi_or_url="https://github.com/tqwei05/PlantSeg",
                licence="CC-BY-4.0"
            ),
            DatasetIdentifier(
                name="Coffee Leaf Rust CV Dataset",
                doi_or_url="https://huggingface.co/datasets/MaryPazRB/Paper_CLR_CV",
                licence="MIT"
            )
        ]

    return ManifestSchema(
        bundle_version=bundle_version,
        created_at=created_at,
        git_commit=git_commit,
        library_versions=lib_versions,
        feature_schema_hash=feature_schema_hash,
        files=file_entries,
        metrics_summary=metrics_summary or {},
        dataset_identifiers=dataset_identifiers
    )
