"""
Phase 10: FastAPI Application Configuration Settings
Manages environment variables, default model paths, security limits, CORS origins, and rate limits.
"""

import os
from pathlib import Path
from typing import List, Union
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with pydantic validation and environment variable overrides."""
    
    app_name: str = "crop-prognosis-api"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Model Bundle & Inference Settings
    bundle_dir: str = Field(default="models", description="Directory path to extracted model bundle")
    dev_stub_models: bool = Field(default=False, description="Force lightweight stub predictors for testing")
    
    # Security & Upload Constraints
    max_upload_size_bytes: int = Field(default=10 * 1024 * 1024, description="Maximum image upload size (10 MB)")
    max_image_pixels: int = Field(default=25_000_000, description="Decompression bomb limit (25 MP)")
    store_uploads: bool = Field(default=False, description="Persist user upload images to disk (default False)")
    upload_store_dir: str = Field(default="data/interim/user_uploads", description="Directory for stored uploads")
    
    # Operations, CORS & Rate Limits
    cors_origins: List[str] = Field(default_factory=lambda: ["*"], description="Allowed CORS origins")
    rate_limit_per_minute: int = Field(default=60, description="Rate limit per client IP per minute")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
