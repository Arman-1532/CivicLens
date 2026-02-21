"""
Application configuration settings.
Loads environment variables and provides typed configuration.
"""

import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from functools import lru_cache
import json


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "CivicLens"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS
    CORS_ORIGINS: str = '["http://localhost:5173","http://localhost:3000"]'

    # Model paths (relative to backend directory)
    MODEL_PATH: str = "trained_models/classifier.pkl"
    VECTORIZER_PATH: str = "trained_models/tfidf_vectorizer.pkl"

    # Logging
    LOG_LEVEL: str = "INFO"

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./civiclens.db"

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from JSON string to list."""
        try:
            return json.loads(self.CORS_ORIGINS)
        except json.JSONDecodeError:
            return ["http://localhost:5173", "http://localhost:3000"]

    @property
    def base_dir(self) -> Path:
        """Get the base directory of the backend."""
        return Path(__file__).resolve().parent.parent.parent

    @property
    def model_full_path(self) -> Path:
        """Get full path to the classifier model."""
        return self.base_dir / self.MODEL_PATH

    @property
    def vectorizer_full_path(self) -> Path:
        """Get full path to the TF-IDF vectorizer."""
        return self.base_dir / self.VECTORIZER_PATH

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to avoid reading .env file on every call.
    """
    return Settings()


# Export settings instance
settings = get_settings()

