"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings for the inference API.

    Every setting can be overridden with a ``TRIAGE_`` prefixed environment
    variable, for example ``TRIAGE_MODEL_BACKEND=sklearn``.
    """

    model_config = SettingsConfigDict(
        env_prefix="TRIAGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Medical Triage API"
    app_version: str = "0.1.0"
    model_backend: Literal["auto", "sklearn", "onnx"] = "auto"
    sklearn_model_path: Path = PROJECT_ROOT / "artifacts" / "model.joblib"
    sklearn_metadata_path: Path = PROJECT_ROOT / "artifacts" / "model_metadata.json"
    onnx_model_path: Path = PROJECT_ROOT / "artifacts" / "model.onnx"
    onnx_metadata_path: Path = PROJECT_ROOT / "artifacts" / "onnx_metadata.json"
    model_version: str = "unknown"
    max_text_length: int = Field(default=10_000, ge=1, le=100_000)


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide, immutable-by-convention settings instance."""

    return Settings()
