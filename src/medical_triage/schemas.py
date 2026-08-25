"""HTTP request and response schemas."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictRequest(BaseModel):
    """Medical text submitted for academic triage classification."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=10_000, examples=["Persistent chest pain."])

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


class PredictResponse(BaseModel):
    """Model result plus its operational provenance."""

    condition: str
    priority: str
    priority_source: str
    priority_reason: str | None = None
    backend: Literal["sklearn", "onnx"]
    model_version: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    inference_ms: float = Field(ge=0.0)


class HealthResponse(BaseModel):
    """Readiness state. Degraded means the process is alive but cannot infer."""

    status: Literal["healthy", "degraded"]
    model_loaded: bool
    backend: Literal["sklearn", "onnx"] | None = None
    model_version: str | None = None
    detail: str | None = None


class ErrorResponse(BaseModel):
    detail: str
