"""Long-lived sklearn and ONNX inference adapters."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from medical_triage.config import Settings


class ModelUnavailableError(RuntimeError):
    """Raised when inference is requested without a loaded artifact."""


@dataclass(frozen=True)
class Prediction:
    condition: str
    priority: str
    priority_source: str
    priority_reason: str | None
    backend: str
    model_version: str
    confidence: float | None
    inference_ms: float


def _read_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"metadata must contain a JSON object: {path}")
    return value


def _safe_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return min(1.0, max(0.0, result))


def _unpack_prediction(value: Any) -> tuple[str, float | None]:
    """Accept the modeling helper's tuple/dict as well as a plain label."""

    if isinstance(value, dict):
        label = value.get("condition", value.get("label", value.get("prediction")))
        confidence = value.get("confidence", value.get("score"))
        return str(label), _safe_confidence(confidence)
    if isinstance(value, tuple):
        return str(value[0]), _safe_confidence(value[1] if len(value) > 1 else None)
    return str(value), None


def _unpack_priority(value: Any) -> tuple[str, str | None]:
    """Normalize the priority helper's ``(priority, reason)`` output."""

    if isinstance(value, dict):
        priority = value.get("priority", "unknown")
        reason = value.get("priority_reason", value.get("reason"))
        return str(priority), None if reason is None else str(reason)
    if isinstance(value, tuple):
        reason = value[1] if len(value) > 1 else None
        return str(value[0]), None if reason is None else str(reason)
    return str(value), None


class InferenceEngine:
    """Loads exactly one configured backend and reuses it across requests."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.backend: str | None = None
        self.model_version: str | None = None
        self.load_error: str | None = None
        self._model: Any = None
        self._metadata: dict[str, Any] = {}

    @property
    def ready(self) -> bool:
        return self._model is not None and self.backend is not None

    def load(self) -> None:
        """Load one model artifact; record a concise error for readiness checks."""

        self.backend = None
        self.model_version = None
        self.load_error = None
        self._model = None

        candidates = self._candidates()
        failures: list[str] = []
        for backend, model_path, metadata_path in candidates:
            if not model_path.is_file():
                failures.append(f"{backend} artifact not found")
                continue
            try:
                metadata = _read_metadata(metadata_path)
                model = self._load_backend(backend, model_path)
            except Exception as exc:  # readiness reports errors; process remains alive
                failures.append(f"{backend} load failed: {type(exc).__name__}")
                continue
            self._model = model
            self._metadata = metadata
            self.backend = backend
            self.model_version = str(metadata.get("model_version", self.settings.model_version))
            return

        self.load_error = "; ".join(failures) or "no model backend configured"

    def _candidates(self) -> list[tuple[str, Path, Path]]:
        configured = self.settings.model_backend
        candidates = {
            "onnx": (
                "onnx",
                self.settings.onnx_model_path,
                self.settings.onnx_metadata_path,
            ),
            "sklearn": (
                "sklearn",
                self.settings.sklearn_model_path,
                self.settings.sklearn_metadata_path,
            ),
        }
        if configured == "auto":
            return [candidates["onnx"], candidates["sklearn"]]
        return [candidates[configured]]

    @staticmethod
    def _load_backend(backend: str, model_path: Path) -> Any:
        if backend == "sklearn":
            from medical_triage.modeling import load_model

            return load_model(model_path)

        import onnxruntime as ort

        return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def predict(self, text: str) -> Prediction:
        if not self.ready:
            raise ModelUnavailableError(self.load_error or "model is not loaded")

        started = time.perf_counter()
        if self.backend == "sklearn":
            condition, confidence = self._predict_sklearn(text)
        else:
            condition, confidence = self._predict_onnx(text)
        priority, priority_reason = self._assign_priority(text, condition)
        inference_ms = (time.perf_counter() - started) * 1_000
        return Prediction(
            condition=condition,
            priority=priority,
            priority_source="business_rule",
            priority_reason=priority_reason,
            backend=self.backend,
            model_version=self.model_version or self.settings.model_version,
            confidence=confidence,
            inference_ms=inference_ms,
        )

    def _predict_sklearn(self, text: str) -> tuple[str, float | None]:
        from medical_triage.modeling import predict_with_confidence

        return _unpack_prediction(predict_with_confidence(self._model, text))

    def _predict_onnx(self, text: str) -> tuple[str, float | None]:
        input_meta = self._model.get_inputs()[0]
        shape = getattr(input_meta, "shape", None) or []
        values = np.asarray([[text]] if len(shape) == 2 else [text], dtype=object)
        outputs = self._model.run(None, {input_meta.name: values})
        if not outputs:
            raise RuntimeError("ONNX model returned no outputs")

        labels = np.asarray(outputs[0]).reshape(-1)
        if not len(labels):
            raise RuntimeError("ONNX model returned no labels")
        condition = str(labels[0])
        confidence = self._onnx_confidence(outputs[1:] if len(outputs) > 1 else [])
        return condition, confidence

    @staticmethod
    def _onnx_confidence(outputs: list[Any]) -> float | None:
        if not outputs:
            return None
        probabilities = outputs[0]
        if isinstance(probabilities, list) and probabilities and isinstance(probabilities[0], dict):
            return _safe_confidence(max(probabilities[0].values(), default=None))
        try:
            values = np.asarray(probabilities, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        return _safe_confidence(values.max()) if values.size else None

    @staticmethod
    def _assign_priority(text: str, condition: str) -> tuple[str, str | None]:
        from medical_triage.priority import assign_priority

        return _unpack_priority(assign_priority(text, condition))
