"""Focused tests for model loading and inference backend adapters."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from medical_triage.config import Settings
from medical_triage.inference import (
    InferenceEngine,
    ModelUnavailableError,
    _read_metadata,
    _safe_confidence,
    _unpack_prediction,
    _unpack_priority,
)
from medical_triage.modeling import build_pipeline, save_model


def settings_for(tmp_path: Path, backend: str = "sklearn") -> Settings:
    return Settings(
        model_backend=backend,
        sklearn_model_path=tmp_path / "model.joblib",
        sklearn_metadata_path=tmp_path / "model_metadata.json",
        onnx_model_path=tmp_path / "model.onnx",
        onnx_metadata_path=tmp_path / "onnx_metadata.json",
        model_version="fallback-version",
    )


def test_candidates_respect_explicit_backend_and_auto_order(tmp_path: Path) -> None:
    sklearn_engine = InferenceEngine(settings_for(tmp_path, "sklearn"))
    auto_engine = InferenceEngine(settings_for(tmp_path, "auto"))

    assert [candidate[0] for candidate in sklearn_engine._candidates()] == ["sklearn"]
    assert [candidate[0] for candidate in auto_engine._candidates()] == ["onnx", "sklearn"]


def test_load_and_predict_with_real_small_sklearn_pipeline(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    model = build_pipeline()
    model.fit(
        [
            "routine skin rash follow up",
            "itchy skin allergy symptoms",
            "persistent cough and breathing symptoms",
            "lung infection with persistent cough",
        ],
        ["dermatological", "dermatological", "respiratory", "respiratory"],
    )
    save_model(
        model,
        settings.sklearn_model_path,
        metadata={"model_version": "tiny-v1"},
        metadata_path=settings.sklearn_metadata_path,
    )

    engine = InferenceEngine(settings)
    engine.load()
    prediction = engine.predict("Persistent cough with difficulty breathing")

    assert engine.ready is True
    assert engine.load_error is None
    assert prediction.backend == "sklearn"
    assert prediction.model_version == "tiny-v1"
    assert prediction.condition == "respiratory"
    assert prediction.priority == "high"
    assert prediction.priority_source == "business_rule"
    assert "difficulty breathing" in (prediction.priority_reason or "")
    assert prediction.confidence is not None and 0.5 <= prediction.confidence <= 1.0
    assert prediction.inference_ms >= 0


def test_missing_artifacts_and_unavailable_prediction(tmp_path: Path) -> None:
    engine = InferenceEngine(settings_for(tmp_path, "auto"))

    engine.load()

    assert engine.ready is False
    assert engine.load_error == "onnx artifact not found; sklearn artifact not found"
    with pytest.raises(ModelUnavailableError, match="artifact not found"):
        engine.predict("some clinical text")


def test_corrupt_model_is_reported_as_load_failure(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    settings.sklearn_model_path.write_bytes(b"not a joblib artifact")
    engine = InferenceEngine(settings)

    engine.load()

    assert engine.ready is False
    assert engine.load_error is not None
    assert engine.load_error.startswith("sklearn load failed:")


def test_invalid_metadata_degrades_instead_of_crashing_process(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    settings.sklearn_model_path.write_bytes(b"artifact exists")
    settings.sklearn_metadata_path.write_text("[]", encoding="utf-8")
    engine = InferenceEngine(settings)

    engine.load()

    assert engine.ready is False
    assert engine.load_error == "sklearn load failed: ValueError"


def test_metadata_reader_handles_missing_object_and_invalid_json(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"model_version": "v2"}), encoding="utf-8")

    assert _read_metadata(missing) == {}
    assert _read_metadata(metadata) == {"model_version": "v2"}

    metadata.write_text("not-json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        _read_metadata(metadata)


class FakeOnnxSession:
    def __init__(self, _path: str, providers: list[str]) -> None:
        self.providers = providers
        self.last_values: np.ndarray[Any, Any] | None = None
        self.outputs: list[Any] = [
            np.asarray(["respiratory"]),
            [{"respiratory": 0.84, "dermatological": 0.16}],
        ]

    @staticmethod
    def get_inputs() -> list[SimpleNamespace]:
        return [SimpleNamespace(name="medical_text", shape=[None])]

    def run(self, _output_names: Any, feeds: dict[str, np.ndarray[Any, Any]]) -> list[Any]:
        self.last_values = feeds["medical_text"]
        return self.outputs


def test_onnx_load_and_predict_with_fake_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = settings_for(tmp_path, "onnx")
    settings.onnx_model_path.write_bytes(b"fake onnx")
    settings.onnx_metadata_path.write_text(
        json.dumps({"model_version": "onnx-v1"}), encoding="utf-8"
    )
    monkeypatch.setattr("onnxruntime.InferenceSession", FakeOnnxSession)
    engine = InferenceEngine(settings)

    engine.load()
    prediction = engine.predict("difficulty breathing")

    assert engine.ready is True
    assert isinstance(engine._model, FakeOnnxSession)
    assert engine._model.providers == ["CPUExecutionProvider"]
    assert engine._model.last_values.tolist() == ["difficulty breathing"]
    assert prediction.condition == "respiratory"
    assert prediction.confidence == pytest.approx(0.84)
    assert prediction.priority == "high"
    assert prediction.backend == "onnx"
    assert prediction.model_version == "onnx-v1"


def test_onnx_adapter_supports_rank_two_input_and_array_probabilities(tmp_path: Path) -> None:
    engine = InferenceEngine(settings_for(tmp_path, "onnx"))
    session = FakeOnnxSession("unused", ["CPUExecutionProvider"])
    session.get_inputs = lambda: [SimpleNamespace(name="medical_text", shape=[None, 1])]
    session.outputs = [np.asarray(["condition-a"]), np.asarray([[0.25, 0.75]])]
    engine._model = session
    engine.backend = "onnx"
    engine.model_version = "test"

    condition, confidence = engine._predict_onnx("rank two")

    assert session.last_values.tolist() == [["rank two"]]
    assert condition == "condition-a"
    assert confidence == pytest.approx(0.75)


def test_onnx_adapter_rejects_missing_or_empty_label_output(tmp_path: Path) -> None:
    engine = InferenceEngine(settings_for(tmp_path, "onnx"))
    session = FakeOnnxSession("unused", [])
    engine._model = session

    session.outputs = []
    with pytest.raises(RuntimeError, match="no outputs"):
        engine._predict_onnx("text")

    session.outputs = [np.asarray([])]
    with pytest.raises(RuntimeError, match="no labels"):
        engine._predict_onnx("text")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("not-a-number", None),
        (float("nan"), None),
        (-0.5, 0.0),
        (1.5, 1.0),
        (0.42, 0.42),
    ],
)
def test_safe_confidence(raw: Any, expected: float | None) -> None:
    assert _safe_confidence(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"condition": "a", "confidence": 0.8}, ("a", 0.8)),
        ({"label": "b", "score": "0.6"}, ("b", 0.6)),
        ({"prediction": "c"}, ("c", None)),
        (("d", 0.9), ("d", 0.9)),
        (("e",), ("e", None)),
        ("f", ("f", None)),
    ],
)
def test_unpack_prediction(raw: Any, expected: tuple[str, float | None]) -> None:
    assert _unpack_prediction(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"priority": "high", "reason": "red flag"}, ("high", "red flag")),
        ({"priority": "medium", "priority_reason": "rule"}, ("medium", "rule")),
        ({}, ("unknown", None)),
        (("low", "no match"), ("low", "no match")),
        (("low",), ("low", None)),
        ("medium", ("medium", None)),
    ],
)
def test_unpack_priority(raw: Any, expected: tuple[str, str | None]) -> None:
    assert _unpack_priority(raw) == expected


def test_onnx_confidence_handles_no_output_bad_array_and_empty_map(tmp_path: Path) -> None:
    engine = InferenceEngine(settings_for(tmp_path))

    assert engine._onnx_confidence([]) is None
    assert engine._onnx_confidence([[{}]]) is None
    assert engine._onnx_confidence([["not-a-number"]]) is None
