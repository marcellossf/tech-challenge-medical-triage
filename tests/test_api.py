"""Contract tests for the HTTP API."""

from fastapi.testclient import TestClient

from medical_triage.api import create_app
from medical_triage.config import Settings
from medical_triage.inference import ModelUnavailableError, Prediction


class FakeEngine:
    def __init__(self, settings: Settings, *, available: bool = True) -> None:
        self.settings = settings
        self.available = available
        self.backend = "sklearn" if available else None
        self.model_version = "test-1" if available else None
        self.load_error = None if available else "sklearn artifact not found"
        self.load_calls = 0

    @property
    def ready(self) -> bool:
        return self.available

    def load(self) -> None:
        self.load_calls += 1

    def predict(self, text: str) -> Prediction:
        if not self.ready:
            raise ModelUnavailableError(self.load_error)
        return Prediction(
            condition="cardiovascular",
            priority="urgent",
            priority_source="business_rule",
            priority_reason="urgent symptom keyword",
            backend="sklearn",
            model_version="test-1",
            confidence=0.91,
            inference_ms=1.25,
        )


def test_health_and_predict_contract_and_single_load() -> None:
    engine = FakeEngine(Settings())
    app = create_app(engine_factory=lambda _settings: engine)

    with TestClient(app) as client:
        health = client.get("/health")
        first = client.post("/predict", json={"text": "  severe chest pain  "})
        second = client.post("/predict", json={"text": "shortness of breath"})

    assert engine.load_calls == 1
    assert health.status_code == 200
    assert health.json() == {
        "status": "healthy",
        "model_loaded": True,
        "backend": "sklearn",
        "model_version": "test-1",
        "detail": None,
    }
    assert first.status_code == 200
    assert first.json() == {
        "condition": "cardiovascular",
        "priority": "urgent",
        "priority_source": "business_rule",
        "priority_reason": "urgent symptom keyword",
        "backend": "sklearn",
        "model_version": "test-1",
        "confidence": 0.91,
        "inference_ms": 1.25,
    }
    assert second.status_code == 200


def test_missing_artifact_degrades_health_and_blocks_prediction() -> None:
    engine = FakeEngine(Settings(), available=False)
    app = create_app(engine_factory=lambda _settings: engine)

    with TestClient(app) as client:
        health = client.get("/health")
        prediction = client.post("/predict", json={"text": "medical text"})

    assert health.status_code == 200
    assert health.json() == {
        "status": "degraded",
        "model_loaded": False,
        "backend": None,
        "model_version": None,
        "detail": "sklearn artifact not found",
    }
    assert prediction.status_code == 503
    assert prediction.json() == {"detail": "sklearn artifact not found"}


def test_predict_rejects_blank_text_and_extra_fields() -> None:
    app = create_app(engine_factory=lambda settings: FakeEngine(settings))

    with TestClient(app) as client:
        blank = client.post("/predict", json={"text": "   "})
        extra = client.post("/predict", json={"text": "valid", "patient_id": "secret"})

    assert blank.status_code == 422
    assert extra.status_code == 422


def test_metrics_contain_operations_but_never_submitted_text() -> None:
    app = create_app(engine_factory=lambda settings: FakeEngine(settings))
    submitted_text = "unique private clinical sentence 83429"

    with TestClient(app) as client:
        assert client.post("/predict", json={"text": submitted_text}).status_code == 200
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "triage_predictions_total" in response.text
    assert 'backend="sklearn"' in response.text
    assert 'priority="urgent"' in response.text
    assert submitted_text not in response.text
    assert "cardiovascular" not in response.text
