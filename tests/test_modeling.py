from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from medical_triage.modeling import (
    evaluate_model,
    load_model,
    predict_with_confidence,
    save_model,
    train_model,
)


@pytest.fixture
def training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "medical_abstract": [
                "malignant tumor oncology treatment",
                "cancer neoplasm tumor study",
                "chemotherapy for malignant cancer",
                "digestive stomach bowel disorder",
                "intestinal disease digestion study",
                "gastrointestinal stomach treatment",
            ],
            "condition_name": ["neoplasms"] * 3 + ["digestive"] * 3,
        }
    )


def test_training_evaluation_and_prediction_are_serializable(
    training_frame: pd.DataFrame,
) -> None:
    model = train_model(training_frame)

    label, confidence = predict_with_confidence(model, "malignant cancer tumor")
    metrics = evaluate_model(model, training_frame)

    assert label == "neoplasms"
    assert 0.5 <= confidence <= 1.0
    assert metrics["accuracy"] >= 0.9
    json.dumps(metrics)


def test_model_round_trip(tmp_path: Path, training_frame: pd.DataFrame) -> None:
    model = train_model(training_frame)
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"

    save_model(model, model_path, metadata={"version": 1}, metadata_path=metadata_path)
    loaded = load_model(model_path)

    assert loaded.predict(["digestive bowel disease"])[0] == "digestive"
    assert json.loads(metadata_path.read_text()) == {"version": 1}


def test_training_rejects_one_class() -> None:
    frame = pd.DataFrame({"medical_abstract": ["one", "two"], "condition_name": ["same", "same"]})
    with pytest.raises(ValueError, match="at least two classes"):
        train_model(frame)


def test_prediction_rejects_blank_text(training_frame: pd.DataFrame) -> None:
    model = train_model(training_frame)
    with pytest.raises(ValueError, match="blank"):
        predict_with_confidence(model, "   ")
