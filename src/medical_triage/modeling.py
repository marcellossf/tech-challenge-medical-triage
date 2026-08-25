"""Training, evaluation, persistence, and inference helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

TEXT_COLUMN = "medical_abstract"
TARGET_COLUMN = "condition_name"


def build_pipeline(*, random_state: int = 42) -> Pipeline:
    """Build a deterministic and CPU-friendly text classification baseline."""

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 1),
                    max_features=40_000,
                    min_df=1,
                    # Raw term frequency preserves numerical parity in skl2onnx.
                    sublinear_tf=False,
                    # Kept at None because skl2onnx does not convert accent stripping.
                    strip_accents=None,
                    # Explicit ASCII regex keeps sklearn and ONNX token boundaries identical.
                    token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _training_columns(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if TEXT_COLUMN not in frame or TARGET_COLUMN not in frame:
        raise ValueError(f"Training data must contain {TEXT_COLUMN!r} and {TARGET_COLUMN!r}")
    if frame.empty or frame[[TEXT_COLUMN, TARGET_COLUMN]].isna().any().any():
        raise ValueError("Training data cannot be empty or contain null values")

    texts = frame[TEXT_COLUMN].astype(str).str.strip()
    targets = frame[TARGET_COLUMN].astype(str).str.strip()
    if texts.eq("").any() or targets.eq("").any():
        raise ValueError("Training texts and labels cannot be blank")
    if targets.nunique() < 2:
        raise ValueError("Training data must contain at least two classes")
    return texts, targets


def train_model(train_frame: pd.DataFrame, *, random_state: int = 42) -> Pipeline:
    """Fit the baseline pipeline from a validated processed dataframe."""

    texts, targets = _training_columns(train_frame)
    model = build_pipeline(random_state=random_state)
    model.fit(texts, targets)
    return model


def evaluate_model(model: Pipeline, test_frame: pd.DataFrame) -> dict[str, Any]:
    """Return JSON-serializable classification metrics."""

    if TEXT_COLUMN not in test_frame or TARGET_COLUMN not in test_frame or test_frame.empty:
        raise ValueError(f"Test data must contain non-empty {TEXT_COLUMN!r} and {TARGET_COLUMN!r}")
    texts = test_frame[TEXT_COLUMN].astype(str)
    targets = test_frame[TARGET_COLUMN].astype(str)
    predictions = model.predict(texts)
    report = classification_report(targets, predictions, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "f1_macro": float(f1_score(targets, predictions, average="macro")),
        "classification_report": report,
        "test_samples": int(len(test_frame)),
    }


def save_model(
    model: Pipeline,
    model_path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    metadata_path: str | Path | None = None,
) -> None:
    """Persist a fitted pipeline and, optionally, its JSON metadata atomically."""

    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    try:
        joblib.dump(model, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    if metadata is not None:
        metadata_destination = Path(metadata_path or destination.with_suffix(".metadata.json"))
        metadata_destination.parent.mkdir(parents=True, exist_ok=True)
        metadata_destination.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def load_model(model_path: str | Path) -> Pipeline:
    """Load a trusted model artifact created by :func:`save_model`."""

    model = joblib.load(Path(model_path))
    if not isinstance(model, Pipeline):
        raise TypeError("The model artifact does not contain a scikit-learn Pipeline")
    return model


def predict_with_confidence(model: Pipeline, text: str) -> tuple[str, float]:
    """Predict one condition label and the corresponding class probability."""

    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("Text cannot be blank")
    probabilities = model.predict_proba([normalized_text])[0]
    best_index = int(probabilities.argmax())
    label = str(model.classes_[best_index])
    return label, float(probabilities[best_index])
