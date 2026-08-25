"""Export the trained scikit-learn pipeline to ONNX and verify parity."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

from medical_triage.data import sha256_file
from medical_triage.modeling import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument(
        "--source-metadata", type=Path, default=Path("artifacts/model_metadata.json")
    )
    parser.add_argument("--test-data", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.onnx"))
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--target-opset", type=int, default=17)
    parser.add_argument("--probability-tolerance", type=float, default=1e-4)
    return parser.parse_args()


def _outputs_by_shape(outputs: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(outputs[0]).reshape(-1)
    probabilities = outputs[1]
    if isinstance(probabilities, list) and probabilities and isinstance(probabilities[0], dict):
        ordered_keys = sorted(probabilities[0])
        probabilities = np.asarray(
            [[row[key] for key in ordered_keys] for row in probabilities], dtype=float
        )
    return labels.astype(str), np.asarray(probabilities, dtype=float)


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")
    model = load_model(args.model)
    test_frame = pd.read_csv(args.test_data)
    texts = test_frame["medical_abstract"].astype(str).head(args.samples).to_numpy()
    if len(texts) == 0:
        raise ValueError("Test data is empty")

    onnx_model = convert_sklearn(
        model,
        name="medical_triage_tfidf_logistic_regression",
        initial_types=[("text", StringTensorType([None, 1]))],
        options={"zipmap": False},
        target_opset=args.target_opset,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(onnx_model.SerializeToString())

    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_outputs = session.run(None, {input_name: texts.reshape(-1, 1)})
    onnx_labels, onnx_probabilities = _outputs_by_shape(onnx_outputs)
    native_labels = model.predict(texts).astype(str)
    native_probabilities = model.predict_proba(texts)
    agreement = float(np.mean(onnx_labels == native_labels))
    max_probability_difference = float(np.max(np.abs(onnx_probabilities - native_probabilities)))
    parity = {
        "samples": int(len(texts)),
        "prediction_agreement": agreement,
        "max_probability_difference": max_probability_difference,
        "probability_tolerance": args.probability_tolerance,
        "passed": agreement == 1.0 and max_probability_difference <= args.probability_tolerance,
    }
    if not parity["passed"]:
        args.output.unlink(missing_ok=True)
        raise RuntimeError(f"ONNX parity check failed: {parity}")

    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_version": "unknown",
        "source_model_sha256": sha256_file(args.model),
        "onnx_sha256": sha256_file(args.output),
        "onnx_size_bytes": args.output.stat().st_size,
        "target_opset": args.target_opset,
        "providers": session.get_providers(),
        "parity": parity,
    }
    if args.source_metadata.is_file():
        source_metadata = json.loads(args.source_metadata.read_text(encoding="utf-8"))
        metadata["model_version"] = str(source_metadata.get("model_version", "unknown"))
        metadata["classes"] = source_metadata.get("classes", [])
    metadata_path = args.output.with_name("onnx_metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
