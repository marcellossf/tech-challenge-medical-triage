"""Train and evaluate the TF-IDF plus Logistic Regression model."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn

from medical_triage.data import DATASET_COMMIT, prepare_dataset, sha256_file
from medical_triage.modeling import evaluate_model, save_model, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-prepare", action="store_true")
    return parser.parse_args()


def _json_safe_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in params.items()
        if isinstance(value, (str, int, float, bool, type(None), tuple))
    }


def main() -> None:
    args = parse_args()
    train_path = args.processed_dir / "train.csv"
    test_path = args.processed_dir / "test.csv"
    dataset_metadata_path = args.processed_dir / "dataset_metadata.json"
    if not args.skip_prepare:
        prepare_dataset(args.raw_dir, args.processed_dir)
    for required_path in (train_path, test_path, dataset_metadata_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required processed file not found: {required_path}")

    train_frame = pd.read_csv(train_path)
    test_frame = pd.read_csv(test_path)
    model = train_model(train_frame, random_state=args.random_state)
    metrics = evaluate_model(model, test_frame)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifacts_dir / "model.joblib"
    metadata_path = args.artifacts_dir / "model_metadata.json"
    dataset_metadata = json.loads(dataset_metadata_path.read_text(encoding="utf-8"))
    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_version": "0.1.0",
        "model_type": "tfidf_logistic_regression",
        "dataset_commit": DATASET_COMMIT,
        "dataset_processed_checksums": dataset_metadata["processed_files"],
        "random_state": args.random_state,
        "train_samples": int(len(train_frame)),
        "classes": [str(label) for label in model.classes_],
        "parameters": _json_safe_params(model.get_params(deep=True)),
        "metrics": metrics,
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
        },
    }
    save_model(model, model_path)
    metadata["model_sha256"] = sha256_file(model_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"model": str(model_path), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
