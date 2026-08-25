from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from medical_triage.data import DatasetIntegrityError, load_dataset, prepare_dataset, sha256_file


def _write_raw_dataset(raw_dir: Path, *, unknown_label: bool = False) -> None:
    raw_dir.mkdir()
    pd.DataFrame({"condition_label": [1, 2], "condition_name": ["neoplasms", "digestive"]}).to_csv(
        raw_dir / "medical_tc_labels.csv", index=False
    )
    label = 9 if unknown_label else 1
    pd.DataFrame(
        {
            "condition_label": [label, 2],
            "medical_abstract": ["Tumor pathology report", "Digestive tract inflammation"],
        }
    ).to_csv(raw_dir / "medical_tc_train.csv", index=False)
    pd.DataFrame(
        {
            "condition_label": [1, 2],
            "medical_abstract": ["Neoplasm study", "Gastrointestinal disease"],
        }
    ).to_csv(raw_dir / "medical_tc_test.csv", index=False)


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_bytes(b"medical triage")
    assert sha256_file(path) == hashlib.sha256(b"medical triage").hexdigest()


def test_load_dataset_maps_human_readable_labels(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_raw_dataset(raw_dir)

    train, test, labels = load_dataset(raw_dir)

    assert labels == {1: "neoplasms", 2: "digestive"}
    assert train["condition_name"].tolist() == ["neoplasms", "digestive"]
    assert len(test) == 2


def test_load_dataset_rejects_unknown_labels(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_raw_dataset(raw_dir, unknown_label=True)

    with pytest.raises(DatasetIntegrityError, match="Unknown condition labels"):
        load_dataset(raw_dir)


def test_prepare_dataset_writes_checksummed_metadata(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _write_raw_dataset(raw_dir)

    metadata = prepare_dataset(raw_dir, processed_dir, download=False)

    saved_metadata = json.loads((processed_dir / "dataset_metadata.json").read_text())
    assert metadata["splits"]["train"]["rows"] == 2
    assert saved_metadata["processed_files"]["train.csv"] == sha256_file(
        processed_dir / "train.csv"
    )
