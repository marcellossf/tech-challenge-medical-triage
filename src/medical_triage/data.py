"""Reproducible download and validation of the Medical Abstracts dataset."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import pandas as pd

DATASET_REPOSITORY = "https://github.com/sebischair/Medical-Abstracts-TC-Corpus"
DATASET_COMMIT = "70a2d9106c724729be8b3c4ddb00d1b14ec300c8"
RAW_BASE_URL = (
    f"https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/{DATASET_COMMIT}"
)


@dataclass(frozen=True)
class DatasetFile:
    """An immutable upstream file specification."""

    filename: str
    sha256: str

    @property
    def url(self) -> str:
        return f"{RAW_BASE_URL}/{self.filename}"


DATASET_FILES = (
    DatasetFile(
        "medical_tc_train.csv",
        "ad53aebc682d6b87a5647f619a079bb446d286fdc93bf0159b812418f5758609",
    ),
    DatasetFile(
        "medical_tc_test.csv",
        "1eecea73c9ecad292c55e10403bd139fab9580545d6878482997c5564d51ac05",
    ),
    DatasetFile(
        "medical_tc_labels.csv",
        "8a27ae03339c798103678efa8012f744a723ff71a80f2b2c1355ee249564adc5",
    ),
)


class DatasetIntegrityError(ValueError):
    """Raised when downloaded or local data does not match the expected schema."""


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a file SHA-256 without loading the whole file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        while chunk := file_handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(spec: DatasetFile, destination: Path, timeout: float) -> None:
    temporary_path = destination.with_suffix(f"{destination.suffix}.part")
    try:
        with urlopen(spec.url, timeout=timeout) as response, temporary_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)

        actual_checksum = sha256_file(temporary_path)
        if actual_checksum != spec.sha256:
            raise DatasetIntegrityError(
                f"Checksum mismatch for {spec.filename}: expected {spec.sha256}, "
                f"got {actual_checksum}"
            )
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def download_dataset(
    raw_dir: str | Path = "data/raw", *, force: bool = False, timeout: float = 120
) -> dict[str, Path]:
    """Download the dataset at the pinned commit and verify every file checksum."""

    destination_dir = Path(raw_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, Path] = {}

    for spec in DATASET_FILES:
        destination = destination_dir / spec.filename
        if destination.exists() and not force:
            actual_checksum = sha256_file(destination)
            if actual_checksum != spec.sha256:
                raise DatasetIntegrityError(
                    f"Existing file {destination} has checksum {actual_checksum}; "
                    "remove it or pass force=True to download a verified copy"
                )
        else:
            _download_file(spec, destination, timeout)
        downloaded[spec.filename] = destination

    return downloaded


def _read_labels(path: Path) -> dict[int, str]:
    labels = pd.read_csv(path)
    expected_columns = {"condition_label", "condition_name"}
    if set(labels.columns) != expected_columns:
        raise DatasetIntegrityError(
            f"Invalid label schema in {path}; expected {sorted(expected_columns)}"
        )

    labels = labels.dropna().copy()
    labels["condition_label"] = labels["condition_label"].astype(int)
    labels["condition_name"] = labels["condition_name"].astype(str).str.strip()
    if labels["condition_label"].duplicated().any() or labels["condition_name"].eq("").any():
        raise DatasetIntegrityError(f"Duplicate or empty labels in {path}")
    return dict(zip(labels["condition_label"], labels["condition_name"], strict=True))


def _read_split(path: Path, label_names: dict[int, str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected_columns = {"condition_label", "medical_abstract"}
    if set(frame.columns) != expected_columns:
        raise DatasetIntegrityError(
            f"Invalid data schema in {path}; expected {sorted(expected_columns)}"
        )
    if frame.empty or frame[list(expected_columns)].isna().any().any():
        raise DatasetIntegrityError(f"Empty dataset or null values in {path}")

    normalized = frame.copy()
    normalized["condition_label"] = normalized["condition_label"].astype(int)
    normalized["medical_abstract"] = normalized["medical_abstract"].astype(str).str.strip()
    if normalized["medical_abstract"].eq("").any():
        raise DatasetIntegrityError(f"Empty medical abstract in {path}")

    unknown_labels = sorted(set(normalized["condition_label"]) - set(label_names))
    if unknown_labels:
        raise DatasetIntegrityError(f"Unknown condition labels in {path}: {unknown_labels}")
    normalized["condition_name"] = normalized["condition_label"].map(label_names)
    return normalized[["condition_label", "condition_name", "medical_abstract"]]


def load_dataset(
    raw_dir: str | Path = "data/raw",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    """Load and validate the original train/test split plus its label mapping."""

    source_dir = Path(raw_dir)
    paths = {spec.filename: source_dir / spec.filename for spec in DATASET_FILES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Dataset files not found: {', '.join(missing)}")

    label_names = _read_labels(paths["medical_tc_labels.csv"])
    train = _read_split(paths["medical_tc_train.csv"], label_names)
    test = _read_split(paths["medical_tc_test.csv"], label_names)
    return train, test, label_names


def _distribution(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["condition_name"].value_counts().sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def prepare_dataset(
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    *,
    download: bool = True,
    force_download: bool = False,
) -> dict[str, Any]:
    """Create validated processed CSVs and provenance metadata.

    The official train/test split is preserved. Raw files remain in ``data/raw``,
    which is intentionally ignored by Git.
    """

    raw_path = Path(raw_dir)
    if download:
        download_dataset(raw_path, force=force_download)
    train, test, label_names = load_dataset(raw_path)

    output_dir = Path(processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train.to_csv(train_path, index=False, lineterminator="\n")
    test.to_csv(test_path, index=False, lineterminator="\n")

    raw_checksums = {spec.filename: sha256_file(raw_path / spec.filename) for spec in DATASET_FILES}
    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "repository": DATASET_REPOSITORY,
            "commit": DATASET_COMMIT,
            "license": "CC BY-SA 3.0",
            "files": [asdict(spec) | {"url": spec.url} for spec in DATASET_FILES],
            "raw_checksums": raw_checksums,
        },
        "labels": {str(key): value for key, value in sorted(label_names.items())},
        "splits": {
            "train": {"rows": len(train), "distribution": _distribution(train)},
            "test": {"rows": len(test), "distribution": _distribution(test)},
        },
        "processed_files": {
            "train.csv": sha256_file(train_path),
            "test.csv": sha256_file(test_path),
        },
    }
    metadata_path = output_dir / "dataset_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata
