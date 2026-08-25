"""Download, validate, and materialize the processed dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medical_triage.data import prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare_dataset(
        args.raw_dir,
        args.processed_dir,
        force_download=args.force_download,
    )
    summary = {
        "train_rows": metadata["splits"]["train"]["rows"],
        "test_rows": metadata["splits"]["test"]["rows"],
        "source_commit": metadata["source"]["commit"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
