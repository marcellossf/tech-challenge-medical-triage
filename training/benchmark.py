"""Benchmark native and ONNX single-record inference with warmup."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from medical_triage.data import sha256_file
from medical_triage.modeling import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--onnx", type=Path, default=Path("artifacts/model.onnx"))
    parser.add_argument("--test-data", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
    return parser.parse_args()


def _measure(call: Callable[[str], object], texts: list[str], warmup: int) -> list[float]:
    for index in range(warmup):
        call(texts[index % len(texts)])

    timings_ms: list[float] = []
    for text in texts:
        start = time.perf_counter_ns()
        call(text)
        timings_ms.append((time.perf_counter_ns() - start) / 1_000_000)
    return timings_ms


def _summary(timings_ms: list[float]) -> dict[str, float | int]:
    values = np.asarray(timings_ms, dtype=float)
    return {
        "runs": int(len(values)),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "throughput_per_second": float(1_000 / values.mean()),
    }


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("--warmup must be non-negative and --iterations must be at least 1")

    model = load_model(args.model)
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    available_texts = pd.read_csv(args.test_data)["medical_abstract"].astype(str).tolist()
    if not available_texts:
        raise ValueError("Test data is empty")
    texts = [available_texts[index % len(available_texts)] for index in range(args.iterations)]

    native_timings = _measure(lambda text: model.predict([text]), texts, args.warmup)
    onnx_timings = _measure(
        lambda text: session.run(None, {input_name: np.asarray([[text]])}),
        texts,
        args.warmup,
    )
    native = _summary(native_timings)
    onnx = _summary(onnx_timings)
    result = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "methodology": {
            "workload": "single-record end-to-end prediction",
            "warmup_runs_per_runtime": args.warmup,
            "measured_runs_per_runtime": args.iterations,
            "clock": "time.perf_counter_ns",
        },
        "artifacts": {
            "native_sha256": sha256_file(args.model),
            "onnx_sha256": sha256_file(args.onnx),
        },
        "native": native,
        "onnx": onnx,
        "p50_speedup": float(native["p50_ms"] / onnx["p50_ms"]),
        "p95_speedup": float(native["p95_ms"] / onnx["p95_ms"]),
        "p99_speedup": float(native["p99_ms"] / onnx["p99_ms"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
