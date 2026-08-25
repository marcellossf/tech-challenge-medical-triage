"""Generate repeatable traffic for the Prometheus and Grafana demonstration."""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

EXAMPLES = (
    "Dor leve no joelho há duas semanas, sem febre ou inchaço.",
    "Febre alta, tosse persistente e dificuldade para respirar desde ontem.",
    "Dor forte no peito, suor frio e falta de ar iniciados agora.",
    "Coceira nos olhos e espirros frequentes ao acordar.",
    "Confusão repentina, fala enrolada e fraqueza no lado direito do corpo.",
    "Náusea moderada e dor abdominal depois do almoço.",
)


@dataclass(frozen=True)
class Result:
    status: int
    elapsed_ms: float
    error: str | None = None


def send_request(url: str, text: str, timeout: float) -> Result:
    payload = json.dumps({"text": text}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            return Result(response.status, (time.perf_counter() - started) * 1000)
    except urllib.error.HTTPError as exc:
        return Result(exc.code, (time.perf_counter() - started) * 1000, str(exc))
    except urllib.error.URLError as exc:
        return Result(0, (time.perf_counter() - started) * 1000, str(exc.reason))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--requests", type=int, default=100, dest="request_count")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.request_count < 1 or args.concurrency < 1:
        raise SystemExit("--requests and --concurrency must be positive")

    randomizer = random.Random(args.seed)
    texts = [randomizer.choice(EXAMPLES) for _ in range(args.request_count)]
    results: list[Result] = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(send_request, args.url, text, args.timeout) for text in texts]
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 25 == 0 or completed == args.request_count:
                print(f"Completed {completed}/{args.request_count}")

    successes = sum(200 <= result.status < 300 for result in results)
    failures = len(results) - successes
    latencies = sorted(result.elapsed_ms for result in results)
    p95_index = max(0, round(0.95 * len(latencies)) - 1)
    print(
        f"success={successes} failure={failures} "
        f"mean_ms={sum(latencies) / len(latencies):.2f} p95_ms={latencies[p95_index]:.2f}"
    )
    if failures:
        first_error = next(result for result in results if result.error)
        print(f"first_error=status:{first_error.status} detail:{first_error.error}")
    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
