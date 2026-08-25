"""Prometheus instrumentation without medical text or patient identifiers."""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class ApiMetrics:
    """Metrics owned by one application instance.

    A dedicated registry keeps TestClient/application factory usage isolated and
    prevents duplicated collectors. Labels deliberately contain only bounded
    operational values; submitted text is never observed or exported.
    """

    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        self.http_requests = Counter(
            "triage_http_requests_total",
            "HTTP requests handled by the triage service.",
            ("method", "endpoint", "status"),
            registry=self.registry,
        )
        self.http_latency = Histogram(
            "triage_http_request_duration_seconds",
            "HTTP request latency in seconds.",
            ("method", "endpoint"),
            registry=self.registry,
        )
        self.predictions = Counter(
            "triage_predictions_total",
            "Predictions produced by backend and normalized priority.",
            ("backend", "priority"),
            registry=self.registry,
        )
        self.inference_latency = Histogram(
            "triage_inference_duration_seconds",
            "Model-only inference latency in seconds.",
            ("backend",),
            registry=self.registry,
        )
        self.model_ready = Gauge(
            "triage_model_ready",
            "Whether a model artifact was loaded and is ready for inference.",
            registry=self.registry,
        )
        self.prediction_errors = Counter(
            "triage_prediction_errors_total",
            "Prediction failures grouped by bounded error category.",
            ("category",),
            registry=self.registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)


KNOWN_ENDPOINTS = frozenset({"/health", "/predict", "/metrics", "/docs", "/openapi.json"})


def normalized_endpoint(path: str) -> str:
    """Avoid unbounded metrics labels from arbitrary URLs."""

    return path if path in KNOWN_ENDPOINTS else "unmatched"
