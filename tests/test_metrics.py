"""Unit tests for safe Prometheus metric construction."""

from medical_triage.metrics import ApiMetrics, normalized_endpoint


def test_arbitrary_paths_are_collapsed_to_bounded_label() -> None:
    assert normalized_endpoint("/predict") == "/predict"
    assert normalized_endpoint("/patient/secret-identifier") == "unmatched"


def test_each_application_gets_an_isolated_registry() -> None:
    first = ApiMetrics()
    second = ApiMetrics()

    first.predictions.labels("sklearn", "normal").inc()

    first_payload = first.render().decode()
    second_payload = second.render().decode()
    assert 'triage_predictions_total{backend="sklearn",priority="normal"} 1.0' in first_payload
    assert 'triage_predictions_total{backend="sklearn",priority="normal"}' not in second_payload


def test_metric_labels_do_not_accept_medical_text_or_condition() -> None:
    metrics = ApiMetrics()

    assert metrics.predictions._labelnames == ("backend", "priority")
    assert metrics.http_requests._labelnames == ("method", "endpoint", "status")
    assert "text" not in metrics.render().decode().lower()
