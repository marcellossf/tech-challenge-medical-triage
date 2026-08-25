"""FastAPI application for medical text classification."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, status

from medical_triage.config import Settings, get_settings
from medical_triage.inference import InferenceEngine, ModelUnavailableError
from medical_triage.metrics import ApiMetrics, normalized_endpoint
from medical_triage.schemas import ErrorResponse, HealthResponse, PredictRequest, PredictResponse

EngineFactory = Callable[[Settings], InferenceEngine]


def create_app(
    settings: Settings | None = None,
    engine_factory: EngineFactory = InferenceEngine,
) -> FastAPI:
    """Build an isolated app instance, which also makes tests deterministic."""

    runtime_settings = settings or get_settings()
    metrics = ApiMetrics()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        engine = engine_factory(runtime_settings)
        engine.load()
        application.state.engine = engine
        application.state.metrics = metrics
        metrics.model_ready.set(1 if engine.ready else 0)
        yield

    application = FastAPI(
        title=runtime_settings.app_name,
        version=runtime_settings.app_version,
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def observe_http(request: Request, call_next: Callable) -> Response:
        started = time.perf_counter()
        endpoint = normalized_endpoint(request.url.path)
        try:
            response = await call_next(request)
        except Exception:
            metrics.http_requests.labels(request.method, endpoint, "500").inc()
            metrics.http_latency.labels(request.method, endpoint).observe(
                time.perf_counter() - started
            )
            raise
        metrics.http_requests.labels(request.method, endpoint, str(response.status_code)).inc()
        metrics.http_latency.labels(request.method, endpoint).observe(time.perf_counter() - started)
        return response

    @application.get("/health", response_model=HealthResponse, tags=["operations"])
    def health(request: Request) -> HealthResponse:
        engine: InferenceEngine = request.app.state.engine
        return HealthResponse(
            status="healthy" if engine.ready else "degraded",
            model_loaded=engine.ready,
            backend=engine.backend,
            model_version=engine.model_version,
            detail=None if engine.ready else engine.load_error,
        )

    @application.post(
        "/predict",
        response_model=PredictResponse,
        responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse}},
        tags=["inference"],
    )
    def predict(payload: PredictRequest, request: Request) -> PredictResponse:
        engine: InferenceEngine = request.app.state.engine
        try:
            prediction = engine.predict(payload.text)
        except ModelUnavailableError as exc:
            metrics.prediction_errors.labels("model_unavailable").inc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            metrics.prediction_errors.labels("inference_failure").inc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="model inference failed",
            ) from exc

        metrics.predictions.labels(prediction.backend, prediction.priority).inc()
        metrics.inference_latency.labels(prediction.backend).observe(
            prediction.inference_ms / 1_000
        )
        return PredictResponse(**prediction.__dict__)

    @application.get("/metrics", include_in_schema=False, tags=["operations"])
    def prometheus_metrics() -> Response:
        return Response(content=metrics.render(), media_type="text/plain; version=0.0.4")

    return application


app = create_app()
