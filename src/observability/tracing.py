"""
Distributed tracing with OpenTelemetry.

Traces:
- HTTP requests (automatic via FastAPI instrumentation)
- LLM calls (manual spans in generate_node)
- Retrieval operations (manual spans in retrieve_node)

Export backends:
- OTLP/gRPC (production) — point OTLP_ENDPOINT at Grafana Tempo, Jaeger, or
  any OpenTelemetry Collector.
- Console (development) — prints spans to stdout when OTLP_ENDPOINT is unset.

Usage in application code:
    from src.observability.tracing import get_tracer
    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("my_operation") as span:
        span.set_attribute("key", "value")
        # ... do work
"""

from typing import Optional

from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from src.observability.logging_config import get_logger

log = get_logger(__name__)

_provider: Optional[TracerProvider] = None


def configure_tracing(app=None) -> TracerProvider:
    """
    Initialise OpenTelemetry tracing.

    Should be called once during application startup (in the FastAPI lifespan).
    Subsequent calls are no-ops — the provider is only set globally once.

    Args:
        app: FastAPI instance. When provided, HTTP requests are auto-instrumented
             and every incoming request gets its own root span.

    Returns:
        The configured TracerProvider.
    """
    global _provider
    if _provider is not None:
        return _provider

    from src.config.settings import settings

    resource = Resource.create({
        "service.name": "k8s-rag-chatbot",
        "service.version": "1.0.0",
        "deployment.environment": settings.environment,
    })

    provider = TracerProvider(resource=resource)

    if settings.otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        exporter = OTLPSpanExporter(
            endpoint=settings.otlp_endpoint,
            insecure=True,
        )
        log.info("tracing_configured", exporter="otlp", endpoint=settings.otlp_endpoint)
    else:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        exporter = ConsoleSpanExporter()
        log.info("tracing_configured", exporter="console")

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    if app is not None:
        FastAPIInstrumentor.instrument_app(app)
        log.info("fastapi_instrumented", message="Auto-tracing enabled for HTTP requests")

    _provider = provider
    return provider


def get_tracer(name: str = __name__) -> trace.Tracer:
    """
    Return a named tracer for manual span instrumentation.

    Safe to call before configure_tracing() — OpenTelemetry returns a
    no-op tracer if no provider has been configured yet.
    """
    return trace.get_tracer(name)
