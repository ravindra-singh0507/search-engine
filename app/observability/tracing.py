"""
Distributed Tracing — Phase 8 Batch 4

=== THEORY ===

Distributed tracing follows a request across multiple services and
operations.  It answers "why was request X slow?" by showing exactly
which service/operation consumed the time.

A Trace is a tree of Spans.  Each Span represents one operation (a
function call, an RPC, a DB query).  The root Span represents the
entire request; child spans represent sub-operations.

Spans are connected via:
  trace_id  — identical for every span in one request (UUID4)
  span_id   — unique ID for this span
  parent_id — span_id of the parent (None for root spans)

Key metadata:
  operation  — human-readable name ("search", "embed", "bm25_rank")
  service    — which microservice ("search-engine", "indexer")
  start_time — Unix timestamp (time.time())
  end_time   — set when span.finish() is called
  status     — "ok" or "error"
  tags       — key-value annotations (query, user_id, result_count…)
  events     — timestamped log entries within the span

=== OPENTELEMETRY (OTLP) ===

OpenTelemetry is the CNCF standard for distributed tracing.  It defines:
  - API:      how to instrument code (start_span, add_event, set_attribute)
  - SDK:      how to collect/process/export spans
  - Protocol: OTLP — gRPC or HTTP/protobuf export format

Backends: Jaeger, Zipkin, Grafana Tempo, AWS X-Ray, Datadog APM.

This implementation uses an in-process ring buffer (bounded deque) for
development and optionally exports via OTLP if opentelemetry-sdk is
installed.

=== SAMPLING ===

Recording every span in production is too expensive.  Sampling strategies:
  - Head sampling:         decide at the root span whether to record (this file)
  - Tail sampling:         decide after the trace completes (latency/errors)
  - Probabilistic:         record N% of traces (trace_sample_rate in config)

=== PRODUCTION EQUIVALENTS ===

  Jaeger (Uber):    jaeger-client-python, exported as UDP thrift
  Zipkin (Twitter): py_zipkin, exported as JSON over HTTP
  AWS X-Ray:        aws-xray-sdk-python, daemon UDP forwarding
  Datadog APM:      ddtrace, agent forwarding
  Grafana Tempo:    OpenTelemetry SDK + OTLP exporter
"""

import logging
import random
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from app.config import ObservabilityConfig2

logger = logging.getLogger(__name__)


# ── Span ──────────────────────────────────────────────────────────────────────


@dataclass
class Span:
    """
    Lightweight span for tracing a single operation.

    A Span is the fundamental unit of distributed tracing.  It records
    one logical operation: its name, timing, status, and any annotations.

    Thread safety: each Span carries its own threading.Lock so tags/events
    can be safely written from any thread (e.g. callbacks that annotate a
    parent span while child spans run concurrently).
    """

    trace_id:   str
    span_id:    str
    parent_id:  str | None
    operation:  str
    service:    str
    start_time: float
    end_time:   float | None = None
    status:     str          = "ok"   # ok | error
    tags:       dict         = field(default_factory=dict)
    events:     list[dict]   = field(default_factory=list)
    # Internal lock — excluded from repr/eq so Span is still easily comparable.
    _lock: Any = field(
        default_factory=threading.Lock, repr=False, compare=False, hash=False
    )

    def finish(self, status: str = "ok") -> None:
        """Record end_time and final status.  Idempotent: only the first call wins."""
        with self._lock:
            if self.end_time is None:
                self.end_time = time.time()
                self.status   = status

    def set_tag(self, key: str, value: Any) -> None:
        """Annotate the span with a key-value tag (overwrites existing key)."""
        with self._lock:
            self.tags[key] = value

    def add_event(self, name: str, attrs: dict = {}) -> None:
        """Append a timestamped log event to the span."""
        with self._lock:
            self.events.append({
                "name":  name,
                "ts":    time.time(),
                "attrs": dict(attrs),
            })

    def duration_ms(self) -> float:
        """Elapsed time in milliseconds.  Uses time.time() if span is not yet finished."""
        end = self.end_time or time.time()
        return round((end - self.start_time) * 1000, 3)

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict (snapshot, not live)."""
        with self._lock:
            return {
                "trace_id":    self.trace_id,
                "span_id":     self.span_id,
                "parent_id":   self.parent_id,
                "operation":   self.operation,
                "service":     self.service,
                "start_time":  self.start_time,
                "end_time":    self.end_time,
                "duration_ms": self.duration_ms(),
                "status":      self.status,
                "tags":        dict(self.tags),
                "events":      list(self.events),
            }


# ── Tracer ────────────────────────────────────────────────────────────────────


class Tracer:
    """
    Distributed tracer.  Manages trace context and span lifecycle.

    In-process: completed spans are stored in a bounded deque (ring buffer)
    accessible via get_recent_traces().  Buffer size is capped at 10 000 spans
    so memory is bounded even under high traffic.

    OTLP export: if `opentelemetry-sdk` is installed AND
    config.tracing_enabled is True, spans are additionally exported to the
    OTLP collector (Jaeger / Grafana Tempo / Datadog / etc.) via the gRPC
    OTLP exporter.  If the library is absent the tracer silently falls back
    to in-process-only mode.

    === THEORY ===

    A trace represents a complete request across all services.
    Each operation is a Span.  Spans form a tree via parent_id.
    The root span has no parent — it represents the full request.

    Sampling controls how many traces are recorded.  1.0 = 100% (all),
    0.1 = 10%.  Head-based probabilistic sampling is used: the decision
    is made when start_span() is called for the root span.

    === PRODUCTION EQUIVALENTS ===

    Jaeger (Uber), Zipkin (Twitter), AWS X-Ray, Datadog APM,
    Grafana Tempo, Honeycomb, Lightstep.
    """

    _MAX_BUFFER: int = 10_000

    def __init__(self, config: ObservabilityConfig2) -> None:
        self._cfg         = config
        self._service     = config.service_name
        self._sample_rate = config.trace_sample_rate
        self._lock        = threading.Lock()
        # Ring buffer of completed span dicts (not live Span objects)
        self._buffer: deque[dict] = deque(maxlen=self._MAX_BUFFER)
        self._total_spans  = 0
        self._total_errors = 0
        self._otlp         = self._init_otlp() if config.tracing_enabled else None
        logger.debug(
            "Tracer initialised: service=%s sample_rate=%.2f otlp=%s",
            self._service, self._sample_rate, bool(self._otlp),
        )

    # ── OTLP bootstrap (optional dependency) ──────────────────────────────

    def _init_otlp(self):
        """
        Attempt to initialise the OpenTelemetry OTLP gRPC exporter.
        Returns an OTel tracer if the SDK is installed, else None.
        """
        try:
            from opentelemetry import trace as otel_trace
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            resource = Resource.create({"service.name": self._service})
            provider = TracerProvider(resource=resource)
            exporter = OTLPSpanExporter(
                endpoint=self._cfg.otlp_endpoint, insecure=True
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
            otel_trace.set_tracer_provider(provider)
            logger.info("OTLP tracing enabled → %s", self._cfg.otlp_endpoint)
            return otel_trace.get_tracer(self._service)
        except ImportError:
            logger.debug(
                "opentelemetry-sdk not installed — OTLP export disabled; "
                "install with: pip install opentelemetry-sdk opentelemetry-exporter-otlp"
            )
            return None
        except Exception as exc:
            logger.warning("OTLP tracer init failed: %s", exc)
            return None

    # ── Span lifecycle ────────────────────────────────────────────────────

    def start_span(
        self,
        operation: str,
        parent_id: str | None = None,
        trace_id:  str | None = None,
    ) -> Span:
        """
        Create and start a new Span.

        If `trace_id` is provided (propagated from an upstream service via
        a W3C traceparent header, for example), the span joins that existing
        trace.  Otherwise a new trace_id is generated, making this span the
        root of a new trace.

        Sampling: the `sampled` tag is set to True/False based on
        self._sample_rate.  Unsampled spans are still returned and usable;
        finish_span() simply skips buffering/exporting them.
        """
        sampled = random.random() <= self._sample_rate
        span = Span(
            trace_id   = trace_id or str(uuid.uuid4()),
            span_id    = str(uuid.uuid4()),
            parent_id  = parent_id,
            operation  = operation,
            service    = self._service,
            start_time = time.time(),
            tags       = {"sampled": sampled},
        )
        return span

    def finish_span(self, span: Span, status: str = "ok") -> None:
        """
        Finish `span` and store it in the in-process ring buffer.
        Also exports via OTLP if tracing_enabled and SDK is available.
        """
        span.finish(status)
        sampled = span.tags.get("sampled", True)

        with self._lock:
            self._total_spans += 1
            if status == "error":
                self._total_errors += 1
            if sampled:
                self._buffer.append(span.to_dict())

        if self._otlp and sampled:
            self._export_otlp(span)

    def _export_otlp(self, span: Span) -> None:
        """Best-effort OTLP export — errors are logged and swallowed."""
        try:
            otel_span = self._otlp.start_span(span.operation)
            for k, v in span.tags.items():
                if k != "sampled":
                    otel_span.set_attribute(k, str(v))
            for evt in span.events:
                otel_span.add_event(evt["name"], evt.get("attrs", {}))
            otel_span.end()
        except Exception as exc:
            logger.debug("OTLP export error (span=%s): %s", span.span_id[:8], exc)

    # ── Context manager ───────────────────────────────────────────────────

    @contextmanager
    def trace(
        self,
        operation: str,
        parent_id: str | None = None,
        trace_id:  str | None = None,
    ) -> Iterator[Span]:
        """
        Context manager: start_span → yield span → finish_span.

        On exception the span is finished with status="error" and the
        exception is re-raised so callers still see it.

        Usage::

            with tracer.trace("bm25_search") as span:
                span.set_tag("query", query)
                results = bm25.search(query)
                span.set_tag("result_count", len(results))
        """
        span = self.start_span(operation, parent_id=parent_id, trace_id=trace_id)
        try:
            yield span
            self.finish_span(span, status="ok")
        except Exception as exc:
            span.add_event(
                "exception",
                {"type": type(exc).__name__, "message": str(exc)},
            )
            self.finish_span(span, status="error")
            raise

    # ── Query API ─────────────────────────────────────────────────────────

    def get_recent_traces(self, limit: int = 50) -> list[dict]:
        """
        Return up to `limit` most-recently completed span dicts,
        newest first.
        """
        with self._lock:
            spans = list(self._buffer)
        # Reverse so newest is index 0
        return list(reversed(spans))[:limit]

    def stats(self) -> dict:
        """Return a JSON-serialisable stats snapshot."""
        with self._lock:
            buffered = len(self._buffer)
        return {
            "service":        self._service,
            "total_spans":    self._total_spans,
            "total_errors":   self._total_errors,
            "buffered_spans": buffered,
            "sample_rate":    self._sample_rate,
            "otlp_enabled":   bool(self._otlp),
        }


# ── traced decorator ──────────────────────────────────────────────────────────


class traced:
    """
    Decorator to automatically trace a function call.

    The decorator starts a span before calling the function and finishes
    it (with status "ok" or "error") after it returns or raises.

    If no tracer is provided, the decorator is a transparent no-op so
    that call sites don't need conditional guards.

    Usage::

        tracer = Tracer(config)

        @traced("bm25_search", tracer=tracer)
        def search(query: str) -> list:
            ...

        # Without a tracer — no-op, function unchanged:
        @traced("embed", tracer=None)
        def embed(text: str) -> list:
            ...
    """

    def __init__(self, operation: str, tracer: "Tracer | None" = None) -> None:
        self._operation = operation
        self._tracer    = tracer

    def __call__(self, func):
        if self._tracer is None:
            return func

        tracer    = self._tracer
        operation = self._operation

        def wrapper(*args, **kwargs):
            with tracer.trace(operation) as span:
                span.set_tag("function", func.__qualname__)
                return func(*args, **kwargs)

        wrapper.__name__    = func.__name__
        wrapper.__doc__     = func.__doc__
        wrapper.__wrapped__ = func
        return wrapper
