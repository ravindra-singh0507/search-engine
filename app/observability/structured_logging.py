"""
Structured (JSON) Logging — Phase 8 Batch 4

=== THEORY ===

Traditional text logs are easy to write but hard to query at scale.
Structured logging emits machine-readable JSON, enabling:

  - Full-text search (Elasticsearch / Loki)
  - Aggregations (count errors by service, latency percentiles)
  - Correlation via trace_id / request_id across services
  - Alerting rules (error rate > threshold)

Each log entry is a JSON object (one per line — NDJSON) containing:
  ts          ISO 8601 timestamp (sortable, timezone-aware)
  level       INFO | WARNING | ERROR | DEBUG
  service     emitting service name (search-engine, indexer, …)
  trace_id    (optional) links the log entry to a distributed trace
  tenant_id   (optional) multi-tenant filtering
  request_id  (optional) per-request correlation
  msg         human-readable message
  …kwargs     caller-supplied extra fields (query, latency_ms, …)

=== LOG CONTEXT (THREAD-LOCAL) ===

LogContext uses threading.local to propagate trace_id / tenant_id /
request_id across function calls within one thread without passing them
explicitly as arguments.

Typical usage:
  1. Request middleware calls LogContext.set(trace_id=..., tenant_id=...)
  2. All log lines within the request automatically include these fields
  3. After the response is sent, middleware calls LogContext.clear()

=== PRODUCTION EQUIVALENTS ===

  ELK Stack:             Elasticsearch + Logstash + Kibana
  Grafana Loki:          log aggregation with label indexing + LogQL
  Splunk:                enterprise log management with SPL
  Datadog Logs:          SaaS with APM trace correlation
  Google Cloud Logging:  structured JSON with trace_id linking to Cloud Trace
  AWS CloudWatch Logs:   JSON structured logs with Logs Insights querying
"""

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import ClassVar

from app.config import ObservabilityConfig2

_py_logger = logging.getLogger(__name__)


# ── LogContext ─────────────────────────────────────────────────────────────────


class LogContext:
    """
    Thread-local log context: trace_id, tenant_id, request_id.

    Set once at the start of a request (in middleware or at task entry)
    and automatically included in every log line emitted from that thread.

    Cleared after the request completes to prevent context from leaking
    into the next request handled by the same thread-pool thread.

    Thread isolation: each thread maintains its own independent context
    dict via threading.local, so concurrent requests do not interfere.
    """

    _local: ClassVar[threading.local] = threading.local()

    @classmethod
    def set(cls, **kwargs) -> None:
        """
        Set one or more context fields.  Merges with existing context
        (does not wipe fields not mentioned in kwargs).

        Example::

            LogContext.set(trace_id="abc-123", tenant_id="acme")
        """
        ctx = getattr(cls._local, "ctx", {})
        ctx.update({k: v for k, v in kwargs.items() if v is not None})
        cls._local.ctx = ctx

    @classmethod
    def get(cls) -> dict:
        """Return a copy of the current thread's context dict."""
        return dict(getattr(cls._local, "ctx", {}))

    @classmethod
    def clear(cls) -> None:
        """Wipe all context fields for the current thread."""
        cls._local.ctx = {}


# ── StructuredLogger ──────────────────────────────────────────────────────────


class StructuredLogger:
    """
    Emits JSON log lines with consistent fields.  Works alongside Python's
    standard logging — it writes structured output to stdout and also
    forwards entries to the stdlib logging framework for compatibility
    with existing log handlers (file, Syslog, etc.).

    Each log entry includes: timestamp, level, service, trace_id (if set),
    tenant_id (if set), message, and any extra fields passed as kwargs.

    Thread-safe: the internal deque and counters are protected by a Lock.

    === LOG FORMAT ===

    When config.log_format == "json" (default):
      {"ts":"2025-01-01T00:00:00.000+00:00","level":"INFO","service":"search-engine","msg":"..."}

    When config.log_format == "text":
      [2025-01-01T00:00:00.000+00:00] INFO    search-engine: ...  key=value

    === PRODUCTION EQUIVALENTS ===

    ELK Stack (Elasticsearch + Logstash + Kibana), Grafana Loki,
    Splunk, Datadog Logs, Google Cloud Logging, AWS CloudWatch Logs.
    """

    _LEVEL_ORDER: ClassVar[dict[str, int]] = {
        "DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3,
    }
    _MAX_BUFFER: ClassVar[int] = 10_000

    def __init__(self, service_name: str, config: ObservabilityConfig2) -> None:
        self._service    = service_name
        self._cfg        = config
        self._min_level  = self._LEVEL_ORDER.get(config.log_level.upper(), 1)
        self._lock       = threading.Lock()
        self._buffer: deque[dict] = deque(maxlen=self._MAX_BUFFER)
        self._counts: dict[str, int] = {
            "DEBUG": 0, "INFO": 0, "WARNING": 0, "ERROR": 0,
        }
        self._start_time = time.time()

    # ── Public API ────────────────────────────────────────────────────────

    def info(self, msg: str, **kwargs) -> None:
        """Emit an INFO-level log entry."""
        self._emit("INFO", msg, **kwargs)

    def warning(self, msg: str, **kwargs) -> None:
        """Emit a WARNING-level log entry."""
        self._emit("WARNING", msg, **kwargs)

    def error(self, msg: str, **kwargs) -> None:
        """Emit an ERROR-level log entry."""
        self._emit("ERROR", msg, **kwargs)

    def debug(self, msg: str, **kwargs) -> None:
        """Emit a DEBUG-level log entry (suppressed if log_level > DEBUG)."""
        self._emit("DEBUG", msg, **kwargs)

    # ── Internal ──────────────────────────────────────────────────────────

    def _emit(self, level: str, msg: str, **kwargs) -> None:
        """
        Build the structured log record, write to stdout, store in buffer.

        JSON format (one object per line)::

            {"ts": "2025-01-01T00:00:00.000+00:00",
             "level": "INFO",
             "service": "search-engine",
             "msg": "Search completed",
             "query": "...",
             "latency_ms": 42.1}

        The thread-local LogContext (trace_id, tenant_id, request_id) is
        automatically merged into every record.
        """
        if self._LEVEL_ORDER.get(level, 0) < self._min_level:
            return

        ctx = LogContext.get()

        record: dict = {
            "ts":      datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level":   level,
            "service": self._service,
            "msg":     msg,
        }
        # Merge thread-local context fields (trace_id, tenant_id, request_id)
        if ctx:
            record.update(ctx)
        # Merge caller-supplied extra fields
        if kwargs:
            record.update(kwargs)

        line = json.dumps(record, default=str)

        with self._lock:
            self._counts[level] = self._counts.get(level, 0) + 1
            self._buffer.append(record)

        # Write to stdout
        if self._cfg.log_format == "json":
            print(line, flush=True)
        else:
            # Human-readable fallback for local development
            extra = {
                k: v for k, v in record.items()
                if k not in ("ts", "level", "service", "msg")
            }
            extra_str = "  " + "  ".join(f"{k}={v}" for k, v in extra.items()) if extra else ""
            print(
                f"[{record['ts']}] {level:<7s} {self._service}: {msg}{extra_str}",
                flush=True,
            )

        # Forward to Python stdlib logging for compatibility with
        # existing handlers (file handlers, Syslog, etc.)
        py_level = getattr(logging, level, logging.INFO)
        _py_logger.log(py_level, msg, stacklevel=3)

    def get_recent(self, limit: int = 100, level: str | None = None) -> list[dict]:
        """
        Return recent log entries, newest first.

        Args:
          limit: maximum number of entries to return
          level: if set, filter to this level only (case-insensitive)
        """
        with self._lock:
            entries = list(self._buffer)
        if level:
            level_upper = level.upper()
            entries = [e for e in entries if e.get("level") == level_upper]
        # Newest first
        return list(reversed(entries))[:limit]

    def stats(self) -> dict:
        """Return a JSON-serialisable stats snapshot."""
        with self._lock:
            counts  = dict(self._counts)
            buffered = len(self._buffer)
        return {
            "service":    self._service,
            "min_level":  self._cfg.log_level,
            "format":     self._cfg.log_format,
            "buffered":   buffered,
            "counts":     counts,
            "uptime_sec": round(time.time() - self._start_time, 1),
        }
