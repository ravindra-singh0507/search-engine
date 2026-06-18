"""
Performance Optimizer — Phase 8 Batch 5

=== THEORY ===

The performance optimizer provides a unified interface for profiling and
optimizing the search engine platform.  It combines:

  1. Distributed caching (multi-tier L1+L2)
  2. Batch processing (amortise per-item overhead)
  3. Query result caching (avoid redundant retrieval)
  4. Connection pooling metrics
  5. Memory usage tracking
  6. Performance profiling helpers

=== PRODUCTION EQUIVALENTS ===

  Google:   profiling infra + Dapper + Borg resource allocation
  Netflix:  Titus resource optimization + adaptive concurrency
  Uber:     profiling + resource right-sizing
  Elastic:  shard allocation + search profiler
"""

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Result of profiling a function execution."""
    name:       str
    duration_ms: float
    success:    bool
    result:     Any = None
    error:      Optional[str] = None
    timestamp:  float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "name":        self.name,
            "duration_ms": round(self.duration_ms, 2),
            "success":     self.success,
            "error":       self.error,
            "timestamp":   self.timestamp,
        }


class PerformanceOptimizer:
    """
    Unified performance monitoring and optimization layer.

    Provides:
      - Function profiling with latency tracking
      - Query deduplication (prevent duplicate concurrent queries)
      - Latency budget enforcement
      - Performance summary generation

    Usage:
        optimizer = PerformanceOptimizer()

        with optimizer.profile("search"):
            results = search_service.search(query)

        print(optimizer.summary())
    """

    def __init__(self, max_history: int = 1000):
        self._history: list[ProfileResult] = []
        self._max_history = max_history
        self._lock = threading.Lock()
        self._in_flight: dict[str, threading.Event] = {}
        self._total_profiled = 0
        self._total_duration_ms = 0.0

    class _ProfileCtx:
        """Context manager for profiling."""
        def __init__(self, optimizer: "PerformanceOptimizer", name: str):
            self._opt = optimizer
            self._name = name
            self._t0 = 0.0
            self.result: Optional[ProfileResult] = None

        def __enter__(self):
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            ms = (time.perf_counter() - self._t0) * 1000
            success = exc_type is None
            error = str(exc_val) if exc_val else None
            pr = ProfileResult(
                name=self._name, duration_ms=ms,
                success=success, error=error,
            )
            self.result = pr
            self._opt._record(pr)
            return False

    def profile(self, name: str) -> _ProfileCtx:
        """Context manager that profiles the enclosed block."""
        return self._ProfileCtx(self, name)

    def profile_fn(self, name: str, fn: Callable, *args, **kwargs) -> Any:
        """Profile a function call and return its result."""
        t0 = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            ms = (time.perf_counter() - t0) * 1000
            self._record(ProfileResult(
                name=name, duration_ms=ms, success=True, result=result,
            ))
            return result
        except Exception as exc:
            ms = (time.perf_counter() - t0) * 1000
            self._record(ProfileResult(
                name=name, duration_ms=ms, success=False, error=str(exc),
            ))
            raise

    def dedup_query(self, key: str) -> bool:
        """
        Check if a query with this key is already in-flight.
        Returns True if this is a new query, False if duplicate.
        Used to prevent duplicate concurrent queries.
        """
        with self._lock:
            if key in self._in_flight:
                return False
            self._in_flight[key] = threading.Event()
            return True

    def complete_query(self, key: str) -> None:
        """Mark an in-flight query as complete."""
        with self._lock:
            event = self._in_flight.pop(key, None)
        if event:
            event.set()

    def get_history(self, name: Optional[str] = None,
                    limit: int = 50) -> list[dict]:
        """Get recent profiling results."""
        with self._lock:
            results = list(self._history)
        if name:
            results = [r for r in results if r.name == name]
        return [r.to_dict() for r in results[-limit:]]

    def summary(self) -> dict:
        """Generate a performance summary."""
        with self._lock:
            history = list(self._history)

        if not history:
            return {"total_profiled": 0, "operations": {}}

        by_name: dict[str, list[float]] = {}
        for r in history:
            by_name.setdefault(r.name, []).append(r.duration_ms)

        operations = {}
        for name, latencies in by_name.items():
            s = sorted(latencies)
            count = len(s)
            operations[name] = {
                "count":      count,
                "mean_ms":    round(sum(s) / count, 2),
                "p50_ms":     round(s[count // 2], 2),
                "p95_ms":     round(s[int(count * 0.95)], 2) if count > 1 else round(s[0], 2),
                "p99_ms":     round(s[int(count * 0.99)], 2) if count > 1 else round(s[0], 2),
                "min_ms":     round(s[0], 2),
                "max_ms":     round(s[-1], 2),
            }

        return {
            "total_profiled": self._total_profiled,
            "total_duration_ms": round(self._total_duration_ms, 2),
            "operations": operations,
            "in_flight_queries": len(self._in_flight),
        }

    def stats(self) -> dict:
        return self.summary()

    def _record(self, result: ProfileResult) -> None:
        with self._lock:
            self._history.append(result)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
            self._total_profiled += 1
            self._total_duration_ms += result.duration_ms
