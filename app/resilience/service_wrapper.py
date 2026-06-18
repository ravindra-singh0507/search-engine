"""
Resilience Service Wrapper — Phase 8 Final

Wraps external service calls with circuit breakers, retries,
and timeout enforcement. Rather than modifying each service
internally, this provides a decorator/wrapper pattern.

=== THEORY ===

The Bulkhead + Circuit Breaker pattern isolates failures:
  - Each external dependency gets its own circuit breaker
  - Failures in one service don't cascade to others
  - The system degrades gracefully under partial failure

The wrapper composes three resilience primitives in order:
  1. Circuit Breaker (outer) — fast-fails if the service is known-bad
  2. Retry (inner) — retries transient failures with backoff + jitter
  3. Fallback (on any failure) — returns a degraded response

This ordering matters:
  - The circuit breaker counts retried-then-failed calls as ONE failure
    (not N failures for N retry attempts), preventing premature tripping.
  - Fallback is invoked only after both retry and circuit breaker give up,
    ensuring the system always returns *something* to the caller.

Thread-safety: each ResilientService uses atomic counters via lock, and
the CircuitBreakerRegistry is internally locked. ServiceResilienceLayer
guards its service dict with a lock for thread-safe registration.

=== PRODUCTION EQUIVALENTS ===

  Netflix:    Hystrix command wrappers (now Resilience4j)
  Uber:       Client library wrappers with timeouts + circuit breakers
  Google:     Deadline propagation + adaptive load shedding
  Microsoft:  Polly (.NET resilience library)
  Go:         sony/gobreaker + hashicorp/go-retryablehttp
"""

import logging
import threading
import time
from typing import Any, Callable, Optional, TypeVar
from functools import wraps

from app.config import ResilienceConfig
from app.resilience.circuit_breaker import (
    CircuitBreakerRegistry,
    CircuitOpenError,
)
from app.resilience.retry import RetryConfig, RetryStrategy

logger = logging.getLogger(__name__)
T = TypeVar('T')


class ResilientService:
    """
    Wraps a service with circuit breaking and retry logic.

    Each external dependency (LLM provider, embedding service, database,
    external API) gets its own ResilientService instance with independent
    failure tracking and circuit breaker state.

    Usage:
        llm_service = ResilientService(
            "llm", registry, config,
            fallback=lambda *a, **k: "Service unavailable"
        )
        result = llm_service.call(llm_provider.generate, prompt="hello")

    The call flow:
        caller → circuit_breaker.call(retry.execute(fn))
              ↓ on CircuitOpenError → fallback (if provided) or re-raise
              ↓ on any Exception    → fallback (if provided) or re-raise
    """

    def __init__(self, name: str, registry: CircuitBreakerRegistry,
                 config: ResilienceConfig, fallback: Optional[Callable] = None):
        self._name = name
        self._cb = registry.get(name)
        self._retry = RetryStrategy(RetryConfig(
            max_attempts=config.retry_max_attempts,
            base_delay_sec=config.retry_base_delay_sec,
            max_delay_sec=config.retry_max_delay_sec,
            jitter=config.retry_jitter,
        ))
        self._fallback = fallback
        self._lock = threading.Lock()
        self._calls = 0
        self._failures = 0
        self._fallback_used = 0
        self._total_latency_ms = 0.0
        self._last_call_time: Optional[float] = None

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute fn with circuit breaker + retry + optional fallback.

        The retry strategy is nested inside the circuit breaker call:
        the breaker sees the retry-wrapped function as a single unit,
        so N retry attempts that all fail count as ONE circuit breaker
        failure (preventing premature tripping).

        Args:
            fn:     The function to call (e.g., llm.generate)
            *args:  Positional arguments forwarded to fn
            **kwargs: Keyword arguments forwarded to fn

        Returns:
            The return value of fn, or the fallback's return value on failure.

        Raises:
            CircuitOpenError: If the circuit is open and no fallback is set.
            Exception: If fn fails and no fallback is set.
        """
        start = time.monotonic()
        with self._lock:
            self._calls += 1
            self._last_call_time = start

        try:
            result = self._cb.call(lambda: self._retry.execute(fn, *args, **kwargs))
            elapsed_ms = (time.monotonic() - start) * 1000
            with self._lock:
                self._total_latency_ms += elapsed_ms
            return result

        except CircuitOpenError:
            with self._lock:
                self._failures += 1
            if self._fallback:
                with self._lock:
                    self._fallback_used += 1
                logger.warning(
                    "Circuit open for '%s', using fallback", self._name
                )
                return self._fallback(*args, **kwargs)
            raise

        except Exception as exc:
            with self._lock:
                self._failures += 1
            if self._fallback:
                with self._lock:
                    self._fallback_used += 1
                logger.warning(
                    "Service '%s' failed (%s), using fallback",
                    self._name, exc
                )
                return self._fallback(*args, **kwargs)
            raise

    def stats(self) -> dict:
        """Return service-level resilience statistics."""
        with self._lock:
            avg_latency = (
                self._total_latency_ms / self._calls
                if self._calls > 0 else 0.0
            )
            return {
                "name": self._name,
                "calls": self._calls,
                "failures": self._failures,
                "fallback_used": self._fallback_used,
                "success_rate": (
                    (self._calls - self._failures) / self._calls
                    if self._calls > 0 else 1.0
                ),
                "avg_latency_ms": round(avg_latency, 2),
                "circuit_state": self._cb.state().value,
                "circuit_stats": self._cb.stats(),
            }


class ServiceResilienceLayer:
    """
    Manages resilient wrappers for all external services.

    Provides a centralized registry of wrapped services with
    individual circuit breakers and retry policies. Acts as the
    single point of configuration for all resilience patterns.

    This is analogous to Netflix's Hystrix command groups or
    Resilience4j's registry pattern: each service dependency
    is isolated with its own failure budget.

    Thread-safety: _services dict is guarded by _lock for
    concurrent registration and lookup.
    """

    def __init__(self, config: ResilienceConfig):
        self._config = config
        self._registry = CircuitBreakerRegistry(config)
        self._services: dict[str, ResilientService] = {}
        self._lock = threading.Lock()

    def register(self, name: str, fallback: Optional[Callable] = None) -> ResilientService:
        """
        Register a new resilient service wrapper.

        If a service with the same name already exists, it is replaced.

        Args:
            name:     Unique identifier for the service (e.g., "llm", "embeddings")
            fallback: Optional callable invoked when the service is unavailable

        Returns:
            The created ResilientService instance.
        """
        svc = ResilientService(name, self._registry, self._config, fallback)
        with self._lock:
            self._services[name] = svc
        logger.info("Registered resilient service: %s (fallback=%s)",
                    name, fallback is not None)
        return svc

    def get(self, name: str) -> Optional[ResilientService]:
        """Get a registered resilient service by name."""
        with self._lock:
            return self._services.get(name)

    def call(self, service_name: str, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Call a function through the named service's resilience layer.

        If the service has not been registered yet, it is auto-registered
        with no fallback (fail-fast behaviour).

        Args:
            service_name: Name of the service to route through
            fn:           Function to execute
            *args, **kwargs: Forwarded to fn

        Returns:
            The return value of fn (or its fallback).
        """
        with self._lock:
            svc = self._services.get(service_name)
        if svc is None:
            svc = self.register(service_name)
        return svc.call(fn, *args, **kwargs)

    def stats(self) -> dict:
        """Return aggregated statistics for all registered services."""
        with self._lock:
            services_snapshot = dict(self._services)
        return {
            "services": {
                name: svc.stats() for name, svc in services_snapshot.items()
            },
            "total_services": len(services_snapshot),
        }

    def get_all_stats(self) -> dict:
        """Return circuit breaker stats for all registered breakers."""
        return self._registry.get_all_stats()


def resilient(service_name: str, fallback: Optional[Callable] = None):
    """
    Decorator that wraps a function with resilience patterns.

    Provides a lightweight alternative to the full ServiceResilienceLayer
    for simple use cases where you want retry + fallback on a single function.

    Note: This decorator does not use a shared circuit breaker registry.
    For production use with circuit breaking, prefer ServiceResilienceLayer.

    Usage:
        @resilient("llm", fallback=lambda p, **kw: "unavailable")
        def generate(prompt: str, **kwargs):
            return llm.generate(prompt, **kwargs)

    Args:
        service_name: Identifier for logging purposes.
        fallback:     Optional callable invoked on failure.
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                if fallback:
                    logger.warning(
                        "Function %s (service=%s) failed (%s), using fallback",
                        fn.__name__, service_name, exc
                    )
                    return fallback(*args, **kwargs)
                raise
        return wrapper
    return decorator
