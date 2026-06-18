"""
Circuit Breaker Pattern — Phase 8 Batch 4

The Circuit Breaker pattern (Michael Nygard, "Release It!", 2007) prevents
cascading failures in distributed systems by wrapping remote calls in a
state machine:

  CLOSED    → tracks failures in a sliding count; trips to OPEN when
              failure_threshold is exceeded within the window.
  OPEN      → fast-fails every call with CircuitOpenError; after
              recovery_timeout_sec elapses → HALF_OPEN.
  HALF_OPEN → allows up to half_open_max_calls probe requests; a success
              closes the circuit; a failure re-opens it.

This is analogous to an electrical circuit breaker: it "opens" (breaks the
circuit) when the load is too high, protecting downstream services from
being overwhelmed while giving them time to recover.

Thread-safety: all state mutations are guarded by threading.Lock.
Timing: time.monotonic() is used to avoid wall-clock drift issues.

Production equivalents:
  Netflix:  Hystrix (now Resilience4j)
  AWS SDK:  Built-in retry + circuit breaker
  gRPC:     Deadline propagation + status codes
"""

import threading
import time
from enum import Enum
from typing import Any, Callable

from app.config import ResilienceConfig


class CircuitState(Enum):
    """State machine states for the circuit breaker."""
    CLOSED    = "closed"     # Normal operation; failures are tracked
    OPEN      = "open"       # Fast-failing; recovery timer running
    HALF_OPEN = "half_open"  # Probing recovery; limited calls allowed


class CircuitOpenError(Exception):
    """Raised when a call is attempted while the circuit is OPEN."""


class CircuitBreaker:
    """
    Thread-safe circuit breaker that wraps any callable.

    Uses a cumulative failure count (sliding window approximation via
    success-side decrement) to track the failure rate.  time.monotonic()
    ensures timing is unaffected by NTP adjustments or DST transitions.

    Args:
        name:   Human-readable identifier used in error messages and stats.
        config: ResilienceConfig providing threshold and timeout values.
    """

    def __init__(self, name: str, config: ResilienceConfig) -> None:
        self.name = name
        self._config = config
        self._lock = threading.Lock()

        # State
        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._total_calls: int = 0
        self._rejected_calls: int = 0
        self._half_open_calls: int = 0
        self._last_failure_time: float | None = None
        self._opened_at: float | None = None

    # ── Public API ───────────────────────────────────────────────────────────

    def call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute fn(*args, **kwargs) subject to circuit breaker logic.

        Raises:
            CircuitOpenError: If the circuit is OPEN (or HALF_OPEN and the
                              probe call quota is exhausted).
            Exception:        The underlying exception propagated from fn on
                              failure (the failure is recorded first).
        """
        with self._lock:
            state = self._get_state_locked()

            if state == CircuitState.OPEN:
                self._rejected_calls += 1
                raise CircuitOpenError(
                    f"Circuit '{self.name}' is OPEN — fast-failing call"
                )

            if state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self._config.half_open_max_calls:
                    self._rejected_calls += 1
                    raise CircuitOpenError(
                        f"Circuit '{self.name}' is HALF_OPEN — "
                        f"max probe calls ({self._config.half_open_max_calls}) reached"
                    )
                self._half_open_calls += 1

            self._total_calls += 1

        # Execute outside the lock so we don't block other threads
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self._on_failure()
            raise

        self._on_success()
        return result

    def is_open(self) -> bool:
        """Return True if the circuit is currently OPEN."""
        with self._lock:
            return self._get_state_locked() == CircuitState.OPEN

    def state(self) -> CircuitState:
        """Return the current CircuitState."""
        with self._lock:
            return self._get_state_locked()

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0
            self._opened_at = None
            self._last_failure_time = None

    def stats(self) -> dict:
        """Return current statistics as a JSON-serialisable dict."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._get_state_locked().value,
                "total_calls": self._total_calls,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "rejected_calls": self._rejected_calls,
                "half_open_calls": self._half_open_calls,
                "opened_at": self._opened_at,
                "failure_threshold": self._config.failure_threshold,
                "recovery_timeout_sec": self._config.recovery_timeout_sec,
                "half_open_max_calls": self._config.half_open_max_calls,
            }

    # ── Internal helpers (must be called with _lock held) ───────────────────

    def _get_state_locked(self) -> CircuitState:
        """
        Return effective state; transitions OPEN → HALF_OPEN when the
        recovery timeout has elapsed.  Called with _lock held.
        """
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._config.recovery_timeout_sec:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
        return self._state

    def _on_failure(self) -> None:
        """Record a failed call; trips or re-opens the circuit as needed."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Probe failed — re-open the circuit
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self._half_open_calls = 0

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self._config.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._opened_at = time.monotonic()

    def _on_success(self) -> None:
        """Record a successful call; closes the circuit from HALF_OPEN."""
        with self._lock:
            self._success_count += 1
            if self._state == CircuitState.HALF_OPEN:
                # Probe succeeded — close the circuit
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
                self._opened_at = None
            elif self._state == CircuitState.CLOSED:
                # Sliding-window approximation: decay failure count on success
                self._failure_count = max(0, self._failure_count - 1)


class CircuitBreakerRegistry:
    """
    Central registry for all named CircuitBreaker instances.

    The registry pattern allows monitoring tools to query all breakers
    in a single call without needing to know their names in advance.
    All breakers share the same ResilienceConfig.

    Thread-safe via threading.Lock.
    """

    def __init__(self, config: ResilienceConfig) -> None:
        self._config = config
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> CircuitBreaker:
        """Return the named breaker, creating it lazily if absent."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, self._config)
            return self._breakers[name]

    def get_all_stats(self) -> dict:
        """Return stats for every registered circuit breaker."""
        with self._lock:
            return {name: cb.stats() for name, cb in self._breakers.items()}
