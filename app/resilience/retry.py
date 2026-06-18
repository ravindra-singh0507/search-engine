"""
Retry with Exponential Backoff and Jitter — Phase 8 Batch 4

=== THEORY ===

Transient failures (network blips, temporary overload) benefit from
automatic retries. The key challenge: if many clients fail simultaneously
and all retry at the same time, they create a thundering herd that can
prevent the recovering service from coming back up.

Exponential backoff: delay doubles on each retry (0.5s, 1s, 2s, 4s…).
Jitter: randomize the delay by ±50% to desynchronize retries across clients.

With jitter:
  delay = base_delay * 2^attempt * random.uniform(0.5, 1.5)
  capped at max_delay_sec

=== PRODUCTION EQUIVALENTS ===

  AWS SDK:    built-in retry with full jitter
  gRPC:       deadline propagation + retry policy in service config
  Celery:     task.retry(countdown=delay, max_retries=n)
  Temporal:   activity retry policy with backoff coefficient
"""

import logging
import random
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type

from app.config import ResilienceConfig

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """
    Retry policy parameters.

    max_attempts:           Total attempts including the first (not just retries).
    base_delay_sec:         Initial delay for backoff calculation.
    max_delay_sec:          Upper cap on delay regardless of backoff.
    jitter:                 Add ±50% randomness to prevent thundering herd.
    retryable_exceptions:   Only retry on these exception types.
    """
    max_attempts:          int              = 3
    base_delay_sec:        float            = 0.5
    max_delay_sec:         float            = 30.0
    jitter:                bool             = True
    retryable_exceptions:  Tuple[Type, ...] = field(default_factory=lambda: (Exception,))


class RetryStrategy:
    """
    Retry executor with exponential backoff and optional jitter.

    Usage:
        strategy = RetryStrategy()
        result = strategy.execute(my_fn, arg1, arg2)
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        self._config = config or RetryConfig()

    def execute(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute fn with retry on failure."""
        last_exc: Optional[Exception] = None
        for attempt in range(self._config.max_attempts):
            try:
                return fn(*args, **kwargs)
            except self._config.retryable_exceptions as exc:
                last_exc = exc
                if not self.should_retry(attempt, exc):
                    break
                delay = self.delay_for(attempt)
                logger.warning(
                    "Attempt %d/%d failed (%s) — retrying in %.2fs",
                    attempt + 1, self._config.max_attempts, exc, delay,
                )
                if delay > 0:
                    time.sleep(delay)
        raise last_exc  # type: ignore[misc]

    def delay_for(self, attempt: int) -> float:
        """Return sleep duration in seconds for this attempt (0-indexed)."""
        if attempt == 0:
            return 0.0
        raw = self._config.base_delay_sec * (2 ** (attempt - 1))
        if self._config.jitter:
            raw *= random.uniform(0.5, 1.5)
        return min(raw, self._config.max_delay_sec)

    def should_retry(self, attempt: int, exc: Exception) -> bool:
        """Return True if another attempt should be made."""
        return (attempt + 1) < self._config.max_attempts


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 0.5,
    jitter: bool = True,
    exceptions: Tuple[Type, ...] = (Exception,),
) -> Callable:
    """
    Decorator factory that wraps a function with retry behaviour.

    Usage:
        @with_retry(max_attempts=3, base_delay=1.0)
        def call_external_api():
            ...
    """
    def decorator(fn: Callable) -> Callable:
        cfg = RetryConfig(
            max_attempts=max_attempts,
            base_delay_sec=base_delay,
            jitter=jitter,
            retryable_exceptions=exceptions,
        )
        strategy = RetryStrategy(cfg)

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return strategy.execute(fn, *args, **kwargs)

        return wrapper
    return decorator
