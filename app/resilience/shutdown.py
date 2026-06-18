"""
Graceful Shutdown — Phase 8 Batch 4

=== THEORY ===

Graceful shutdown prevents data loss and connection errors during deploys:
  1. Stop accepting new requests
  2. Finish in-flight requests (with deadline)
  3. Flush caches / write buffers
  4. Close DB connections
  5. Terminate

Handlers are called in LIFO order (reverse registration) — the last
thing registered is the first to run on shutdown, matching how resource
dependencies should be torn down (close consumers before producers).

=== PRODUCTION EQUIVALENTS ===

  Kubernetes: SIGTERM → terminationGracePeriodSeconds → SIGKILL
  Docker:     docker stop --time=N sends SIGTERM then SIGKILL
  uvicorn:    --timeout-graceful-shutdown
  gunicorn:   worker_exit hook
"""

import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from app.config import ResilienceConfig

logger = logging.getLogger(__name__)


@dataclass
class _ShutdownHandler:
    name:        str
    handler:     Callable
    timeout_sec: float


class GracefulShutdown:
    """
    Orchestrates graceful shutdown by calling registered handlers in LIFO order.

    Each handler has an individual timeout. If a handler exceeds its
    timeout it is skipped and logged as an error.

    Usage:
        shutdown = GracefulShutdown(config)
        shutdown.register("close_db", db.close, timeout_sec=5.0)
        shutdown.setup_signal_handlers()

        # On SIGTERM:
        results = shutdown.shutdown()
    """

    def __init__(self, config: ResilienceConfig):
        self._config = config
        self._handlers: List[_ShutdownHandler] = []
        self._shutting_down = False
        self._lock = threading.Lock()

    def register(
        self,
        name:        str,
        handler:     Callable,
        timeout_sec: float = 10.0,
    ) -> None:
        """Register a shutdown handler. Called in LIFO order during shutdown."""
        with self._lock:
            self._handlers.append(_ShutdownHandler(name, handler, timeout_sec))
        logger.debug("Registered shutdown handler: %s (timeout=%.1fs)", name, timeout_sec)

    def shutdown(self) -> Dict[str, str]:
        """
        Execute all registered shutdown handlers in LIFO order.

        Returns a dict mapping handler name → "ok" or error description.
        """
        with self._lock:
            if self._shutting_down:
                logger.warning("Shutdown already in progress")
                return {}
            self._shutting_down = True
            handlers = list(reversed(self._handlers))

        logger.info("Graceful shutdown started — %d handlers", len(handlers))
        results: Dict[str, str] = {}
        deadline = time.perf_counter() + self._config.graceful_shutdown_sec

        for h in handlers:
            remaining = max(0.0, deadline - time.perf_counter())
            timeout = min(h.timeout_sec, remaining)
            if timeout <= 0:
                results[h.name] = "skipped: overall deadline exceeded"
                logger.warning("Skipping %s — overall deadline exceeded", h.name)
                continue

            result_holder: list = [None]
            error_holder:  list = [None]

            def _run(handler=h) -> None:
                try:
                    handler.handler()
                    result_holder[0] = "ok"
                except Exception as exc:
                    error_holder[0] = str(exc)

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(timeout=timeout)

            if t.is_alive():
                msg = f"timed out after {timeout:.1f}s"
                results[h.name] = msg
                logger.error("Shutdown handler %s %s", h.name, msg)
            elif error_holder[0]:
                msg = f"error: {error_holder[0]}"
                results[h.name] = msg
                logger.error("Shutdown handler %s failed: %s", h.name, error_holder[0])
            else:
                results[h.name] = "ok"
                logger.info("Shutdown handler %s completed", h.name)

        logger.info("Graceful shutdown complete — results: %s", results)
        return results

    def is_shutting_down(self) -> bool:
        return self._shutting_down

    def setup_signal_handlers(self) -> None:
        """Register SIGTERM and SIGINT signal handlers for automatic shutdown."""
        def _handler(signum: int, frame) -> None:
            logger.info("Received signal %d — initiating graceful shutdown", signum)
            self.shutdown()

        try:
            signal.signal(signal.SIGTERM, _handler)
        except (OSError, ValueError):
            logger.debug("SIGTERM not available on this platform")

        try:
            signal.signal(signal.SIGINT, _handler)
        except (OSError, ValueError):
            logger.debug("SIGINT not available on this platform")
