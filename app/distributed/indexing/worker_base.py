"""
Worker Base — Phase 8 Batch 2

=== THEORY ===

Workers are the fundamental execution units in an event-driven processing
pipeline.  Each worker subscribes to one or more event topics on the EventBus
and processes incoming events asynchronously.

The Worker pattern originates from the Actor Model (Hewitt, Bishop, Steiger,
1973): each worker is an isolated actor that communicates exclusively through
messages (events).  This enables horizontal scaling — add more workers of the
same type to increase throughput without changing the pipeline topology.

Key properties of a well-designed worker:
  1. Single responsibility — each worker handles one processing stage
  2. Idempotent processing — re-processing the same event produces the
     same result (safe retries)
  3. Self-describing — reports health, metrics, and current status
  4. Graceful lifecycle — clean start/stop with in-flight work completion

=== ARCHITECTURE ===

  EventBus
    │
    │  subscribe(topic, handler)
    ▼
  WorkerBase._handle_event(event)
    │
    │  1. Acquire lock (thread safety)
    │  2. Increment attempt counter
    │  3. Record start time
    │  4. Delegate to process(event)   ◀── subclass implements
    │  5. Record latency
    │  6. Update metrics (processed / failed)
    │  7. Retry on failure (exponential backoff)
    ▼
  Metrics: processed_count, failed_count, avg_latency_ms

=== COMPLEXITY ===

  _handle_event(): O(1) framework overhead + O(process) subclass cost
  start():         O(T) where T = number of topics to subscribe
  stop():          O(S) where S = number of active subscriptions
  stats():         O(1)

=== SPACE COMPLEXITY ===

  O(S) where S = subscription count (typically 1-3 per worker)

=== TRADEOFFS ===

  + ABC enforcement ensures subclasses implement process()
  + Thread-safe via Lock — safe for multi-threaded dispatch
  + Built-in retry with configurable max_retries
  + Metrics tracking without external dependencies
  + Graceful shutdown: stop() waits for current event to finish
  - Synchronous processing (one event at a time per worker)
  - In-process only (no network transport or serialisation)
  - No backpressure (relies on EventBus dispatch rate)

=== PRODUCTION EQUIVALENTS ===

  Kafka Consumer:     ConsumerGroup with poll loop and offset commit
  Celery Worker:      Worker process with task routing and retry
  AWS Lambda:         Event-driven function with retry policy
  Flink Operator:     Stateful operator in a dataflow DAG
  Temporal Activity:  Activity worker with heartbeat and retry
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

from app.events.models import Event
from app.events.bus import EventBus
from app.config import DistributedIndexingConfig

logger = logging.getLogger(__name__)


class WorkerBase(ABC):
    """
    Base class for distributed processing workers.

    Provides:
      - Lifecycle management (start / stop)
      - Event bus integration (subscribe to topics)
      - Health reporting via stats()
      - Graceful shutdown (finish current event, then stop)
      - Retry on failure with configurable max_retries
      - Metrics tracking (processed, failed, uptime, avg_latency)

    Subclasses must implement:
      - process(event) -> dict   — core processing logic
      - topics: list[str]        — event topics to subscribe to

    Usage:
        class MyWorker(WorkerBase):
            topics = ["document.indexed"]

            def process(self, event: Event) -> dict:
                # do work
                return {"status": "ok"}

        worker = MyWorker("worker-1", event_bus=bus)
        worker.start()
        # ... events flow through ...
        worker.stop()
    """

    # Subclasses override this with the list of topics to subscribe to.
    topics: list[str] = []

    def __init__(
        self,
        worker_id: str,
        event_bus: EventBus | None = None,
        config: DistributedIndexingConfig | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.event_bus = event_bus
        self.config = config or DistributedIndexingConfig()

        # Lifecycle state
        self._running = False
        self._started_at: float | None = None
        self._subscription_ids: list[str] = []

        # Thread safety
        self._lock = threading.Lock()

        # Metrics
        self._processed_count = 0
        self._failed_count = 0
        self._total_latency_ms = 0.0

    # ── Abstract interface ────────────────────────────────────────────────

    @abstractmethod
    def process(self, event: Event) -> dict:
        """
        Process a single event.

        Subclasses implement the domain-specific logic here.  The method
        should return a result dict on success.  Exceptions are caught by
        _handle_event() and trigger retry logic.

        Args:
            event: The incoming event to process.

        Returns:
            A dict describing the processing result (keys are domain-specific).
        """

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Start the worker: subscribe to configured topics and set running flag.

        Idempotent — calling start() on an already-running worker is a no-op.
        """
        if self._running:
            logger.debug("Worker %s already running, ignoring start()", self.worker_id)
            return

        if self.event_bus is not None:
            for topic in self.topics:
                sub_id = self.event_bus.subscribe(topic, self._handle_event)
                self._subscription_ids.append(sub_id)
                logger.info(
                    "Worker %s subscribed to topic '%s' (sub=%s)",
                    self.worker_id, topic, sub_id[:8],
                )

        self._running = True
        self._started_at = time.time()
        logger.info("Worker %s started", self.worker_id)

    def stop(self) -> None:
        """
        Stop the worker: unsubscribe from all topics and clear running flag.

        Any event currently being processed will finish before the worker
        fully stops (the lock in _handle_event guarantees this).

        Idempotent — calling stop() on a stopped worker is a no-op.
        """
        if not self._running:
            logger.debug("Worker %s already stopped, ignoring stop()", self.worker_id)
            return

        self._running = False

        if self.event_bus is not None:
            for sub_id in self._subscription_ids:
                self.event_bus.unsubscribe(sub_id)
            self._subscription_ids.clear()

        logger.info(
            "Worker %s stopped (processed=%d, failed=%d)",
            self.worker_id, self._processed_count, self._failed_count,
        )

    def is_running(self) -> bool:
        """Return whether the worker is currently active."""
        return self._running

    # ── Metrics ───────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """
        Return worker health and performance metrics.

        Returns:
            Dict with keys: worker_id, running, processed, failed,
            uptime_seconds, avg_latency_ms.
        """
        uptime = 0.0
        if self._started_at is not None:
            uptime = time.time() - self._started_at

        avg_latency = 0.0
        if self._processed_count > 0:
            avg_latency = self._total_latency_ms / self._processed_count

        return {
            "worker_id":      self.worker_id,
            "worker_type":    self.__class__.__name__,
            "running":        self._running,
            "processed":      self._processed_count,
            "failed":         self._failed_count,
            "uptime_seconds": round(uptime, 2),
            "avg_latency_ms": round(avg_latency, 2),
        }

    # ── Event handling (internal) ─────────────────────────────────────────

    def _handle_event(self, event: Event) -> None:
        """
        Wrap process() with retry logic, metrics tracking, and error handling.

        This method is registered as the EventBus handler.  It:
          1. Checks that the worker is still running
          2. Acquires the processing lock (one event at a time)
          3. Calls process() with retry on failure
          4. Updates metrics (processed/failed count, latency)

        Thread-safe: the lock prevents concurrent process() calls within
        a single worker instance.  Multiple worker instances can process
        in parallel.
        """
        if not self._running:
            return

        max_retries = self.config.max_retries if self.config.retry_on_failure else 1

        with self._lock:
            t0 = time.perf_counter()
            last_error: str | None = None

            for attempt in range(max_retries):
                try:
                    result = self.process(event)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    self._processed_count += 1
                    self._total_latency_ms += elapsed_ms

                    logger.debug(
                        "Worker %s processed event %s (%.1f ms, attempt %d): %s",
                        self.worker_id, event.metadata.event_id[:8],
                        elapsed_ms, attempt + 1, result,
                    )
                    return  # success

                except Exception as exc:
                    last_error = str(exc)
                    logger.warning(
                        "Worker %s failed to process event %s (attempt %d/%d): %s",
                        self.worker_id, event.metadata.event_id[:8],
                        attempt + 1, max_retries, exc,
                    )
                    # Brief backoff before retry (exponential: 0.1s, 0.2s, 0.4s, ...)
                    if attempt < max_retries - 1:
                        backoff = 0.1 * (2 ** attempt)
                        time.sleep(min(backoff, 2.0))

            # All retries exhausted
            self._failed_count += 1
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._total_latency_ms += elapsed_ms
            logger.error(
                "Worker %s permanently failed event %s after %d attempts: %s",
                self.worker_id, event.metadata.event_id[:8],
                max_retries, last_error,
            )
