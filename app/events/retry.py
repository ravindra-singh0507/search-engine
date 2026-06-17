"""
Event Retry & Dead-Letter Queue — Phase 8

=== THEORY ===

In distributed systems, transient failures are inevitable.  The retry
pattern with exponential backoff provides resilience by re-attempting
failed operations with increasing delays:

  attempt 0:  immediate
  attempt 1:  base_delay * 2^0 = 1s
  attempt 2:  base_delay * 2^1 = 2s
  attempt 3:  base_delay * 2^2 = 4s
  ...
  capped at:  max_delay

Exponential backoff prevents thundering-herd effects when a downstream
service recovers from an outage.

When all retries are exhausted, the event is moved to a Dead-Letter
Queue (DLQ) — a holding area for messages that could not be processed.
DLQs enable:
  1. Manual inspection of persistent failures
  2. Selective retry after root-cause fix
  3. Alerting on DLQ depth (operational signal)

=== ARCHITECTURE ===

  EventBus.publish(event)
    │
    ├── handler succeeds → done
    │
    └── handler fails
          │
          ├── retry_count < max_retries?
          │     YES → increment retry_count, wait delay_for(attempt), re-publish
          │     NO  → move to DeadLetterQueue
          │
          ▼
  DeadLetterQueue
    │  .add(event, error)
    │  .get_all()
    │  .retry(event_id, bus)   — re-publish from DLQ
    │  .remove(event_id)
    └  .clear()

=== COMPLEXITY ===

  EventRetryPolicy:
    delay_for():      O(1)
    should_retry():   O(1)

  DeadLetterQueue:
    add():            O(1)
    get_all():        O(min(n, limit))
    retry():          O(1)
    remove():         O(n) worst-case for deque rebuild
    clear():          O(1) (deque.clear)
    count():          O(1)

=== SPACE COMPLEXITY ===

  DeadLetterQueue: O(max_size) — bounded by deque capacity

=== TRADEOFFS ===

  + Exponential backoff prevents thundering herd
  + Bounded DLQ prevents memory leak
  + Retry from DLQ enables operational recovery
  + Thread-safe via Lock
  - No jitter (production systems add random jitter to spread retries)
  - No per-topic retry policies
  - In-memory only (DLQ contents lost on restart)

=== PRODUCTION EQUIVALENTS ===

  Apache Kafka:     retry topics + dead-letter topic
  RabbitMQ:         x-dead-letter-exchange header
  AWS SQS:          redrive policy to DLQ after maxReceiveCount
  Google Pub/Sub:   dead-letter topic with max delivery attempts
  Azure Service Bus: dead-letter sub-queue per subscription
"""

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass

from app.events.bus import EventBus
from app.events.models import Event

logger = logging.getLogger(__name__)


# ── Retry Policy ─────────────────────────────────────────────────────────────

@dataclass
class EventRetryPolicy:
    """
    Exponential backoff retry policy for event delivery.

    Fields:
      max_retries:     maximum number of retry attempts (0 = no retries)
      base_delay_sec:  initial delay in seconds (doubles each attempt)
      max_delay_sec:   upper bound on the delay (cap)

    The delay formula:
      delay = min(base_delay * 2^(attempt - 1), max_delay)

    Example with defaults (base=1.0, max=30.0):
      attempt 1: 1.0s
      attempt 2: 2.0s
      attempt 3: 4.0s
      ...
      attempt 5: 16.0s
      attempt 6: 30.0s (capped)
    """
    max_retries:    int   = 3
    base_delay_sec: float = 1.0
    max_delay_sec:  float = 30.0

    def delay_for(self, attempt: int) -> float:
        """
        Return the sleep duration in seconds for a given attempt number.

        Args:
            attempt: 0-indexed attempt number

        Returns:
            Delay in seconds.  attempt 0 returns 0 (no delay for first try).
        """
        if attempt <= 0:
            return 0.0
        raw = self.base_delay_sec * (2 ** (attempt - 1))
        return min(raw, self.max_delay_sec)

    def should_retry(self, attempt: int) -> bool:
        """
        Whether another retry should be attempted.

        Args:
            attempt: 0-indexed attempt number (0 = first try)

        Returns:
            True if attempt < max_retries.
        """
        return attempt < self.max_retries


# ── Dead-Letter Queue ────────────────────────────────────────────────────────

class DeadLetterQueue:
    """
    Bounded, thread-safe dead-letter queue for events that exhausted
    all retry attempts.

    Each entry stores the original event, the error message, and
    the timestamp when it was dead-lettered.

    Usage:
        dlq = DeadLetterQueue(max_size=500)
        dlq.add(failed_event, "Connection refused")
        entries = dlq.get_all()
        dlq.retry("event-id", bus)   # re-publish to bus
        dlq.remove("event-id")
        dlq.clear()
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._max_size = max_size
        self._entries: deque[dict] = deque(maxlen=max_size)
        self._index: dict[str, dict] = {}
        self._lock = threading.Lock()

    def add(self, event: Event, error: str) -> None:
        """
        Add a failed event to the dead-letter queue.

        Args:
            event: the event that could not be delivered
            error: the error message from the last delivery attempt
        """
        entry = {
            "event":     event,
            "error":     error,
            "timestamp": time.time(),
            "event_id":  event.metadata.event_id,
        }
        with self._lock:
            # If at capacity, evict the oldest entry from the index
            if len(self._entries) == self._max_size:
                evicted = self._entries[0]
                self._index.pop(evicted["event_id"], None)
            self._entries.append(entry)
            self._index[event.metadata.event_id] = entry
        logger.warning(
            "Dead-lettered event id=%s topic='%s' error='%s'",
            event.metadata.event_id[:8], event.topic, error[:80],
        )

    def get_all(self, limit: int = 100) -> list[dict]:
        """
        Retrieve dead-lettered entries.

        Returns a list of dicts, each containing:
          - event:     the Event object
          - error:     the error string
          - timestamp: when it was dead-lettered
          - event_id:  the event's unique ID

        Args:
            limit: maximum number of entries to return

        Returns:
            List of entry dicts, most recent first.
        """
        with self._lock:
            results: list[dict] = []
            for entry in reversed(self._entries):
                results.append(entry)
                if len(results) >= limit:
                    break
            return results

    def retry(self, event_id: str, bus: EventBus) -> bool:
        """
        Re-publish a dead-lettered event to the bus.

        Removes the event from the DLQ and resets its retry count
        before re-publishing.

        Args:
            event_id: the event_id to retry
            bus:      the EventBus to publish to

        Returns:
            True if the event was found and re-published, False otherwise.
        """
        with self._lock:
            entry = self._index.get(event_id)
            if entry is None:
                logger.warning("DLQ retry: event %s not found", event_id[:8])
                return False
            event = entry["event"]
            # Remove from DLQ
            self._entries = deque(
                (e for e in self._entries if e["event_id"] != event_id),
                maxlen=self._max_size,
            )
            self._index.pop(event_id, None)

        # Reset retry count and re-publish (outside lock to avoid deadlock)
        event.metadata.retry_count = 0
        bus.publish(event)
        logger.info("DLQ retried event id=%s topic='%s'", event_id[:8], event.topic)
        return True

    def remove(self, event_id: str) -> bool:
        """
        Remove an event from the DLQ without retrying.

        Args:
            event_id: the event_id to remove

        Returns:
            True if the event was found and removed, False otherwise.
        """
        with self._lock:
            if event_id not in self._index:
                return False
            self._entries = deque(
                (e for e in self._entries if e["event_id"] != event_id),
                maxlen=self._max_size,
            )
            self._index.pop(event_id, None)
        logger.debug("DLQ removed event id=%s", event_id[:8])
        return True

    def clear(self) -> int:
        """
        Remove all entries from the DLQ.

        Returns:
            The number of entries that were cleared.
        """
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._index.clear()
        logger.info("DLQ cleared %d entries", count)
        return count

    def count(self) -> int:
        """Return the number of entries in the DLQ."""
        with self._lock:
            return len(self._entries)
