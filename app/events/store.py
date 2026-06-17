"""
Event Store — Phase 8

=== THEORY ===

The Event Store implements the Event Sourcing pattern (Fowler 2005,
Young 2010).  Instead of persisting only the current state of an
entity, we persist the full sequence of events that led to it.

Benefits of Event Sourcing:
  1. Complete audit trail (every state change is recorded)
  2. Temporal queries ("what was the state at time T?")
  3. Replay (rebuild state by replaying events)
  4. Debugging (inspect the exact sequence of events)

The store also tracks delivery status, enabling:
  - Retry of failed events
  - Dead-letter inspection
  - Delivery rate monitoring

=== ARCHITECTURE ===

  EventBus.publish(event)
    │
    ▼
  EventStore.append(event)
    │
    ├── deque (bounded, FIFO eviction)  — chronological storage
    └── dict index                      — O(1) lookup by event_id

  Query methods:
    get_events(topic, limit)          — filter by topic
    get_event(event_id)               — O(1) by ID
    get_by_status(status, limit)      — filter by delivery status
    mark_status(event_id, status)     — update delivery state
    count()                           — total stored events

=== DATA STRUCTURES ===

  collections.deque(maxlen=N)  — bounded FIFO; O(1) append, O(1) eviction
  dict[event_id → EventEnvelope] — O(1) lookup by ID

=== COMPLEXITY ===

  append():       O(1) amortised
  get_event():    O(1) via dict lookup
  get_events():   O(n) scan (n = stored events)
  get_by_status(): O(n) scan
  mark_status():  O(1) via dict lookup
  count():        O(1)

=== SPACE COMPLEXITY ===

  O(maxlen) — bounded by deque capacity

=== TRADEOFFS ===

  + Bounded storage via deque (no memory leak)
  + O(1) lookup by event_id via dict index
  + Thread-safe via Lock
  - In-memory only (lost on restart)
  - O(n) topic/status queries (no secondary index)

=== PRODUCTION EQUIVALENTS ===

  EventStoreDB:       purpose-built event store with projections
  Apache Kafka:       distributed commit log (immutable, append-only)
  PostgreSQL + Outbox: transactional event storage with CDC
  DynamoDB Streams:   change data capture as event stream
  Axon Server:        CQRS/ES infrastructure with event store
"""

import logging
import threading
import time
from collections import deque
from typing import Protocol, runtime_checkable

from app.events.models import Event, EventEnvelope, EventStatus

logger = logging.getLogger(__name__)


# ── Protocol ─────────────────────────────────────────────────────────────────

@runtime_checkable
class EventStore(Protocol):
    """
    Protocol defining the event store interface.

    Any implementation (in-memory, PostgreSQL, EventStoreDB) must
    satisfy this contract.
    """

    def append(self, event: Event) -> None:
        """Append an event to the store."""
        ...

    def get_events(self, topic: str | None = None, limit: int = 100) -> list[Event]:
        """Retrieve events, optionally filtered by topic."""
        ...

    def get_event(self, event_id: str) -> Event | None:
        """Retrieve a single event by its ID.  Returns None if not found."""
        ...

    def mark_status(self, event_id: str, status: EventStatus) -> None:
        """Update the delivery status of an event."""
        ...

    def get_by_status(self, status: EventStatus, limit: int = 100) -> list[Event]:
        """Retrieve events filtered by delivery status."""
        ...

    def count(self) -> int:
        """Return the total number of stored events."""
        ...


# ── In-Memory Implementation ────────────────────────────────────────────────

class InMemoryEventStore:
    """
    Thread-safe, bounded, in-memory event store.

    Uses a deque with maxlen for FIFO eviction and a dict index
    for O(1) event lookup by ID.

    When the deque reaches capacity, the oldest event is evicted
    from the deque.  The dict index is also cleaned up to prevent
    stale references.

    Usage:
        store = InMemoryEventStore(max_size=10000)
        store.append(event)
        event = store.get_event("some-uuid")
        failed = store.get_by_status(EventStatus.FAILED)
    """

    def __init__(self, max_size: int = 10000) -> None:
        self._max_size = max_size
        self._events: deque[EventEnvelope] = deque(maxlen=max_size)
        self._index: dict[str, EventEnvelope] = {}
        self._lock = threading.Lock()

    def append(self, event: Event) -> None:
        """
        Append an event to the store.

        If the deque is at capacity, the oldest event is evicted
        automatically by the deque.  We also remove it from the index.
        """
        envelope = EventEnvelope(event=event, status=EventStatus.PENDING)
        with self._lock:
            # If at capacity, the deque will evict the leftmost item
            if len(self._events) == self._max_size:
                evicted = self._events[0]
                self._index.pop(evicted.event.metadata.event_id, None)
            self._events.append(envelope)
            self._index[event.metadata.event_id] = envelope
        logger.debug(
            "Stored event id=%s topic='%s'",
            event.metadata.event_id[:8], event.topic,
        )

    def get_events(self, topic: str | None = None, limit: int = 100) -> list[Event]:
        """
        Retrieve events in reverse chronological order.

        Args:
            topic: optional filter; None returns all topics
            limit: maximum number of events to return
        """
        with self._lock:
            results: list[Event] = []
            for envelope in reversed(self._events):
                if topic is not None and envelope.event.topic != topic:
                    continue
                results.append(envelope.event)
                if len(results) >= limit:
                    break
            return results

    def get_event(self, event_id: str) -> Event | None:
        """O(1) lookup by event_id.  Returns None if not found."""
        with self._lock:
            envelope = self._index.get(event_id)
            return envelope.event if envelope else None

    def mark_status(self, event_id: str, status: EventStatus) -> None:
        """
        Update the delivery status of an event.

        Also sets delivered_at timestamp when status is DELIVERED.
        """
        with self._lock:
            envelope = self._index.get(event_id)
            if envelope is None:
                logger.warning("Cannot mark status: event %s not found", event_id[:8])
                return
            envelope.status = status
            if status == EventStatus.DELIVERED:
                envelope.delivered_at = time.time()
            logger.debug(
                "Event %s status → %s",
                event_id[:8], status.value,
            )

    def get_by_status(self, status: EventStatus, limit: int = 100) -> list[Event]:
        """Retrieve events filtered by delivery status."""
        with self._lock:
            results: list[Event] = []
            for envelope in reversed(self._events):
                if envelope.status == status:
                    results.append(envelope.event)
                    if len(results) >= limit:
                        break
            return results

    def count(self) -> int:
        """Return the total number of stored events."""
        with self._lock:
            return len(self._events)
