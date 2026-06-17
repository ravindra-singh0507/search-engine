"""
Event Bus — Phase 8

=== THEORY ===

The Event Bus implements the Publish-Subscribe (Pub/Sub) messaging
pattern (Hohpe & Woolf, *Enterprise Integration Patterns*, 2003).

In Pub/Sub, publishers and subscribers are decoupled:
  - Publishers emit events without knowing who (if anyone) listens.
  - Subscribers register interest in topics without knowing who publishes.

This decoupling is the foundation of event-driven architecture (EDA).

=== ARCHITECTURE ===

  Publisher ─── publish(event) ───▶ EventBus
                                      │
                                      ├── topic "document.indexed"
                                      │      ├── handler_A
                                      │      └── handler_B
                                      │
                                      └── topic "search.executed"
                                             └── handler_C

The InMemoryEventBus dispatches synchronously within the same process.
Each subscription is identified by a UUID, enabling targeted unsubscribe.

=== COMPLEXITY ===

  publish():          O(k) where k = number of subscribers for the topic
  subscribe():        O(1) amortised (list append)
  unsubscribe():      O(n) worst-case where n = total subscriptions
  subscriber_count(): O(1) (len of subscription list)

=== SPACE COMPLEXITY ===

  O(T * S) where T = topics, S = avg subscribers per topic

=== TRADEOFFS ===

  + Synchronous dispatch: simple, deterministic, easy to debug
  + Thread-safe via Lock: safe for multi-threaded producer/consumer
  + Error isolation: one failing handler does not block others
  - No backpressure (unbounded dispatch rate)
  - No ordering guarantees across topics
  - Single-process only (no network transport)

=== PRODUCTION EQUIVALENTS ===

  Apache Kafka:         distributed commit log with consumer groups
  RabbitMQ:             AMQP broker with exchange/queue binding
  Redis Pub/Sub:        in-memory broadcast (no persistence)
  Google Cloud Pub/Sub: managed, at-least-once, pull/push delivery
  AWS SNS + SQS:        fan-out pub/sub with durable queues
"""

import logging
import threading
import uuid
from typing import Callable, Protocol, runtime_checkable

from app.events.models import Event

logger = logging.getLogger(__name__)


# ── Protocol ─────────────────────────────────────────────────────────────────

@runtime_checkable
class EventBus(Protocol):
    """
    Protocol defining the event bus interface.

    Any implementation (in-memory, Kafka, Redis) must satisfy this
    contract.  The @runtime_checkable decorator enables isinstance()
    checks at runtime.
    """

    def publish(self, event: Event) -> None:
        """Dispatch an event to all subscribers of its topic."""
        ...

    def subscribe(self, topic: str, handler: Callable[[Event], None]) -> str:
        """
        Register a handler for a topic.

        Returns a subscription_id that can be used to unsubscribe.
        """
        ...

    def unsubscribe(self, subscription_id: str) -> None:
        """Remove a subscription by its ID."""
        ...

    def subscriber_count(self, topic: str) -> int:
        """Return the number of active subscribers for a topic."""
        ...


# ── In-Memory Implementation ────────────────────────────────────────────────

class InMemoryEventBus:
    """
    Thread-safe, synchronous, in-memory event bus.

    Subscriptions are stored as:
      {topic: [(subscription_id, handler), ...]}

    Dispatch is synchronous — publish() blocks until every handler
    for that topic has been called.  Errors in individual handlers
    are logged but do not prevent remaining handlers from executing.
    """

    def __init__(self) -> None:
        self._subscriptions: dict[str, list[tuple[str, Callable[[Event], None]]]] = {}
        self._lock = threading.Lock()

    def publish(self, event: Event) -> None:
        """
        Dispatch event to all handlers subscribed to event.topic,
        plus any handlers subscribed to the catch-all topic "*".

        Thread-safe: acquires lock to snapshot handlers, then releases
        before calling them (to avoid holding the lock during handler
        execution, which could deadlock if a handler publishes).
        """
        with self._lock:
            handlers = list(self._subscriptions.get(event.topic, []))
            if event.topic != "*":
                handlers += list(self._subscriptions.get("*", []))

        for sub_id, handler in handlers:
            try:
                handler(event)
            except Exception as exc:
                logger.error(
                    "Handler %s for topic '%s' (sub=%s) raised: %s",
                    handler.__name__ if hasattr(handler, "__name__") else repr(handler),
                    event.topic,
                    sub_id[:8],
                    exc,
                )

    def subscribe(self, topic: str, handler: Callable[[Event], None]) -> str:
        """
        Register a handler for a topic.

        Returns a unique subscription_id (UUID string).
        """
        sub_id = str(uuid.uuid4())
        with self._lock:
            if topic not in self._subscriptions:
                self._subscriptions[topic] = []
            self._subscriptions[topic].append((sub_id, handler))
        logger.debug("Subscribed %s to topic '%s' (sub=%s)", handler, topic, sub_id[:8])
        return sub_id

    def unsubscribe(self, subscription_id: str) -> None:
        """
        Remove a subscription by its ID.

        Scans all topics to find and remove the matching subscription.
        """
        with self._lock:
            for topic, subs in self._subscriptions.items():
                self._subscriptions[topic] = [
                    (sid, h) for sid, h in subs if sid != subscription_id
                ]
        logger.debug("Unsubscribed sub=%s", subscription_id[:8])

    def subscriber_count(self, topic: str) -> int:
        """Return the number of active subscribers for a given topic."""
        with self._lock:
            return len(self._subscriptions.get(topic, []))
