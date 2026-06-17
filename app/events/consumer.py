"""
Event Consumer — Phase 8

=== THEORY ===

The Consumer pattern (also called Subscriber or Listener) encapsulates
the mechanics of receiving and dispatching events from the event bus.

In enterprise messaging, consumers belong to *consumer groups*.  Within
a group, each message is delivered to exactly one consumer (competing
consumers pattern).  Across groups, each message is delivered to all
groups (fan-out).

For the in-memory implementation, the group is informational only —
all handlers receive all events.  The group field becomes meaningful
when migrating to Kafka or RabbitMQ.

=== ARCHITECTURE ===

  EventBus
    │
    │  dispatches event to subscribed handlers
    ▼
  EventConsumer
    │  .on("document.indexed", handler)   — subscribe
    │  .off(subscription_id)              — unsubscribe
    │  .start() / .stop()                 — lifecycle (no-op for in-memory)
    ▼
  Application handler functions

=== COMPLEXITY ===

  on():             O(1) — delegates to bus.subscribe()
  off():            O(n) — delegates to bus.unsubscribe()
  handler_count():  O(1) — len of internal list

=== SPACE COMPLEXITY ===

  O(s) where s = number of active subscriptions for this consumer

=== TRADEOFFS ===

  + Clean API for subscription management
  + Group field prepares for Kafka migration
  + start/stop lifecycle hooks for resource management
  - No message acknowledgement (at-most-once for in-memory)
  - No consumer offset tracking

=== PRODUCTION EQUIVALENTS ===

  Kafka Consumer:    KafkaConsumer with group_id and poll loop
  RabbitMQ:          Channel.basic_consume with queue binding
  Redis Streams:     XREADGROUP with consumer group
  AWS SQS:           ReceiveMessage with visibility timeout
"""

import logging
from typing import Callable

from app.events.bus import EventBus
from app.events.models import Event

logger = logging.getLogger(__name__)


class EventConsumer:
    """
    Manages event subscriptions for a logical consumer group.

    Wraps the EventBus subscribe/unsubscribe API with lifecycle
    management and subscription tracking.

    Usage:
        bus = InMemoryEventBus()
        consumer = EventConsumer(bus, group="indexer-workers")
        sub_id = consumer.on("document.indexed", my_handler)
        consumer.start()   # no-op for in-memory
        ...
        consumer.off(sub_id)
        consumer.stop()
    """

    def __init__(self, bus: EventBus, group: str = "default") -> None:
        self._bus = bus
        self._group = group
        self._subscriptions: list[str] = []
        self._running = False

    def on(self, topic: str, handler: Callable[[Event], None]) -> str:
        """
        Subscribe a handler to a topic.

        Returns a subscription_id for later unsubscription.
        """
        sub_id = self._bus.subscribe(topic, handler)
        self._subscriptions.append(sub_id)
        logger.debug(
            "Consumer group='%s' subscribed to '%s' (sub=%s)",
            self._group, topic, sub_id[:8],
        )
        return sub_id

    def off(self, subscription_id: str) -> None:
        """Unsubscribe a handler by its subscription ID."""
        self._bus.unsubscribe(subscription_id)
        self._subscriptions = [s for s in self._subscriptions if s != subscription_id]
        logger.debug(
            "Consumer group='%s' unsubscribed sub=%s",
            self._group, subscription_id[:8],
        )

    def handler_count(self) -> int:
        """Return the number of active subscriptions for this consumer."""
        return len(self._subscriptions)

    def start(self) -> None:
        """
        Start the consumer.

        No-op for the in-memory bus.  For Kafka/RabbitMQ, this would
        start the poll loop or begin consuming from the queue.
        """
        self._running = True
        logger.debug("Consumer group='%s' started (%d handlers)", self._group, self.handler_count())

    def stop(self) -> None:
        """
        Stop the consumer.

        No-op for the in-memory bus.  For Kafka/RabbitMQ, this would
        close the connection and commit offsets.
        """
        self._running = False
        logger.debug("Consumer group='%s' stopped", self._group)

    @property
    def is_running(self) -> bool:
        """Whether the consumer is in the running state."""
        return self._running

    @property
    def group(self) -> str:
        """The consumer group name."""
        return self._group
