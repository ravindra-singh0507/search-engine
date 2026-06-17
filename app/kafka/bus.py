"""
Kafka Event Bus — Phase 8 Batch 2

=== THEORY ===

The KafkaEventBus implements the EventBus protocol (from app.events.bus)
with Apache Kafka as the transport layer.  This enables a drop-in swap
from InMemoryEventBus to distributed messaging:

  config.events.backend = "memory"  → InMemoryEventBus (single process)
  config.events.backend = "kafka"   → KafkaEventBus   (distributed)

The key architectural difference:
  - InMemoryEventBus: synchronous, same-process dispatch via function calls
  - KafkaEventBus:    asynchronous, cross-process dispatch via Kafka topics

Each subscription creates a dedicated KafkaEventConsumer with its own
consumer group, ensuring that every subscriber receives every message
(fan-out).  Within a consumer group, Kafka handles partition assignment
and offset management automatically.

=== ARCHITECTURE ===

  Publisher ──► KafkaEventBus.publish(event)
                  │
                  ├── KafkaEventProducer.produce(topic, event)
                  │       │
                  │       ▼
                  │   Kafka Broker (topic partitions)
                  │       │
                  │       ▼
                  ├── KafkaEventConsumer (sub_1, group=sub-{uuid})
                  │       └── handler_A(event)
                  │
                  └── KafkaEventConsumer (sub_2, group=sub-{uuid})
                          └── handler_B(event)

=== COMPLEXITY ===

  publish():          O(n) serialisation + O(1) Kafka produce
  subscribe():        O(1) consumer creation + O(P) partition assignment
  unsubscribe():      O(1) consumer shutdown
  subscriber_count(): O(1)

  where n = event payload size, P = number of partitions

=== SPACE COMPLEXITY ===

  O(S) where S = number of active subscriptions (one consumer per sub)

=== TRADEOFFS ===

  + Distributed: works across multiple processes and machines
  + Durable: messages persisted to disk with configurable retention
  + Scalable: add partitions for higher throughput
  + Protocol-compatible: implements EventBus, drop-in replacement
  - Higher latency than in-memory (network + disk I/O)
  - Requires running Kafka cluster
  - Each subscription creates a consumer thread

=== PRODUCTION EQUIVALENTS ===

  Netflix:    Hermes — abstraction layer over Kafka with routing rules
  Uber:       Cherami / Kafka with unified messaging API
  LinkedIn:   Kafka with custom consumer framework (Brooklin)
  Confluent:  Schema Registry + Kafka Streams for typed event buses
"""

import logging
import threading
import uuid
from typing import Callable

from app.config import KafkaConfig
from app.events.models import Event
from app.kafka.producer import KafkaEventProducer
from app.kafka.consumer import KafkaEventConsumer

logger = logging.getLogger(__name__)


class KafkaEventBus:
    """
    Implements the EventBus protocol backed by Apache Kafka.

    Each publish() call produces a message to the event's topic.
    Each subscribe() call creates a new consumer with a unique group
    so that every subscriber receives every message (fan-out semantics).

    This class enables swapping InMemoryEventBus for KafkaEventBus
    by changing config.events.backend from "memory" to "kafka".

    Usage:
        config = KafkaConfig(bootstrap_servers="localhost:9092")
        bus = KafkaEventBus(config)

        sub_id = bus.subscribe("document.indexed", my_handler)
        bus.publish(event)

        bus.unsubscribe(sub_id)
        bus.close()
    """

    def __init__(self, config: KafkaConfig) -> None:
        self._config = config
        self._producer = KafkaEventProducer(config)
        self._lock = threading.Lock()

        # Map subscription_id → (topic, KafkaEventConsumer)
        self._subscriptions: dict[str, tuple[str, KafkaEventConsumer]] = {}

        logger.info(
            "KafkaEventBus initialised: brokers=%s",
            config.bootstrap_servers,
        )

    def publish(self, event: Event) -> None:
        """
        Publish an event to its topic via the Kafka producer.

        The event is serialised to JSON and sent to the Kafka topic
        matching event.topic.

        Args:
            event: Event to publish
        """
        self._producer.produce(event.topic, event)
        logger.debug(
            "Published event id=%s to topic='%s'",
            event.metadata.event_id[:8], event.topic,
        )

    def subscribe(self, topic: str, handler: Callable[[Event], None]) -> str:
        """
        Subscribe a handler to a Kafka topic.

        Creates a new KafkaEventConsumer with a unique consumer group
        (sub-{uuid}) so this subscriber receives all messages on the
        topic independently of other subscribers.

        Args:
            topic:   Kafka topic to subscribe to
            handler: callable that receives Event objects

        Returns:
            A subscription_id (UUID string) for unsubscription.
        """
        sub_id = str(uuid.uuid4())
        group_id = f"sub-{sub_id[:12]}"

        consumer = KafkaEventConsumer(
            config=self._config,
            topics=[topic],
            group_id=group_id,
        )
        consumer.subscribe(handler)
        consumer.start()

        with self._lock:
            self._subscriptions[sub_id] = (topic, consumer)

        logger.debug(
            "Subscribed to topic='%s' sub=%s group=%s",
            topic, sub_id[:8], group_id,
        )
        return sub_id

    def unsubscribe(self, subscription_id: str) -> None:
        """
        Remove a subscription and stop its consumer.

        Args:
            subscription_id: the ID returned by subscribe()
        """
        with self._lock:
            entry = self._subscriptions.pop(subscription_id, None)

        if entry is None:
            logger.debug("Subscription not found: %s", subscription_id[:8])
            return

        topic, consumer = entry
        consumer.stop()
        logger.debug(
            "Unsubscribed from topic='%s' sub=%s",
            topic, subscription_id[:8],
        )

    def subscriber_count(self, topic: str) -> int:
        """
        Return the number of active subscribers for a topic.

        Args:
            topic: topic to count subscribers for

        Returns:
            Number of active subscriptions for the given topic.
        """
        with self._lock:
            count = sum(
                1 for t, _ in self._subscriptions.values()
                if t == topic
            )
        return count

    def close(self) -> None:
        """
        Stop all consumers, flush and close the producer.

        After close(), the bus should not be used.
        """
        with self._lock:
            subs = list(self._subscriptions.items())
            self._subscriptions.clear()

        for sub_id, (topic, consumer) in subs:
            try:
                consumer.stop()
            except Exception as exc:
                logger.warning(
                    "Error stopping consumer sub=%s: %s",
                    sub_id[:8], exc,
                )

        self._producer.close()
        logger.info("KafkaEventBus closed")
