"""
Kafka Event Consumer — Phase 8 Batch 2

=== THEORY ===

A Kafka consumer reads records from topic partitions.  Consumers belong
to a *consumer group*; Kafka assigns each partition to exactly one
consumer in the group (the group coordinator manages this via the
rebalance protocol).

Key concepts:
  - Offset:  sequential position of a record within a partition.
             Consumers track their offset to know where to resume.
  - Commit:  persisting the current offset so that, on restart, the
             consumer resumes from the last committed position.
  - Poll loop:  the consumer calls poll() in a loop to fetch batches
                of records.  poll() also triggers heartbeats and
                rebalance callbacks.
  - Deserialization:  converting raw bytes back into application objects.

Delivery semantics per consumer group:
  At-most-once:   commit before processing (may lose messages on crash)
  At-least-once:  commit after processing (may reprocess on crash)
  Exactly-once:   transactional consume-process-produce (Kafka Streams)

=== ARCHITECTURE ===

  Kafka broker (topic, partition)
    │
    │  consumer.poll(timeout)
    ▼
  KafkaEventConsumer
    │  1. Deserialises JSON value → Event via Event.from_dict()
    │  2. Dispatches to all registered handlers
    │  3. Commits offset (auto or manual)
    ▼
  Application handler functions

=== COMPLEXITY ===

  poll_once():  O(n * h) where n = messages polled, h = number of handlers
  start():      O(1) — spawns background thread
  stop():       O(1) — sets flag, thread exits on next poll

=== SPACE COMPLEXITY ===

  O(H) where H = registered handlers
  O(B) where B = max.poll.records per poll batch (transient)

=== TRADEOFFS ===

  + Background thread polling — non-blocking for application code
  + Handler-based dispatch — same pattern as InMemoryEventBus
  + Graceful shutdown via threading.Event
  - Single-threaded consumption (one thread per consumer instance)
  - No manual offset management (relies on auto-commit or per-poll commit)

=== PRODUCTION EQUIVALENTS ===

  LinkedIn:  librdkafka consumer with cooperative rebalancing
  Uber:      Consumer with custom partition assignment and offset storage
  Netflix:   Consumer with Avro deserialization and schema registry
  Stripe:    Consumer with exactly-once transactional processing
"""

import json
import logging
import threading
from typing import Callable

from app.config import KafkaConfig
from app.events.models import Event

logger = logging.getLogger(__name__)


class KafkaEventConsumer:
    """
    Kafka consumer that polls messages and dispatches to registered handlers.

    Runs a background polling thread that fetches messages from subscribed
    topics, deserialises them into Event objects, and invokes all registered
    handler functions.

    Usage:
        config = KafkaConfig(bootstrap_servers="localhost:9092")
        consumer = KafkaEventConsumer(config, topics=["document.indexed"])
        consumer.subscribe(my_handler)
        consumer.start()
        # ... later ...
        consumer.stop()
    """

    def __init__(
        self,
        config: KafkaConfig,
        topics: list[str],
        group_id: str | None = None,
    ) -> None:
        try:
            from confluent_kafka import Consumer as _KafkaConsumer  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "confluent-kafka library is required for KafkaEventConsumer. "
                "Install with: pip install confluent-kafka"
            ) from exc

        self._config = config
        self._topics = list(topics)
        self._group_id = group_id or config.group_id
        self._handlers: list[Callable[[Event], None]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False

        # Build consumer configuration from KafkaConfig
        consumer_conf = {
            "bootstrap.servers": config.bootstrap_servers,
            "group.id": self._group_id,
            "auto.offset.reset": config.auto_offset_reset,
            "enable.auto.commit": config.enable_auto_commit,
            "max.poll.interval.ms": config.session_timeout_ms * 2,
            "session.timeout.ms": config.session_timeout_ms,
        }

        from confluent_kafka import Consumer as _KafkaConsumer
        self._consumer = _KafkaConsumer(consumer_conf)
        self._consumer.subscribe(self._topics)

        logger.info(
            "KafkaEventConsumer initialised: brokers=%s topics=%s group=%s",
            config.bootstrap_servers, self._topics, self._group_id,
        )

    def subscribe(self, handler: Callable[[Event], None]) -> None:
        """
        Register a handler function to receive deserialized events.

        All registered handlers are called for every message consumed.
        Handlers are called synchronously in registration order.

        Args:
            handler: callable that accepts an Event object
        """
        with self._lock:
            self._handlers.append(handler)
        logger.debug(
            "Handler registered: %s (total=%d)",
            getattr(handler, "__name__", repr(handler)),
            len(self._handlers),
        )

    def start(self) -> None:
        """
        Start the background polling thread.

        The thread runs a loop calling poll() and dispatching messages
        to all registered handlers until stop() is called.
        """
        if self._running:
            logger.warning("Consumer already running")
            return

        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            name=f"kafka-consumer-{self._group_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("Consumer started: group=%s", self._group_id)

    def stop(self) -> None:
        """
        Stop the background polling thread and close the consumer.

        Signals the poll loop to exit and waits for the thread to finish.
        """
        if not self._running:
            return

        self._stop_event.set()
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

        try:
            self._consumer.close()
        except Exception as exc:
            logger.warning("Error closing consumer: %s", exc)

        logger.info("Consumer stopped: group=%s", self._group_id)

    def poll_once(self, timeout: float = 1.0) -> Event | None:
        """
        Poll for a single message (useful for testing).

        Deserialises the message and dispatches to handlers, then returns
        the Event.  Returns None if no message is available within the
        timeout.

        Args:
            timeout: maximum seconds to wait for a message

        Returns:
            The deserialized Event, or None if no message was available.
        """
        msg = self._consumer.poll(timeout)

        if msg is None:
            return None

        if msg.error():
            logger.error("Consumer poll error: %s", msg.error())
            return None

        event = self._deserialize(msg)
        if event is None:
            return None

        self._dispatch(event)
        return event

    def _poll_loop(self) -> None:
        """
        Background polling loop.

        Runs until the stop event is set.  Each iteration polls for a
        batch of messages and dispatches them to all registered handlers.
        """
        logger.debug("Poll loop started: group=%s", self._group_id)

        while not self._stop_event.is_set():
            try:
                msg = self._consumer.poll(timeout=1.0)

                if msg is None:
                    continue

                if msg.error():
                    logger.error("Consumer poll error: %s", msg.error())
                    continue

                event = self._deserialize(msg)
                if event is not None:
                    self._dispatch(event)

            except Exception as exc:
                logger.error("Poll loop error: %s", exc)
                if self._stop_event.is_set():
                    break

        logger.debug("Poll loop exited: group=%s", self._group_id)

    def _deserialize(self, msg) -> Event | None:
        """
        Deserialise a Kafka message value (JSON bytes) into an Event.

        Returns None if deserialisation fails.
        """
        try:
            value = msg.value()
            if value is None:
                return None
            data = json.loads(value.decode("utf-8"))
            return Event.from_dict(data)
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as exc:
            logger.error(
                "Failed to deserialize message from topic=%s: %s",
                msg.topic(), exc,
            )
            return None

    def _dispatch(self, event: Event) -> None:
        """
        Dispatch a deserialized event to all registered handlers.

        Errors in individual handlers are logged but do not prevent
        remaining handlers from executing.
        """
        with self._lock:
            handlers = list(self._handlers)

        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:
                logger.error(
                    "Handler %s failed for event id=%s topic='%s': %s",
                    getattr(handler, "__name__", repr(handler)),
                    event.metadata.event_id[:8],
                    event.topic,
                    exc,
                )

    @property
    def is_running(self) -> bool:
        """Whether the consumer poll loop is active."""
        return self._running

    @property
    def handler_count(self) -> int:
        """Number of registered handlers."""
        with self._lock:
            return len(self._handlers)
