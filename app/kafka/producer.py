"""
Kafka Event Producer — Phase 8 Batch 2

=== THEORY ===

A Kafka producer is the client-side component that publishes records
(key-value pairs) to Kafka topics.  Under the hood, records are:

  1. Serialised (key + value → bytes)
  2. Partitioned (hash of key mod num_partitions → target partition)
  3. Batched (linger.ms window to amortise network round-trips)
  4. Compressed (optional: gzip, snappy, lz4, zstd)
  5. Sent to the partition leader broker

Kafka provides three delivery semantics via the `acks` setting:
  acks=0  — fire-and-forget (fastest, may lose messages)
  acks=1  — leader acknowledgement (default, possible loss on leader crash)
  acks=-1 — all in-sync replicas acknowledge (strongest, highest latency)

=== ARCHITECTURE ===

  Application code
    │
    │  producer.produce(topic, event)
    ▼
  KafkaEventProducer
    │  1. Serialises Event to JSON via event.to_dict()
    │  2. Uses event.metadata.event_id as the Kafka message key
    │  3. Calls confluent_kafka.Producer.produce()
    │  4. Calls poll(0) to trigger delivery callbacks
    ▼
  Kafka broker (topic, partition)

=== COMPLEXITY ===

  produce():  O(n) where n = size of serialised event payload
  flush():    O(m) where m = number of buffered messages
  close():    O(m) flush + connection teardown

=== SPACE COMPLEXITY ===

  O(B) where B = producer buffer size (queue.buffering.max.messages)

=== TRADEOFFS ===

  + Asynchronous batching for high throughput
  + Delivery callbacks for error handling
  + Thread-safe (confluent-kafka handles internal locking)
  - Requires Kafka broker to be running
  - No graceful fallback (raises ImportError if confluent_kafka missing)

=== PRODUCTION EQUIVALENTS ===

  LinkedIn:  librdkafka-based producer (same as confluent-kafka)
  Uber:      Custom Kafka producer with schema registry integration
  Netflix:   Kafka producer with Avro serialisation and Confluent Schema Registry
  Stripe:    Kafka producer with exactly-once semantics (idempotent producer)
"""

import json
import logging
import threading

from app.config import KafkaConfig
from app.events.models import Event

logger = logging.getLogger(__name__)


class KafkaEventProducer:
    """
    Kafka-backed event producer using confluent-kafka.

    Serialises Event objects to JSON and publishes them to Kafka topics.
    The event's metadata.event_id is used as the Kafka message key to
    ensure deterministic partitioning (same event always lands on the
    same partition, preserving per-entity ordering).

    Usage:
        config = KafkaConfig(bootstrap_servers="localhost:9092")
        producer = KafkaEventProducer(config)
        producer.produce("document.indexed", event)
        producer.flush()
        producer.close()
    """

    def __init__(self, config: KafkaConfig) -> None:
        try:
            from confluent_kafka import Producer as _KafkaProducer  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "confluent-kafka library is required for KafkaEventProducer. "
                "Install with: pip install confluent-kafka"
            ) from exc

        self._config = config
        self._lock = threading.Lock()
        self._closed = False

        # Build producer configuration from KafkaConfig
        producer_conf = {
            "bootstrap.servers": config.bootstrap_servers,
            "client.id": f"search-engine-producer",
            "acks": "all",
            "retries": 3,
            "retry.backoff.ms": 100,
            "linger.ms": 5,
            "compression.type": "snappy",
            "request.timeout.ms": config.request_timeout_ms,
        }

        from confluent_kafka import Producer as _KafkaProducer
        self._producer = _KafkaProducer(producer_conf)
        self._delivery_errors: list[str] = []

        logger.info(
            "KafkaEventProducer initialised: brokers=%s",
            config.bootstrap_servers,
        )

    def _delivery_callback(self, err, msg) -> None:
        """
        Callback invoked by confluent-kafka when a message is delivered
        (or fails to deliver).  Called from the producer's poll() thread.
        """
        if err is not None:
            error_msg = f"Delivery failed for topic={msg.topic()}: {err}"
            logger.error(error_msg)
            with self._lock:
                self._delivery_errors.append(error_msg)
        else:
            logger.debug(
                "Delivered to topic=%s partition=%d offset=%d",
                msg.topic(), msg.partition(), msg.offset(),
            )

    def produce(self, topic: str, event: Event) -> None:
        """
        Publish an event to a Kafka topic.

        The event is serialised to JSON.  The event_id is used as the
        message key for deterministic partitioning.

        Args:
            topic: Kafka topic name (e.g. "document.indexed")
            event: Event object to publish

        Raises:
            RuntimeError: if the producer has been closed
            BufferError:  if the internal producer queue is full
        """
        if self._closed:
            raise RuntimeError("Cannot produce: KafkaEventProducer is closed")

        # Serialise event to JSON bytes
        key = event.metadata.event_id.encode("utf-8")
        value = json.dumps(event.to_dict()).encode("utf-8")

        self._producer.produce(
            topic=topic,
            key=key,
            value=value,
            callback=self._delivery_callback,
        )

        # Trigger delivery callbacks without blocking
        self._producer.poll(0)

        logger.debug(
            "Produced event id=%s to topic='%s' (%d bytes)",
            event.metadata.event_id[:8], topic, len(value),
        )

    def flush(self, timeout: float = 10.0) -> int:
        """
        Wait for all buffered messages to be delivered.

        Args:
            timeout: maximum seconds to wait for delivery

        Returns:
            Number of messages still in the queue (0 if all delivered)
        """
        if self._closed:
            return 0
        remaining = self._producer.flush(timeout)
        if remaining > 0:
            logger.warning(
                "Flush timeout: %d messages still in queue", remaining,
            )
        return remaining

    def close(self) -> None:
        """
        Flush remaining messages and close the producer.

        After close(), any call to produce() will raise RuntimeError.
        """
        if self._closed:
            return
        with self._lock:
            self._closed = True
        self.flush(timeout=10.0)
        logger.info("KafkaEventProducer closed")

    @property
    def delivery_errors(self) -> list[str]:
        """Return a copy of the delivery error log."""
        with self._lock:
            return list(self._delivery_errors)
