"""
Event Producer — Phase 8

=== THEORY ===

The Producer pattern (also called Publisher or Emitter) encapsulates the
mechanics of creating well-formed events and dispatching them through
the event bus.

By centralising event creation in a Producer, we ensure:
  1. Consistent metadata (auto-generated IDs, timestamps, source)
  2. Correlation tracking (correlation_id propagation across flows)
  3. Causation chains (causation_id links cause → effect)
  4. Single point of instrumentation for outgoing events

=== ARCHITECTURE ===

  Application code
    │
    │  producer.emit("document.indexed", {"doc_id": "abc"})
    ▼
  EventProducer
    │  1. Creates EventMetadata (id, timestamp, source)
    │  2. Wraps topic + payload + metadata into Event
    │  3. Publishes through the injected EventBus
    ▼
  EventBus.publish(event)

=== COMPLEXITY ===

  emit(): O(1) for event creation + O(k) for bus dispatch
          where k = number of subscribers for the topic

=== SPACE COMPLEXITY ===

  O(1) — the producer holds no state beyond a bus reference and source name

=== TRADEOFFS ===

  + Thin wrapper: minimal overhead, easy to reason about
  + Correlation/causation support enables distributed tracing
  + Source tagging identifies the originating component
  - No batching (one event per emit call)
  - No schema validation (consumer's responsibility)

=== PRODUCTION EQUIVALENTS ===

  Kafka Producer:      KafkaProducer with serializer and partitioner
  AWS EventBridge:     PutEvents API with source and detail-type
  RabbitMQ:            Channel.basic_publish with exchange and routing key
  OpenTelemetry:       Span creation with trace_id and parent_span_id
"""

import logging
from typing import Callable

from app.events.bus import EventBus
from app.events.models import Event, EventMetadata

logger = logging.getLogger(__name__)


class EventProducer:
    """
    Convenience wrapper for publishing well-formed events.

    Encapsulates metadata generation so callers only need to provide
    the topic and payload.  The source field tags every outgoing event
    with the originating component name.

    Usage:
        bus = InMemoryEventBus()
        producer = EventProducer(bus, source="indexer")
        event = producer.emit("document.indexed", {"doc_id": "abc"})
    """

    def __init__(self, bus: EventBus, source: str = "search-engine") -> None:
        self._bus = bus
        self._source = source

    def emit(
        self,
        topic: str,
        payload: dict,
        correlation_id: str | None = None,
        causation_id: str | None = None,
    ) -> Event:
        """
        Create and publish an event.

        Args:
            topic:          well-known topic string (see topics.py)
            payload:        domain-specific data
            correlation_id: optional correlation identifier for tracing
            causation_id:   optional ID of the event that caused this one

        Returns:
            The published Event (with auto-generated metadata).
        """
        metadata = EventMetadata(
            source=self._source,
            correlation_id=correlation_id or "",
            causation_id=causation_id or "",
        )
        event = Event(topic=topic, payload=payload, metadata=metadata)
        logger.debug(
            "Emitting event topic='%s' id=%s source='%s'",
            topic, metadata.event_id[:8], self._source,
        )
        self._bus.publish(event)
        return event
