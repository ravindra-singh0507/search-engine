"""
Event Models — Phase 8

=== THEORY ===

Events are immutable records of *something that happened* in the system.
They follow the Event Sourcing pattern (Fowler 2005): instead of storing
only current state, we capture every state change as a first-class object.

Key properties of a well-designed event:
  1. Immutable      — once created, never mutated
  2. Self-describing — carries its own metadata (id, timestamp, source)
  3. Correlatable   — correlation_id links events across service boundaries
  4. Versioned      — schema version enables backward-compatible evolution

=== ARCHITECTURE ===

  EventMetadata
    │  (id, timestamp, source, correlation_id, causation_id, version)
    ▼
  Event
    │  topic  — what happened (e.g. "document.indexed")
    │  payload — domain-specific data (arbitrary dict)
    │  metadata — envelope info
    ▼
  EventEnvelope
       wraps Event with delivery status tracking
       (delivered_at, status, error)

=== DATA STRUCTURES ===

  EventMetadata  — lightweight envelope; all fields have sensible defaults
  Event          — the unit of communication between producers and consumers
  EventEnvelope  — tracks delivery lifecycle (PENDING → DELIVERED | FAILED)
  EventStatus    — enum for delivery state machine

=== COMPLEXITY ===

  Event.to_dict():   O(n) where n = size of payload
  Event.from_dict(): O(n) where n = size of dict

=== SPACE COMPLEXITY ===

  Event:         O(len(payload)) — bounded by producer
  EventEnvelope: O(len(event))   — thin wrapper

=== TRADEOFFS ===

  + dataclass-based: fast construction, built-in __eq__, repr
  + Default factories for id/timestamp eliminate boilerplate
  + Serialisation via to_dict / from_dict (no Pydantic dependency)
  - No schema validation on payload (consumer's responsibility)
  - No Avro/Protobuf encoding (JSON only for simplicity)

=== PRODUCTION EQUIVALENTS ===

  CloudEvents spec: standardised event envelope (type, source, id, time)
  Apache Kafka:     ProducerRecord with key, value, headers
  AWS EventBridge:  PutEventsRequestEntry (source, detail-type, detail)
  Confluent Schema Registry: Avro/Protobuf envelope with schema ID
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Enumerations ─────────────────────────────────────────────────────────────

class EventStatus(str, Enum):
    """Delivery status of an event through the bus."""
    PENDING       = "pending"
    DELIVERED     = "delivered"
    FAILED        = "failed"
    DEAD_LETTERED = "dead_lettered"


# ── Metadata ─────────────────────────────────────────────────────────────────

@dataclass
class EventMetadata:
    """
    Envelope metadata attached to every event.

    Fields:
      event_id       — globally unique identifier (auto-generated UUID)
      timestamp      — Unix epoch seconds when the event was created
      source         — originating component (e.g. "crawler", "indexer")
      correlation_id — groups related events across an end-to-end flow
      causation_id   — the event_id that directly caused this event
      version        — schema version for backward-compatible evolution
      retry_count    — how many delivery attempts have been made
      max_retries    — upper bound on retry attempts
    """
    event_id:       str   = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:      float = field(default_factory=time.time)
    source:         str   = ""
    correlation_id: str   = ""
    causation_id:   str   = ""
    version:        int   = 1
    retry_count:    int   = 0
    max_retries:    int   = 3


# ── Event ────────────────────────────────────────────────────────────────────

@dataclass
class Event:
    """
    The fundamental unit of communication in the event-driven architecture.

    An event represents a fact — something that already happened.
    It is immutable after creation and self-describing via its metadata.

    Fields:
      topic    — well-known topic string (see topics.py)
      payload  — domain-specific data as a plain dict
      metadata — envelope info (id, timestamp, source, correlation)
    """
    topic:    str
    payload:  dict[str, Any]
    metadata: EventMetadata = field(default_factory=EventMetadata)

    def to_dict(self) -> dict:
        """Serialise the event to a plain dictionary."""
        return {
            "topic":   self.topic,
            "payload": self.payload,
            "metadata": {
                "event_id":       self.metadata.event_id,
                "timestamp":      self.metadata.timestamp,
                "source":         self.metadata.source,
                "correlation_id": self.metadata.correlation_id,
                "causation_id":   self.metadata.causation_id,
                "version":        self.metadata.version,
                "retry_count":    self.metadata.retry_count,
                "max_retries":    self.metadata.max_retries,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Event":
        """Deserialise an event from a plain dictionary."""
        meta_data = data.get("metadata", {})
        metadata = EventMetadata(
            event_id=meta_data.get("event_id", str(uuid.uuid4())),
            timestamp=meta_data.get("timestamp", time.time()),
            source=meta_data.get("source", ""),
            correlation_id=meta_data.get("correlation_id", ""),
            causation_id=meta_data.get("causation_id", ""),
            version=meta_data.get("version", 1),
            retry_count=meta_data.get("retry_count", 0),
            max_retries=meta_data.get("max_retries", 3),
        )
        return cls(
            topic=data["topic"],
            payload=data.get("payload", {}),
            metadata=metadata,
        )


# ── Envelope ─────────────────────────────────────────────────────────────────

@dataclass
class EventEnvelope:
    """
    Wraps an Event with delivery-lifecycle tracking.

    Used by the EventStore to record when an event was delivered,
    its current status, and any error message if delivery failed.

    Fields:
      event        — the wrapped event
      delivered_at — Unix epoch when delivery was confirmed (None if pending)
      status       — current delivery status
      error        — error message if status == FAILED or DEAD_LETTERED
    """
    event:        Event
    delivered_at: float | None  = None
    status:       EventStatus   = EventStatus.PENDING
    error:        str | None    = None
