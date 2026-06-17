"""
Kafka Infrastructure — Phase 8 Batch 2

Provides Apache Kafka integration for the search engine platform,
enabling distributed event streaming as an alternative to the
in-memory event bus.

Modules:
  producer      — KafkaEventProducer:  publish events to Kafka topics
  consumer      — KafkaEventConsumer:  poll and dispatch Kafka messages
  topic_manager — KafkaTopicManager:   create, list, delete Kafka topics
  bus           — KafkaEventBus:       EventBus protocol implementation
"""

from app.kafka.producer import KafkaEventProducer
from app.kafka.consumer import KafkaEventConsumer
from app.kafka.topic_manager import KafkaTopicManager
from app.kafka.bus import KafkaEventBus

__all__ = [
    "KafkaEventProducer",
    "KafkaEventConsumer",
    "KafkaTopicManager",
    "KafkaEventBus",
]
