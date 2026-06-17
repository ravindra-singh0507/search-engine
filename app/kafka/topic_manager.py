"""
Kafka Topic Manager — Phase 8 Batch 2

=== THEORY ===

Kafka topics are logical channels for organising messages.  Each topic
is divided into one or more *partitions*, which are the unit of parallelism
and replication.

Topic management involves:
  1. Creation:   define a topic with partition count and replication factor
  2. Listing:    discover existing topics in the cluster
  3. Deletion:   remove a topic and all its data
  4. Configuration: alter topic-level settings (retention, cleanup policy)

Partition count determines the maximum consumer parallelism: each
partition is consumed by exactly one consumer in a consumer group.

Replication factor determines fault tolerance: with replication factor N,
the cluster can tolerate N-1 broker failures without data loss.

=== ARCHITECTURE ===

  KafkaTopicManager
    │
    │  create_topic() / list_topics() / delete_topic()
    ▼
  AdminClient (confluent-kafka)
    │
    │  Metadata requests / CreateTopics / DeleteTopics RPCs
    ▼
  Kafka Broker (Controller)

=== COMPLEXITY ===

  create_topic():   O(1) — single RPC to controller
  list_topics():    O(T) — fetches metadata for all T topics
  delete_topic():   O(1) — single RPC to controller
  ensure_topics():  O(T) — list + create missing

=== SPACE COMPLEXITY ===

  O(1) — holds only a reference to the AdminClient

=== TRADEOFFS ===

  + Centralised topic management (no manual kafka-topics.sh)
  + ensure_topics() for idempotent initialisation at startup
  + Timeout-based error handling for robust operations
  - Requires admin privileges on the Kafka cluster
  - Topic deletion may be disabled on production clusters

=== PRODUCTION EQUIVALENTS ===

  Confluent Cloud:   REST API / Terraform provider for topic management
  AWS MSK:           AWS CLI / CloudFormation for topic provisioning
  Strimzi (K8s):     KafkaTopic CRD for declarative topic management
  LinkedIn:          Custom topic governance service with approval workflows
"""

import logging
import threading

from app.config import KafkaConfig

logger = logging.getLogger(__name__)


class KafkaTopicManager:
    """
    Manages Kafka topic creation, listing, and deletion.

    Wraps the confluent-kafka AdminClient to provide a clean interface
    for topic lifecycle management.  Used at application startup to
    ensure all required topics exist.

    Usage:
        config = KafkaConfig(bootstrap_servers="localhost:9092")
        manager = KafkaTopicManager(config)
        manager.ensure_topics(["document.indexed", "crawl.started"])
        print(manager.list_topics())
    """

    def __init__(self, config: KafkaConfig) -> None:
        try:
            from confluent_kafka.admin import AdminClient as _AdminClient  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "confluent-kafka library is required for KafkaTopicManager. "
                "Install with: pip install confluent-kafka"
            ) from exc

        self._config = config
        self._lock = threading.Lock()

        admin_conf = {
            "bootstrap.servers": config.bootstrap_servers,
            "request.timeout.ms": config.request_timeout_ms,
        }

        from confluent_kafka.admin import AdminClient as _AdminClient
        self._admin = _AdminClient(admin_conf)

        logger.info(
            "KafkaTopicManager initialised: brokers=%s",
            config.bootstrap_servers,
        )

    def create_topic(
        self,
        name: str,
        num_partitions: int = 3,
        replication_factor: int = 1,
    ) -> bool:
        """
        Create a Kafka topic.

        Args:
            name:               topic name (e.g. "document.indexed")
            num_partitions:     number of partitions (default 3)
            replication_factor: replication factor (default 1)

        Returns:
            True if the topic was created successfully, False on error.
        """
        from confluent_kafka.admin import NewTopic

        new_topic = NewTopic(
            topic=name,
            num_partitions=num_partitions,
            replication_factor=replication_factor,
        )

        with self._lock:
            futures = self._admin.create_topics([new_topic])

        for topic_name, future in futures.items():
            try:
                future.result()  # blocks until topic is created or fails
                logger.info(
                    "Topic created: %s (partitions=%d, replication=%d)",
                    topic_name, num_partitions, replication_factor,
                )
                return True
            except Exception as exc:
                error_str = str(exc)
                # TOPIC_ALREADY_EXISTS is not an error for idempotent creation
                if "TOPIC_ALREADY_EXISTS" in error_str:
                    logger.debug("Topic already exists: %s", topic_name)
                    return True
                logger.error("Failed to create topic '%s': %s", topic_name, exc)
                return False

        return False

    def list_topics(self) -> list[str]:
        """
        List all topics in the Kafka cluster.

        Returns:
            List of topic name strings.
        """
        try:
            with self._lock:
                metadata = self._admin.list_topics(timeout=10.0)
            topics = list(metadata.topics.keys())
            logger.debug("Listed %d topics", len(topics))
            return topics
        except Exception as exc:
            logger.error("Failed to list topics: %s", exc)
            return []

    def delete_topic(self, name: str) -> bool:
        """
        Delete a Kafka topic.

        Note: topic deletion must be enabled on the broker
        (delete.topic.enable=true, which is the default since Kafka 1.0).

        Args:
            name: topic name to delete

        Returns:
            True if deletion succeeded, False on error.
        """
        with self._lock:
            futures = self._admin.delete_topics([name])

        for topic_name, future in futures.items():
            try:
                future.result()  # blocks until deletion completes
                logger.info("Topic deleted: %s", topic_name)
                return True
            except Exception as exc:
                error_str = str(exc)
                if "UNKNOWN_TOPIC_OR_PARTITION" in error_str:
                    logger.debug("Topic does not exist: %s", topic_name)
                    return True
                logger.error("Failed to delete topic '%s': %s", topic_name, exc)
                return False

        return False

    def ensure_topics(self, topics: list[str]) -> None:
        """
        Ensure that all specified topics exist, creating any that are missing.

        This is idempotent: topics that already exist are left unchanged.
        Uses the default partition count and replication factor from config.

        Args:
            topics: list of topic names to ensure exist
        """
        existing = set(self.list_topics())

        for topic_name in topics:
            if topic_name not in existing:
                self.create_topic(
                    name=topic_name,
                    num_partitions=self._config.num_partitions,
                    replication_factor=self._config.replication_factor,
                )
            else:
                logger.debug("Topic already exists: %s", topic_name)

        logger.info(
            "Ensured %d topics (%d already existed)",
            len(topics),
            len(set(topics) & existing),
        )
