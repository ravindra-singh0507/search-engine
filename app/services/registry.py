"""
Service Registry — Microservice Architecture

=== THEORY ===

A service registry is a database of service instances.  It is the core
component of the service discovery pattern.  Services register themselves
on startup and deregister on shutdown.  Consumers query the registry to
locate available instances.

Registration lifecycle:
  1. Service starts -> register(name, host, port, metadata)
  2. Service sends heartbeats -> heartbeat(instance_id) every N seconds
  3. Registry marks instances as unhealthy if heartbeat is stale
  4. Service shutting down -> deregister(instance_id)
  5. Registry cleanup removes stale entries periodically

Health model:
  HEALTHY   — receiving heartbeats within threshold
  UNHEALTHY — heartbeat missed beyond stale_threshold
  DRAINING  — instance is shutting down gracefully (no new traffic)
  UNKNOWN   — just registered, no heartbeat yet

=== COMPLEXITY ===

  register():       O(1) — dict insertion
  deregister():     O(1) — dict removal
  heartbeat():      O(1) — timestamp update
  get_instances():  O(K) — K = instances of that service
  get_healthy():    O(K) — filter by status
  cleanup_stale():  O(N) — N = total instances across all services

=== PRODUCTION EQUIVALENTS ===

  Netflix Eureka:     AP (available + partition-tolerant) registry with replication
  Consul:             CP (consistent + partition-tolerant) with Raft consensus
  Kubernetes:         etcd-backed Service/Endpoints with kubelet health probes
  ZooKeeper:          Ephemeral nodes for service registration (LinkedIn, Kafka)
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.config import ServiceRegistryConfig

logger = logging.getLogger(__name__)


class ServiceStatus(str, Enum):
    """Health status of a service instance."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    UNKNOWN = "unknown"


@dataclass
class ServiceInstance:
    """
    Represents one running instance of a service.

    Fields:
      service_name   — logical name (e.g. "indexer", "retrieval", "gateway")
      instance_id    — unique identifier for this process (UUID)
      host           — hostname or IP address
      port           — TCP port the service listens on
      status         — current health status
      metadata       — arbitrary key-value metadata (version, region, etc.)
      registered_at  — unix timestamp of initial registration
      last_heartbeat — unix timestamp of most recent heartbeat
      health_check_url — optional HTTP endpoint for active health probing
    """
    service_name:    str
    instance_id:     str
    host:            str
    port:            int
    status:          ServiceStatus = ServiceStatus.UNKNOWN
    metadata:        dict[str, Any] = field(default_factory=dict)
    registered_at:   float = field(default_factory=time.time)
    last_heartbeat:  float = field(default_factory=time.time)
    health_check_url: str = ""

    def address(self) -> str:
        """Return host:port string."""
        return f"{self.host}:{self.port}"

    def to_dict(self) -> dict:
        return {
            "service_name": self.service_name,
            "instance_id": self.instance_id,
            "host": self.host,
            "port": self.port,
            "status": self.status.value,
            "metadata": self.metadata,
            "registered_at": self.registered_at,
            "last_heartbeat": self.last_heartbeat,
            "health_check_url": self.health_check_url,
        }


class ServiceRegistry:
    """
    In-process service registry for service discovery.

    === THEORY ===

    Service registry maintains a list of available service instances.
    Services register on startup and send periodic heartbeats.
    Consumers query the registry to find healthy instances.
    Stale instances (no heartbeat) are marked unhealthy.

    This implementation is an in-process registry suitable for:
      - Single-node deployments (all services in one process)
      - Development and testing
      - Educational purposes (understanding the pattern)

    For distributed deployments, this would be backed by a coordination
    service (etcd, Consul, ZooKeeper) or a dedicated registry (Eureka).

    === PRODUCTION EQUIVALENTS ===

    Netflix Eureka, Consul, Kubernetes Service Discovery
    """

    def __init__(self, config: ServiceRegistryConfig, redis_client: Any = None):
        """
        Args:
            config: ServiceRegistryConfig with heartbeat/stale thresholds
            redis_client: Optional RedisClient for distributed state
        """
        self._config = config
        self._redis = redis_client
        self._lock = threading.Lock()
        # instance_id -> ServiceInstance
        self._instances: dict[str, ServiceInstance] = {}
        # service_name -> set of instance_ids
        self._services: dict[str, set[str]] = {}
        logger.info(
            "ServiceRegistry initialized: heartbeat_interval=%.1fs, stale_threshold=%.1fs",
            config.heartbeat_interval,
            config.stale_threshold_sec,
        )

    def register(
        self,
        service_name: str,
        host: str,
        port: int,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Register a new service instance.

        Returns:
            instance_id (UUID string) for use in heartbeat/deregister calls.

        Raises:
            ValueError: if max_instances limit is reached.
        """
        if metadata is None:
            metadata = {}

        with self._lock:
            if len(self._instances) >= self._config.max_instances:
                raise ValueError(
                    f"Registry full: max {self._config.max_instances} instances"
                )

            instance_id = str(uuid.uuid4())
            now = time.time()

            instance = ServiceInstance(
                service_name=service_name,
                instance_id=instance_id,
                host=host,
                port=port,
                status=ServiceStatus.HEALTHY,
                metadata=metadata,
                registered_at=now,
                last_heartbeat=now,
                health_check_url=f"http://{host}:{port}/health",
            )

            self._instances[instance_id] = instance

            if service_name not in self._services:
                self._services[service_name] = set()
            self._services[service_name].add(instance_id)

        logger.info(
            "Registered service %s instance %s at %s:%d",
            service_name, instance_id[:8], host, port,
        )
        return instance_id

    def deregister(self, instance_id: str) -> bool:
        """
        Remove a service instance from the registry.

        Returns:
            True if the instance was found and removed, False otherwise.
        """
        with self._lock:
            instance = self._instances.pop(instance_id, None)
            if instance is None:
                return False

            service_ids = self._services.get(instance.service_name)
            if service_ids:
                service_ids.discard(instance_id)
                if not service_ids:
                    del self._services[instance.service_name]

        logger.info(
            "Deregistered service %s instance %s",
            instance.service_name, instance_id[:8],
        )
        return True

    def heartbeat(self, instance_id: str) -> None:
        """
        Update the heartbeat timestamp for an instance.

        If the instance was previously marked unhealthy due to a stale
        heartbeat, this restores it to healthy status.

        Raises:
            KeyError: if instance_id is not registered.
        """
        with self._lock:
            instance = self._instances.get(instance_id)
            if instance is None:
                raise KeyError(f"Instance {instance_id} not registered")

            instance.last_heartbeat = time.time()
            if instance.status == ServiceStatus.UNHEALTHY:
                instance.status = ServiceStatus.HEALTHY
                logger.info(
                    "Instance %s recovered (heartbeat received)",
                    instance_id[:8],
                )

    def get_instances(self, service_name: str) -> list[ServiceInstance]:
        """
        Get all instances (any status) for a given service name.

        Returns:
            List of ServiceInstance objects. Empty list if service unknown.
        """
        with self._lock:
            instance_ids = self._services.get(service_name, set())
            return [
                self._instances[iid]
                for iid in instance_ids
                if iid in self._instances
            ]

    def get_healthy(self, service_name: str) -> list[ServiceInstance]:
        """
        Get only healthy instances for a given service name.

        This is the primary method used by ServiceDiscovery to find
        routable endpoints.

        Returns:
            List of healthy ServiceInstance objects.
        """
        with self._lock:
            instance_ids = self._services.get(service_name, set())
            return [
                self._instances[iid]
                for iid in instance_ids
                if iid in self._instances
                and self._instances[iid].status == ServiceStatus.HEALTHY
            ]

    def get_all_services(self) -> dict[str, list[ServiceInstance]]:
        """
        Get all registered services and their instances.

        Returns:
            Dict mapping service_name -> list of instances.
        """
        with self._lock:
            result: dict[str, list[ServiceInstance]] = {}
            for service_name, instance_ids in self._services.items():
                result[service_name] = [
                    self._instances[iid]
                    for iid in instance_ids
                    if iid in self._instances
                ]
            return result

    def health_check(self, instance_id: str) -> ServiceStatus:
        """
        Check and return the health status of a specific instance.

        If the heartbeat is stale (beyond stale_threshold_sec), the
        instance is automatically marked unhealthy.

        Returns:
            Current ServiceStatus of the instance.

        Raises:
            KeyError: if instance_id is not registered.
        """
        with self._lock:
            instance = self._instances.get(instance_id)
            if instance is None:
                raise KeyError(f"Instance {instance_id} not registered")

            now = time.time()
            elapsed = now - instance.last_heartbeat

            if elapsed > self._config.stale_threshold_sec:
                if instance.status == ServiceStatus.HEALTHY:
                    instance.status = ServiceStatus.UNHEALTHY
                    logger.warning(
                        "Instance %s marked unhealthy (no heartbeat for %.1fs)",
                        instance_id[:8], elapsed,
                    )

            return instance.status

    def mark_unhealthy(self, instance_id: str) -> None:
        """
        Manually mark an instance as unhealthy.

        Used when active health probes fail or when a circuit breaker
        trips for this instance.

        Raises:
            KeyError: if instance_id is not registered.
        """
        with self._lock:
            instance = self._instances.get(instance_id)
            if instance is None:
                raise KeyError(f"Instance {instance_id} not registered")

            instance.status = ServiceStatus.UNHEALTHY
            logger.warning("Instance %s manually marked unhealthy", instance_id[:8])

    def cleanup_stale(self) -> int:
        """
        Remove instances whose heartbeat exceeds the stale threshold.

        This is typically called periodically by a background thread
        or timer.  In production, the registry server runs this as
        part of its eviction loop.

        Returns:
            Number of instances removed.
        """
        now = time.time()
        removed = 0

        with self._lock:
            stale_ids = [
                iid for iid, inst in self._instances.items()
                if (now - inst.last_heartbeat) > self._config.stale_threshold_sec
                and inst.status != ServiceStatus.DRAINING
            ]

            for iid in stale_ids:
                instance = self._instances.pop(iid)
                service_ids = self._services.get(instance.service_name)
                if service_ids:
                    service_ids.discard(iid)
                    if not service_ids:
                        del self._services[instance.service_name]
                removed += 1
                logger.info(
                    "Evicted stale instance %s (%s)",
                    iid[:8], instance.service_name,
                )

        if removed:
            logger.info("Cleanup removed %d stale instances", removed)
        return removed

    def stats(self) -> dict:
        """
        Return registry statistics.

        Returns:
            Dict with total_instances, total_services, healthy_count,
            unhealthy_count, and per-service instance counts.
        """
        with self._lock:
            total = len(self._instances)
            healthy = sum(
                1 for inst in self._instances.values()
                if inst.status == ServiceStatus.HEALTHY
            )
            unhealthy = sum(
                1 for inst in self._instances.values()
                if inst.status == ServiceStatus.UNHEALTHY
            )
            per_service = {
                name: len(ids)
                for name, ids in self._services.items()
            }

        return {
            "total_instances": total,
            "total_services": len(per_service),
            "healthy_count": healthy,
            "unhealthy_count": unhealthy,
            "services": per_service,
        }
