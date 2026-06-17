"""
Service Discovery — Client-Side Discovery Pattern

=== THEORY ===

Service discovery is the mechanism by which a client (consumer) locates
a service instance (provider) in a dynamic environment where instances
come and go (scaling, failures, deployments).

Two main patterns:
  1. Client-side discovery (this implementation):
     - Client queries the registry directly
     - Client applies load balancing logic (round-robin, random, etc.)
     - Pros: simple, no extra infrastructure, client controls routing
     - Cons: client coupled to registry, must implement LB logic

  2. Server-side discovery (not implemented here):
     - Client sends requests to a load balancer (LB)
     - LB queries the registry and routes to an instance
     - Pros: client is simple, LB handles everything
     - Cons: extra hop, LB is a single point of failure
     - Examples: AWS ALB, Kubernetes Service (kube-proxy)

Load balancing strategies implemented:
  - Round-robin: distribute requests evenly across instances
  - Random: pick a random healthy instance (simple, effective)

=== COMPLEXITY ===

  discover():      O(K) — K = healthy instances of the service
  discover_all():  O(K) — returns all healthy instances

=== PRODUCTION EQUIVALENTS ===

  Netflix Ribbon:    Client-side LB with round-robin, weighted, zone-aware
  Envoy:            Proxy-based discovery + LB (server-side pattern)
  gRPC:             Built-in client-side LB with name resolution
  Kubernetes:       kube-proxy iptables rules (server-side, transparent)
"""

import logging
import random
import threading
from typing import Optional

from app.services.registry import ServiceRegistry, ServiceInstance

logger = logging.getLogger(__name__)


class ServiceDiscovery:
    """
    Client-side service discovery.

    Queries the registry to find service endpoints.
    Supports round-robin and random load balancing.

    Usage:
        discovery = ServiceDiscovery(registry)

        # Get one instance (round-robin)
        instance = discovery.discover("indexer")
        if instance:
            url = f"http://{instance.host}:{instance.port}/api"

        # Get all healthy instances
        all_instances = discovery.discover_all("retrieval")
    """

    def __init__(self, registry: ServiceRegistry) -> None:
        """
        Args:
            registry: ServiceRegistry to query for instances.
        """
        self._registry = registry
        self._lock = threading.Lock()
        # Round-robin counters per service name
        self._counters: dict[str, int] = {}
        logger.info("ServiceDiscovery initialized (strategy=round-robin)")

    def discover(self, service_name: str) -> Optional[ServiceInstance]:
        """
        Discover one healthy instance using round-robin load balancing.

        The round-robin counter advances on each call, distributing
        requests evenly across all healthy instances.

        Args:
            service_name: Logical service name to look up.

        Returns:
            A healthy ServiceInstance, or None if no healthy instances exist.
        """
        instances = self._registry.get_healthy(service_name)
        if not instances:
            logger.debug("No healthy instances for service '%s'", service_name)
            return None

        with self._lock:
            counter = self._counters.get(service_name, 0)
            selected = instances[counter % len(instances)]
            self._counters[service_name] = counter + 1

        logger.debug(
            "Discovered %s instance %s at %s",
            service_name, selected.instance_id[:8], selected.address(),
        )
        return selected

    def discover_all(self, service_name: str) -> list[ServiceInstance]:
        """
        Discover all healthy instances for a service.

        Useful when the client wants to implement its own routing logic
        or fan out requests to all instances (scatter-gather pattern).

        Args:
            service_name: Logical service name to look up.

        Returns:
            List of healthy ServiceInstance objects. Empty if none available.
        """
        instances = self._registry.get_healthy(service_name)
        logger.debug(
            "Discovered %d healthy instances for '%s'",
            len(instances), service_name,
        )
        return instances

    def discover_random(self, service_name: str) -> Optional[ServiceInstance]:
        """
        Discover one healthy instance using random selection.

        Random selection provides good load distribution without
        maintaining state (no counter).  Useful for stateless services
        where strict round-robin is not required.

        Args:
            service_name: Logical service name to look up.

        Returns:
            A healthy ServiceInstance, or None if no healthy instances exist.
        """
        instances = self._registry.get_healthy(service_name)
        if not instances:
            return None
        return random.choice(instances)

    def reset_counters(self) -> None:
        """Reset all round-robin counters (e.g., after topology change)."""
        with self._lock:
            self._counters.clear()
