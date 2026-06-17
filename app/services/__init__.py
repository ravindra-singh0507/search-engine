"""
Microservice Architecture — Phase 8 Batch 3

=== THEORY ===

Microservice architecture decomposes a monolithic application into a suite
of small, independently deployable services, each running in its own process
and communicating via lightweight protocols (HTTP/gRPC).

Key principles:
  - Single Responsibility: each service owns one bounded context
  - Independent Deployment: change one service without redeploying others
  - Decentralised Data: each service owns its data store
  - Design for Failure: services must handle downstream failures gracefully

This module implements the foundational infrastructure:
  - ServiceRegistry: tracks running instances and their health
  - ServiceDiscovery: client-side lookup with load balancing
  - HealthCheck: liveness and readiness probes

=== PRODUCTION EQUIVALENTS ===

  Netflix:    Eureka (registry) + Ribbon (client-side LB) + Hystrix (circuit breaker)
  Kubernetes: kube-dns + Service + Endpoints + readinessProbe/livenessProbe
  Consul:     Service mesh with health checks and DNS-based discovery
  Uber:       Hyperbahn (TChannel) + YARPC service framework
"""

from app.services.registry import ServiceRegistry, ServiceInstance, ServiceStatus
from app.services.health import HealthCheck, HealthStatus
from app.services.discovery import ServiceDiscovery

__all__ = [
    "ServiceRegistry",
    "ServiceInstance",
    "ServiceStatus",
    "HealthCheck",
    "HealthStatus",
    "ServiceDiscovery",
]
