"""
Health Check System — Microservice Architecture

=== THEORY ===

Health checks are the foundation of service reliability in microservice
architectures.  They answer two distinct questions:

  Liveness:  "Is the process alive and running?"
    - If NO: the orchestrator should restart the process
    - Checks: uptime, basic memory/thread sanity
    - Should NEVER call external dependencies

  Readiness: "Can this instance serve traffic?"
    - If NO: the load balancer should stop routing to this instance
    - Checks: database connectivity, cache availability, model loaded, etc.
    - May call external dependencies (with short timeouts)

The separation matters because:
  - A service can be alive but not ready (e.g., loading a large model)
  - Routing traffic to a not-ready service causes errors
  - Restarting a not-ready-but-alive service wastes resources

=== KUBERNETES MAPPING ===

  livenessProbe:  -> liveness()
  readinessProbe: -> readiness()
  startupProbe:   -> (not implemented, but liveness covers startup)

=== PRODUCTION EQUIVALENTS ===

  Kubernetes: livenessProbe + readinessProbe (HTTP GET /healthz, /readyz)
  AWS ELB:    Health check endpoint with configurable interval/threshold
  Consul:     Service health checks (HTTP, TCP, script, TTL)
  Envoy:      Active health checking with ejection on failure

=== COMPLEXITY ===

  liveness():   O(1) — no external calls
  readiness():  O(K) — K = number of registered checks
  run_checks(): O(K) — runs each check function
"""

import logging
import threading
import time
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Overall health status of a service."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheck:
    """
    Health check system for service liveness and readiness.

    Liveness: is the process running?
    Readiness: can it serve traffic?

    Usage:
        health = HealthCheck()
        health.add_check("database", lambda: db.ping())
        health.add_check("cache", lambda: redis.ping())

        # Liveness probe (no deps)
        health.liveness()  # {"status": "alive", "uptime_seconds": 123.4}

        # Readiness probe (checks deps)
        health.readiness()  # {"status": "ready", "checks": {"database": True, ...}}
    """

    def __init__(self) -> None:
        self._checks: dict[str, Callable[[], bool]] = {}
        self._lock = threading.Lock()
        self._start_time = time.time()
        logger.info("HealthCheck system initialized")

    def add_check(self, name: str, check_fn: Callable[[], bool]) -> None:
        """
        Register a named readiness check.

        Args:
            name: Human-readable check name (e.g. "database", "redis")
            check_fn: Callable returning True (healthy) or False (unhealthy).
                      Must not raise exceptions -- catch internally.
        """
        with self._lock:
            self._checks[name] = check_fn
        logger.debug("Added health check: %s", name)

    def liveness(self) -> dict:
        """
        Liveness probe -- is the process alive?

        This should NEVER call external dependencies.  If this fails,
        the process should be restarted.

        Returns:
            {"status": "alive", "uptime_seconds": float}
        """
        uptime = time.time() - self._start_time
        return {
            "status": "alive",
            "uptime_seconds": round(uptime, 2),
        }

    def readiness(self) -> dict:
        """
        Readiness probe -- can the service serve traffic?

        Runs all registered health checks and reports aggregate status.

        Returns:
            {
                "status": "ready" | "not_ready",
                "checks": {"name": True/False, ...},
                "healthy_checks": int,
                "total_checks": int,
            }
        """
        check_results = self.run_checks()
        all_pass = all(check_results.values()) if check_results else True
        status = "ready" if all_pass else "not_ready"

        return {
            "status": status,
            "checks": check_results,
            "healthy_checks": sum(1 for v in check_results.values() if v),
            "total_checks": len(check_results),
        }

    def is_ready(self) -> bool:
        """
        Quick boolean readiness check.

        Returns:
            True if all checks pass, False otherwise.
        """
        check_results = self.run_checks()
        if not check_results:
            return True
        return all(check_results.values())

    def run_checks(self) -> dict[str, bool]:
        """
        Execute all registered health checks.

        Each check function is called with exception protection --
        if a check raises, it is treated as failed (False).

        Returns:
            Dict mapping check_name -> pass/fail boolean.
        """
        with self._lock:
            checks_snapshot = dict(self._checks)

        results: dict[str, bool] = {}
        for name, check_fn in checks_snapshot.items():
            try:
                results[name] = bool(check_fn())
            except Exception as exc:
                logger.warning("Health check '%s' raised exception: %s", name, exc)
                results[name] = False

        return results

    def overall_status(self) -> HealthStatus:
        """
        Compute the overall health status.

        Returns:
            HEALTHY   — all checks pass
            DEGRADED  — some checks pass (service partially functional)
            UNHEALTHY — no checks pass or critical failure
        """
        check_results = self.run_checks()
        if not check_results:
            return HealthStatus.HEALTHY

        passing = sum(1 for v in check_results.values() if v)
        total = len(check_results)

        if passing == total:
            return HealthStatus.HEALTHY
        elif passing > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNHEALTHY

    def stats(self) -> dict:
        """Return health check statistics."""
        check_results = self.run_checks()
        return {
            "uptime_seconds": round(time.time() - self._start_time, 2),
            "overall_status": self.overall_status().value,
            "checks": check_results,
            "total_checks": len(check_results),
            "passing_checks": sum(1 for v in check_results.values() if v),
        }
