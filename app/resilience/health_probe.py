"""
Health Probes — Phase 8 Batch 4

=== THEORY ===

Kubernetes and load balancers distinguish two probe types:
  - Liveness:   is the process alive? (restart if fails)
  - Readiness:  can it serve traffic? (remove from LB pool if fails)

Health probes actively test downstream dependencies within a timeout.
A service is ready only when ALL required probes pass.

=== PRODUCTION EQUIVALENTS ===

  Kubernetes:  livenessProbe, readinessProbe in Pod spec
  AWS:         Target Group health checks
  Consul:      service health checks
  Nginx:       upstream health checks
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from app.config import ResilienceConfig

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Result of a single health probe execution."""
    probe_name:  str
    healthy:     bool
    latency_ms:  float
    message:     str  = ""
    checked_at:  float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "probe":      self.probe_name,
            "healthy":    self.healthy,
            "latency_ms": round(self.latency_ms, 2),
            "message":    self.message,
            "checked_at": self.checked_at,
        }


class HealthProbe:
    """
    Actively probes downstream dependencies and reports readiness.

    Each registered probe is a Callable[[], bool] that returns True if
    healthy.  Probes run with an individual timeout; a probe that hangs
    or raises is marked unhealthy.

    === COMPLEXITY ===
      probe_all(): O(P) probes, each with its own timeout thread.
    """

    def __init__(self, config: ResilienceConfig):
        self._config = config
        self._probes: Dict[str, tuple] = {}   # name -> (fn, timeout_sec)
        self._lock = threading.Lock()
        self._results: Dict[str, ProbeResult] = {}
        self._probe_count = 0

    def add_probe(
        self,
        name: str,
        probe_fn: Callable[[], bool],
        timeout_sec: float = 5.0,
    ) -> None:
        """Register a named health probe."""
        with self._lock:
            self._probes[name] = (probe_fn, timeout_sec)
        logger.debug("Registered health probe: %s (timeout=%.1fs)", name, timeout_sec)

    def probe(self, name: str) -> ProbeResult:
        """Run a single named probe and return its result."""
        with self._lock:
            entry = self._probes.get(name)
        if entry is None:
            return ProbeResult(probe_name=name, healthy=False,
                               latency_ms=0.0, message="Probe not registered")

        probe_fn, timeout_sec = entry
        t0 = time.perf_counter()
        result_holder: list = [None]
        error_holder:  list = [None]

        def _run() -> None:
            try:
                result_holder[0] = probe_fn()
            except Exception as exc:
                error_holder[0] = str(exc)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=timeout_sec)

        latency_ms = (time.perf_counter() - t0) * 1000
        self._probe_count += 1

        if t.is_alive():
            result = ProbeResult(probe_name=name, healthy=False,
                                 latency_ms=latency_ms,
                                 message=f"Probe timed out after {timeout_sec}s")
        elif error_holder[0]:
            result = ProbeResult(probe_name=name, healthy=False,
                                 latency_ms=latency_ms,
                                 message=f"Probe raised: {error_holder[0]}")
        else:
            healthy = bool(result_holder[0])
            result = ProbeResult(probe_name=name, healthy=healthy,
                                 latency_ms=latency_ms,
                                 message="ok" if healthy else "Probe returned False")

        with self._lock:
            self._results[name] = result
        return result

    def probe_all(self) -> List[ProbeResult]:
        """Run all registered probes concurrently and return results."""
        with self._lock:
            names = list(self._probes.keys())
        return [self.probe(name) for name in names]

    def is_healthy(self) -> bool:
        """Return True if all probes pass."""
        results = self.probe_all()
        return all(r.healthy for r in results)

    def get_last_results(self) -> Dict[str, ProbeResult]:
        with self._lock:
            return dict(self._results)

    def stats(self) -> dict:
        with self._lock:
            probes = list(self._probes.keys())
            last = {n: r.to_dict() for n, r in self._results.items()}
        return {
            "registered_probes": probes,
            "probe_count": self._probe_count,
            "last_results": last,
        }
