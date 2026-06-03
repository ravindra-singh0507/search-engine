"""
Search Observability & Metrics

Provides a lightweight, Prometheus-compatible in-memory metrics registry.

=== METRICS COLLECTED ===

  search_requests_total       — counter
  search_latency_ms           — histogram (buckets: 10, 50, 100, 200, 500, 1000)
  index_operations_total      — counter
  index_latency_ms            — histogram
  crawl_pages_total           — counter
  cache_hits_total            — counter
  cache_misses_total          — counter
  ranking_latency_ms          — histogram
  slow_queries_total          — counter (queries > threshold_ms)

=== PROMETHEUS TEXT FORMAT ===

The /metrics endpoint returns text in the standard Prometheus exposition
format so it can be scraped directly by a Prometheus server:

  # HELP search_requests_total Total number of search requests
  # TYPE search_requests_total counter
  search_requests_total 1024

=== AT GOOGLE SCALE ===

Google uses Monarch (internal time-series DB), Streamz (streaming counters)
and Dapper (distributed tracing) for observability.  Prometheus + Grafana
is the standard open-source equivalent.
"""

import time
import threading
import logging
from collections import defaultdict
from dataclasses import dataclass, field

from app.config import ObservabilityConfig

logger = logging.getLogger(__name__)


# ── Histogram ──────────────────────────────────────────────────────────────


class Histogram:
    """
    Simple histogram over configurable buckets.
    Tracks count, sum, and per-bucket counts (Prometheus-style).
    """
    def __init__(self, name: str, buckets: tuple[float, ...] = (10, 50, 100, 200, 500, 1000)):
        self.name = name
        self.buckets = sorted(buckets)
        self._lock = threading.Lock()
        self._count = 0
        self._sum   = 0.0
        self._bucket_counts: dict[float, int] = {b: 0 for b in self.buckets}
        self._bucket_counts[float("inf")] = 0

    def observe(self, value: float) -> None:
        with self._lock:
            self._count += 1
            self._sum   += value
            for b in self.buckets:
                if value <= b:
                    self._bucket_counts[b] += 1
            self._bucket_counts[float("inf")] += 1

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> float:
        return self._sum / self._count if self._count else 0.0

    def to_prometheus(self) -> list[str]:
        with self._lock:
            lines = [
                f"# HELP {self.name} Histogram",
                f"# TYPE {self.name} histogram",
            ]
            for b, cnt in self._bucket_counts.items():
                label = '+Inf' if b == float('inf') else str(b)
                lines.append(f'{self.name}_bucket{{le="{label}"}} {cnt}')
            lines.append(f"{self.name}_sum {self._sum}")
            lines.append(f"{self.name}_count {self._count}")
            return lines

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "count": self._count,
                "sum":   round(self._sum, 3),
                "mean":  round(self.mean, 3),
                "buckets": {str(k): v for k, v in self._bucket_counts.items()},
            }


# ── Counter ────────────────────────────────────────────────────────────────


class Counter:
    def __init__(self, name: str):
        self.name = name
        self._value = 0
        self._lock  = threading.Lock()

    def inc(self, delta: int = 1) -> None:
        with self._lock:
            self._value += delta

    @property
    def value(self) -> int:
        return self._value

    def to_prometheus(self) -> list[str]:
        return [
            f"# HELP {self.name} Counter",
            f"# TYPE {self.name} counter",
            f"{self.name} {self._value}",
        ]


# ── MetricsCollector ───────────────────────────────────────────────────────


class MetricsCollector:
    """
    Central registry for all search engine metrics.
    One singleton is created in the application factory and shared
    across all components via dependency injection.
    """

    def __init__(self, config: ObservabilityConfig | None = None):
        self.config = config or ObservabilityConfig()

        self.search_requests      = Counter("search_requests_total")
        self.index_operations     = Counter("index_operations_total")
        self.crawl_pages          = Counter("crawl_pages_total")
        self.cache_hits           = Counter("cache_hits_total")
        self.cache_misses         = Counter("cache_misses_total")
        self.slow_queries         = Counter("slow_queries_total")

        # Phase 4 counters
        self.semantic_searches    = Counter("semantic_searches_total")
        self.hybrid_searches      = Counter("hybrid_searches_total")
        self.embedding_operations = Counter("embedding_operations_total")
        self.embedding_cache_hits = Counter("embedding_cache_hits_total")

        # Phase 5 counters
        self.pipeline_searches    = Counter("pipeline_searches_total")
        self.reranking_operations = Counter("reranking_operations_total")
        self.query_classifications = Counter("query_classifications_total")

        latency_buckets = (5, 10, 25, 50, 100, 200, 500, 1000, 2000)
        self.search_latency   = Histogram("search_latency_ms",   latency_buckets)
        self.index_latency    = Histogram("index_latency_ms",    latency_buckets)
        self.ranking_latency  = Histogram("ranking_latency_ms",  latency_buckets)
        self.crawl_latency    = Histogram("crawl_latency_ms",    (500, 1000, 2000, 5000))

        # Phase 4 histograms
        embed_buckets = (10, 50, 100, 200, 500, 1000, 2000)
        self.embedding_latency        = Histogram("embedding_latency_ms",        embed_buckets)
        self.semantic_search_latency  = Histogram("semantic_search_latency_ms",  latency_buckets)
        self.hybrid_search_latency    = Histogram("hybrid_search_latency_ms",    latency_buckets)
        self.vector_index_latency     = Histogram("vector_index_latency_ms",     embed_buckets)

        # Phase 5 histograms
        rerank_buckets = (50, 100, 250, 500, 1000, 2500, 5000)
        self.reranking_latency        = Histogram("reranking_latency_ms",        rerank_buckets)
        self.pipeline_latency         = Histogram("pipeline_latency_ms",         rerank_buckets)
        self.fusion_latency           = Histogram("fusion_latency_ms",           (1, 2, 5, 10, 25))
        self.query_understanding_latency = Histogram("query_understanding_latency_ms", (1, 2, 5, 10))

        self._start_time = time.time()

    # ── Record helpers ─────────────────────────────────────────────────────

    def record_search(self, latency_ms: float) -> None:
        self.search_requests.inc()
        self.search_latency.observe(latency_ms)
        if latency_ms > self.config.slow_query_threshold_ms:
            self.slow_queries.inc()
            logger.warning("Slow query: %.1f ms", latency_ms)

    def record_index(self, latency_ms: float) -> None:
        self.index_operations.inc()
        self.index_latency.observe(latency_ms)

    def record_ranking(self, latency_ms: float) -> None:
        self.ranking_latency.observe(latency_ms)

    def record_crawl_page(self, latency_ms: float) -> None:
        self.crawl_pages.inc()
        self.crawl_latency.observe(latency_ms)

    def record_cache_hit(self) -> None:
        self.cache_hits.inc()

    def record_cache_miss(self) -> None:
        self.cache_misses.inc()

    # Phase 4 record helpers
    def record_semantic_search(self, latency_ms: float) -> None:
        self.semantic_searches.inc()
        self.semantic_search_latency.observe(latency_ms)

    def record_hybrid_search(self, latency_ms: float) -> None:
        self.hybrid_searches.inc()
        self.hybrid_search_latency.observe(latency_ms)

    def record_embedding(self, latency_ms: float) -> None:
        self.embedding_operations.inc()
        self.embedding_latency.observe(latency_ms)

    def record_embedding_cache_hit(self) -> None:
        self.embedding_cache_hits.inc()

    def record_vector_index(self, latency_ms: float) -> None:
        self.vector_index_latency.observe(latency_ms)

    # Phase 5 record helpers
    def record_pipeline_search(self, latency_ms: float) -> None:
        self.pipeline_searches.inc()
        self.pipeline_latency.observe(latency_ms)

    def record_reranking(self, latency_ms: float) -> None:
        self.reranking_operations.inc()
        self.reranking_latency.observe(latency_ms)

    def record_query_classification(self, latency_ms: float) -> None:
        self.query_classifications.inc()
        self.query_understanding_latency.observe(latency_ms)

    def record_fusion(self, latency_ms: float) -> None:
        self.fusion_latency.observe(latency_ms)

    # ── Reporting ─────────────────────────────────────────────────────────

    def to_prometheus_text(self) -> str:
        """Generate Prometheus exposition format output."""
        lines: list[str] = []
        lines += [
            "# HELP engine_uptime_seconds Seconds since engine started",
            "# TYPE engine_uptime_seconds gauge",
            f"engine_uptime_seconds {round(time.time() - self._start_time, 1)}",
        ]
        for metric in [
            self.search_requests, self.index_operations,
            self.crawl_pages, self.cache_hits, self.cache_misses, self.slow_queries,
            # Phase 4
            self.semantic_searches, self.hybrid_searches,
            self.embedding_operations, self.embedding_cache_hits,
            # Phase 5
            self.pipeline_searches, self.reranking_operations, self.query_classifications,
        ]:
            lines += metric.to_prometheus()

        for hist in [
            self.search_latency, self.index_latency,
            self.ranking_latency, self.crawl_latency,
            # Phase 4
            self.embedding_latency, self.semantic_search_latency,
            self.hybrid_search_latency, self.vector_index_latency,
            # Phase 5
            self.reranking_latency, self.pipeline_latency,
            self.fusion_latency, self.query_understanding_latency,
        ]:
            lines += hist.to_prometheus()

        return "\n".join(lines) + "\n"

    def snapshot(self) -> dict:
        """Return a JSON-serialisable metrics snapshot."""
        total          = self.cache_hits.value + self.cache_misses.value
        cache_hit_rate = self.cache_hits.value / total if total else 0.0

        emb_total    = self.embedding_operations.value + self.embedding_cache_hits.value
        emb_hit_rate = self.embedding_cache_hits.value / emb_total if emb_total else 0.0

        return {
            "uptime_seconds":           round(time.time() - self._start_time, 1),
            "search_requests_total":    self.search_requests.value,
            "index_operations_total":   self.index_operations.value,
            "crawl_pages_total":        self.crawl_pages.value,
            "slow_queries_total":       self.slow_queries.value,
            "cache_hit_rate":           round(cache_hit_rate, 4),
            # Phase 4
            "semantic_searches_total":  self.semantic_searches.value,
            "hybrid_searches_total":    self.hybrid_searches.value,
            "embedding_operations":     self.embedding_operations.value,
            "embedding_cache_hit_rate": round(emb_hit_rate, 4),
            # Phase 5
            "pipeline_searches_total":  self.pipeline_searches.value,
            "reranking_operations":     self.reranking_operations.value,
            "query_classifications":    self.query_classifications.value,
            "search_latency":           self.search_latency.snapshot(),
            "index_latency":            self.index_latency.snapshot(),
            "ranking_latency":          self.ranking_latency.snapshot(),
            "embedding_latency":        self.embedding_latency.snapshot(),
            "semantic_search_latency":  self.semantic_search_latency.snapshot(),
            "hybrid_search_latency":    self.hybrid_search_latency.snapshot(),
            "reranking_latency":        self.reranking_latency.snapshot(),
            "pipeline_latency":         self.pipeline_latency.snapshot(),
        }
