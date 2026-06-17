"""
Retrieval Gateway Service

=== THEORY ===

The Gateway pattern (also called API Gateway or Edge Service) centralises
cross-cutting concerns into a single entry point.  Clients interact with
one service instead of knowing about every retrieval backend.

Cross-cutting concerns handled here:

  1. QUERY ROUTING — Selects the best retrieval backend (BM25, semantic,
     hybrid, pipeline) based on query analysis.

  2. RESULT CACHING — Two-tier cache (L1 in-process + L2 Redis) avoids
     redundant retrieval for repeated queries.

  3. RATE LIMITING — Per-client request throttling using a sliding window
     counter.  Prevents abuse and ensures fair resource allocation.

  4. FUSION STRATEGY SELECTION — Different queries may benefit from
     different fusion strategies; the gateway selects based on config.

  5. TIMEOUT MANAGEMENT — Per-request timeouts prevent slow backends
     from blocking the gateway indefinitely.

  6. METRICS COLLECTION — Latency, throughput, cache hit rate, and
     error counts for observability.

=== ARCHITECTURE ===

  Client
    |
    v
  RetrievalGateway
    ├── GatewayCache (L1 + L2)
    ├── QueryRouter (intent classification)
    ├── Rate limiter (per client_id)
    |
    ├── SearchService       (BM25)
    ├── SemanticSearchService (FAISS/Qdrant)
    ├── HybridSearchService (BM25 + semantic + fusion)
    └── RetrievalPipeline   (full multi-stage)

=== PRODUCTION EQUIVALENTS ===

  Google:     GFE (Google Front End) -> mix servers -> index serving
  Elastic:    Coordinating node in Elasticsearch cluster
  Netflix:    Zuul API gateway for microservice routing
  Vespa:      Container / search chain with multiple rank profiles
  Uber:       Edge gateway with service mesh routing

=== COMPLEXITY ===

  search (cache hit):    O(1) cache lookup
  search (cache miss):   O(backend) + O(1) cache write
  invalidate_cache:      O(N) key scan
  stats:                 O(1)

=== RATE LIMITING ===

Uses a fixed-window counter per client_id.  Each minute window allows
config.rate_limit_rpm requests.  When exceeded, the gateway returns an
empty response with a rate_limited flag in metadata.

Production systems use:
  - Sliding window log (more accurate)
  - Token bucket (burst-friendly)
  - Distributed rate limiting via Redis (INCR + EXPIRE)
"""

import logging
import threading
import time
from dataclasses import asdict

from app.config import GatewayConfig
from app.gateway.cache import GatewayCache
from app.gateway.models import GatewayRequest, GatewayResponse
from app.gateway.router import QueryRouter

logger = logging.getLogger(__name__)


class RetrievalGateway:
    """
    Central retrieval orchestration service.

    Sits between clients and retrieval backends.  All retrieval requests
    flow through this gateway, which applies caching, rate limiting,
    routing, and timeout management before dispatching to the appropriate
    backend.

    All backend services are optional — if a backend is None, queries
    routed to it fall back to the next available backend.
    """

    def __init__(
        self,
        config:           GatewayConfig | None = None,
        search_service=None,
        semantic_service=None,
        hybrid_service=None,
        pipeline=None,
        redis_client=None,
        metrics=None,
        classifier=None,
    ):
        """
        Parameters
        ----------
        config           : GatewayConfig with cache, rate limit, timeout settings
        search_service   : SearchService (BM25)
        semantic_service : SemanticSearchService
        hybrid_service   : HybridSearchService
        pipeline         : RetrievalPipeline
        redis_client     : RedisClient for L2 cache (None = L1 only)
        metrics          : MetricsCollector for observability
        classifier       : QueryClassifier for intent-based routing
        """
        self._config = config or GatewayConfig()

        # Backend services
        self._search   = search_service
        self._semantic = semantic_service
        self._hybrid   = hybrid_service
        self._pipeline = pipeline

        # Infrastructure
        self._router = QueryRouter(classifier=classifier)
        self._cache = GatewayCache(
            redis_client=redis_client,
            l1_capacity=self._config.cache_max_size,
            l2_ttl=self._config.cache_ttl,
        )
        self._metrics = metrics

        # Rate limiting: {client_id: (window_start, count)}
        self._rate_limits: dict[str, tuple[float, int]] = {}
        self._rate_lock = threading.Lock()

        # Gateway-level stats
        self._lock = threading.Lock()
        self._total_requests = 0
        self._total_errors = 0
        self._total_latency_ms = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        self._rate_limited = 0

    # ── Public API ───────────────────────────────────────────────────────

    def search(
        self,
        query:     str,
        mode:      str  = "hybrid",
        top_k:     int  = 10,
        fusion:    str  = "rrf",
        rerank:    bool = True,
        client_id: str  = "",
    ) -> GatewayResponse:
        """
        Execute a retrieval request through the gateway.

        Parameters
        ----------
        query     : raw search query
        mode      : "bm25", "semantic", "hybrid", "pipeline", or "auto"
                    "auto" uses the QueryRouter to select the best mode
        top_k     : number of results to return
        fusion    : fusion strategy (rrf, combsum, combmnz, weighted, borda)
        rerank    : whether to apply reranking (pipeline mode only)
        client_id : caller identifier for rate limiting

        Returns
        -------
        GatewayResponse with results, latency, cache status, and metadata.
        """
        start = time.perf_counter()

        with self._lock:
            self._total_requests += 1

        # ── Rate limiting ─────────────────────────────────────────────
        if client_id and self._is_rate_limited(client_id):
            with self._lock:
                self._rate_limited += 1
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("Rate limited: client=%s, query=%r", client_id, query)
            return GatewayResponse(
                query=query, mode=mode, results=[], total_results=0,
                latency_ms=round(elapsed, 2), cache_hit=False,
                metadata={"rate_limited": True, "client_id": client_id},
            )

        # ── Route query ───────────────────────────────────────────────
        effective_mode = mode
        if mode == "auto":
            effective_mode = self._router.route(query)

        # ── Cache check ───────────────────────────────────────────────
        if self._config.enable_cache:
            cache_key = GatewayCache.make_key(query, effective_mode, top_k, fusion)
            cached = self._cache.get(cache_key)
            if cached is not None:
                with self._lock:
                    self._cache_hits += 1
                elapsed = (time.perf_counter() - start) * 1000
                if self._metrics:
                    self._metrics.record_cache_hit()
                return GatewayResponse(
                    query=query, mode=effective_mode,
                    results=cached.get("results", []),
                    total_results=cached.get("total_results", 0),
                    latency_ms=round(elapsed, 2),
                    cache_hit=True,
                    fusion_strategy=cached.get("fusion_strategy", ""),
                    reranked=cached.get("reranked", False),
                    metadata={"source": "cache"},
                )

        with self._lock:
            self._cache_misses += 1
        if self._metrics:
            self._metrics.record_cache_miss()

        # ── Execute retrieval ─────────────────────────────────────────
        try:
            results, result_meta = self._dispatch(
                query, effective_mode, top_k, fusion, rerank,
            )
        except Exception as exc:
            with self._lock:
                self._total_errors += 1
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(
                "Gateway retrieval error: mode=%s, query=%r, error=%s",
                effective_mode, query, exc,
            )
            return GatewayResponse(
                query=query, mode=effective_mode,
                results=[], total_results=0,
                latency_ms=round(elapsed, 2),
                metadata={"error": str(exc)},
            )

        elapsed = (time.perf_counter() - start) * 1000

        with self._lock:
            self._total_latency_ms += elapsed

        # ── Cache write ───────────────────────────────────────────────
        if self._config.enable_cache and results:
            cache_value = {
                "results": results,
                "total_results": len(results),
                "fusion_strategy": result_meta.get("fusion_strategy", ""),
                "reranked": result_meta.get("reranked", False),
            }
            self._cache.put(cache_key, cache_value)

        response = GatewayResponse(
            query=query,
            mode=effective_mode,
            results=results,
            total_results=len(results),
            latency_ms=round(elapsed, 2),
            cache_hit=False,
            fusion_strategy=result_meta.get("fusion_strategy", ""),
            reranked=result_meta.get("reranked", False),
            metadata=result_meta,
        )

        logger.info(
            "Gateway search: mode=%s, query=%r, results=%d, latency=%.1f ms",
            effective_mode, query, len(results), elapsed,
        )

        if self._metrics:
            self._metrics.record_search(elapsed)

        return response

    def invalidate_cache(self, pattern: str = "*") -> int:
        """
        Invalidate cached results matching a pattern.

        Parameters
        ----------
        pattern : glob-style pattern for cache key matching.
                  "*" invalidates all entries.

        Returns
        -------
        Number of entries invalidated.
        """
        count = self._cache.invalidate(pattern)
        logger.info("Gateway cache invalidated: pattern=%r, count=%d", pattern, count)
        return count

    def stats(self) -> dict:
        """
        Return gateway operational statistics.

        Includes request counts, latency averages, cache stats,
        rate limiting stats, and per-tier cache metrics.
        """
        with self._lock:
            avg_latency = (
                self._total_latency_ms / self._total_requests
                if self._total_requests > 0 else 0.0
            )
            total_cache = self._cache_hits + self._cache_misses
            cache_hit_rate = (
                self._cache_hits / total_cache if total_cache > 0 else 0.0
            )

            return {
                "total_requests":  self._total_requests,
                "total_errors":    self._total_errors,
                "avg_latency_ms":  round(avg_latency, 2),
                "cache_hits":      self._cache_hits,
                "cache_misses":    self._cache_misses,
                "cache_hit_rate":  round(cache_hit_rate, 4),
                "rate_limited":    self._rate_limited,
                "cache_stats":     self._cache.stats(),
                "config": {
                    "cache_ttl":       self._config.cache_ttl,
                    "rate_limit_rpm":  self._config.rate_limit_rpm,
                    "timeout_sec":     self._config.timeout_sec,
                    "default_fusion":  self._config.default_fusion,
                    "default_rerank":  self._config.default_rerank,
                },
            }

    # ── Retrieval dispatch ───────────────────────────────────────────────

    def _dispatch(
        self,
        query:  str,
        mode:   str,
        top_k:  int,
        fusion: str,
        rerank: bool,
    ) -> tuple[list[dict], dict]:
        """
        Dispatch a query to the appropriate backend.

        Returns (results_list, metadata_dict).
        Falls back to available backends when the requested one is None.
        """
        meta: dict = {"fusion_strategy": fusion, "reranked": False}

        if mode == "bm25":
            return self._search_bm25(query, top_k, meta)
        elif mode == "semantic":
            return self._search_semantic(query, top_k, meta)
        elif mode == "pipeline":
            return self._search_pipeline(query, top_k, fusion, rerank, meta)
        else:  # hybrid (default)
            return self._search_hybrid(query, top_k, meta)

    def _search_bm25(
        self, query: str, top_k: int, meta: dict
    ) -> tuple[list[dict], dict]:
        """Execute BM25 keyword search."""
        if self._search is None:
            # Fall back to hybrid if available
            if self._hybrid is not None:
                return self._search_hybrid(query, top_k, meta)
            return [], meta

        result = self._search.search(query, top_k=top_k)
        results = [
            {
                "doc_id":  r.doc_id,
                "title":   r.title,
                "snippet": r.snippet,
                "score":   round(r.score, 6),
            }
            for r in result.results
        ]
        meta["bm25_matches"] = result.total_matches
        meta["search_time_ms"] = result.search_time_ms
        return results, meta

    def _search_semantic(
        self, query: str, top_k: int, meta: dict
    ) -> tuple[list[dict], dict]:
        """Execute semantic (vector) search."""
        if self._semantic is None:
            # Fall back to hybrid if available
            if self._hybrid is not None:
                return self._search_hybrid(query, top_k, meta)
            return [], meta

        result = self._semantic.search(query, top_k=top_k)
        results = [
            {
                "doc_id":         r.doc_id,
                "title":          r.title,
                "snippet":        r.snippet,
                "score":          round(r.semantic_score, 6),
                "chunk_id":       r.chunk_id,
            }
            for r in result.results
        ]
        meta["model_name"] = result.model_name
        meta["search_time_ms"] = result.search_time_ms
        return results, meta

    def _search_hybrid(
        self, query: str, top_k: int, meta: dict
    ) -> tuple[list[dict], dict]:
        """Execute hybrid (BM25 + semantic) search."""
        if self._hybrid is None:
            # Fall back to BM25 if available
            if self._search is not None:
                return self._search_bm25(query, top_k, meta)
            if self._semantic is not None:
                return self._search_semantic(query, top_k, meta)
            return [], meta

        result = self._hybrid.search(query, top_k=top_k)
        results = [
            {
                "doc_id":          r.doc_id,
                "title":           r.title,
                "snippet":         r.snippet,
                "score":           round(r.fusion_score, 6),
                "bm25_score":      round(r.bm25_score, 6),
                "semantic_score":  round(r.semantic_score, 6),
                "bm25_rank":       r.bm25_rank,
                "semantic_rank":   r.semantic_rank,
            }
            for r in result.results
        ]
        meta["fusion_strategy"] = result.fusion_strategy
        meta["bm25_count"] = result.bm25_count
        meta["semantic_count"] = result.semantic_count
        meta["search_time_ms"] = result.search_time_ms
        return results, meta

    def _search_pipeline(
        self,
        query:  str,
        top_k:  int,
        fusion: str,
        rerank: bool,
        meta:   dict,
    ) -> tuple[list[dict], dict]:
        """Execute the full multi-stage retrieval pipeline."""
        if self._pipeline is None:
            # Fall back to hybrid
            return self._search_hybrid(query, top_k, meta)

        override = {
            "fusion_strategy": fusion,
            "use_reranker": rerank,
        }
        result = self._pipeline.search(query, top_k=top_k, override=override)
        results = [
            {
                "doc_id":          c.doc_id,
                "title":           c.title,
                "snippet":         c.snippet,
                "score":           round(c.final_score, 6),
                "bm25_score":      round(c.bm25_score, 6),
                "semantic_score":  round(c.semantic_score, 6),
                "fusion_score":    round(c.fusion_score, 6),
                "reranker_score":  round(c.reranker_score, 6),
                "final_rank":      c.final_rank,
            }
            for c in result.results
        ]
        meta["fusion_strategy"] = fusion
        meta["reranked"] = result.reranked_count > 0
        meta["stage_latencies"] = result.stage_latencies
        meta["total_latency_ms"] = result.total_latency_ms
        meta["retrieval_count"] = result.retrieval_count
        meta["reranked_count"] = result.reranked_count
        return results, meta

    # ── Rate limiting ────────────────────────────────────────────────────

    def _is_rate_limited(self, client_id: str) -> bool:
        """
        Check whether a client has exceeded the rate limit.

        Uses a fixed-window counter: each 60-second window allows
        config.rate_limit_rpm requests.  The window resets when the
        minute boundary passes.
        """
        now = time.time()
        window = int(now / 60)  # minute-granularity window

        with self._rate_lock:
            entry = self._rate_limits.get(client_id)
            if entry is None or entry[0] != window:
                # New window
                self._rate_limits[client_id] = (window, 1)
                return False

            current_count = entry[1]
            if current_count >= self._config.rate_limit_rpm:
                return True

            self._rate_limits[client_id] = (window, current_count + 1)
            return False
