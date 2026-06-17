"""
Gateway Cache — Two-Tier Caching for Retrieval Results

=== THEORY ===

Multi-tier caching exploits the trade-off between access speed, capacity,
and sharing scope:

  L1: In-process LRU cache
    - Access: ~100 ns (no I/O, no serialization)
    - Scope:  per-process (not shared across instances)
    - Size:   limited by process heap (hundreds to low thousands of entries)
    - Eviction: LRU (Least Recently Used)

  L2: Redis cache
    - Access: ~0.5 ms (network + serialization)
    - Scope:  shared across all gateway instances
    - Size:   limited by Redis server RAM (millions of entries)
    - Eviction: TTL-based + Redis maxmemory policy

Read path:
  1. Check L1 (fast, local)  -> hit? return immediately
  2. Check L2 (shared)       -> hit? promote to L1, return
  3. Miss                    -> caller executes backend, then put() into L1+L2

Write path:
  put(key, value) -> write to L1 AND L2

This mirrors how CPU caches work (L1/L2/L3) and how CDNs work
(edge cache / origin cache / origin server).

=== CACHE KEY DESIGN ===

Keys are deterministic hashes of (query, mode, top_k, fusion) so that:
  - Same query with different top_k = different cache entries
  - Same query with different fusion = different cache entries
  - Case-insensitive (queries are lowercased before hashing)

=== PRODUCTION EQUIVALENTS ===

  Google:     In-memory result cache + Bigtable serving cache
  Netflix:    Local Guava cache + EVCache (Redis-backed)
  Uber:       LRU + Redis with consistent hashing
  OpenSearch: Node query cache + shard request cache

=== COMPLEXITY ===

  get: O(1) L1 lookup + O(1) L2 lookup (amortised)
  put: O(1) L1 insert + O(1) L2 insert
  invalidate: O(N) scan matching keys (N = cache size)
"""

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


class GatewayCache:
    """
    Two-tier cache (L1 in-process LRU + L2 Redis) for retrieval results.

    Thread safety: L1 operations are guarded by a threading.Lock.
    L2 (Redis) operations are atomic by nature of the Redis protocol.
    """

    def __init__(
        self,
        redis_client=None,
        l1_capacity: int = 256,
        l2_ttl: int = 300,
    ):
        """
        Parameters
        ----------
        redis_client : RedisClient or None.  If None, L2 is disabled and
                       only the in-process L1 cache is used.
        l1_capacity  : maximum entries in the L1 LRU cache.
        l2_ttl       : TTL in seconds for L2 (Redis) entries.
        """
        self._redis = redis_client
        self._l1_capacity = l1_capacity
        self._l2_ttl = l2_ttl
        self._l2_prefix = "gw:cache:"

        # L1: OrderedDict used as LRU — most recent access moves to end
        self._l1: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

        # Stats
        self._l1_hits = 0
        self._l1_misses = 0
        self._l2_hits = 0
        self._l2_misses = 0

    # ── Public API ───────────────────────────────────────────────────────

    def get(self, cache_key: str) -> Any | None:
        """
        Look up a cached value.

        Checks L1 first, then L2.  On L2 hit, promotes the value to L1.
        Returns None on miss.
        """
        # L1 check
        with self._lock:
            if cache_key in self._l1:
                self._l1.move_to_end(cache_key)
                self._l1_hits += 1
                return self._l1[cache_key]
            self._l1_misses += 1

        # L2 check
        if self._redis is not None:
            try:
                raw = self._redis.get(f"{self._l2_prefix}{cache_key}")
                if raw is not None:
                    value = json.loads(raw)
                    # Promote to L1
                    with self._lock:
                        self._l1[cache_key] = value
                        self._evict_l1_if_needed()
                        self._l2_hits += 1
                    return value
                with self._lock:
                    self._l2_misses += 1
            except Exception as exc:
                logger.debug("L2 cache get failed: %s", exc)
                with self._lock:
                    self._l2_misses += 1

        return None

    def put(self, cache_key: str, value: Any) -> None:
        """
        Store a value in both L1 and L2.

        The value must be JSON-serializable for L2 storage.
        Non-serializable values are stored in L1 only.
        """
        # L1 write
        with self._lock:
            self._l1[cache_key] = value
            self._l1.move_to_end(cache_key)
            self._evict_l1_if_needed()

        # L2 write
        if self._redis is not None:
            try:
                serialized = json.dumps(value, ensure_ascii=False)
                self._redis.set(
                    f"{self._l2_prefix}{cache_key}",
                    serialized,
                    ex=self._l2_ttl,
                )
            except (TypeError, ValueError) as exc:
                logger.debug("L2 cache put skipped (not serializable): %s", exc)
            except Exception as exc:
                logger.debug("L2 cache put failed: %s", exc)

    def invalidate(self, pattern: str = "*") -> int:
        """
        Invalidate cache entries matching a pattern.

        If pattern is "*", clears all entries in both tiers.
        Otherwise, uses fnmatch-style glob matching on keys.

        Returns the number of entries invalidated.
        """
        import fnmatch
        count = 0

        # L1 invalidation
        with self._lock:
            if pattern == "*":
                count += len(self._l1)
                self._l1.clear()
            else:
                to_delete = [
                    k for k in self._l1
                    if fnmatch.fnmatch(k, pattern)
                ]
                for k in to_delete:
                    del self._l1[k]
                count += len(to_delete)

        # L2 invalidation
        if self._redis is not None:
            try:
                redis_pattern = f"{self._l2_prefix}{pattern}"
                matching = self._redis.keys(redis_pattern)
                for k in matching:
                    self._redis.delete(k)
                count += len(matching)
            except Exception as exc:
                logger.debug("L2 cache invalidate failed: %s", exc)

        return count

    def stats(self) -> dict:
        """Return cache performance statistics for both tiers."""
        with self._lock:
            l1_total = self._l1_hits + self._l1_misses
            l2_total = self._l2_hits + self._l2_misses
            return {
                "l1_size":     len(self._l1),
                "l1_capacity": self._l1_capacity,
                "l1_hits":     self._l1_hits,
                "l1_misses":   self._l1_misses,
                "l1_hit_rate": round(self._l1_hits / l1_total, 4) if l1_total else 0.0,
                "l2_hits":     self._l2_hits,
                "l2_misses":   self._l2_misses,
                "l2_hit_rate": round(self._l2_hits / l2_total, 4) if l2_total else 0.0,
                "l2_ttl":      self._l2_ttl,
                "l2_enabled":  self._redis is not None,
            }

    # ── Static helpers ───────────────────────────────────────────────────

    @staticmethod
    def make_key(query: str, mode: str, top_k: int, fusion: str) -> str:
        """
        Build a deterministic cache key from query parameters.

        Uses SHA-256 of the normalised query + parameters to produce a
        fixed-length key that is safe for use as a Redis key.
        """
        normalised = f"{query.lower().strip()}|{mode}|{top_k}|{fusion}"
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:32]

    # ── Internal helpers ─────────────────────────────────────────────────

    def _evict_l1_if_needed(self) -> None:
        """Evict oldest entries if L1 exceeds capacity.  Must hold _lock."""
        while len(self._l1) > self._l1_capacity:
            self._l1.popitem(last=False)  # remove oldest (LRU)
