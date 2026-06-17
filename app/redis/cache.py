"""
Redis-Backed Cache

=== THEORY ===

A cache is a fast-access storage layer that sits between the application and
a slower backend (database, search index, LLM API).  The goal is to avoid
redundant computation by storing previously computed results.

Redis caches operate at the L2 level in a typical cache hierarchy:

  L1: In-process LRU cache (app/cache/lru_cache.py)
      - Sub-microsecond access, per-process, limited by heap
  L2: Redis cache (this module)
      - Sub-millisecond access, shared across processes/nodes, limited by RAM
  L3: Database / search index
      - Millisecond-to-second access, persistent

=== SERIALIZATION ===

Cache values are serialized to JSON via json.dumps/json.loads.  This limits
cached values to JSON-serializable types (dicts, lists, strings, numbers,
booleans, None).  For binary data or complex objects, a msgpack or pickle
serializer could be substituted.

=== CACHE STATISTICS ===

We track hits (key found in cache) and misses (key not found or expired)
to compute hit rate.  A healthy cache should maintain >80% hit rate for
repeated query workloads.  Low hit rates indicate:
  - TTL too short (values expire before reuse)
  - Key space too large (unique queries dominate)
  - Cache too small (evictions outpace inserts)

=== QUERY CACHE ===

RedisQueryCache wraps RedisCache with the same interface as
app/cache/lru_cache.py QueryCache: normalised lowercase key + top_k so
different top_k values for the same query are cached separately.

=== COMPLEXITY ===

  get:        O(1)  -- Redis GET + json.loads
  put:        O(1)  -- Redis SET + json.dumps
  invalidate: O(1)  -- Redis DEL
  clear:      O(N)  -- Redis KEYS + DEL (N = matching keys)

=== AT PRODUCTION SCALE ===

  Google:   Memcache + Bigtable serving layer cache
  Netflix:  EVCache (Redis-backed, multi-region)
  Uber:     Schemaless caching + Redis query cache
  Twitter:  Redis + Memcache hybrid (Manhattan for heavy reads)
"""

import json
import logging
import threading
from typing import Any

from app.redis.client import RedisClient

logger = logging.getLogger(__name__)


# -- Redis-backed generic cache -----------------------------------------------

class RedisCache:
    """
    Redis-backed cache with JSON serialization and TTL.

    All values are stored as JSON strings in Redis.  Keys are prefixed to
    avoid collisions with other Redis users in the same database.

    Thread safety: individual Redis operations are atomic; hit/miss counters
    are protected by a local lock.
    """

    def __init__(self, client: RedisClient, prefix: str = "cache:", ttl: int = 300):
        self._client = client
        self._prefix = prefix
        self._ttl = ttl
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _full_key(self, key: str) -> str:
        """Build the Redis key with namespace prefix."""
        return f"{self._prefix}{key}"

    def get(self, key: str) -> Any | None:
        """
        Retrieve a cached value.

        Returns None on miss (key not found or expired).
        Deserialization failures are treated as misses.
        """
        raw = self._client.get(self._full_key(key))
        if raw is None:
            with self._lock:
                self._misses += 1
            return None
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            with self._lock:
                self._misses += 1
            return None
        with self._lock:
            self._hits += 1
        return value

    def put(self, key: str, value: Any) -> None:
        """
        Store a value in cache with TTL.

        The value must be JSON-serializable.  Non-serializable values are
        silently dropped with a warning log.
        """
        try:
            serialized = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            logger.warning("Cache put failed (not serializable): %s — %s", key, exc)
            return
        self._client.set(self._full_key(key), serialized, ex=self._ttl)

    def invalidate(self, key: str) -> None:
        """Remove a specific key from cache."""
        self._client.delete(self._full_key(key))

    def clear(self) -> None:
        """Remove all keys matching this cache's prefix."""
        pattern = f"{self._prefix}*"
        matching_keys = self._client.keys(pattern)
        for k in matching_keys:
            self._client.delete(k)
        with self._lock:
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict:
        """Return cache performance statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            # Count current keys matching prefix
            size = len(self._client.keys(f"{self._prefix}*"))
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
                "size": size,
                "prefix": self._prefix,
                "ttl": self._ttl,
            }


# -- Query-specific Redis cache -----------------------------------------------

class RedisQueryCache:
    """
    Query-specific cache backed by Redis.

    Same interface as app/cache/lru_cache.py QueryCache so the two can
    be used interchangeably.  The canonical key is:
        {prefix}{query_normalised}|{top_k}

    This ensures different top_k values for the same query are cached
    separately, and queries are case/whitespace-normalised.
    """

    def __init__(self, client: RedisClient, prefix: str = "qcache:", ttl: int = 300):
        self._cache = RedisCache(client=client, prefix=prefix, ttl=ttl)

    def _key(self, query: str, top_k: int) -> str:
        """Normalised cache key: lowercase, stripped, with top_k suffix."""
        return f"{query.lower().strip()}|{top_k}"

    def get(self, query: str, top_k: int) -> Any | None:
        """Retrieve cached query results or None on miss."""
        return self._cache.get(self._key(query, top_k))

    def put(self, query: str, top_k: int, results: Any) -> None:
        """Store query results in cache."""
        self._cache.put(self._key(query, top_k), results)

    def invalidate_all(self) -> None:
        """Clear all cached query results."""
        self._cache.clear()

    def stats(self) -> dict:
        """Return cache performance statistics."""
        return self._cache.stats()
