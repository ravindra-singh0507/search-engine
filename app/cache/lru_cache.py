"""
LRU Cache

=== THEORY ===

An LRU (Least Recently Used) cache evicts the entry that was accessed
least recently when the cache is full.

IMPLEMENTATION: doubly-linked list + hash map

  - Hash map:  O(1) key lookup → node pointer
  - DLL:       O(1) move-to-front and evict-tail

The DLL head = most recently used.
The DLL tail = least recently used (next eviction candidate).

    HEAD ← [recent] ↔ [mid] ↔ [least] → TAIL

On get(key):
  1. Look up key in hash map.        O(1)
  2. Move node to HEAD.              O(1)
  3. Return value.

On put(key, value):
  1. If exists: update, move to HEAD. O(1)
  2. Else insert at HEAD.             O(1)
  3. If over capacity: remove TAIL.   O(1)

=== COMPLEXITY ===

  get:  O(1)    put: O(1)    evict: O(1)
  Space: O(capacity)

=== AT GOOGLE SCALE ===

Google's serving layer uses:
  - Memcache / Redis clusters for distributed caching
  - L1 in-process cache (this module's equivalent) per serving thread
  - Cache TTL + LRU hybrid (evict on both staleness AND cold access)
  - Negative caching (cache "no results" for extremely common zero-result queries)
"""

import threading
import time
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── LRU Cache (generic) ───────────────────────────────────────────────────────


class LRUCache:
    """
    Thread-safe LRU cache backed by OrderedDict.

    The OrderedDict is used in insertion order; move_to_end(key, last=False)
    puts the key at the front (most recent), and popitem(last=True) removes
    the back (least recent).  We flip the convention so last=True means MRU.
    """

    def __init__(self, capacity: int = 512, ttl_seconds: float | None = None):
        self._capacity   = max(1, capacity)
        self._ttl        = ttl_seconds
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock       = threading.Lock()
        self._hits       = 0
        self._misses     = 0

    def get(self, key: str) -> Optional[Any]:
        """Return cached value or None on miss / expiry."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            value, ts = self._cache[key]
            if self._ttl is not None and (time.time() - ts) > self._ttl:
                del self._cache[key]
                self._misses += 1
                return None
            # Move to end = most recently used
            self._cache.move_to_end(key, last=True)
            self._hits += 1
            return value

    def put(self, key: str, value: Any) -> None:
        """Insert or update a key."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key, last=True)
            self._cache[key] = (value, time.time())
            if len(self._cache) > self._capacity:
                # Evict least recently used (first item)
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug("LRU evicted: %s", evicted_key)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)

    def stats(self) -> dict:
        return {
            "capacity": self._capacity,
            "size": self.size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
            "ttl_seconds": self._ttl,
        }


# ── Query-specific cache ──────────────────────────────────────────────────────


class QueryCache:
    """
    Wraps LRUCache with a canonical key derived from the query string.
    Cache key = normalised lowercase query + top_k so different top_k
    values for the same query are cached separately.
    """

    def __init__(self, capacity: int = 512, ttl_seconds: float = 300.0):
        self._lru = LRUCache(capacity=capacity, ttl_seconds=ttl_seconds)

    def _key(self, query: str, top_k: int) -> str:
        return f"{query.lower().strip()}|{top_k}"

    def get(self, query: str, top_k: int) -> Optional[Any]:
        return self._lru.get(self._key(query, top_k))

    def put(self, query: str, top_k: int, results: Any) -> None:
        self._lru.put(self._key(query, top_k), results)

    def invalidate_all(self) -> None:
        self._lru.clear()

    def stats(self) -> dict:
        return self._lru.stats()
