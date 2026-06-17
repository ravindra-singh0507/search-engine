"""
Redis Client Abstraction Layer

=== THEORY ===

Redis (Remote Dictionary Server) is an in-memory key-value data structure store
that supports strings, hashes, lists, sets, sorted sets, bitmaps, HyperLogLogs,
and streams.  All operations are atomic and most run in O(1) or O(log N) time.

Redis's single-threaded event loop processes commands sequentially, which
eliminates data races without locking.  For clients, however, thread safety
is still needed because multiple application threads may share a connection
pool.  The official redis-py client handles this internally with connection
pooling, but our in-memory fallback must use explicit locks.

=== ARCHITECTURE ===

We define a Protocol (structural subtyping) so that any object implementing
the required methods can be used as a Redis client.  Two implementations:

  RealRedisClient   -- wraps the redis-py library; lazy connection on first
                       operation; raises ImportError if redis-py is not
                       installed.

  InMemoryRedisClient -- complete in-memory implementation using Python dicts
                         and threading.Lock.  Supports ALL Redis data types
                         used by our application: strings, hashes, lists,
                         sets, and sorted sets.  TTL is tracked per key and
                         expired lazily on access.

=== DATA STRUCTURES (in-memory) ===

  _store: dict[str, Any]
    str keys    -> str values     (GET/SET)
    hash keys   -> dict[str,str]  (HGET/HSET/HGETALL)
    list keys   -> list[str]      (LPUSH/RPOP/LRANGE)
    set keys    -> set[str]       (SADD/SISMEMBER/SMEMBERS)
    zset keys   -> list[tuple[str,float]]  (ZADD/ZRANGEBYSCORE)

  _expiry: dict[str, float]
    key -> absolute timestamp when the key expires

=== COMPLEXITY ===

  RealRedisClient:     All operations O(1) amortised (network RTT ~0.1ms local)
  InMemoryRedisClient: All operations O(1) amortised (no network)
    keys(pattern):     O(N) scan -- same as Redis KEYS (avoid in production)
    zrangebyscore:     O(N) scan -- real Redis uses skip-lists for O(log N + M)

=== GRACEFUL FALLBACK ===

  try:
      client = RealRedisClient(config)
  except (ImportError, ConnectionError):
      client = InMemoryRedisClient()

=== AT PRODUCTION SCALE ===

  Google:    Memorystore for Redis (managed, 300 GB instances, 99.9% SLA)
  Netflix:   EVCache -- Redis-compatible, multi-region replication
  Uber:      Redis Cluster with custom sharding + client-side consistent hashing
  Twitter:   Twemproxy (nutcracker) in front of Redis shards
  OpenSearch: Redis for query/result caching layer
"""

import fnmatch
import logging
import threading
import time
from typing import Any, Protocol, runtime_checkable

from app.config import RedisConfig

logger = logging.getLogger(__name__)


# -- Protocol -----------------------------------------------------------------

@runtime_checkable
class RedisClient(Protocol):
    """
    Structural interface for a Redis-compatible key-value client.

    Covers the subset of Redis commands used across the search engine
    infrastructure: strings, hashes, lists, sets, sorted sets, and key
    lifecycle (TTL, delete, exists, flush).
    """

    # ---- String commands ----------------------------------------------------
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, ex: int | None = None) -> None: ...
    def delete(self, key: str) -> bool: ...
    def exists(self, key: str) -> bool: ...
    def expire(self, key: str, seconds: int) -> None: ...
    def incr(self, key: str) -> int: ...
    def keys(self, pattern: str) -> list[str]: ...

    # ---- Hash commands ------------------------------------------------------
    def hget(self, key: str, field: str) -> str | None: ...
    def hset(self, key: str, field: str, value: str) -> None: ...
    def hgetall(self, key: str) -> dict[str, str]: ...
    def hdel(self, key: str, field: str) -> bool: ...

    # ---- List commands ------------------------------------------------------
    def lpush(self, key: str, value: str) -> int: ...
    def rpop(self, key: str) -> str | None: ...
    def lrange(self, key: str, start: int, stop: int) -> list[str]: ...
    def llen(self, key: str) -> int: ...

    # ---- Set commands -------------------------------------------------------
    def sadd(self, key: str, *values: str) -> int: ...
    def sismember(self, key: str, value: str) -> bool: ...
    def smembers(self, key: str) -> set[str]: ...
    def srem(self, key: str, value: str) -> int: ...

    # ---- Sorted set commands ------------------------------------------------
    def zadd(self, key: str, mapping: dict[str, float]) -> int: ...
    def zrangebyscore(self, key: str, min_score: float, max_score: float) -> list[str]: ...
    def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int: ...
    def zcard(self, key: str) -> int: ...

    # ---- Connection / admin -------------------------------------------------
    def ping(self) -> bool: ...
    def close(self) -> None: ...
    def flushdb(self) -> None: ...


# -- Real Redis client --------------------------------------------------------

class RealRedisClient:
    """
    Wraps the redis-py library with lazy connection.

    The underlying redis.Redis instance is created on the first operation
    (not in __init__) so that import-time errors are deferred and the
    application can fall back to InMemoryRedisClient if Redis is unavailable.

    Thread safety is provided by redis-py's internal connection pool.
    """

    def __init__(self, config: RedisConfig):
        try:
            import redis as _redis_lib  # noqa: F401 — availability check
        except ImportError as exc:
            raise ImportError(
                "redis library is required for RealRedisClient. "
                "Install with: pip install redis"
            ) from exc

        self._config = config
        self._client: Any = None
        self._lock = threading.Lock()

    def _ensure_connected(self) -> None:
        """Create the redis.Redis connection on first use."""
        if self._client is not None:
            return
        with self._lock:
            if self._client is not None:
                return  # double-check after acquiring lock
            import redis
            self._client = redis.Redis(
                host=self._config.host,
                port=self._config.port,
                db=self._config.db,
                password=self._config.password or None,
                ssl=self._config.ssl,
                socket_timeout=self._config.socket_timeout,
                decode_responses=True,
            )
            logger.info(
                "Redis connected: %s:%d db=%d",
                self._config.host, self._config.port, self._config.db,
            )

    # ---- String commands ----------------------------------------------------

    def get(self, key: str) -> str | None:
        self._ensure_connected()
        return self._client.get(key)

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self._ensure_connected()
        self._client.set(key, value, ex=ex)

    def delete(self, key: str) -> bool:
        self._ensure_connected()
        return bool(self._client.delete(key))

    def exists(self, key: str) -> bool:
        self._ensure_connected()
        return bool(self._client.exists(key))

    def expire(self, key: str, seconds: int) -> None:
        self._ensure_connected()
        self._client.expire(key, seconds)

    def incr(self, key: str) -> int:
        self._ensure_connected()
        return self._client.incr(key)

    def keys(self, pattern: str) -> list[str]:
        self._ensure_connected()
        return self._client.keys(pattern)

    # ---- Hash commands ------------------------------------------------------

    def hget(self, key: str, field: str) -> str | None:
        self._ensure_connected()
        return self._client.hget(key, field)

    def hset(self, key: str, field: str, value: str) -> None:
        self._ensure_connected()
        self._client.hset(key, field, value)

    def hgetall(self, key: str) -> dict[str, str]:
        self._ensure_connected()
        return self._client.hgetall(key)

    def hdel(self, key: str, field: str) -> bool:
        self._ensure_connected()
        return bool(self._client.hdel(key, field))

    # ---- List commands ------------------------------------------------------

    def lpush(self, key: str, value: str) -> int:
        self._ensure_connected()
        return self._client.lpush(key, value)

    def rpop(self, key: str) -> str | None:
        self._ensure_connected()
        return self._client.rpop(key)

    def lrange(self, key: str, start: int, stop: int) -> list[str]:
        self._ensure_connected()
        return self._client.lrange(key, start, stop)

    def llen(self, key: str) -> int:
        self._ensure_connected()
        return self._client.llen(key)

    # ---- Set commands -------------------------------------------------------

    def sadd(self, key: str, *values: str) -> int:
        self._ensure_connected()
        return self._client.sadd(key, *values)

    def sismember(self, key: str, value: str) -> bool:
        self._ensure_connected()
        return bool(self._client.sismember(key, value))

    def smembers(self, key: str) -> set[str]:
        self._ensure_connected()
        return self._client.smembers(key)

    def srem(self, key: str, value: str) -> int:
        self._ensure_connected()
        return self._client.srem(key, value)

    # ---- Sorted set commands ------------------------------------------------

    def zadd(self, key: str, mapping: dict[str, float]) -> int:
        self._ensure_connected()
        return self._client.zadd(key, mapping)

    def zrangebyscore(self, key: str, min_score: float, max_score: float) -> list[str]:
        self._ensure_connected()
        return self._client.zrangebyscore(key, min_score, max_score)

    def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        self._ensure_connected()
        return self._client.zremrangebyscore(key, min_score, max_score)

    def zcard(self, key: str) -> int:
        self._ensure_connected()
        return self._client.zcard(key)

    # ---- Connection / admin -------------------------------------------------

    def ping(self) -> bool:
        self._ensure_connected()
        try:
            return self._client.ping()
        except Exception:
            return False

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Redis connection closed")

    def flushdb(self) -> None:
        self._ensure_connected()
        self._client.flushdb()


# -- In-memory Redis client ---------------------------------------------------

class InMemoryRedisClient:
    """
    Complete in-memory Redis-compatible client using Python dicts.

    Thread-safe via threading.Lock.  All Redis data types used by the
    application are supported: strings, hashes, lists, sets, sorted sets.

    TTL is implemented via lazy expiry: each access checks whether the key
    has expired.  A periodic cleanup runs during set() to prevent unbounded
    growth of expired keys.

    This class serves as the graceful fallback when the redis-py library
    is not installed or when the Redis server is unreachable.
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._expiry: dict[str, float] = {}
        self._lock = threading.Lock()
        self._closed = False

    # ---- Internal helpers ---------------------------------------------------

    def _is_expired(self, key: str) -> bool:
        """Check whether a key has passed its TTL deadline."""
        if key not in self._expiry:
            return False
        if time.time() >= self._expiry[key]:
            # Evict expired key
            self._store.pop(key, None)
            self._expiry.pop(key, None)
            return True
        return False

    def _cleanup_expired(self, sample_size: int = 20) -> None:
        """
        Probabilistic lazy cleanup -- mirrors Redis's active expiry.

        Redis samples 20 random keys with TTL each cycle and deletes those
        that have expired.  We do the same on every set() to keep memory
        bounded without a background thread.
        """
        if not self._expiry:
            return
        now = time.time()
        expired_keys = [
            k for k in list(self._expiry.keys())[:sample_size]
            if now >= self._expiry.get(k, float("inf"))
        ]
        for k in expired_keys:
            self._store.pop(k, None)
            self._expiry.pop(k, None)

    def _ensure_type(self, key: str, expected_type: type) -> bool:
        """Return True if key doesn't exist or is the expected type."""
        if key not in self._store:
            return True
        return isinstance(self._store[key], expected_type)

    # ---- String commands ----------------------------------------------------

    def get(self, key: str) -> str | None:
        with self._lock:
            if self._is_expired(key):
                return None
            val = self._store.get(key)
            if val is None or not isinstance(val, str):
                return None
            return val

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        with self._lock:
            self._store[key] = value
            if ex is not None:
                self._expiry[key] = time.time() + ex
            else:
                self._expiry.pop(key, None)
            self._cleanup_expired()

    def delete(self, key: str) -> bool:
        with self._lock:
            existed = key in self._store
            self._store.pop(key, None)
            self._expiry.pop(key, None)
            return existed

    def exists(self, key: str) -> bool:
        with self._lock:
            if self._is_expired(key):
                return False
            return key in self._store

    def expire(self, key: str, seconds: int) -> None:
        with self._lock:
            if key in self._store and not self._is_expired(key):
                self._expiry[key] = time.time() + seconds

    def incr(self, key: str) -> int:
        with self._lock:
            self._is_expired(key)
            val = self._store.get(key)
            if val is None:
                self._store[key] = "1"
                return 1
            new_val = int(val) + 1
            self._store[key] = str(new_val)
            return new_val

    def keys(self, pattern: str) -> list[str]:
        with self._lock:
            # Clean up expired keys first
            now = time.time()
            expired = [k for k, exp in self._expiry.items() if now >= exp]
            for k in expired:
                self._store.pop(k, None)
                self._expiry.pop(k, None)
            # fnmatch glob matching (Redis KEYS uses glob-style patterns)
            return [k for k in self._store if fnmatch.fnmatch(k, pattern)]

    # ---- Hash commands ------------------------------------------------------

    def hget(self, key: str, field: str) -> str | None:
        with self._lock:
            if self._is_expired(key):
                return None
            h = self._store.get(key)
            if not isinstance(h, dict):
                return None
            return h.get(field)

    def hset(self, key: str, field: str, value: str) -> None:
        with self._lock:
            self._is_expired(key)
            if key not in self._store or not isinstance(self._store.get(key), dict):
                self._store[key] = {}
            self._store[key][field] = value

    def hgetall(self, key: str) -> dict[str, str]:
        with self._lock:
            if self._is_expired(key):
                return {}
            h = self._store.get(key)
            if not isinstance(h, dict):
                return {}
            return dict(h)

    def hdel(self, key: str, field: str) -> bool:
        with self._lock:
            if self._is_expired(key):
                return False
            h = self._store.get(key)
            if not isinstance(h, dict):
                return False
            if field in h:
                del h[field]
                return True
            return False

    # ---- List commands ------------------------------------------------------

    def lpush(self, key: str, value: str) -> int:
        with self._lock:
            self._is_expired(key)
            if key not in self._store or not isinstance(self._store.get(key), list):
                self._store[key] = []
            self._store[key].insert(0, value)
            return len(self._store[key])

    def rpop(self, key: str) -> str | None:
        with self._lock:
            if self._is_expired(key):
                return None
            lst = self._store.get(key)
            if not isinstance(lst, list) or not lst:
                return None
            return lst.pop()

    def lrange(self, key: str, start: int, stop: int) -> list[str]:
        with self._lock:
            if self._is_expired(key):
                return []
            lst = self._store.get(key)
            if not isinstance(lst, list):
                return []
            # Redis LRANGE is inclusive on both ends
            return lst[start: stop + 1] if stop >= 0 else lst[start:]

    def llen(self, key: str) -> int:
        with self._lock:
            if self._is_expired(key):
                return 0
            lst = self._store.get(key)
            if not isinstance(lst, list):
                return 0
            return len(lst)

    # ---- Set commands -------------------------------------------------------

    def sadd(self, key: str, *values: str) -> int:
        with self._lock:
            self._is_expired(key)
            if key not in self._store or not isinstance(self._store.get(key), set):
                self._store[key] = set()
            s: set[str] = self._store[key]
            before = len(s)
            s.update(values)
            return len(s) - before

    def sismember(self, key: str, value: str) -> bool:
        with self._lock:
            if self._is_expired(key):
                return False
            s = self._store.get(key)
            if not isinstance(s, set):
                return False
            return value in s

    def smembers(self, key: str) -> set[str]:
        with self._lock:
            if self._is_expired(key):
                return set()
            s = self._store.get(key)
            if not isinstance(s, set):
                return set()
            return set(s)

    def srem(self, key: str, value: str) -> int:
        with self._lock:
            if self._is_expired(key):
                return 0
            s = self._store.get(key)
            if not isinstance(s, set):
                return 0
            if value in s:
                s.discard(value)
                return 1
            return 0

    # ---- Sorted set commands ------------------------------------------------

    def zadd(self, key: str, mapping: dict[str, float]) -> int:
        with self._lock:
            self._is_expired(key)
            if key not in self._store or not isinstance(self._store.get(key), list):
                self._store[key] = []
            zset: list[tuple[str, float]] = self._store[key]
            existing_members = {member for member, _ in zset}
            added = 0
            for member, score in mapping.items():
                if member in existing_members:
                    # Update score for existing member
                    zset[:] = [(m, s) if m != member else (m, score) for m, s in zset]
                else:
                    zset.append((member, score))
                    added += 1
            # Keep sorted by score for efficient range queries
            zset.sort(key=lambda x: x[1])
            return added

    def zrangebyscore(self, key: str, min_score: float, max_score: float) -> list[str]:
        with self._lock:
            if self._is_expired(key):
                return []
            zset = self._store.get(key)
            if not isinstance(zset, list):
                return []
            return [member for member, score in zset if min_score <= score <= max_score]

    def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        with self._lock:
            if self._is_expired(key):
                return 0
            zset = self._store.get(key)
            if not isinstance(zset, list):
                return 0
            original_len = len(zset)
            self._store[key] = [
                (member, score) for member, score in zset
                if not (min_score <= score <= max_score)
            ]
            return original_len - len(self._store[key])

    def zcard(self, key: str) -> int:
        with self._lock:
            if self._is_expired(key):
                return 0
            zset = self._store.get(key)
            if not isinstance(zset, list):
                return 0
            return len(zset)

    # ---- Connection / admin -------------------------------------------------

    def ping(self) -> bool:
        return not self._closed

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._store.clear()
            self._expiry.clear()

    def flushdb(self) -> None:
        with self._lock:
            self._store.clear()
            self._expiry.clear()
