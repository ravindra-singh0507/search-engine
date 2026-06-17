"""
Redis Infrastructure Module

Provides Redis-backed data structures for the search engine platform:
  - RedisClient (Protocol) + RealRedisClient / InMemoryRedisClient
  - RedisCache / RedisQueryCache — distributed caching with JSON serialization
  - RedisSessionStore — server-side session management via Redis hashes
  - DistributedLock / InMemoryLock — mutual exclusion across processes
  - RedisRateLimiter — sliding window rate limiting via sorted sets
"""

from app.redis.client import (
    RedisClient,
    RealRedisClient,
    InMemoryRedisClient,
)
from app.redis.cache import (
    RedisCache,
    RedisQueryCache,
)
from app.redis.sessions import RedisSessionStore
from app.redis.locks import (
    DistributedLock,
    InMemoryLock,
)
from app.redis.rate_limiter import RedisRateLimiter

__all__ = [
    "RedisClient",
    "RealRedisClient",
    "InMemoryRedisClient",
    "RedisCache",
    "RedisQueryCache",
    "RedisSessionStore",
    "DistributedLock",
    "InMemoryLock",
    "RedisRateLimiter",
]
