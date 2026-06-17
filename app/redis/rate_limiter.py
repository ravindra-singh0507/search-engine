"""
Redis Rate Limiter

=== THEORY ===

Rate limiting controls the number of requests a client can make within
a time window.  It protects backend services from overload, ensures fair
resource sharing, and mitigates abuse (scraping, brute-force attacks).

=== ALGORITHMS ===

  Fixed Window:
    Count requests in fixed time buckets (e.g., 00:00-00:59, 01:00-01:59).
    Problem: burst at window boundary allows 2x the limit.

  Sliding Window Log (this implementation):
    Store each request's timestamp in a sorted set.
    To check: remove timestamps older than `now - window`, count remaining.
    Advantage: no boundary burst; exact sliding window.
    Disadvantage: stores one entry per request (memory proportional to limit).

  Sliding Window Counter (production alternative):
    Hybrid of fixed + sliding.  Uses two adjacent fixed windows and
    interpolates based on elapsed time.  O(1) memory per client.

  Token Bucket:
    Tokens replenish at a fixed rate; each request consumes one.
    Allows controlled bursts up to bucket capacity.

  Leaky Bucket:
    Requests enter a FIFO queue drained at constant rate.
    Smooths traffic but adds latency.

=== REDIS SORTED SET IMPLEMENTATION ===

  Key:    {prefix}{client_id}
  Score:  Unix timestamp of the request
  Member: Unique string per request (we use str(timestamp) — sufficient
          for our single-threaded-per-client model; in production, append
          a random suffix to handle sub-microsecond duplicates)

  Algorithm for is_allowed(client_id):
    1. now = time.time()
    2. cutoff = now - window
    3. ZREMRANGEBYSCORE key -inf cutoff       # remove expired entries
    4. count = ZCARD key                       # count remaining
    5. if count >= max_requests: return False   # rate exceeded
    6. ZADD key {str(now): now}                # record this request
    7. EXPIRE key int(window) + 1              # auto-cleanup safety net
    8. return True

  All steps are O(1) amortised (ZREMRANGEBYSCORE is O(log N + M) where M
  is the number of removed elements, but M is bounded by max_requests).

=== COMPLEXITY ===

  is_allowed:  O(log N + M)  per call (N = set size, M = expired entries)
  Space:       O(max_requests) per client_id (bounded by rate limit)

=== AT PRODUCTION SCALE ===

  Cloudflare:  Sliding window counter in distributed KV
  Stripe:      Token bucket in Redis (lua scripts for atomicity)
  GitHub:      Sliding window with per-endpoint limits
  Google:      Adaptive rate limiting with client priority classes
  Netflix:     Zuul API gateway with Redis-backed rate limiter (Sentinel)
"""

import logging
import time

from app.redis.client import RedisClient

logger = logging.getLogger(__name__)


class RedisRateLimiter:
    """
    Sliding window rate limiter using Redis sorted sets.

    Each client_id gets a sorted set where:
      - Members are unique request identifiers (timestamp strings)
      - Scores are Unix timestamps

    The window slides with the current time: on each check, entries
    older than (now - window) are pruned, and the remaining count is
    compared against max_requests.

    Thread safety: all operations are atomic Redis commands.  The
    InMemoryRedisClient's internal lock ensures correctness in the
    in-memory fallback.
    """

    def __init__(
        self,
        client: RedisClient,
        prefix: str = "ratelimit:",
        max_requests: int = 60,
        window: float = 60.0,
    ):
        self._client = client
        self._prefix = prefix
        self._max_requests = max_requests
        self._window = window

    def is_allowed(self, client_id: str) -> bool:
        """
        Check whether a request from client_id is allowed under the rate limit.

        Returns True if the request is allowed (and records it).
        Returns False if the rate limit has been exceeded.

        Algorithm:
          1. Remove entries outside the sliding window
          2. Count remaining entries
          3. If count >= limit, reject
          4. Otherwise, add current request and accept
        """
        key = f"{self._prefix}{client_id}"
        now = time.time()
        cutoff = now - self._window

        # Step 1: Remove expired entries (timestamps before cutoff)
        self._client.zremrangebyscore(key, float("-inf"), cutoff)

        # Step 2: Count remaining entries in the window
        count = self._client.zcard(key)

        # Step 3: Check against limit
        if count >= self._max_requests:
            logger.debug(
                "Rate limit exceeded: client=%s count=%d limit=%d",
                client_id, count, self._max_requests,
            )
            return False

        # Step 4: Record this request
        # Using str(now) as the member; score is the timestamp itself
        self._client.zadd(key, {str(now): now})

        # Step 5: Set expiry on the key as a safety net
        # (in case no more requests come, the key will auto-expire)
        self._client.expire(key, int(self._window) + 1)

        return True
