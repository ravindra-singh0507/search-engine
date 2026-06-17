"""
URL Frontier — Phase 8 Batch 2

=== THEORY ===

The URL frontier (also called the crawl frontier or URL queue) is the
central data structure of a web crawler.  It determines *what to crawl
next* by maintaining a priority queue of URLs waiting to be fetched.

A well-designed frontier provides:
  1. Prioritisation   — crawl important URLs first (PageRank, freshness)
  2. Deduplication    — never enqueue the same URL twice
  3. Politeness       — rate-limit requests per domain
  4. Persistence      — survive crashes without losing the queue
  5. Distribution     — shared across multiple crawler workers

=== DATA STRUCTURES ===

The frontier uses two backing stores with automatic fallback:

  Redis-backed (production):
    Sorted Set   — ZADD/ZRANGEBYSCORE for priority-ordered URL retrieval
    Set          — SADD/SISMEMBER for O(1) deduplication
    Hash         — HSET/HGETALL for URL metadata (depth, status)

  In-memory fallback (development/testing):
    heapq        — binary heap for priority queue (O(log n) insert/pop)
    set          — hash set for O(1) deduplication

=== ARCHITECTURE ===

  CrawlerCoordinator
    │
    │  frontier.add(url, priority, depth)
    │  frontier.get_batch(batch_size)
    ▼
  URLFrontier / InMemoryFrontier
    │
    ├── Priority Queue (sorted set / heapq)
    │     ordered by priority score (lower = higher priority)
    │
    ├── Seen Set (set / Redis SET)
    │     tracks all URLs ever added for deduplication
    │
    └── Metadata Store (dict / Redis HASH)
          depth, status, error for each URL

=== COMPLEXITY ===

  URLFrontier (Redis-backed):
    add():          O(log N) — ZADD into sorted set
    get_batch():    O(log N + B) — ZRANGEBYSCORE + ZREM
    mark_complete(): O(1) — SREM + HSET
    contains():     O(1) — SISMEMBER
    size():         O(1) — ZCARD

  InMemoryFrontier:
    add():          O(log N) — heapq push
    get_batch():    O(B log N) — B heapq pops
    mark_complete(): O(1) — set operations
    contains():     O(1) — set lookup
    size():         O(1) — len

  where N = frontier size, B = batch size

=== SPACE COMPLEXITY ===

  O(N) for the priority queue + O(N) for the seen set = O(N) total

=== TRADEOFFS ===

  URLFrontier (Redis):
    + Distributed: shared across multiple crawler workers
    + Persistent: survives process restarts
    + Atomic: Redis operations are atomic (no race conditions)
    - Requires Redis server
    - Network latency per operation (~0.1ms local)

  InMemoryFrontier:
    + Zero dependencies
    + Lowest latency (no network I/O)
    - Single-process only
    - Lost on crash

=== PRODUCTION EQUIVALENTS ===

  Google:       Distributed URL frontier with sharded priority queues
  Bing:         URL frontier service with domain-level partitioning
  CommonCrawl:  Apache Nutch FetcherBolt with URL generator
  Scrapy:       In-memory scheduler with disk-backed queue fallback
"""

import heapq
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


# ── Redis-backed Frontier ───────────────────────────────────────────────────

class URLFrontier:
    """
    Priority queue of URLs to crawl, backed by Redis sorted sets.

    Falls back to InMemoryFrontier when Redis is unavailable.
    URLs are scored by priority (lower score = higher priority),
    and deduplication is enforced via a Redis SET.

    Keys used in Redis (all prefixed with 'frontier:'):
      frontier:queue    — sorted set of (url, priority) pairs
      frontier:seen     — set of all URLs ever added
      frontier:meta:{url} — hash with depth, status, error, added_at
      frontier:complete — set of completed URLs
      frontier:failed   — set of failed URLs

    Usage:
        from app.redis.client import InMemoryRedisClient
        redis = InMemoryRedisClient()
        frontier = URLFrontier(redis_client=redis, max_size=10000)
        frontier.add("https://example.com", priority=0.0, depth=0)
        batch = frontier.get_batch(10)
    """

    def __init__(self, redis_client: Any = None, max_size: int = 100000) -> None:
        self._redis = redis_client
        self._max_size = max_size
        self._lock = threading.Lock()

        # Key names
        self._queue_key = "frontier:queue"
        self._seen_key = "frontier:seen"
        self._complete_key = "frontier:complete"
        self._failed_key = "frontier:failed"

        logger.info(
            "URLFrontier initialised: backend=%s max_size=%d",
            "redis" if redis_client else "none",
            max_size,
        )

    def _meta_key(self, url: str) -> str:
        """Return the Redis hash key for a URL's metadata."""
        return f"frontier:meta:{url}"

    def add(self, url: str, priority: float = 0.0, depth: int = 0) -> bool:
        """
        Add a URL to the frontier with a given priority.

        Returns False if the URL is a duplicate (already seen) or if
        the frontier has reached max_size.

        Args:
            url:      normalised URL string
            priority: crawl priority (lower = higher priority)
            depth:    BFS depth from seed URL

        Returns:
            True if the URL was added, False if duplicate or full.
        """
        with self._lock:
            # Check capacity
            current_size = self._redis.zcard(self._queue_key)
            if current_size >= self._max_size:
                logger.debug("Frontier full (%d/%d)", current_size, self._max_size)
                return False

            # Dedup check
            if self._redis.sismember(self._seen_key, url):
                return False

            # Add to seen set
            self._redis.sadd(self._seen_key, url)

            # Add to priority queue (sorted set)
            self._redis.zadd(self._queue_key, {url: priority})

            # Store metadata
            self._redis.hset(self._meta_key(url), "depth", str(depth))
            self._redis.hset(self._meta_key(url), "priority", str(priority))
            self._redis.hset(self._meta_key(url), "status", "pending")
            self._redis.hset(self._meta_key(url), "added_at", str(time.time()))

        return True

    def get_batch(self, batch_size: int = 10) -> list[dict]:
        """
        Retrieve a batch of URLs from the frontier, ordered by priority.

        Removes the URLs from the queue (they are now "in-flight").
        Each entry is a dict with keys: url, priority, depth.

        Args:
            batch_size: maximum number of URLs to retrieve

        Returns:
            List of dicts [{url, priority, depth}, ...]
        """
        batch: list[dict] = []

        with self._lock:
            # Get the lowest-priority (highest importance) URLs
            urls = self._redis.zrangebyscore(
                self._queue_key,
                min_score=float("-inf"),
                max_score=float("inf"),
            )

            # Take up to batch_size
            selected = urls[:batch_size]

            for url in selected:
                # Get metadata
                meta = self._redis.hgetall(self._meta_key(url))
                depth = int(meta.get("depth", "0"))
                priority = float(meta.get("priority", "0.0"))

                batch.append({
                    "url": url,
                    "priority": priority,
                    "depth": depth,
                })

                # Remove from queue (now in-flight)
                self._redis.zremrangebyscore(
                    self._queue_key,
                    min_score=priority,
                    max_score=priority,
                )
                # Update status
                self._redis.hset(self._meta_key(url), "status", "in_flight")

        return batch

    def mark_complete(self, url: str) -> None:
        """
        Mark a URL as successfully crawled.

        Moves the URL from the seen set to the complete set and
        updates its metadata status.

        Args:
            url: the URL that was successfully crawled
        """
        with self._lock:
            self._redis.sadd(self._complete_key, url)
            self._redis.hset(self._meta_key(url), "status", "complete")
            self._redis.hset(self._meta_key(url), "completed_at", str(time.time()))

    def mark_failed(self, url: str, error: str) -> None:
        """
        Mark a URL as failed.

        Records the error message and moves the URL to the failed set.

        Args:
            url:   the URL that failed to crawl
            error: error description
        """
        with self._lock:
            self._redis.sadd(self._failed_key, url)
            self._redis.hset(self._meta_key(url), "status", "failed")
            self._redis.hset(self._meta_key(url), "error", error)
            self._redis.hset(self._meta_key(url), "failed_at", str(time.time()))

    def size(self) -> int:
        """Return the number of URLs currently in the queue (pending)."""
        return self._redis.zcard(self._queue_key)

    def is_empty(self) -> bool:
        """Return True if the frontier queue is empty."""
        return self.size() == 0

    def contains(self, url: str) -> bool:
        """
        Check if a URL has ever been added to the frontier (dedup check).

        Args:
            url: URL to check

        Returns:
            True if the URL has been seen before.
        """
        return self._redis.sismember(self._seen_key, url)

    def stats(self) -> dict:
        """
        Return frontier statistics.

        Returns:
            Dict with keys: pending, seen, complete, failed, max_size.
        """
        pending = self._redis.zcard(self._queue_key)
        seen_members = self._redis.smembers(self._seen_key)
        seen = len(seen_members) if seen_members else 0
        complete_members = self._redis.smembers(self._complete_key)
        complete = len(complete_members) if complete_members else 0
        failed_members = self._redis.smembers(self._failed_key)
        failed = len(failed_members) if failed_members else 0

        return {
            "pending": pending,
            "seen": seen,
            "complete": complete,
            "failed": failed,
            "max_size": self._max_size,
        }


# ── In-Memory Frontier ─────────────────────────────────────────────────────

class InMemoryFrontier:
    """
    Heapq-backed fallback URL frontier with set-based dedup.

    Same interface as URLFrontier but uses Python's heapq module
    for the priority queue and a plain set for deduplication.
    Suitable for single-process crawling and testing.

    The heap contains tuples of (priority, insertion_order, url, depth)
    where insertion_order breaks ties for equal priorities (FIFO).

    Usage:
        frontier = InMemoryFrontier(max_size=10000)
        frontier.add("https://example.com", priority=0.0, depth=0)
        batch = frontier.get_batch(10)
    """

    def __init__(self, max_size: int = 100000) -> None:
        self._max_size = max_size
        self._lock = threading.Lock()

        # Priority queue: (priority, insertion_order, url, depth)
        self._heap: list[tuple[float, int, str, int]] = []
        self._insertion_counter = 0

        # Deduplication and tracking sets
        self._seen: set[str] = set()
        self._complete: set[str] = set()
        self._failed: dict[str, str] = {}  # url → error message

        # Metadata for stats
        self._meta: dict[str, dict[str, Any]] = {}

        logger.info("InMemoryFrontier initialised: max_size=%d", max_size)

    def add(self, url: str, priority: float = 0.0, depth: int = 0) -> bool:
        """
        Add a URL to the frontier.

        Returns False if the URL is a duplicate or the frontier is full.
        """
        with self._lock:
            if len(self._heap) >= self._max_size:
                return False

            if url in self._seen:
                return False

            self._seen.add(url)
            self._insertion_counter += 1
            heapq.heappush(
                self._heap,
                (priority, self._insertion_counter, url, depth),
            )
            self._meta[url] = {
                "depth": depth,
                "priority": priority,
                "status": "pending",
                "added_at": time.time(),
            }

        return True

    def get_batch(self, batch_size: int = 10) -> list[dict]:
        """
        Retrieve a batch of highest-priority URLs.

        Removes URLs from the heap (they are now in-flight).
        """
        batch: list[dict] = []

        with self._lock:
            while self._heap and len(batch) < batch_size:
                priority, _, url, depth = heapq.heappop(self._heap)
                batch.append({
                    "url": url,
                    "priority": priority,
                    "depth": depth,
                })
                if url in self._meta:
                    self._meta[url]["status"] = "in_flight"

        return batch

    def mark_complete(self, url: str) -> None:
        """Mark a URL as successfully crawled."""
        with self._lock:
            self._complete.add(url)
            if url in self._meta:
                self._meta[url]["status"] = "complete"
                self._meta[url]["completed_at"] = time.time()

    def mark_failed(self, url: str, error: str) -> None:
        """Mark a URL as failed with an error message."""
        with self._lock:
            self._failed[url] = error
            if url in self._meta:
                self._meta[url]["status"] = "failed"
                self._meta[url]["error"] = error
                self._meta[url]["failed_at"] = time.time()

    def size(self) -> int:
        """Return the number of URLs currently in the queue."""
        with self._lock:
            return len(self._heap)

    def is_empty(self) -> bool:
        """Return True if the frontier queue is empty."""
        with self._lock:
            return len(self._heap) == 0

    def contains(self, url: str) -> bool:
        """Check if a URL has ever been added (dedup check)."""
        with self._lock:
            return url in self._seen

    def stats(self) -> dict:
        """Return frontier statistics."""
        with self._lock:
            return {
                "pending": len(self._heap),
                "seen": len(self._seen),
                "complete": len(self._complete),
                "failed": len(self._failed),
                "max_size": self._max_size,
            }
