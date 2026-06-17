"""
URL Deduplicator — Phase 8 Batch 2

=== THEORY ===

URL deduplication is a critical component of any web crawler.  Without
it, a crawler would endlessly re-fetch the same pages, wasting bandwidth
and compute resources.

The challenge: the web contains an enormous number of URLs, many of which
refer to the same content via different representations:
  https://example.com/page
  https://example.com/page/
  https://example.com/page?ref=twitter
  https://www.example.com/page

Deduplication happens at two levels:
  1. URL-level:     normalise URLs and check against a seen set (this module)
  2. Content-level: fingerprint page content (e.g., SimHash) to detect
                    near-duplicate pages with different URLs (not implemented
                    here — would be a separate ContentDeduplicator)

=== DATA STRUCTURES ===

  Redis SET (distributed):
    - O(1) membership test (SISMEMBER)
    - O(1) insert (SADD)
    - Shared across all crawler workers
    - Persistent across restarts

  Python set (in-memory fallback):
    - O(1) membership test (hash table)
    - O(1) insert
    - Single-process only
    - Lost on crash

=== ARCHITECTURE ===

  CrawlerWorker / URLFrontier
    │
    │  dedup.is_seen(url) → skip if True
    │  dedup.mark_seen(url) → add to seen set
    ▼
  URLDeduplicator
    │
    ├── normalize_url(url)  → canonical form
    │     (from app.crawler.url_normalize)
    │
    └── seen set (Redis SET or Python set)

=== COMPLEXITY ===

  is_seen():         O(1) — set membership test
  mark_seen():       O(1) — set insert
  mark_seen_batch(): O(B) — B inserts
  count():           O(1) — set cardinality
  clear():           O(N) — clear all entries

=== SPACE COMPLEXITY ===

  O(N) where N = number of unique URLs seen
  At ~100 bytes per URL, 1M URLs ≈ 100 MB

=== TRADEOFFS ===

  + Simple and fast: O(1) lookups via hash-based sets
  + URL normalization catches most duplicates
  + Redis backend enables distributed dedup across workers
  - No content-level dedup (different URLs, same content)
  - No probabilistic dedup (e.g., Bloom filter for memory savings)

=== PRODUCTION EQUIVALENTS ===

  Google:       Combination of URL canonicalization + SimHash content dedup
  Bing:         URL seen service with Bloom filter + content fingerprinting
  CommonCrawl:  Nutch uses CrawlDB with fetch status per URL
  Scrapy:       RFPDupeFilter using SHA-1 fingerprints of canonical URLs
"""

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class URLDeduplicator:
    """
    Tracks already-crawled URLs to prevent re-crawling.

    Uses Redis SET for distributed deduplication across multiple
    crawler workers.  Falls back to a Python set when Redis is
    unavailable.

    URLs are normalised via app.crawler.url_normalize.normalize_url
    before checking, ensuring that different representations of the
    same URL are treated as duplicates.

    Usage:
        from app.redis.client import InMemoryRedisClient
        redis = InMemoryRedisClient()
        dedup = URLDeduplicator(redis_client=redis)

        if not dedup.is_seen("https://example.com/page"):
            dedup.mark_seen("https://example.com/page")
            # ... crawl the page ...
    """

    # Redis key for the dedup set
    _REDIS_KEY = "dedup:seen_urls"

    def __init__(self, redis_client: Any = None) -> None:
        self._redis = redis_client
        self._lock = threading.Lock()

        # In-memory fallback when Redis is not available
        self._local_set: set[str] | None = None
        if redis_client is None:
            self._local_set = set()

        backend = "redis" if redis_client else "memory"
        logger.info("URLDeduplicator initialised: backend=%s", backend)

    def _normalize(self, url: str) -> str:
        """
        Normalise a URL to its canonical form.

        Uses the project's existing URL normalisation logic to ensure
        consistent deduplication.  If normalisation fails (invalid URL),
        returns the original URL unchanged.
        """
        try:
            from app.crawler.url_normalize import normalize_url
            normalised = normalize_url(url)
            return normalised if normalised is not None else url
        except Exception:
            return url

    def is_seen(self, url: str) -> bool:
        """
        Check whether a URL has already been seen.

        Args:
            url: the URL to check

        Returns:
            True if the URL (after normalisation) has been seen before.
        """
        normalised = self._normalize(url)

        if self._local_set is not None:
            with self._lock:
                return normalised in self._local_set

        return self._redis.sismember(self._REDIS_KEY, normalised)

    def mark_seen(self, url: str) -> None:
        """
        Mark a URL as seen (already crawled or queued).

        Args:
            url: the URL to mark
        """
        normalised = self._normalize(url)

        if self._local_set is not None:
            with self._lock:
                self._local_set.add(normalised)
            return

        self._redis.sadd(self._REDIS_KEY, normalised)

    def mark_seen_batch(self, urls: list[str]) -> None:
        """
        Mark multiple URLs as seen in a single operation.

        For the Redis backend, each URL is added individually (Redis SADD
        supports variadic arguments, but we normalise each URL first).

        Args:
            urls: list of URLs to mark as seen
        """
        normalised_urls = [self._normalize(url) for url in urls]

        if self._local_set is not None:
            with self._lock:
                self._local_set.update(normalised_urls)
            return

        # Redis SADD with multiple values
        if normalised_urls:
            self._redis.sadd(self._REDIS_KEY, *normalised_urls)

    def count(self) -> int:
        """
        Return the number of unique URLs seen.

        Returns:
            Count of URLs in the seen set.
        """
        if self._local_set is not None:
            with self._lock:
                return len(self._local_set)

        members = self._redis.smembers(self._REDIS_KEY)
        return len(members) if members else 0

    def clear(self) -> None:
        """
        Clear all seen URLs.

        Warning: this resets deduplication state, meaning previously
        crawled URLs may be re-crawled.
        """
        if self._local_set is not None:
            with self._lock:
                self._local_set.clear()
            logger.info("URLDeduplicator cleared (in-memory)")
            return

        self._redis.delete(self._REDIS_KEY)
        logger.info("URLDeduplicator cleared (redis)")
