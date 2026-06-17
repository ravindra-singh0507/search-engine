"""
Crawler Coordinator — Phase 8 Batch 2

=== THEORY ===

The Coordinator is the brain of a distributed crawler.  It implements
the Manager/Worker pattern (also called Master/Slave in older literature):

  1. Manages the URL frontier (what to crawl next)
  2. Assigns URL batches to available workers
  3. Tracks worker health via heartbeats and timeouts
  4. Handles failures by reassigning work from dead workers
  5. Emits events for observability (crawl.started, crawl.completed)

The coordinator does NOT fetch pages itself — it only orchestrates.
Workers do the actual HTTP fetching and content extraction.

=== ARCHITECTURE ===

  Seed URLs
    │
    ▼
  CrawlerCoordinator
    │
    ├── URLFrontier (priority queue)
    │     │
    │     ├── add_seeds() → enqueue seed URLs
    │     ├── assign_batch() → dequeue URLs for worker
    │     └── report_result() → mark complete or re-enqueue
    │
    ├── Worker Registry
    │     │
    │     ├── register_worker() → add to active set
    │     ├── deregister_worker() → remove + reassign work
    │     └── get_workers() → list active workers
    │
    └── EventBus (optional)
          │
          └── emit crawl.started / crawl.page_fetched / crawl.completed

=== COMPLEXITY ===

  add_seeds():       O(S) where S = number of seed URLs
  assign_batch():    O(B log N) where B = batch_size, N = frontier size
  report_result():   O(1) — update tracking sets
  register_worker(): O(1) — dict insert
  get_workers():     O(W) where W = number of workers
  is_complete():     O(1) — check frontier empty + no in-flight

=== SPACE COMPLEXITY ===

  O(N + W) where N = frontier size, W = active workers

=== TRADEOFFS ===

  + Centralised coordination simplifies worker logic
  + Event-driven observability via optional EventBus
  + Failure recovery via in-flight tracking and reassignment
  + Workers are stateless (can be added/removed dynamically)
  - Single coordinator = single point of failure
  - No domain-level politeness partitioning (simplified model)

=== PRODUCTION EQUIVALENTS ===

  Google:       Distributed scheduler with sharded URL servers
  Bing:         Coordinator service with domain-level partitioning
  CommonCrawl:  Apache Nutch GeneratorJob + FetcherJob
  Scrapy:       Scrapyd scheduler with project/spider management
"""

import logging
import threading
import time
from collections import deque
from typing import Any

from app.config import DistributedCrawlerConfig
from app.distributed.crawler.frontier import URLFrontier, InMemoryFrontier

logger = logging.getLogger(__name__)


class CrawlerCoordinator:
    """
    Manages the distributed crawler cluster.

    Assigns URL batches to workers, tracks worker health, handles
    failures, and emits events through the optional event bus.

    Usage:
        config = DistributedCrawlerConfig(max_workers=4, batch_size=10)
        frontier = InMemoryFrontier(max_size=10000)
        coordinator = CrawlerCoordinator(config, frontier)

        coordinator.add_seeds(["https://example.com"])
        coordinator.register_worker("worker-1")

        batch = coordinator.assign_batch("worker-1")
        # ... worker processes batch ...
        coordinator.report_result("worker-1", url, success=True, doc_id=1)
    """

    def __init__(
        self,
        config: DistributedCrawlerConfig,
        frontier: URLFrontier | InMemoryFrontier,
        event_bus: Any = None,
        redis_client: Any = None,
    ) -> None:
        self._config = config
        self._frontier = frontier
        self._event_bus = event_bus
        self._redis = redis_client
        self._lock = threading.Lock()

        # Worker tracking: worker_id → {status, registered_at, last_heartbeat,
        #                                urls_assigned, urls_completed, urls_failed}
        self._workers: dict[str, dict[str, Any]] = {}

        # In-flight tracking: url → {worker_id, assigned_at}
        self._in_flight: dict[str, dict[str, Any]] = {}

        # Results tracking
        self._results: deque[dict] = deque(maxlen=10000)
        self._total_completed = 0
        self._total_failed = 0
        self._start_time = time.time()

        logger.info(
            "CrawlerCoordinator initialised: max_workers=%d batch_size=%d",
            config.max_workers, config.batch_size,
        )

    def add_seeds(self, urls: list[str]) -> int:
        """
        Add seed URLs to the frontier.

        Seeds are added with priority 0.0 (highest) and depth 0.

        Args:
            urls: list of seed URL strings

        Returns:
            Count of URLs actually added (excludes duplicates).
        """
        count = 0
        for url in urls:
            if self._frontier.add(url, priority=0.0, depth=0):
                count += 1

        if count > 0 and self._event_bus is not None:
            try:
                from app.events.models import Event, EventMetadata
                event = Event(
                    topic="crawl.started",
                    payload={
                        "seed_count": count,
                        "seeds": urls[:10],  # limit payload size
                    },
                    metadata=EventMetadata(source="coordinator"),
                )
                self._event_bus.publish(event)
            except Exception as exc:
                logger.warning("Failed to emit crawl.started event: %s", exc)

        logger.info("Added %d seed URLs to frontier (requested %d)", count, len(urls))
        return count

    def assign_batch(self, worker_id: str) -> list[dict]:
        """
        Get a batch of URLs for a worker to crawl.

        Dequeues URLs from the frontier and marks them as in-flight,
        assigned to the given worker.

        Args:
            worker_id: identifier of the requesting worker

        Returns:
            List of dicts [{url, priority, depth}, ...].
            Empty list if frontier is empty or worker not registered.
        """
        with self._lock:
            if worker_id not in self._workers:
                logger.warning("Unknown worker '%s' requesting batch", worker_id)
                return []

            if self._workers[worker_id]["status"] != "active":
                logger.warning("Inactive worker '%s' requesting batch", worker_id)
                return []

        batch = self._frontier.get_batch(self._config.batch_size)

        with self._lock:
            for item in batch:
                self._in_flight[item["url"]] = {
                    "worker_id": worker_id,
                    "assigned_at": time.time(),
                }
                self._workers[worker_id]["urls_assigned"] += 1

            self._workers[worker_id]["last_heartbeat"] = time.time()

        logger.debug(
            "Assigned %d URLs to worker '%s'", len(batch), worker_id,
        )
        return batch

    def report_result(
        self,
        worker_id: str,
        url: str,
        success: bool,
        doc_id: int | None = None,
    ) -> None:
        """
        Report the result of crawling a URL.

        On success, the URL is marked complete. On failure, it may be
        re-enqueued for retry depending on the retry configuration.

        Args:
            worker_id: the worker that processed the URL
            url:       the URL that was processed
            success:   True if crawling succeeded
            doc_id:    document ID if the page was indexed
        """
        with self._lock:
            # Remove from in-flight
            self._in_flight.pop(url, None)

            # Update worker stats
            if worker_id in self._workers:
                self._workers[worker_id]["last_heartbeat"] = time.time()

        if success:
            self._frontier.mark_complete(url)
            with self._lock:
                self._total_completed += 1
                if worker_id in self._workers:
                    self._workers[worker_id]["urls_completed"] += 1

            self._results.append({
                "url": url,
                "worker_id": worker_id,
                "success": True,
                "doc_id": doc_id,
                "timestamp": time.time(),
            })

            # Emit event
            if self._event_bus is not None:
                try:
                    from app.events.models import Event, EventMetadata
                    event = Event(
                        topic="crawl.page_fetched",
                        payload={
                            "url": url,
                            "worker_id": worker_id,
                            "doc_id": doc_id,
                        },
                        metadata=EventMetadata(source="coordinator"),
                    )
                    self._event_bus.publish(event)
                except Exception as exc:
                    logger.warning("Failed to emit crawl.page_fetched: %s", exc)

            logger.debug("URL completed: %s (worker=%s)", url, worker_id)
        else:
            self.report_failure(worker_id, url, "crawl_failed")

    def report_failure(self, worker_id: str, url: str, error: str) -> None:
        """
        Report a crawl failure for a URL.

        The URL is marked as failed in the frontier.

        Args:
            worker_id: the worker that encountered the failure
            url:       the URL that failed
            error:     error description
        """
        with self._lock:
            self._in_flight.pop(url, None)
            self._total_failed += 1
            if worker_id in self._workers:
                self._workers[worker_id]["urls_failed"] += 1
                self._workers[worker_id]["last_heartbeat"] = time.time()

        self._frontier.mark_failed(url, error)

        self._results.append({
            "url": url,
            "worker_id": worker_id,
            "success": False,
            "error": error,
            "timestamp": time.time(),
        })

        logger.debug("URL failed: %s (worker=%s, error=%s)", url, worker_id, error)

    def register_worker(self, worker_id: str) -> None:
        """
        Register a crawler worker with the coordinator.

        Args:
            worker_id: unique identifier for the worker
        """
        with self._lock:
            if len(self._workers) >= self._config.max_workers:
                logger.warning(
                    "Max workers reached (%d), rejecting worker '%s'",
                    self._config.max_workers, worker_id,
                )
                return

            self._workers[worker_id] = {
                "status": "active",
                "registered_at": time.time(),
                "last_heartbeat": time.time(),
                "urls_assigned": 0,
                "urls_completed": 0,
                "urls_failed": 0,
            }

        logger.info("Worker registered: %s", worker_id)

    def deregister_worker(self, worker_id: str) -> None:
        """
        Remove a worker from the coordinator.

        Any URLs assigned to this worker that are still in-flight are
        re-enqueued into the frontier for reassignment.

        Args:
            worker_id: the worker to remove
        """
        with self._lock:
            self._workers.pop(worker_id, None)

            # Re-enqueue in-flight URLs from this worker
            urls_to_requeue: list[tuple[str, dict]] = []
            for url, info in list(self._in_flight.items()):
                if info["worker_id"] == worker_id:
                    urls_to_requeue.append((url, info))
                    del self._in_flight[url]

        # Re-add to frontier outside the lock
        for url, _ in urls_to_requeue:
            self._frontier.add(url, priority=1.0, depth=0)

        if urls_to_requeue:
            logger.info(
                "Re-enqueued %d URLs from deregistered worker '%s'",
                len(urls_to_requeue), worker_id,
            )
        logger.info("Worker deregistered: %s", worker_id)

    def get_workers(self) -> list[dict]:
        """
        List all registered workers with their status.

        Returns:
            List of dicts [{worker_id, status, urls_assigned, ...}, ...]
        """
        with self._lock:
            workers = []
            for worker_id, info in self._workers.items():
                in_flight = sum(
                    1 for v in self._in_flight.values()
                    if v["worker_id"] == worker_id
                )
                workers.append({
                    "worker_id": worker_id,
                    "status": info["status"],
                    "registered_at": info["registered_at"],
                    "last_heartbeat": info["last_heartbeat"],
                    "urls_assigned": info["urls_assigned"],
                    "urls_completed": info["urls_completed"],
                    "urls_failed": info["urls_failed"],
                    "urls_in_flight": in_flight,
                })
            return workers

    def stats(self) -> dict:
        """
        Return overall crawl statistics.

        Returns:
            Dict with keys: total_completed, total_failed, in_flight,
            frontier_size, worker_count, uptime_seconds, frontier_stats.
        """
        with self._lock:
            in_flight = len(self._in_flight)
            worker_count = len(self._workers)

        frontier_stats = self._frontier.stats()

        return {
            "total_completed": self._total_completed,
            "total_failed": self._total_failed,
            "in_flight": in_flight,
            "frontier_size": self._frontier.size(),
            "worker_count": worker_count,
            "uptime_seconds": round(time.time() - self._start_time, 2),
            "frontier": frontier_stats,
        }

    def is_complete(self) -> bool:
        """
        Check if all URLs have been processed.

        The crawl is complete when the frontier is empty and there are
        no URLs in-flight (being processed by workers).

        Returns:
            True if all URLs have been processed or failed.
        """
        with self._lock:
            in_flight_empty = len(self._in_flight) == 0
        return self._frontier.is_empty() and in_flight_empty
