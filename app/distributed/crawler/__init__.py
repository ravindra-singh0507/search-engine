"""
Distributed Crawler — Phase 8 Batch 2

Provides a distributed web crawling system with a coordinator-worker
architecture, priority-based URL frontier, and deduplication.

Modules:
  frontier    — URLFrontier (Redis) + InMemoryFrontier (heapq fallback)
  coordinator — CrawlerCoordinator: assigns work, tracks workers
  worker      — CrawlerWorker: fetches and processes URLs
  dedup       — URLDeduplicator: prevents re-crawling
"""

from app.distributed.crawler.frontier import URLFrontier, InMemoryFrontier
from app.distributed.crawler.coordinator import CrawlerCoordinator
from app.distributed.crawler.worker import CrawlerWorker
from app.distributed.crawler.dedup import URLDeduplicator

__all__ = [
    "URLFrontier",
    "InMemoryFrontier",
    "CrawlerCoordinator",
    "CrawlerWorker",
    "URLDeduplicator",
]
