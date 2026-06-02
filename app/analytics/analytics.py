"""
Search Analytics

Wraps the Database analytics methods in a service layer and provides
higher-level computed metrics such as CTR buckets and hourly heatmaps.

=== WHAT WE TRACK ===

  search_logs  — every query: text, latency, result count, timestamp, session
  click_logs   — every result click: which search led to it, which doc, at rank N
  query_stats  — per-query rollup: total searches, avg latency, zero-result count

=== WHY IT MATTERS ===

Analytics is how a search engine learns what users actually want:
- Zero-result queries reveal vocabulary gaps → good candidates for synonyms
- High-click positions reveal what users actually trust
- Slow queries reveal which retrieval paths need caching
- CTR < 30% may mean ranking is broken or snippets are poor

=== AT GOOGLE SCALE ===

Google's Search Analytics combines:
  - Kafka streams for real-time event ingestion
  - BigQuery for offline aggregation and ML training data
  - Internal dashboards built on Dremel/Looker
  - Click-stream signals feed Learning-to-Rank models directly

Here we use the SQLite database as a lightweight approximation.
"""

import logging
from dataclasses import dataclass

from app.database.db import Database

logger = logging.getLogger(__name__)


@dataclass
class SearchEvent:
    query: str
    results_count: int
    latency_ms: float
    session_id: str | None = None


class AnalyticsService:
    """
    Service layer for search analytics.
    All persistence is delegated to Database; this class adds
    higher-level methods and logging.
    """

    def __init__(self, db: Database):
        self.db = db

    # ── Event Ingestion ────────────────────────────────────────────────────

    def record_search(self, event: SearchEvent) -> int:
        """Persist a search event.  Returns the log_id for click correlation."""
        log_id = self.db.log_search(
            query=event.query,
            results_count=event.results_count,
            latency_ms=event.latency_ms,
            session_id=event.session_id,
        )
        if event.results_count == 0:
            logger.debug("Zero-result query: %r", event.query)
        return log_id

    def record_click(self, log_id: int, doc_id: int, position: int) -> None:
        """Persist a click event tied to a prior search."""
        self.db.log_click(log_id=log_id, doc_id=doc_id, position=position)
        logger.debug("Click recorded: log=%d doc=%d pos=%d", log_id, doc_id, position)

    # ── Query Reports ──────────────────────────────────────────────────────

    def top_queries(self, limit: int = 20) -> list[dict]:
        """Most frequently searched terms."""
        return self.db.get_top_queries(limit=limit)

    def failed_queries(self, limit: int = 20) -> list[dict]:
        """Queries that most often return zero results."""
        return self.db.get_failed_queries(limit=limit)

    def search_volume(self, hours: int = 24) -> list[dict]:
        """Hourly search counts over the last N hours."""
        return self.db.get_search_volume(hours=hours)

    def click_through_rate(self) -> dict:
        """Fraction of searches that led to at least one click."""
        return self.db.get_click_through_rate()

    # ── Dashboard ──────────────────────────────────────────────────────────

    def dashboard(self) -> dict:
        """
        Aggregated snapshot for the analytics dashboard endpoint.
        Returns all key metrics in a single dict.
        """
        ctr_data  = self.click_through_rate()
        top_q     = self.top_queries(limit=10)
        failed_q  = self.failed_queries(limit=10)
        volume    = self.search_volume(hours=24)
        avg_pos   = self.db.get_avg_click_position()

        return {
            "click_through_rate":   ctr_data,
            "avg_click_position":   avg_pos,
            "top_queries":          top_q,
            "failed_queries":       failed_q,
            "search_volume_24h":    volume,
        }
