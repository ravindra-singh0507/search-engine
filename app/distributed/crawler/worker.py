"""
Crawler Worker — Phase 8 Batch 2

=== THEORY ===

A crawler worker is a unit of execution that fetches web pages.  In a
distributed crawling system, multiple workers run concurrently (across
threads, processes, or machines) to maximise throughput.

Each worker follows a simple loop:
  1. Request a batch of URLs from the coordinator
  2. Fetch each URL (HTTP GET)
  3. Extract content (title, text, links)
  4. Store and index the content
  5. Report results back to the coordinator
  6. Repeat

Workers are *stateless* — they hold no critical data.  If a worker
crashes, its in-flight URLs are reassigned by the coordinator.

=== ARCHITECTURE ===

  CrawlerCoordinator
    │
    │  assign_batch(worker_id) → [{url, priority, depth}, ...]
    ▼
  CrawlerWorker
    │
    ├── fetch_url(url) → {url, title, content, status_code}
    │     │
    │     └── HTTP GET → parse HTML → extract text
    │
    ├── process_batch(batch) → [results...]
    │     │
    │     └── for each url: fetch → index → report
    │
    └── report_result() / report_failure() → coordinator

  CrawlerCoordinator
    │
    └── marks URL complete or re-enqueues for retry

=== COMPLEXITY ===

  fetch_url():      O(S) where S = page size (HTML parsing)
  process_batch():  O(B * S) where B = batch size
  run_once():       O(B * S) = one iteration of the worker loop
  start():          O(1) — spawns background thread

=== SPACE COMPLEXITY ===

  O(S) transient per page (HTML + extracted text in memory)

=== TRADEOFFS ===

  + Stateless: easy to scale horizontally
  + Reuses WebCrawler._fetch_page logic (no duplication)
  + Background thread for non-blocking operation
  + Stats tracking for monitoring
  - Single-threaded fetching per worker (no async I/O)
  - No domain-level rate limiting (relies on coordinator)

=== PRODUCTION EQUIVALENTS ===

  Google:       Googlebot workers with distributed fetch queue
  Bing:         MSNBot worker fleet with domain assignment
  CommonCrawl:  Nutch FetcherBolt as Storm topology workers
  Scrapy:       Spider instances with Twisted reactor for async I/O
"""

import logging
import threading
import time
from typing import Any

import requests
from bs4 import BeautifulSoup

from app.config import DistributedCrawlerConfig

logger = logging.getLogger(__name__)


class CrawlerWorker:
    """
    Individual crawler worker that fetches assigned URLs.

    Uses HTTP GET with BeautifulSoup for content extraction,
    matching the logic in app.crawler.crawler.WebCrawler._fetch_page.

    Workers are stateless processors: they receive URL batches from
    the coordinator, fetch and parse pages, optionally index content,
    and report results back.

    Usage:
        from app.distributed.crawler.coordinator import CrawlerCoordinator
        from app.distributed.crawler.frontier import InMemoryFrontier

        frontier = InMemoryFrontier()
        coordinator = CrawlerCoordinator(config, frontier)
        coordinator.register_worker("worker-1")

        worker = CrawlerWorker("worker-1", coordinator)
        worker.run_once()  # fetch one batch
        # or
        worker.start()     # continuous background fetching
        worker.stop()
    """

    def __init__(
        self,
        worker_id: str,
        coordinator: Any,
        db: Any = None,
        indexer: Any = None,
        config: DistributedCrawlerConfig | None = None,
    ) -> None:
        self._worker_id = worker_id
        self._coordinator = coordinator
        self._db = db
        self._indexer = indexer
        self._config = config or DistributedCrawlerConfig()
        self._lock = threading.Lock()

        # Lifecycle
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False

        # Stats
        self._pages_fetched = 0
        self._pages_failed = 0
        self._start_time = time.time()
        self._last_fetch_time: float | None = None

        # HTTP settings
        self._user_agent = "SearchEngineBot/1.0 (distributed)"
        self._timeout = 10

        logger.info("CrawlerWorker initialised: id=%s", worker_id)

    def fetch_url(self, url: str, depth: int = 0) -> dict:
        """
        Fetch a single URL and extract its content.

        Performs an HTTP GET request, parses the HTML with BeautifulSoup,
        and extracts the title and visible text content.

        Args:
            url:   the URL to fetch
            depth: BFS depth (informational, stored in result)

        Returns:
            Dict with keys: url, title, content, status_code, links, depth.
            On failure, status_code is 0 and content contains the error.
        """
        try:
            response = requests.get(
                url,
                timeout=self._timeout,
                headers={
                    "User-Agent": self._user_agent,
                    "Accept": "text/html,application/xhtml+xml",
                },
                allow_redirects=True,
            )
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                return {
                    "url": url,
                    "title": "",
                    "content": "",
                    "status_code": response.status_code,
                    "links": [],
                    "depth": depth,
                    "error": f"Non-HTML content type: {content_type}",
                }

            html = response.text
            soup = BeautifulSoup(html, "html.parser")

            title = self._extract_title(soup)
            text = self._extract_text(soup)
            links = self._extract_links(soup, url)

            with self._lock:
                self._pages_fetched += 1
                self._last_fetch_time = time.time()

            logger.debug(
                "Fetched %s: %d chars, %d links (worker=%s)",
                url, len(text), len(links), self._worker_id,
            )

            return {
                "url": url,
                "title": title,
                "content": text,
                "status_code": response.status_code,
                "links": links,
                "depth": depth,
            }

        except requests.RequestException as exc:
            with self._lock:
                self._pages_failed += 1

            logger.warning(
                "Failed to fetch %s (worker=%s): %s",
                url, self._worker_id, exc,
            )

            return {
                "url": url,
                "title": "",
                "content": "",
                "status_code": 0,
                "links": [],
                "depth": depth,
                "error": str(exc),
            }

    def process_batch(self, batch: list[dict]) -> list[dict]:
        """
        Fetch and optionally index each URL in the batch.

        For each URL:
          1. Fetch the page
          2. If indexer is available, index the content
          3. Report success/failure to the coordinator

        Args:
            batch: list of dicts [{url, priority, depth}, ...]

        Returns:
            List of result dicts from fetch_url().
        """
        results: list[dict] = []

        for item in batch:
            url = item["url"]
            depth = item.get("depth", 0)

            result = self.fetch_url(url, depth)
            results.append(result)

            # Determine success
            success = result["status_code"] >= 200 and result["status_code"] < 400
            doc_id = None

            # Index if we have an indexer and the fetch succeeded
            if success and self._indexer is not None and result["content"]:
                try:
                    index_result = self._indexer.index_document(
                        title=result["title"],
                        content=result["content"],
                        source=url,
                        doc_type="web",
                    )
                    doc_id = index_result.doc_id
                except Exception as exc:
                    logger.error(
                        "Failed to index %s (worker=%s): %s",
                        url, self._worker_id, exc,
                    )

            # Store in database if available
            if success and self._db is not None and result["content"]:
                try:
                    self._db.insert_crawled_page(
                        url=url,
                        title=result["title"],
                        content=result["content"],
                        html=result.get("html", ""),
                        status_code=result["status_code"],
                        crawl_depth=depth,
                        doc_id=doc_id,
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to store %s (worker=%s): %s",
                        url, self._worker_id, exc,
                    )

            # Report to coordinator
            if success:
                self._coordinator.report_result(
                    self._worker_id, url, success=True, doc_id=doc_id,
                )
            else:
                error = result.get("error", "unknown_error")
                self._coordinator.report_failure(
                    self._worker_id, url, error,
                )

            # Politeness delay between requests
            if self._config.rate_limit_per_domain > 0:
                time.sleep(self._config.rate_limit_per_domain)

        return results

    def run_once(self) -> int:
        """
        Execute one cycle of the worker loop.

        Requests a batch from the coordinator, processes it, and reports
        results.

        Returns:
            Number of URLs processed in this cycle.
        """
        batch = self._coordinator.assign_batch(self._worker_id)

        if not batch:
            return 0

        results = self.process_batch(batch)
        return len(results)

    def start(self) -> None:
        """
        Start the worker in a background thread.

        The worker runs a polling loop: request batch → process → sleep.
        """
        if self._running:
            logger.warning("Worker '%s' already running", self._worker_id)
            return

        self._stop_event.clear()
        self._running = True
        self._start_time = time.time()

        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"crawler-worker-{self._worker_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("Worker started: %s", self._worker_id)

    def stop(self) -> None:
        """
        Stop the background worker thread.
        """
        if not self._running:
            return

        self._stop_event.set()
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

        logger.info("Worker stopped: %s", self._worker_id)

    def _run_loop(self) -> None:
        """
        Background polling loop.

        Continuously requests batches from the coordinator and processes
        them until stop() is called or the coordinator signals completion.
        """
        logger.debug("Worker loop started: %s", self._worker_id)

        while not self._stop_event.is_set():
            try:
                count = self.run_once()

                if count == 0:
                    # No work available — back off before polling again
                    self._stop_event.wait(timeout=2.0)
                    continue

            except Exception as exc:
                logger.error(
                    "Worker loop error (worker=%s): %s",
                    self._worker_id, exc,
                )
                self._stop_event.wait(timeout=5.0)

        logger.debug("Worker loop exited: %s", self._worker_id)

    def stats(self) -> dict:
        """
        Return worker statistics.

        Returns:
            Dict with keys: worker_id, pages_fetched, pages_failed,
            uptime_seconds, last_fetch_time, running.
        """
        with self._lock:
            return {
                "worker_id": self._worker_id,
                "pages_fetched": self._pages_fetched,
                "pages_failed": self._pages_failed,
                "uptime_seconds": round(time.time() - self._start_time, 2),
                "last_fetch_time": self._last_fetch_time,
                "running": self._running,
            }

    # ── HTML Extraction (mirrors WebCrawler logic) ──────────────────────────

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract the page title from HTML."""
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            return title_tag.string.strip()
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        return "Untitled"

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract visible text from HTML, removing scripts and styles."""
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        lines = (line.strip() for line in text.splitlines())
        return " ".join(chunk for chunk in lines if chunk)

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        """Extract all navigable href links from anchor tags."""
        from app.crawler.url_normalize import normalize_url

        links: list[str] = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            if href.startswith((
                "javascript:", "mailto:", "tel:", "#", "data:", "vbscript:",
            )):
                continue
            full_url = normalize_url(href, base_url=base_url)
            if full_url:
                links.append(full_url)
        return links
