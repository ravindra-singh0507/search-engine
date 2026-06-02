"""
Web Crawler

=== THEORY ===

A web crawler (or spider) systematically browses the web to discover and
download pages. It's the data acquisition layer of a search engine.

=== BFS CRAWLING ===

We use Breadth-First Search to crawl:
1. Start with seed URLs (depth 0)
2. Fetch each page, extract links
3. Add new links to the queue (depth + 1)
4. Process all depth-N pages before moving to depth N+1

WHY BFS over DFS?
- BFS discovers "important" pages first (pages closer to the seed)
- DFS can get stuck deep in a single site's hierarchy
- BFS naturally provides a depth limit
- BFS distributes crawling more evenly across domains

=== URL FRONTIER ===

The URL frontier is the queue of URLs waiting to be crawled. It has:
- A FIFO queue for BFS order
- A visited set to avoid recrawling
- Normalization to deduplicate URLs

=== CRAWL PIPELINE ===

For each URL:
1. Check robots.txt → allowed?
2. Check visited set → already crawled?
3. Fetch the page (HTTP GET)
4. Parse HTML → extract title, text, links
5. Store raw page in database
6. Index the page content
7. Add discovered links to the frontier

=== COMPLEXITY ===

- BFS crawl of P pages with avg L links each: O(P * L) URL processing
- Space for visited set: O(P) URLs
- Space for frontier: O(P * L) in worst case

=== AT GOOGLE SCALE ===

Google crawls billions of pages using:
- Distributed URL frontier across many machines
- Prioritized crawling (important pages crawled more frequently)
- Politeness policies (rate limiting per domain)
- Incremental crawling (only recrawl pages that likely changed)
- Caffeine architecture for near-real-time index updates
"""

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from app.database.db import Database
from app.indexer.indexer import Indexer
from app.crawler.url_normalize import normalize_url, is_same_domain
from app.crawler.robots import RobotsParser

logger = logging.getLogger(__name__)


@dataclass
class CrawlResult:
    url: str
    title: str
    content: str
    html: str
    links: list[str]
    status_code: int
    depth: int


@dataclass
class CrawlStats:
    pages_crawled: int = 0
    pages_indexed: int = 0
    pages_failed: int = 0
    pages_skipped_robots: int = 0
    links_discovered: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration_seconds(self) -> float:
        if self.end_time == 0.0:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        return {
            "pages_crawled": self.pages_crawled,
            "pages_indexed": self.pages_indexed,
            "pages_failed": self.pages_failed,
            "pages_skipped_robots": self.pages_skipped_robots,
            "links_discovered": self.links_discovered,
            "duration_seconds": round(self.duration_seconds, 2),
        }


@dataclass
class CrawlJob:
    seed_urls: list[str]
    max_depth: int = 3
    max_pages: int = 100
    stay_on_domain: bool = True
    stats: CrawlStats = field(default_factory=CrawlStats)
    is_running: bool = False
    is_complete: bool = False


class WebCrawler:
    """
    BFS web crawler with robots.txt support, URL normalization,
    and automatic indexing of crawled content.
    """

    def __init__(self, db: Database, indexer: Indexer,
                 user_agent: str = "SearchEngineBot/1.0",
                 request_delay: float = 1.0,
                 timeout: int = 10,
                 respect_robots: bool = True):
        self.db = db
        self.indexer = indexer
        self.user_agent = user_agent
        self.request_delay = request_delay
        self.timeout = timeout
        self.respect_robots = respect_robots
        self.robots_parser = RobotsParser(user_agent)
        self.current_job: CrawlJob | None = None

    def crawl(self, seed_urls: list[str], max_depth: int = 3,
              max_pages: int = 100, stay_on_domain: bool = True) -> CrawlStats:
        """
        Execute a BFS crawl starting from seed URLs.
        """
        job = CrawlJob(
            seed_urls=seed_urls,
            max_depth=max_depth,
            max_pages=max_pages,
            stay_on_domain=stay_on_domain,
        )
        job.stats.start_time = time.time()
        job.is_running = True
        self.current_job = job

        # BFS data structures
        frontier: deque[tuple[str, int]] = deque()  # (url, depth)
        visited: set[str] = set()

        # Seed the frontier — pre-populate visited from DB so a restarted
        # crawl doesn't re-enqueue URLs that are already stored.
        for url in seed_urls:
            normalized = normalize_url(url)
            if normalized and normalized not in visited:
                if not self.db.url_already_crawled(normalized):
                    frontier.append((normalized, 0))
                visited.add(normalized)

        logger.info(
            "Starting crawl: %d seeds, max_depth=%d, max_pages=%d",
            len(seed_urls), max_depth, max_pages
        )

        while frontier and job.stats.pages_crawled < max_pages:
            url, depth = frontier.popleft()

            if depth > max_depth:
                continue

            if self.db.url_already_crawled(url):
                continue

            # Check robots.txt
            if self.respect_robots and not self.robots_parser.is_allowed(url):
                job.stats.pages_skipped_robots += 1
                logger.debug("Blocked by robots.txt: %s", url)
                continue

            result = self._fetch_page(url, depth)
            if result is None:
                job.stats.pages_failed += 1
                continue

            doc_id = self._store_and_index(result)
            job.stats.pages_crawled += 1
            if doc_id:
                job.stats.pages_indexed += 1

            # Add discovered links to frontier
            for link in result.links:
                normalized_link = normalize_url(link, base_url=url)
                if normalized_link is None:
                    continue
                if normalized_link in visited:
                    continue
                if stay_on_domain and not self._is_allowed_domain(normalized_link, seed_urls):
                    continue

                visited.add(normalized_link)
                frontier.append((normalized_link, depth + 1))
                job.stats.links_discovered += 1

            # Politeness delay
            if self.request_delay > 0:
                robots_delay = self.robots_parser.get_crawl_delay(url)
                delay = max(self.request_delay, robots_delay or 0)
                time.sleep(delay)

        job.stats.end_time = time.time()
        job.is_running = False
        job.is_complete = True

        logger.info("Crawl complete: %s", job.stats.to_dict())
        return job.stats

    def _fetch_page(self, url: str, depth: int) -> CrawlResult | None:
        """Fetch a single page and extract its content."""
        try:
            response = requests.get(
                url,
                timeout=self.timeout,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "text/html,application/xhtml+xml",
                },
                allow_redirects=True,
            )
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                logger.debug("Skipping non-HTML: %s (%s)", url, content_type)
                return None

            html = response.text
            soup = BeautifulSoup(html, "html.parser")

            title = self._extract_title(soup)
            text = self._extract_text(soup)
            links = self._extract_links(soup, url)

            logger.debug("Fetched %s: %d chars, %d links", url, len(text), len(links))

            return CrawlResult(
                url=url, title=title, content=text,
                html=html, links=links,
                status_code=response.status_code, depth=depth
            )

        except requests.RequestException as e:
            logger.warning("Failed to fetch %s: %s", url, e)
            return None

    def _extract_title(self, soup: BeautifulSoup) -> str:
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
        # Collapse whitespace
        lines = (line.strip() for line in text.splitlines())
        return " ".join(chunk for chunk in lines if chunk)

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        """Extract all href links from anchor tags, filtering unsafe schemes."""
        _SAFE_SCHEMES = ("http://", "https://", "/", "./", "../")
        links: list[str] = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            # Skip non-navigable and unsafe URI schemes
            if href.startswith((
                "javascript:", "mailto:", "tel:", "#", "data:", "vbscript:",
            )):
                continue
            full_url = normalize_url(href, base_url=base_url)
            if full_url:
                links.append(full_url)
        return links

    def _store_and_index(self, result: CrawlResult) -> int | None:
        """Store a crawled page and index its content."""
        try:
            index_result = self.indexer.index_document(
                title=result.title,
                content=result.content,
                source=result.url,
                doc_type="web"
            )
            doc_id = index_result.doc_id

            self.db.insert_crawled_page(
                url=result.url,
                title=result.title,
                content=result.content,
                html=result.html,
                status_code=result.status_code,
                crawl_depth=result.depth,
                doc_id=doc_id
            )

            return doc_id

        except Exception as e:
            logger.error("Failed to store/index %s: %s", result.url, e)
            return None

    def _is_allowed_domain(self, url: str, seed_urls: list[str]) -> bool:
        """Check if URL belongs to one of the seed domains."""
        return any(is_same_domain(url, seed) for seed in seed_urls)

    def get_status(self) -> dict:
        if self.current_job is None:
            return {"status": "idle", "stats": None}
        return {
            "status": "running" if self.current_job.is_running else "complete",
            "seeds": self.current_job.seed_urls,
            "max_depth": self.current_job.max_depth,
            "max_pages": self.current_job.max_pages,
            "stats": self.current_job.stats.to_dict(),
        }
