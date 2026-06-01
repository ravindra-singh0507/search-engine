"""
Robots.txt Parser

=== THEORY ===

robots.txt is a protocol (Robots Exclusion Protocol) that tells web crawlers
which pages they're allowed to crawl. It sits at the root of every website:
    https://example.com/robots.txt

Format example:
    User-agent: *
    Disallow: /private/
    Disallow: /api/
    Allow: /api/public/
    Crawl-delay: 2

Rules:
- User-agent: which bots this section applies to (* = all bots)
- Disallow: paths the bot must NOT crawl
- Allow: exceptions to Disallow rules
- Crawl-delay: seconds to wait between requests

=== WHY RESPECT ROBOTS.TXT ===

1. Legal: Ignoring robots.txt can have legal consequences
2. Ethical: It's the site owner's right to control crawling
3. Practical: Aggressive crawling can get your IP banned
4. Community: Respectful crawling maintains the open web

=== AT GOOGLE SCALE ===

Google's crawler (Googlebot) checks robots.txt before every crawl.
They cache robots.txt files and refresh them periodically.
Google also supports sitemap directives in robots.txt.
"""

import logging
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)


@dataclass
class RobotsRule:
    path: str
    allowed: bool


@dataclass
class RobotsData:
    rules: list[RobotsRule] = field(default_factory=list)
    crawl_delay: float | None = None
    sitemaps: list[str] = field(default_factory=list)


class RobotsParser:
    """
    Fetches and parses robots.txt, then checks whether a URL is allowed.
    Caches parsed results per domain.
    """

    def __init__(self, user_agent: str = "*"):
        self.user_agent = user_agent
        self._cache: dict[str, RobotsData] = {}

    def fetch_and_parse(self, base_url: str) -> RobotsData:
        domain = urlparse(base_url).hostname
        if domain in self._cache:
            return self._cache[domain]

        robots_url = urljoin(base_url, "/robots.txt")
        data = RobotsData()

        try:
            response = requests.get(robots_url, timeout=10,
                                    headers={"User-Agent": self.user_agent})
            if response.status_code == 200:
                data = self._parse_content(response.text)
                logger.info("Parsed robots.txt for %s: %d rules", domain, len(data.rules))
            else:
                logger.debug("No robots.txt for %s (status %d)", domain, response.status_code)
        except requests.RequestException as e:
            logger.warning("Failed to fetch robots.txt for %s: %s", domain, e)

        self._cache[domain] = data
        return data

    def is_allowed(self, url: str) -> bool:
        """Check if our user-agent is allowed to crawl this URL."""
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        data = self.fetch_and_parse(base_url)

        path = parsed.path or "/"

        # Sort rules by specificity (longest path first)
        matching_rules = [r for r in data.rules if path.startswith(r.path)]
        if not matching_rules:
            return True

        matching_rules.sort(key=lambda r: len(r.path), reverse=True)
        return matching_rules[0].allowed

    def get_crawl_delay(self, base_url: str) -> float | None:
        data = self.fetch_and_parse(base_url)
        return data.crawl_delay

    def _parse_content(self, content: str) -> RobotsData:
        """
        Parse robots.txt content.
        We only care about rules matching our user-agent or *.
        """
        data = RobotsData()
        applies_to_us = False

        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if ":" not in line:
                continue

            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip()

            if key == "user-agent":
                applies_to_us = (
                    value == "*" or
                    value.lower() == self.user_agent.lower()
                )
            elif applies_to_us:
                if key == "disallow" and value:
                    data.rules.append(RobotsRule(path=value, allowed=False))
                elif key == "allow" and value:
                    data.rules.append(RobotsRule(path=value, allowed=True))
                elif key == "crawl-delay":
                    try:
                        data.crawl_delay = float(value)
                    except ValueError:
                        pass
                elif key == "sitemap":
                    data.sitemaps.append(value)

        return data
