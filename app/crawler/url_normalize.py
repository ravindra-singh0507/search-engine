"""
URL Normalization

=== THEORY ===

The same webpage can be referenced by many different URLs:
    https://example.com/page
    https://example.com/page/
    https://example.com/page#section
    https://example.com/page?utm_source=twitter
    https://EXAMPLE.COM/page

Without normalization, we'd crawl and index the same page multiple times,
wasting bandwidth and inflating our index.

=== RULES IMPLEMENTED ===

1. Lowercase the scheme and host: HTTPS://EXAMPLE.COM → https://example.com
2. Remove default ports: :80 for HTTP, :443 for HTTPS
3. Remove fragment identifiers: /page#section → /page
4. Remove trailing slash: /page/ → /page (except root /)
5. Resolve relative URLs: ./about → https://example.com/about
6. Sort query parameters: ?b=2&a=1 → ?a=1&b=2
7. Remove common tracking parameters: utm_source, utm_medium, etc.

=== AT GOOGLE SCALE ===

Google deals with canonicalization at massive scale:
- rel="canonical" tags tell Google which URL is the "official" version
- Hreflang tags handle language variants
- Redirect chains are followed and consolidated
- Content fingerprinting detects duplicates even with different URLs
"""

import logging
from urllib.parse import (
    urlparse, urlunparse, urljoin, parse_qs, urlencode, ParseResult
)

logger = logging.getLogger(__name__)

TRACKING_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "ref", "source",
})


def normalize_url(url: str, base_url: str | None = None) -> str | None:
    """
    Normalize a URL to its canonical form.
    Returns None if the URL is not a valid HTTP(S) URL.
    """
    if not url or not url.strip():
        return None

    url = url.strip()

    if base_url:
        url = urljoin(base_url, url)

    try:
        parsed = urlparse(url)
    except ValueError:
        return None

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        return None

    hostname = parsed.hostname
    if not hostname:
        return None
    hostname = hostname.lower()

    port = parsed.port
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        port = None

    netloc = hostname
    if port:
        netloc = f"{hostname}:{port}"

    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    fragment = ""

    query = parsed.query
    if query:
        params = parse_qs(query, keep_blank_values=True)
        filtered = {
            k: v for k, v in params.items() if k.lower() not in TRACKING_PARAMS
        }
        query = urlencode(sorted(filtered.items()), doseq=True)

    normalized = urlunparse((scheme, netloc, path, parsed.params, query, fragment))
    return normalized


def is_same_domain(url: str, base_url: str) -> bool:
    """Check if two URLs share the same domain."""
    try:
        return urlparse(url).hostname == urlparse(base_url).hostname
    except (ValueError, AttributeError):
        return False


def extract_domain(url: str) -> str | None:
    """Extract the domain from a URL."""
    try:
        return urlparse(url).hostname
    except (ValueError, AttributeError):
        return None
