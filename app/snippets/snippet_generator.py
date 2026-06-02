"""
Search Result Snippet Generator

=== THEORY ===

A snippet is the short excerpt shown below a search result title.
Good snippets answer: "Does this document actually answer my query?"

The algorithm:
  1. Find every position in the document where a query term appears.
  2. Build "windows" — contiguous spans of N words centred on a hit.
  3. Merge overlapping windows into fragments.
  4. Score each fragment by the number of distinct query terms it covers.
  5. Pick the top-K fragments.
  6. Bold-wrap the matching terms: **term**.

=== WHY NOT JUST TRUNCATE ===

A naive "first 200 chars" approach often returns the document header or
navigation text that doesn't mention the query at all.  Hit-centred
snippets are significantly more useful to users.

=== COMPLEXITY ===

  Finding hits:     O(L)  where L = document length (word count)
  Building windows: O(H)  where H = number of hit positions
  Merging windows:  O(H log H) — sort then linear scan
  Total:            O(L + H log H)   ≈ O(L) in practice

=== AT GOOGLE SCALE ===

Google's snippet pipeline:
  - Runs on a dedicated serving layer separate from ranking
  - Uses passage-level indexing (passages retrieved independently from docs)
  - Neural models (Gemini/PaLM) generate dynamic, question-answering snippets
  - Supports rich snippets with schema.org markup
  - Answer boxes ("featured snippets") extract the single best passage
"""

import re
import logging
from dataclasses import dataclass

from app.config import SnippetConfig

logger = logging.getLogger(__name__)


@dataclass
class Snippet:
    text: str            # rendered text with **highlights**
    score: float         # fraction of query terms covered by this snippet


class SnippetGenerator:
    """
    Generates contextual, highlighted snippets for search results.

    Usage:
        gen = SnippetGenerator(config)
        snippet = gen.generate("Python is great for web dev ...", ["python", "web"])
        # → "**Python** is great for **web** development and ..."
    """

    _BOLD_OPEN  = "**"
    _BOLD_CLOSE = "**"
    # Process at most this many characters of raw content before windowing.
    # Prevents building a 50k-element word list for large crawled pages.
    _MAX_CONTENT_CHARS = 50_000

    def __init__(self, config: SnippetConfig | None = None):
        self.config = config or SnippetConfig()

    def generate(self, content: str, query_terms: list[str]) -> str:
        """
        Return the best snippet string for `content` given `query_terms`.
        Falls back to the first `max_length` characters if no terms match.
        """
        if not query_terms or not content:
            return self._truncate(content)

        # Cap content length before building the word list to avoid O(500k)
        # processing on large crawled pages.
        if len(content) > self._MAX_CONTENT_CHARS:
            content = content[: self._MAX_CONTENT_CHARS]

        query_set = {t.lower() for t in query_terms}
        words     = self._split_words(content)

        if not words:
            return self._truncate(content)

        hit_positions = self._find_hits(words, query_set)

        if not hit_positions:
            return self._truncate(content)

        windows   = self._build_windows(hit_positions, len(words))
        merged    = self._merge_windows(windows)
        fragments = self._rank_fragments(merged, words, query_set)
        selected  = fragments[: self.config.max_fragments]

        parts = []
        for start, end in selected:
            raw = " ".join(words[start:end])
            parts.append(self._highlight(raw, query_set))

        joined = " … ".join(parts)
        if len(joined) > self.config.max_length:
            joined = joined[: self.config.max_length].rsplit(" ", 1)[0] + " …"
        return joined

    def generate_rich(self, content: str, query_terms: list[str]) -> Snippet:
        """Return a Snippet dataclass with score metadata."""
        text = self.generate(content, query_terms)
        terms_found = sum(
            1 for t in query_terms if t.lower() in content.lower()
        )
        score = terms_found / len(query_terms) if query_terms else 0.0
        return Snippet(text=text, score=round(score, 3))

    # ── Internals ────────────────────────────────────────────────────────────

    def _split_words(self, text: str) -> list[str]:
        """Split on whitespace, preserving original casing."""
        return re.split(r"\s+", text.strip())

    def _find_hits(self, words: list[str], query_set: set[str]) -> list[int]:
        """Return indices of words that match any query term (case-insensitive)."""
        return [
            i for i, w in enumerate(words)
            if re.sub(r"[^\w]", "", w).lower() in query_set
        ]

    def _build_windows(self, hits: list[int], doc_len: int) -> list[tuple[int, int]]:
        ctx = self.config.context_words
        return [
            (max(0, h - ctx), min(doc_len, h + ctx + 1))
            for h in hits
        ]

    def _merge_windows(
        self, windows: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Merge overlapping (start, end) windows into contiguous spans."""
        if not windows:
            return []
        sorted_w = sorted(windows)
        merged = [sorted_w[0]]
        for start, end in sorted_w[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        return merged

    def _rank_fragments(
        self,
        windows: list[tuple[int, int]],
        words: list[str],
        query_set: set[str],
    ) -> list[tuple[int, int]]:
        """Sort windows by how many distinct query terms they cover, desc."""
        def coverage(span: tuple[int, int]) -> int:
            start, end = span
            present = {
                re.sub(r"[^\w]", "", w).lower()
                for w in words[start:end]
            } & query_set
            return len(present)

        return sorted(windows, key=coverage, reverse=True)

    def _highlight(self, text: str, query_set: set[str]) -> str:
        """
        Wrap matching words with **bold**.
        The compiled regex is cached by frozenset(query_set) so it is not
        recompiled on every snippet call for the same query terms.
        """
        cache_key = frozenset(query_set)
        if not hasattr(self, "_highlight_cache"):
            self._highlight_cache: dict[frozenset, re.Pattern] = {}  # type: ignore[assignment]
        if cache_key not in self._highlight_cache:
            self._highlight_cache[cache_key] = re.compile(
                r"\b(" + "|".join(re.escape(t) for t in query_set) + r")\b",
                flags=re.IGNORECASE,
            )
        pattern = self._highlight_cache[cache_key]
        return pattern.sub(
            lambda m: f"{self._BOLD_OPEN}{m.group(0)}{self._BOLD_CLOSE}", text
        )

    def _truncate(self, text: str) -> str:
        limit = self.config.max_length
        if len(text) <= limit:
            return text
        return text[:limit].rsplit(" ", 1)[0] + " …"
