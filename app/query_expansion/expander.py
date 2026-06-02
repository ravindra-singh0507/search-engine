"""
Query Expansion via Synonym Dictionary

=== THEORY ===

Query expansion improves recall by augmenting the user's query with
semantically related terms before retrieval.

  User types:   "car repair"
  Expanded:     "car repair automobile vehicle"
                → matches documents that say "automobile" even if
                  they never use the word "car"

=== TYPES OF EXPANSION ===

  1. Manual (this implementation) — human-curated synonym dictionary.
     Pros: deterministic, no false positives, auditable.
     Cons: limited coverage, maintenance burden.

  2. Thesaurus-based — WordNet / PPDB (automatic synonym extraction).

  3. Co-occurrence-based — mine query logs for terms that users
     substitute when they don't find results with the original term.

  4. Neural / embedding-based — expand using word2vec / BERT nearest
     neighbours in embedding space.  This is Phase 4 material.

=== EXPANSION STRATEGY ===

We expand only individual terms, not the full query string, so:
  - Boolean operators are preserved.
  - Phrase queries ("machine learning") are expanded term-by-term.
  - Expansion is OR-joined with the original term in retrieval.
  - We cap expansion to avoid query-too-large slowdowns.

=== AT GOOGLE SCALE ===

Google uses multiple expansion signals:
  - Knowledge Graph entity synonyms
  - Spell-correction corpus
  - User reformulation click-through data
  - Multilingual translation (cross-lingual retrieval)
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class QueryExpander:
    """
    Expands query terms with synonyms from a JSON dictionary.

    The JSON file format:
      { "term": ["syn1", "syn2", ...], ... }

    Usage:
        expander = QueryExpander(Path("app/query_expansion/synonyms.json"))
        expander.expand(["car", "repair"])
        # → ["car", "repair", "automobile", "vehicle"]
    """

    def __init__(self, synonyms_path: Path | None = None):
        self._synonyms: dict[str, list[str]] = {}
        if synonyms_path and synonyms_path.exists():
            self.load(synonyms_path)

    def load(self, path: Path) -> None:
        """Load synonym dictionary from a JSON file."""
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            # Normalise keys and values to lowercase
            self._synonyms = {
                k.lower(): [v.lower() for v in vs]
                for k, vs in raw.items()
            }
            logger.info(
                "QueryExpander: loaded %d synonym entries from %s",
                len(self._synonyms), path,
            )
        except Exception as exc:
            logger.warning("Failed to load synonyms from %s: %s", path, exc)

    def expand(self, terms: list[str], max_expansions_per_term: int = 3) -> list[str]:
        """
        Return the original terms plus synonyms.
        Original order is preserved; synonyms are appended.
        Duplicates are removed while preserving order.
        """
        seen: set[str] = set()
        expanded: list[str] = []
        for t in terms:
            if t not in seen:
                expanded.append(t)
                seen.add(t)

        for term in terms:
            syns = self._synonyms.get(term.lower(), [])
            for syn in syns[:max_expansions_per_term]:
                if syn not in seen:
                    expanded.append(syn)
                    seen.add(syn)

        if len(expanded) > len(terms):
            logger.debug(
                "Query expanded: %s → %s", terms, expanded[len(terms):]
            )

        return expanded

    def get_synonyms(self, term: str) -> list[str]:
        """Return synonyms for a single term (empty list if none)."""
        return self._synonyms.get(term.lower(), [])

    def add_synonym(self, term: str, synonym: str) -> None:
        """Add a synonym at runtime (not persisted to disk)."""
        t = term.lower()
        if t not in self._synonyms:
            self._synonyms[t] = []
        if synonym.lower() not in self._synonyms[t]:
            self._synonyms[t].append(synonym.lower())

    @property
    def dictionary_size(self) -> int:
        return len(self._synonyms)
