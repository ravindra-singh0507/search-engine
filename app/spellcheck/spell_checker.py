"""
Spell Checker

Combines BK-tree lookup (fast candidate retrieval) with Levenshtein
scoring to produce ranked correction suggestions.

=== PIPELINE ===

  Input: misspelled word (e.g. "pythn")

  Step 1 – Exact match?  If the word is already in the vocabulary → no correction.

  Step 2 – BK-tree query with max_distance = config.max_edit_distance.
            Returns all vocabulary words within the edit distance threshold.

  Step 3 – Re-rank candidates by:
              (a) edit distance (lower = better)
              (b) word frequency in the index (higher = better)

  Step 4 – Return top-N suggestions with a confidence score:
              confidence = (1 - edit_distance / max_edit_distance)
              in range (0, 1].

=== VOCABULARY BOOTSTRAP ===

The spell checker builds its vocabulary from the search index terms.
This means it only suggests words that are actually in the corpus —
avoiding "corrections" to words the user didn't index.

=== COMPLEXITY ===

  Build vocabulary from N terms:   O(N log N)  — BK-tree build
  Spell-check one word:            O(V^0.36)   — BK-tree search
  With max_distance=2, empirical:  ~50–500 candidates per lookup on a
                                   typical English vocabulary of 100k words
"""

import logging
from dataclasses import dataclass

from app.spellcheck.bk_tree import BKTree
from app.config import SpellCheckConfig

logger = logging.getLogger(__name__)

# Boolean operators and field-search keywords that must never be corrected.
# "AND" → "end", "OR" → something else, "NOT" → "not" (fine but prevents
# subtle semantic changes).  Field prefixes (title:, url:) are split before
# they reach correct_query so they don't need special handling here.
_PROTECTED_TOKENS: frozenset[str] = frozenset({"AND", "OR", "NOT"})


@dataclass
class CorrectionSuggestion:
    original: str
    suggestion: str
    edit_distance: int
    confidence: float   # 1.0 = perfect match, → 0 as distance grows


class SpellChecker:
    """
    Vocabulary-backed spell checker using a BK-tree.

    Typical usage:
        checker = SpellChecker(config)
        checker.build_vocabulary(db.get_all_terms())
        suggestions = checker.correct("pythn")
        # → [CorrectionSuggestion("pythn","python",1,0.5)]
    """

    def __init__(self, config: SpellCheckConfig | None = None):
        self.config = config or SpellCheckConfig()
        self._tree = BKTree()
        self._vocab: set[str] = set()

    def build_vocabulary(self, words: list[str]) -> None:
        """
        Build the BK-tree from a list of vocabulary words.
        This method is ADDITIVE — call clear_vocabulary() first to rebuild
        from scratch (e.g. after document deletion).
        """
        count = 0
        for word in words:
            w = word.lower().strip()
            if w and len(w) >= self.config.min_word_length:
                if w not in self._vocab:
                    self._tree.insert(w)
                    self._vocab.add(w)
                    count += 1
        logger.info("SpellChecker: built vocabulary of %d words", count)

    def clear_vocabulary(self) -> None:
        """
        Discard the current BK-tree and vocabulary set.
        Call this before rebuild_vocabulary() after index deletions so that
        removed terms no longer appear as spell-correction candidates.
        """
        self._tree  = BKTree()
        self._vocab = set()
        logger.info("SpellChecker: vocabulary cleared")

    def correct(self, word: str, top_n: int = 5) -> list[CorrectionSuggestion]:
        """
        Return correction suggestions for `word`.
        Returns [] if the word is already correct or the vocabulary is empty.
        """
        word_lower = word.lower().strip()
        if not word_lower or len(word_lower) < self.config.min_word_length:
            return []

        # Already a known word
        if word_lower in self._vocab:
            return []

        candidates = self._tree.search(
            word_lower, max_distance=self.config.max_edit_distance
        )
        if not candidates:
            return []

        max_d = self.config.max_edit_distance
        suggestions = [
            CorrectionSuggestion(
                original=word,
                suggestion=candidate,
                edit_distance=dist,
                confidence=round(1.0 - dist / (max_d + 1), 4),
            )
            for candidate, dist in candidates
        ]

        # Sort: best confidence first; ties broken by suggestion length
        suggestions.sort(key=lambda s: (-s.confidence, len(s.suggestion)))
        return suggestions[:top_n]

    def correct_query(self, query: str) -> str:
        """
        Attempt to auto-correct each token in a multi-word query.
        Only replaces a token if correction confidence ≥ 0.5.
        Returns the possibly-corrected query string.

        Boolean operators (AND, OR, NOT) are never spell-corrected — they
        would otherwise be corrupted into real words (e.g. "AND" → "end").
        """
        tokens = query.strip().split()
        corrected = []
        for token in tokens:
            # Never spell-correct Boolean operators
            if token.upper() in _PROTECTED_TOKENS:
                corrected.append(token)
                continue
            suggestions = self.correct(token, top_n=1)
            if suggestions and suggestions[0].confidence >= 0.5:
                corrected.append(suggestions[0].suggestion)
            else:
                corrected.append(token)
        return " ".join(corrected)

    def is_known(self, word: str) -> bool:
        return word.lower().strip() in self._vocab

    @property
    def vocabulary_size(self) -> int:
        return len(self._vocab)
