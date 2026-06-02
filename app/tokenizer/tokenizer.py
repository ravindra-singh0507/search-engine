"""
Tokenizer

=== THEORY ===

Tokenization is the first step in any search engine pipeline. It transforms raw text
into a sequence of normalized terms that can be indexed and searched.

The pipeline is:  raw text → split into words → lowercase → remove punctuation
                  → remove stop words → filter by length → final tokens

WHY each step matters:
- Lowercasing: "Python" and "python" should match the same documents.
- Punctuation removal: "hello," and "hello" are the same word.
- Stop word removal: Words like "the", "is", "at" appear in almost every document.
  They waste index space and don't help distinguish documents from each other.
  A query for "the python language" should really just search for "python language".
- Length filtering: Single characters and extremely long strings are rarely useful terms.

=== COMPLEXITY ===

- Tokenize a document of length L: O(L)
- Stop word lookup: O(1) per word (using a set)
- Total for N documents: O(N * avg_doc_length)

=== TRADEOFFS ===

- Aggressive stop word removal can hurt phrase queries ("to be or not to be")
- Stemming (not implemented here) would merge "running" and "run" but can
  create false matches ("universe" and "university" both stem to "univers")
- Lemmatization is more accurate but slower

=== AT GOOGLE SCALE ===

Google uses:
- Language-specific tokenizers (CJK languages don't use spaces)
- Subword tokenization (BPE/WordPiece) for neural models
- Stemming + synonym expansion at query time
- Custom tokenizers per content type (code, math, addresses)
"""

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "not", "but", "with", "by", "from", "as",
    "be", "was", "were", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "shall", "can", "this", "that", "these", "those",
    "am", "are", "if", "then", "than", "so", "no", "nor",
    "up", "out", "about", "into", "over", "after", "before",
    "between", "under", "above", "such", "each", "which", "their",
    "we", "he", "she", "they", "me", "him", "her", "us", "them",
    "my", "his", "its", "our", "your", "all", "any", "both",
    "few", "more", "most", "other", "some", "what", "when", "where",
    "who", "how", "i", "you", "just", "also", "very", "too",
})

# Strip everything that is not a letter, digit, or whitespace.
# Using [^\w\s] kept underscores (part of \w); we explicitly exclude them
# so __init__, some_var etc. are split into meaningful tokens.
PUNCTUATION_PATTERN = re.compile(r"[^\w\s]|_", re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+")
# Pure-digit tokens (e.g. "123") add noise; keep only tokens with ≥1 letter.
HAS_LETTER_PATTERN = re.compile(r"[a-zA-Z]")


@dataclass
class TokenizerConfig:
    min_token_length: int = 2
    max_token_length: int = 50
    custom_stop_words: list[str] = field(default_factory=list)


@dataclass
class TokenizeResult:
    tokens: list[str]
    positions: dict[str, list[int]]
    token_count: int
    unique_count: int


class Tokenizer:
    """
    Converts raw text into normalized, searchable tokens.

    Pipeline: lowercase → remove punctuation → split → remove stop words → filter by length
    """

    def __init__(self, config: TokenizerConfig | None = None):
        self.config = config or TokenizerConfig()
        self.stop_words: frozenset[str] = DEFAULT_STOP_WORDS | frozenset(
            w.lower() for w in self.config.custom_stop_words
        )
        logger.info(
            "Tokenizer initialized: stop_words=%d, min_len=%d, max_len=%d",
            len(self.stop_words), self.config.min_token_length,
            self.config.max_token_length
        )

    def tokenize(self, text: str) -> TokenizeResult:
        """
        Full tokenization pipeline.

        Returns tokens with their positions, which we need for the inverted index.
        Positions let us support phrase queries and proximity scoring later.
        """
        if not text or not text.strip():
            return TokenizeResult(tokens=[], positions={}, token_count=0, unique_count=0)

        lowered = text.lower()

        cleaned = PUNCTUATION_PATTERN.sub(" ", lowered)

        raw_tokens = WHITESPACE_PATTERN.split(cleaned.strip())

        tokens: list[str] = []
        positions: dict[str, list[int]] = {}
        position = 0

        for raw in raw_tokens:
            if not raw:
                continue

            if raw in self.stop_words:
                position += 1
                continue

            if len(raw) < self.config.min_token_length:
                position += 1
                continue

            if len(raw) > self.config.max_token_length:
                position += 1
                continue

            if not any(c.isalnum() for c in raw):
                position += 1
                continue

            # Skip pure-numeric tokens (e.g. "123", "2024") — they're noise
            # in general-purpose document search.
            if not HAS_LETTER_PATTERN.search(raw):
                position += 1
                continue

            tokens.append(raw)
            if raw not in positions:
                positions[raw] = []
            positions[raw].append(position)
            position += 1

        unique_terms = set(tokens)
        logger.debug("Tokenized: %d tokens, %d unique", len(tokens), len(unique_terms))

        return TokenizeResult(
            tokens=tokens,
            positions=positions,
            token_count=len(tokens),
            unique_count=len(unique_terms)
        )

    def get_term_frequencies(self, tokens: list[str]) -> dict[str, int]:
        """Count how many times each term appears in a token list."""
        freq: dict[str, int] = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1
        return freq
