"""
Autocomplete via Trie

=== THEORY ===

A Trie (prefix tree) is a tree where each node represents one character.
A path from root to a marked node spells out a complete word.

Structure:
  root
   ├── 'p' → 'y' → 't' → 'h' → 'o' → 'n'  [END, freq=42]
   │                             └── 'o' → 'n' [END, freq=5]
   └── 'j' → 'a' → 'v' → 'a'  [END, freq=18]

PREFIX SEARCH:
  1. Walk down the trie following prefix characters.
  2. If we fall off → no completions.
  3. Otherwise DFS/BFS from that node collecting all END nodes.
  4. Sort collected words by frequency descending, return top-k.

=== COMPLEXITY ===

  Insert word of length L:        O(L)
  Prefix search prefix of length P, vocabulary V, suggestions K:
    - Navigate to prefix node:    O(P)
    - Collect completions:        O(V)  worst case (all words share prefix)
    - Sort:                       O(K log K)  (only top-k needed — heap)
  Space: O(total characters in vocabulary)

For V=100 000 and typical branching factor the practical depth is ~10.

=== PERSISTENCE ===

The trie is serialised to JSON on disk so it survives restarts.
Schema: { "word": freq, ... } flat dict — we rebuild the in-memory
trie on startup.  This keeps the serialised format simple and
human-readable while the trie provides O(P) prefix lookup in memory.

=== AT GOOGLE SCALE ===

Google's autocomplete (Search Suggest) uses:
  - A distributed prefix-tree sharded by first character
  - Real-time update pipeline (new trends visible in minutes)
  - Personalisation layer on top of global suggestions
  - Policy filters (block harmful/private suggestions)
  - A/B testing framework for ranking model updates
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from app.config import AutocompleteConfig

logger = logging.getLogger(__name__)


# ── Trie Node ─────────────────────────────────────────────────────────────────


class TrieNode:
    __slots__ = ("children", "is_end", "frequency", "word")

    def __init__(self) -> None:
        self.children: dict[str, "TrieNode"] = {}
        self.is_end: bool = False
        self.frequency: int = 0
        self.word: str | None = None   # full word stored at end node


# ── Trie ──────────────────────────────────────────────────────────────────────


class Trie:
    """
    In-memory prefix tree for fast autocomplete suggestions.

    Thread-safety: single-writer / multiple-reader (good enough for a
    single-process FastAPI server).
    """

    def __init__(self) -> None:
        self.root = TrieNode()
        self._size: int = 0

    # ── Mutations ─────────────────────────────────────────────────────────

    def insert(self, word: str, frequency: int = 1) -> None:
        """
        Insert (or update) a word.
        If the word already exists its frequency is *added to*, not replaced.
        """
        if not word:
            return
        word = word.lower().strip()
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        if not node.is_end:
            self._size += 1
        node.is_end = True
        node.frequency += frequency
        node.word = word

    def increment(self, word: str, delta: int = 1) -> None:
        """Increment the frequency of an existing word (no-op if absent)."""
        node = self._find_node(word.lower().strip())
        if node and node.is_end:
            node.frequency += delta

    def remove(self, word: str) -> bool:
        """Remove a word from the trie. Returns True if it existed."""
        return self._remove(self.root, word.lower().strip(), 0)

    def _remove(self, node: TrieNode, word: str, depth: int) -> bool:
        if depth == len(word):
            if not node.is_end:
                return False
            node.is_end = False
            node.word = None
            node.frequency = 0
            self._size -= 1
            return True
        ch = word[depth]
        if ch not in node.children:
            return False
        deleted = self._remove(node.children[ch], word, depth + 1)
        if deleted and not node.children[ch].children and not node.children[ch].is_end:
            del node.children[ch]
        return deleted

    # ── Queries ───────────────────────────────────────────────────────────

    def search_prefix(self, prefix: str, top_k: int = 10) -> list[tuple[str, int]]:
        """
        Return up to top_k (word, frequency) pairs whose prefix matches.
        Results are sorted by frequency descending.

        Returns [] if no words match.
        """
        prefix = prefix.lower().strip()
        if not prefix:
            return []

        node = self._find_node(prefix)
        if node is None:
            return []

        # Collect all completions via DFS
        results: list[tuple[str, int]] = []
        self._collect(node, results)

        # Partial sort — only need top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def contains(self, word: str) -> bool:
        node = self._find_node(word.lower().strip())
        return node is not None and node.is_end

    @property
    def size(self) -> int:
        return self._size

    # ── Internals ─────────────────────────────────────────────────────────

    def _find_node(self, prefix: str) -> TrieNode | None:
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def _collect(self, node: TrieNode, results: list[tuple[str, int]]) -> None:
        """DFS to collect all end-nodes reachable from `node`."""
        if node.is_end and node.word:
            results.append((node.word, node.frequency))
        for child in node.children.values():
            self._collect(child, results)

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, int]:
        """Flatten trie to {word: frequency} for JSON persistence."""
        result: dict[str, int] = {}
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_end and node.word:
                result[node.word] = node.frequency
            stack.extend(node.children.values())
        return result

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "Trie":
        trie = cls()
        for word, freq in data.items():
            trie.insert(word, freq)
        return trie


# ── Autocomplete Service ──────────────────────────────────────────────────────


class AutocompleteService:
    """
    High-level service: manages a Trie, persists to disk, and provides
    the suggest() method consumed by the API endpoint.
    """

    def __init__(self, config: AutocompleteConfig | None = None):
        self.config = config or AutocompleteConfig()
        self._trie = Trie()
        self._loaded = False

    def load(self) -> None:
        """Load the trie from disk (call once on startup)."""
        path = self.config.persist_path
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._trie = Trie.from_dict(data)
                logger.info("Autocomplete: loaded %d words from %s", self._trie.size, path)
            except Exception as exc:
                logger.warning("Failed to load autocomplete trie: %s", exc)
        self._loaded = True

    def save(self) -> None:
        """Persist the current trie to disk."""
        path = self.config.persist_path
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = self._trie.to_dict()
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            logger.debug("Autocomplete: saved %d words to %s", len(data), path)
        except Exception as exc:
            logger.warning("Failed to save autocomplete trie: %s", exc)

    def index_query(self, query: str) -> None:
        """Record a user query so it becomes a suggestion for others."""
        for term in query.lower().split():
            if len(term) >= 2:
                self._trie.increment(term)
                if not self._trie.contains(term):
                    self._trie.insert(term, 1)

        # Also index the full multi-word query
        cleaned = query.lower().strip()
        if cleaned:
            if self._trie.contains(cleaned):
                self._trie.increment(cleaned)
            else:
                self._trie.insert(cleaned, 1)

    def seed_from_vocabulary(self, words: list[str]) -> None:
        """
        Bootstrap the trie from the search index vocabulary.
        Called once after the index is populated so that every indexed
        term is immediately autocomplete-able.
        """
        for word in words:
            if not self._trie.contains(word):
                self._trie.insert(word, 1)
        logger.info("Autocomplete: seeded trie with %d vocabulary words", len(words))

    def suggest(self, prefix: str, top_k: int | None = None) -> list[dict]:
        """
        Return autocomplete suggestions for a prefix.

        Response shape: [{"suggestion": str, "frequency": int}, ...]
        """
        k = top_k or self.config.max_suggestions
        hits = self._trie.search_prefix(prefix, top_k=k)
        return [{"suggestion": word, "frequency": freq} for word, freq in hits]

    @property
    def vocabulary_size(self) -> int:
        return self._trie.size
