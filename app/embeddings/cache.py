"""
Embedding Cache

=== THEORY ===

Embedding is the most expensive operation in the semantic retrieval pipeline
(10-50 ms per text on CPU).  If the same text is re-indexed (e.g. server
restart, re-crawl of unchanged pages), recomputing the embedding wastes time.

We cache embeddings keyed by SHA-256(model_name + "|" + text).
The hash function guarantees:
  - Same text + model → same key → cache hit
  - Different text → different key (with overwhelming probability)
  - Model change is automatically handled (different model_name → different key)

Storage: SQLite `embedding_cache` table (persistent across restarts).

=== COMPLEXITY ===

  Cache lookup:   O(1)  — primary key index on content_hash
  Cache insert:   O(1) amortised
  Cache size:     O(N · D · 4 bytes)  where N = cached texts, D = dimension
                  For bge-small (384 dims): ~1.5 KB per entry
                  10 000 entries ≈ 15 MB of JSON text in SQLite

=== TRADEOFFS ===

  Pro:  Eliminates redundant embedding computation on re-index
  Con:  Slightly slower first-time indexing (extra DB write per chunk)
  Con:  Cache can grow large; periodic eviction is recommended for huge corpora
"""

import hashlib
import logging
from typing import Optional

from app.database.db import Database

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    SQLite-backed content-addressable cache for embedding vectors.
    Thread-safe via SQLite's WAL mode.
    """

    def __init__(self, db: Database):
        self.db = db
        self._hits   = 0
        self._misses = 0

    # ── Public API ────────────────────────────────────────────────────────

    def get(self, text: str, model_name: str) -> Optional[list[float]]:
        """Return cached vector or None if not cached."""
        key = self._hash(text, model_name)
        vector = self.db.get_cached_embedding(key, model_name)
        if vector is not None:
            self._hits += 1
            return vector
        self._misses += 1
        return None

    def put(self, text: str, model_name: str, vector: list[float]) -> None:
        """Store an embedding in the cache."""
        key = self._hash(text, model_name)
        self.db.cache_embedding(key, model_name, vector)

    def clear(self) -> None:
        self.db.clear_embedding_cache()
        self._hits = self._misses = 0
        logger.info("EmbeddingCache cleared")

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits":     self._hits,
            "misses":   self._misses,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
            "db_entries": self.db.embedding_cache_size(),
        }

    # ── Internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _hash(text: str, model_name: str) -> str:
        payload = f"{model_name}|{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
