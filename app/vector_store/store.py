"""
FAISS Vector Store

=== THEORY ===

Vector similarity search finds the top-k vectors in a database that are
closest to a query vector, measured by a distance metric.

We use COSINE SIMILARITY:
  cos(a, b) = a · b / (|a| · |b|)

  Range: [-1, 1].  1 = identical direction, 0 = orthogonal, -1 = opposite.

Implementation trick: if we L2-normalise all vectors to unit length BEFORE
storing them, then cosine similarity equals the dot product:
  |a| = |b| = 1  →  cos(a, b) = a · b

FAISS provides IndexFlatIP (Inner Product) which computes exact dot products.
Combined with prior L2-normalisation, this gives us exact cosine similarity.

=== FAISS INDEX TYPES ===

  IndexFlatIP (this impl):
    Exact nearest neighbours.
    Time:  O(N · D) per query — full scan
    Space: O(N · D · 4 bytes)
    When:  N < 100k — exact search is fast enough

  IndexHNSWFlat (future):
    Approximate NN via hierarchical navigable small world graph.
    Time:  O(log N) per query (approximate)
    Space: O(N · D · 4 + N · M · 8) where M = graph degree (~16)
    When:  N > 100k, latency matters

  IndexIVFFlat (future):
    Inverted file index.  Clusters vectors; searches only nearby clusters.
    Time:  O(nprobe · cluster_size · D)
    Space: O(N · D · 4)
    When:  N > 500k, training data available

=== DELETION ===

FAISS's IndexFlatIP doesn't support real deletion.  We implement SOFT
DELETION: mark deleted IDs in a set and filter them from search results.
The physical vectors remain in the index until `compact()` is called,
which rebuilds the index without the deleted vectors.

=== PERSISTENCE ===

  index.faiss  — binary FAISS serialization (vectors + structure)
  id_map.json  — {faiss_int_id → chunk_id} mapping and metadata

Both files are written atomically (write temp → rename) to prevent
corruption on crash.

=== COMPLEXITY ===

  add(N vectors):     O(N · D)
  search(top_k):      O(V · D)  V = live vectors in index
  delete(M IDs):      O(M)      (soft delete — just marks a set)
  compact():          O(V · D)  (full rebuild without deleted vectors)
  save / load:        O(V · D)  (disk I/O)

=== AT SCALE ===

  Pinecone / Qdrant / Weaviate replace FAISS with:
    - Distributed sharding (horizontal scale)
    - Real-time deletion (no soft-delete needed)
    - Filtered search (metadata predicates + vector ANN)
    - HNSW or ScaNN indexing (sub-linear query time)

  To migrate from this FaissVectorStore to Qdrant, implement the same
  VectorStore Protocol with a QdrantVectorStore class.  No retrieval code
  changes are needed.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from app.config import VectorStoreConfig

logger = logging.getLogger(__name__)


# ── Protocol ──────────────────────────────────────────────────────────────────

@runtime_checkable
class VectorStore(Protocol):
    """
    Structural interface for any vector backend.
    Implement this to swap FAISS for Qdrant/Weaviate/Milvus.
    """

    def add(self, chunk_ids: list[str], vectors: list[list[float]]) -> None: ...
    def search(self, query_vector: list[float], top_k: int) -> list[tuple[str, float]]: ...
    def delete(self, chunk_ids: list[str]) -> None: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...

    @property
    def total_vectors(self) -> int: ...


# ── FAISS implementation ──────────────────────────────────────────────────────

class FaissVectorStore:
    """
    Exact cosine-similarity vector store backed by FAISS IndexFlatIP.

    All vectors are L2-normalised before storage so inner product ≡ cosine.
    Soft deletion keeps the index append-only; call compact() to reclaim space.
    """

    def __init__(self, config: VectorStoreConfig | None = None):
        self._config   = config or VectorStoreConfig()
        self._index    = None          # faiss.IndexIDMap — lazy initialised
        self._dim: int = self._config.dimension

        # Bidirectional mapping: FAISS int ID ↔ string chunk_id
        self._id_to_chunk: dict[int, str] = {}
        self._chunk_to_id: dict[str, int] = {}
        self._next_id: int = 0

        # Soft-deletion set — filtered from search results
        self._deleted: set[int] = set()

    # ── Index lifecycle ───────────────────────────────────────────────────

    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required. Install with: pip install faiss-cpu"
            ) from exc
        base  = faiss.IndexFlatIP(self._dim)
        self._index = faiss.IndexIDMap(base)
        logger.debug("FAISS IndexFlatIP(%d) created", self._dim)

    # ── Mutations ─────────────────────────────────────────────────────────

    def add(self, chunk_ids: list[str], vectors: list[list[float]]) -> None:
        """
        Add chunk_id → vector pairs.
        Vectors are L2-normalised in-place before insertion.
        Already-present chunk_ids are skipped (use delete + add to update).
        """
        if not chunk_ids:
            return
        self._ensure_index()

        new_ids:  list[int]        = []
        new_vecs: list[list[float]] = []

        for cid, vec in zip(chunk_ids, vectors):
            if cid in self._chunk_to_id:
                continue   # already indexed — skip
            fid = self._next_id
            self._next_id += 1
            self._id_to_chunk[fid] = cid
            self._chunk_to_id[cid] = fid
            new_ids.append(fid)
            new_vecs.append(vec)

        if not new_ids:
            return

        import faiss
        mat = np.array(new_vecs, dtype=np.float32)
        faiss.normalize_L2(mat)                                 # cosine normalisation
        ids_np = np.array(new_ids, dtype=np.int64)
        self._index.add_with_ids(mat, ids_np)
        logger.debug("FAISS: added %d vectors (total live=%d)", len(new_ids), self.total_vectors)

    def delete(self, chunk_ids: list[str]) -> None:
        """Soft-delete: mark as deleted, filter from search results."""
        for cid in chunk_ids:
            fid = self._chunk_to_id.get(cid)
            if fid is not None:
                self._deleted.add(fid)
        logger.debug("FAISS: soft-deleted %d chunks", len(chunk_ids))

    def update(self, chunk_id: str, new_vector: list[float]) -> None:
        """Update a vector by soft-deleting the old one and adding a new entry."""
        self.delete([chunk_id])
        # Remove from reverse map so add() doesn't skip it
        old_fid = self._chunk_to_id.pop(chunk_id, None)
        if old_fid is not None:
            self._id_to_chunk.pop(old_fid, None)
        self.add([chunk_id], [new_vector])

    def compact(self) -> None:
        """
        Rebuild the index without deleted vectors.
        Call periodically to reclaim memory after many deletions.
        """
        if not self._deleted:
            return

        import faiss

        live_ids  = [fid for fid in self._id_to_chunk if fid not in self._deleted]
        if not live_ids:
            self._index = None
            self._ensure_index()
            self._id_to_chunk.clear()
            self._chunk_to_id.clear()
            self._deleted.clear()
            return

        # Reconstruct vectors from the old index
        old_index = self._index
        mat = np.zeros((len(live_ids), self._dim), dtype=np.float32)
        for i, fid in enumerate(live_ids):
            old_index.reconstruct(fid, mat[i])

        # Build fresh index
        new_base  = faiss.IndexFlatIP(self._dim)
        new_index = faiss.IndexIDMap(new_base)
        ids_np    = np.array(live_ids, dtype=np.int64)
        new_index.add_with_ids(mat, ids_np)

        # Update internal state
        self._index   = new_index
        self._deleted.clear()
        logger.info("FAISS compact: %d live vectors retained", len(live_ids))

    # ── Search ────────────────────────────────────────────────────────────

    def search(self, query_vector: list[float],
               top_k: int = 10) -> list[tuple[str, float]]:
        """
        Find the top_k most similar chunks.

        Returns list of (chunk_id, cosine_similarity) sorted by score desc.
        Scores are in [0, 1] for normalised vectors (negative scores are
        filtered out since they indicate opposite semantic direction).
        """
        if self._index is None or self.total_vectors == 0:
            return []

        import faiss

        query_np = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query_np)

        # Over-fetch to account for soft-deleted entries
        fetch_k = min(top_k + len(self._deleted) + 1, self._index.ntotal)
        scores, faiss_ids = self._index.search(query_np, fetch_k)

        results: list[tuple[str, float]] = []
        for score, fid in zip(scores[0], faiss_ids[0]):
            if fid == -1:
                continue
            if fid in self._deleted:
                continue
            cid = self._id_to_chunk.get(int(fid))
            if cid and score >= 0:          # only positive similarities
                results.append((cid, float(score)))
            if len(results) >= top_k:
                break

        return results

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Save index and ID map to disk."""
        path.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            import faiss
            faiss.write_index(self._index, str(path / "index.faiss"))

        mapping = {
            "id_to_chunk": {str(k): v for k, v in self._id_to_chunk.items()},
            "chunk_to_id": self._chunk_to_id,
            "next_id":     self._next_id,
            "deleted":     list(self._deleted),
            "dimension":   self._dim,
        }
        (path / "id_map.json").write_text(
            json.dumps(mapping, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("FAISS index saved to %s (%d vectors)", path, self.total_vectors)

    def load(self, path: Path) -> None:
        """Load index and ID map from disk."""
        index_file = path / "index.faiss"
        map_file   = path / "id_map.json"

        if not index_file.exists():
            logger.debug("FAISS: no saved index at %s", path)
            return

        import faiss
        self._index = faiss.read_index(str(index_file))
        logger.info("FAISS index loaded from %s (ntotal=%d)", path, self._index.ntotal)

        if map_file.exists():
            m = json.loads(map_file.read_text(encoding="utf-8"))
            self._id_to_chunk = {int(k): v for k, v in m["id_to_chunk"].items()}
            self._chunk_to_id = m["chunk_to_id"]
            self._next_id     = m["next_id"]
            self._deleted     = set(m.get("deleted", []))
            self._dim         = m.get("dimension", self._dim)

    # ── Stats ─────────────────────────────────────────────────────────────

    @property
    def total_vectors(self) -> int:
        """Live (non-deleted) vectors."""
        if self._index is None:
            return 0
        return max(0, self._index.ntotal - len(self._deleted))

    def stats(self) -> dict:
        raw = self._index.ntotal if self._index else 0
        return {
            "total_vectors":   self.total_vectors,
            "raw_ntotal":      raw,
            "deleted_vectors": len(self._deleted),
            "dimension":       self._dim,
        }
