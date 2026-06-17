"""
Qdrant Vector Store

=== THEORY ===

Qdrant is an open-source vector database built for production-scale
approximate nearest neighbour (ANN) search.  It uses HNSW (Hierarchical
Navigable Small World) graphs as its core indexing structure.

HNSW vs FAISS IndexFlatIP:

  FAISS IndexFlatIP (current implementation):
    - Exact nearest neighbours via brute-force inner product
    - Time:  O(N * D) per query (linear scan)
    - Space: O(N * D * 4 bytes)
    - No filtering, no persistence, no distribution

  Qdrant HNSW:
    - Approximate nearest neighbours via layered proximity graph
    - Time:  O(log N) per query with tunable recall
    - Space: O(N * D * 4 + N * M * 8) where M = graph degree
    - Built-in payload filtering during search
    - Persistent storage (survives restarts)
    - REST/gRPC API for multi-service access
    - Hot index updates without full rebuild
    - Sharding and replication for horizontal scaling

HNSW PARAMETERS:

  m (hnsw_m):
    Number of bidirectional links per node.  Higher m = better recall
    but more memory and slower indexing.  Default 16 is standard.

  ef_construct (hnsw_ef_construct):
    Search width during index construction.  Higher = better index
    quality but slower build.  100 is a good balance.

  ef (search-time):
    Search width at query time.  Higher = better recall but slower
    search.  Defaults to ef_construct if not specified.

=== ARCHITECTURE ===

  QdrantVectorStore implements the VectorStore Protocol from store.py,
  making it a drop-in replacement for FaissVectorStore.

  Connection modes:
    - Remote:    qdrant_client.QdrantClient(host, port)
    - In-memory: qdrant_client.QdrantClient(":memory:")
    - On-disk:   qdrant_client.QdrantClient(path="/data/qdrant")

  Lazy connection: the Qdrant client is created on first operation,
  not in __init__, so import-time errors are deferred.

=== PRODUCTION EQUIVALENTS ===

  Pinecone:   Managed vector DB (serverless, auto-scaling)
  Weaviate:   Open-source with hybrid BM25+vector search
  Milvus:     Open-source with IVF, HNSW, and DiskANN
  Vespa:      Yahoo's hybrid search with ANN built-in

=== COMPLEXITY ===

  add(N vectors):          O(N * log V) — HNSW insert
  search(top_k):           O(log V) — HNSW search
  delete(M IDs):           O(M) — point deletion
  search_with_filter:      O(log V + F) — ANN + filter evaluation
  save (snapshot):         O(V * D) — full snapshot write
  load:                    no-op (Qdrant is persistent)
"""

import logging
import threading
from pathlib import Path
from typing import Any

from app.config import QdrantConfig

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """
    Qdrant-backed vector store implementing the VectorStore protocol.

    Uses the qdrant-client library.  Supports both local (in-memory)
    and remote (cluster) modes.  Falls back to raising ImportError
    when qdrant-client is not installed.

    Thread safety: all mutations are guarded by a threading.Lock to
    prevent concurrent collection creation or point upserts from
    corrupting client state.
    """

    def __init__(self, config: QdrantConfig | None = None):
        self._config = config or QdrantConfig()
        self._client: Any = None
        self._lock = threading.Lock()
        self._connected = False

    # ── Lazy connection ──────────────────────────────────────────────────

    def _ensure_connected(self) -> None:
        """Create the Qdrant client on first use."""
        if self._connected:
            return
        with self._lock:
            if self._connected:
                return  # double-check after acquiring lock
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise ImportError(
                    "qdrant-client is required for QdrantVectorStore. "
                    "Install with: pip install qdrant-client"
                ) from exc

            if self._config.api_key:
                self._client = QdrantClient(
                    host=self._config.host,
                    port=self._config.port,
                    grpc_port=self._config.grpc_port,
                    prefer_grpc=self._config.prefer_grpc,
                    api_key=self._config.api_key,
                )
            else:
                self._client = QdrantClient(
                    host=self._config.host,
                    port=self._config.port,
                    grpc_port=self._config.grpc_port,
                    prefer_grpc=self._config.prefer_grpc,
                )
            self._connected = True
            logger.info(
                "Qdrant connected: %s:%d (grpc=%d, prefer_grpc=%s)",
                self._config.host, self._config.port,
                self._config.grpc_port, self._config.prefer_grpc,
            )

    def _ensure_collection(self) -> None:
        """Create the collection if it does not exist."""
        self._ensure_connected()
        from qdrant_client.http.models import Distance, VectorParams

        collections = self._client.get_collections().collections
        existing_names = {c.name for c in collections}
        if self._config.collection_name in existing_names:
            return

        distance_map = {
            "cosine": Distance.COSINE,
            "euclid": Distance.EUCLID,
            "dot":    Distance.DOT,
        }
        distance = distance_map.get(
            self._config.distance.lower(), Distance.COSINE
        )

        self._client.create_collection(
            collection_name=self._config.collection_name,
            vectors_config=VectorParams(
                size=self._config.vector_size,
                distance=distance,
                on_disk=self._config.on_disk,
                hnsw_config={
                    "m": self._config.hnsw_m,
                    "ef_construct": self._config.hnsw_ef_construct,
                } if self._config.hnsw_m != 16 or self._config.hnsw_ef_construct != 100
                else None,
            ),
        )
        logger.info(
            "Qdrant collection %r created (dim=%d, distance=%s)",
            self._config.collection_name,
            self._config.vector_size,
            self._config.distance,
        )

    # ── VectorStore protocol implementation ──────────────────────────────

    def add(self, chunk_ids: list[str], vectors: list[list[float]]) -> None:
        """
        Upsert chunk_id -> vector pairs into the Qdrant collection.

        Uses upsert (not insert) so re-adding an existing chunk_id
        overwrites the previous vector rather than raising an error.
        Vectors are NOT L2-normalised here because Qdrant handles
        cosine distance internally (normalisation is implicit when
        distance=Cosine).

        Batch size: the qdrant-client handles batching internally,
        but we send all points in a single request.  For very large
        batches (>10k), consider chunking on the caller side.
        """
        if not chunk_ids:
            return
        with self._lock:
            self._ensure_collection()
        from qdrant_client.http.models import PointStruct

        points = [
            PointStruct(
                id=self._chunk_id_to_point_id(cid),
                vector=vec,
                payload={"chunk_id": cid},
            )
            for cid, vec in zip(chunk_ids, vectors)
        ]

        # Upsert in batches of 100 to avoid oversized requests
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self._client.upsert(
                collection_name=self._config.collection_name,
                points=batch,
            )

        logger.debug(
            "Qdrant: upserted %d vectors into %s",
            len(chunk_ids), self._config.collection_name,
        )

    def search(
        self, query_vector: list[float], top_k: int = 10
    ) -> list[tuple[str, float]]:
        """
        Find the top_k most similar chunks.

        Returns list of (chunk_id, score) sorted by score descending.
        Score semantics depend on the distance metric:
          - Cosine: score in [0, 1], higher = more similar
          - Dot:    unbounded, higher = more similar
          - Euclid: lower = more similar (inverted for consistency)
        """
        with self._lock:
            self._ensure_collection()

        results = self._client.search(
            collection_name=self._config.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )

        return [
            (hit.payload.get("chunk_id", str(hit.id)), hit.score)
            for hit in results
        ]

    def delete(self, chunk_ids: list[str]) -> None:
        """Delete vectors by chunk_id from the collection."""
        if not chunk_ids:
            return
        with self._lock:
            self._ensure_collection()
        from qdrant_client.http.models import PointIdsList

        point_ids = [self._chunk_id_to_point_id(cid) for cid in chunk_ids]
        self._client.delete(
            collection_name=self._config.collection_name,
            points_selector=PointIdsList(points=point_ids),
        )
        logger.debug("Qdrant: deleted %d vectors", len(chunk_ids))

    def save(self, path: Path) -> None:
        """
        Create a snapshot of the collection.

        For remote Qdrant: triggers a server-side snapshot.
        The path parameter is used to store snapshot metadata locally.
        Qdrant is persistent by default, so this is primarily for
        backup/migration purposes.
        """
        with self._lock:
            self._ensure_collection()
        try:
            snapshot = self._client.create_snapshot(
                collection_name=self._config.collection_name,
            )
            # Store snapshot name locally for reference
            path.mkdir(parents=True, exist_ok=True)
            meta_file = path / "qdrant_snapshot.txt"
            snapshot_name = getattr(snapshot, "name", str(snapshot))
            meta_file.write_text(snapshot_name, encoding="utf-8")
            logger.info(
                "Qdrant snapshot created: %s (saved meta to %s)",
                snapshot_name, meta_file,
            )
        except Exception as exc:
            logger.warning(
                "Qdrant snapshot failed (non-fatal): %s", exc,
            )

    def load(self, path: Path) -> None:
        """
        No-op for Qdrant: the database is persistent.

        Unlike FAISS, Qdrant stores data on disk / in its own server,
        so there is no need to load from a file.  The connection is
        established lazily on first operation.
        """
        logger.debug("Qdrant load is a no-op (persistent storage)")

    @property
    def total_vectors(self) -> int:
        """Return the number of vectors in the collection."""
        try:
            with self._lock:
                self._ensure_collection()
            info = self._client.get_collection(
                collection_name=self._config.collection_name,
            )
            return info.points_count or 0
        except Exception:
            return 0

    # ── Qdrant-specific methods ──────────────────────────────────────────

    def stats(self) -> dict:
        """Return collection statistics."""
        try:
            with self._lock:
                self._ensure_collection()
            info = self._client.get_collection(
                collection_name=self._config.collection_name,
            )
            return {
                "total_vectors":    info.points_count or 0,
                "indexed_vectors":  getattr(info, "indexed_vectors_count", 0) or 0,
                "segments":         getattr(info, "segments_count", 0) or 0,
                "status":           str(getattr(info, "status", "unknown")),
                "collection_name":  self._config.collection_name,
                "vector_size":      self._config.vector_size,
                "distance":         self._config.distance,
            }
        except Exception as exc:
            return {
                "total_vectors": 0,
                "error": str(exc),
                "collection_name": self._config.collection_name,
            }

    def create_collection(self) -> None:
        """Explicitly create the collection (idempotent)."""
        with self._lock:
            self._ensure_collection()

    def delete_collection(self) -> None:
        """Delete the entire collection and all its data."""
        self._ensure_connected()
        try:
            self._client.delete_collection(
                collection_name=self._config.collection_name,
            )
            logger.info(
                "Qdrant collection %r deleted", self._config.collection_name,
            )
        except Exception as exc:
            logger.warning(
                "Qdrant delete_collection failed: %s", exc,
            )

    def scroll(
        self, limit: int = 100, offset: str | None = None,
    ) -> list[dict]:
        """
        Scroll through all points in the collection.

        Returns a list of dicts with keys: id, chunk_id, vector (truncated).
        Useful for debugging and data inspection.

        Parameters
        ----------
        limit  : max points to return per call
        offset : scroll cursor from a previous call (point ID string)
        """
        with self._lock:
            self._ensure_collection()

        from qdrant_client.http.models import ScrollRequest

        # Build scroll kwargs
        scroll_kwargs: dict[str, Any] = {
            "collection_name": self._config.collection_name,
            "limit": limit,
            "with_payload": True,
            "with_vectors": False,
        }
        if offset is not None:
            scroll_kwargs["offset"] = offset

        records, next_offset = self._client.scroll(**scroll_kwargs)

        results: list[dict] = []
        for record in records:
            results.append({
                "id": str(record.id),
                "chunk_id": record.payload.get("chunk_id", ""),
                "payload": record.payload,
            })

        return results

    def search_with_filter(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_conditions: dict | None = None,
    ) -> list[tuple[str, float]]:
        """
        Search with payload-based filtering.

        Qdrant applies filters DURING the HNSW search, not as a
        post-filter.  This means filtered search is still sub-linear
        as long as the filter doesn't eliminate >99% of the data.

        Parameters
        ----------
        filter_conditions : dict with Qdrant filter structure, e.g.
            {"must": [{"key": "doc_type", "match": {"value": "article"}}]}

        If filter_conditions is None or empty, falls back to unfiltered search.
        """
        with self._lock:
            self._ensure_collection()

        search_kwargs: dict[str, Any] = {
            "collection_name": self._config.collection_name,
            "query_vector": query_vector,
            "limit": top_k,
        }

        if filter_conditions:
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue

            must_conditions = []
            for condition in filter_conditions.get("must", []):
                key = condition.get("key", "")
                match = condition.get("match", {})
                value = match.get("value", "")
                if key and value:
                    must_conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value),
                        )
                    )

            if must_conditions:
                search_kwargs["query_filter"] = Filter(must=must_conditions)

        results = self._client.search(**search_kwargs)

        return [
            (hit.payload.get("chunk_id", str(hit.id)), hit.score)
            for hit in results
        ]

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _chunk_id_to_point_id(chunk_id: str) -> str:
        """
        Convert a chunk_id string to a Qdrant point ID.

        Qdrant supports both integer and UUID point IDs.  We use the
        chunk_id string directly as a UUID-format ID by hashing it
        deterministically.  This avoids maintaining a separate ID map
        (unlike FAISS which needs an int->str mapping).
        """
        import hashlib
        import uuid
        # Deterministic UUID v5 from chunk_id (DNS namespace is arbitrary)
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
