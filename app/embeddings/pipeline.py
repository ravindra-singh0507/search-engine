"""
Embedding Pipeline

Orchestrates the full Document → Chunk → Embed → Store pipeline.

=== PIPELINE ===

  1. Load document text from DB
  2. Check embedding cache (skip already-cached chunks)
  3. Chunk the document (FixedSize or SlidingWindow)
  4. Batch-embed chunks via EmbeddingProvider
  5. Cache each embedding
  6. Store chunk metadata in document_chunks table
  7. Add vectors to FaissVectorStore
  8. Record embedding_ids in document_embeddings table
  9. Update vector_index_metadata

=== INCREMENTAL INDEXING ===

  is_doc_embedded(doc_id, model_name) checks the document_embeddings table.
  Docs that already have embeddings are skipped unless force=True.
  This makes re-indexing a no-op for unchanged content.

=== CONCURRENCY ===

  The pipeline runs synchronously (blocking) for small batches.
  For large corpora, wrap run_all() in a threading.Thread (same pattern as
  the web crawler) and poll via the /embeddings/status endpoint.
"""

import logging
import time
from dataclasses import dataclass, field

from app.database.db import Database
from app.embeddings.provider import EmbeddingProvider
from app.embeddings.cache import EmbeddingCache
from app.chunking.chunker import Chunker, make_chunker, Chunk
from app.vector_store.store import FaissVectorStore
from app.config import EmbeddingConfig, ChunkingConfig, VectorStoreConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    docs_processed:   int = 0
    docs_skipped:     int = 0
    chunks_created:   int = 0
    chunks_embedded:  int = 0
    cache_hits:       int = 0
    total_ms:         float = 0.0

    def as_dict(self) -> dict:
        return {
            "docs_processed":  self.docs_processed,
            "docs_skipped":    self.docs_skipped,
            "chunks_created":  self.chunks_created,
            "chunks_embedded": self.chunks_embedded,
            "cache_hits":      self.cache_hits,
            "latency_ms":      round(self.total_ms, 1),
        }


class EmbeddingPipeline:
    """
    Orchestrates chunking, embedding, and vector-store insertion.
    Stateless except for the injected components — safe to reuse.
    """

    def __init__(
        self,
        db:           Database,
        provider:     EmbeddingProvider,
        cache:        EmbeddingCache,
        vector_store: FaissVectorStore,
        chunker:      Chunker | None = None,
        emb_config:   EmbeddingConfig    = None,
        chunk_config: ChunkingConfig     = None,
        vs_config:    VectorStoreConfig  = None,
    ):
        self.db           = db
        self.provider     = provider
        self.cache        = cache
        self.vector_store = vector_store
        self.chunker      = chunker or make_chunker(chunk_config)
        self.emb_config   = emb_config   or EmbeddingConfig()
        self.vs_config    = vs_config    or VectorStoreConfig()

    # ── Public API ────────────────────────────────────────────────────────

    def index_document(self, doc_id: int, force: bool = False) -> PipelineStats:
        """
        Embed one document.  Skips if already embedded unless force=True.
        Returns empty stats immediately if the DB connection is closed.
        """
        stats = PipelineStats()
        if self.db.conn is None:
            return stats
        t0    = time.perf_counter()

        if not force and self.db.is_doc_embedded(doc_id, self.provider.model_name):
            stats.docs_skipped = 1
            return stats

        doc = self.db.get_document(doc_id)
        if doc is None:
            logger.warning("EmbeddingPipeline: doc %d not found", doc_id)
            return stats

        chunks = self._chunk_and_store(doc_id, doc.content)
        stats.chunks_created = len(chunks)

        if not chunks:
            return stats

        job_id = self.db.create_embedding_job(
            doc_id, self.provider.model_name, len(chunks)
        )
        try:
            self._embed_and_store(chunks, stats)
            self.db.update_embedding_job(
                job_id, "done", chunks_processed=stats.chunks_embedded
            )
        except Exception as exc:
            self.db.update_embedding_job(job_id, "error", error=str(exc))
            logger.error("Embedding failed for doc %d: %s", doc_id, exc)
            raise

        stats.docs_processed = 1
        stats.total_ms = (time.perf_counter() - t0) * 1000
        self._save_index()
        logger.info(
            "Indexed doc %d: %d chunks, %d embedded, %d cache hits (%.1f ms)",
            doc_id, stats.chunks_created, stats.chunks_embedded,
            stats.cache_hits, stats.total_ms,
        )
        return stats

    def index_all(self, force: bool = False) -> PipelineStats:
        """
        Embed all documents that don't have embeddings yet.
        Returns empty stats immediately if the DB connection is closed.
        """
        t0      = time.perf_counter()
        combined = PipelineStats()
        if self.db.conn is None:
            return combined

        if force:
            doc_ids = [d.doc_id for d in self.db.get_all_documents()]
        else:
            doc_ids = self.db.get_unembedded_doc_ids(self.provider.model_name)

        logger.info(
            "EmbeddingPipeline.index_all: %d docs to embed (force=%s)",
            len(doc_ids), force,
        )
        for doc_id in doc_ids:
            s = self.index_document(doc_id, force=force)
            combined.docs_processed  += s.docs_processed
            combined.docs_skipped    += s.docs_skipped
            combined.chunks_created  += s.chunks_created
            combined.chunks_embedded += s.chunks_embedded
            combined.cache_hits      += s.cache_hits

        combined.total_ms = (time.perf_counter() - t0) * 1000
        logger.info("index_all complete: %s", combined.as_dict())
        return combined

    def remove_document(self, doc_id: int) -> None:
        """
        Remove a document's chunks and embeddings from the vector store and DB.
        Call after Document deletion.
        """
        chunks = self.db.get_chunks_for_doc(doc_id)
        chunk_ids = [c.chunk_id for c in chunks]
        if chunk_ids:
            self.vector_store.delete(chunk_ids)
        self.db.delete_embeddings_for_doc(doc_id)
        self.db.delete_chunks_for_doc(doc_id)
        self._save_index()
        logger.info("Removed %d chunks for doc %d", len(chunk_ids), doc_id)

    # ── Internals ─────────────────────────────────────────────────────────

    def _chunk_and_store(self, doc_id: int, content: str) -> list[Chunk]:
        """Chunk document and persist chunk metadata to DB."""
        if self.db.conn is None:
            return []
        chunks = self.chunker.chunk(content, doc_id)
        for chunk in chunks:
            if self.db.conn is None:
                break
            self.db.insert_chunk(
                chunk_id     = chunk.chunk_id,
                doc_id       = chunk.doc_id,
                chunk_index  = chunk.chunk_index,
                text         = chunk.text,
                start_offset = chunk.start_offset,
                end_offset   = chunk.end_offset,
                word_count   = chunk.word_count,
            )
        return chunks

    def _embed_and_store(self, chunks: list[Chunk], stats: PipelineStats) -> None:
        """
        Embed chunks (with cache) and insert into vector store and DB.
        Uses batch processing for efficiency.
        """
        if self.db.conn is None:
            return

        # Separate cache hits from misses
        to_embed_indices: list[int]   = []
        to_embed_texts:   list[str]   = []
        cached_vectors:   dict[int, list[float]] = {}

        for i, chunk in enumerate(chunks):
            if self.db.conn is None:
                return
            if self.emb_config.cache_embeddings:
                cached = self.cache.get(chunk.text, self.provider.model_name)
                if cached is not None:
                    cached_vectors[i] = cached
                    stats.cache_hits += 1
                    continue
            to_embed_indices.append(i)
            to_embed_texts.append(chunk.text)

        # Embed in batch
        new_vectors: dict[int, list[float]] = {}
        if to_embed_texts:
            batch_size = self.emb_config.batch_size
            for start in range(0, len(to_embed_texts), batch_size):
                if self.db.conn is None:
                    return
                batch_texts = to_embed_texts[start: start + batch_size]
                batch_vecs  = self.provider.embed_texts(batch_texts)
                for j, vec in zip(to_embed_indices[start: start + batch_size], batch_vecs):
                    new_vectors[j] = vec
                    if self.emb_config.cache_embeddings and self.db.conn is not None:
                        self.cache.put(chunks[j].text, self.provider.model_name, vec)

        # Insert into vector store and record in DB
        all_chunk_ids: list[str]        = []
        all_vectors:   list[list[float]] = []

        for i, chunk in enumerate(chunks):
            if self.db.conn is None:
                break
            vec = cached_vectors.get(i) or new_vectors.get(i)
            if vec is None:
                logger.warning("No vector for chunk %s — skipping", chunk.chunk_id)
                continue
            all_chunk_ids.append(chunk.chunk_id)
            all_vectors.append(vec)
            self.db.insert_embedding_record(
                chunk_id   = chunk.chunk_id,
                doc_id     = chunk.doc_id,
                model_name = self.provider.model_name,
                vector_dim = len(vec),
            )
            stats.chunks_embedded += 1

        self.vector_store.add(all_chunk_ids, all_vectors)

    def _save_index(self) -> None:
        path = self.vs_config.index_path
        self.vector_store.save(path)
        self.db.upsert_vector_index_metadata(
            model_name    = self.provider.model_name,
            dimension     = self.provider.dimension,
            total_vectors = self.vector_store.total_vectors,
            index_path    = str(path),
        )
