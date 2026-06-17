"""
Embedding Worker — Phase 8 Batch 2

=== THEORY ===

The EmbeddingWorker handles the third and final stage of the distributed
indexing pipeline: vector embedding.  It subscribes to EMBEDDING_STARTED
events (emitted by the ChunkWorker) and generates dense vector
representations for each document chunk.

Embedding is typically the bottleneck in an indexing pipeline:
  - CPU embedding:  ~10-50 ms per chunk (sentence-transformers, BGE)
  - GPU embedding:  ~1-5 ms per chunk (batched, CUDA)
  - API embedding:  ~50-200 ms per chunk (OpenAI, Cohere — network latency)

Batching is critical for throughput.  Instead of embedding one chunk at
a time, the worker accumulates chunks and processes them in configurable
batch sizes (embedding_batch_size in DistributedIndexingConfig).

=== ARCHITECTURE ===

  EventBus
    │
    ├── "embedding.started"  ──▶  EmbeddingWorker.process()
    │                                │  1. Load document from DB
    │                                │  2. Retrieve chunks (or use chunker)
    │                                │  3. Embed chunks via EmbeddingPipeline
    │                                │  4. Store vectors in vector store
    │                                │  5. Emit EMBEDDING_COMPLETED event
    │                                ▼
    └── "embedding.completed" ──▶  (downstream consumers, e.g. search index refresh)

=== COMPLEXITY ===

  process():        O(C * D) where C = chunks, D = embedding dimension
  embed_document(): O(C * D) for one document
  embed_batch():    O(N * C * D) for N documents

  Batch embedding is more efficient due to GPU parallelism and reduced
  Python overhead:
    Sequential: N * overhead + N * forward_pass
    Batched:    1 * overhead + ceil(N/B) * forward_pass

=== TRADEOFFS ===

  + Batch processing for GPU efficiency
  + Incremental embedding (skip already-embedded docs)
  + Emits completion events for downstream consumers
  + Delegates to existing EmbeddingPipeline (no duplication)
  - Synchronous within a single worker (use multiple workers for parallelism)
  - Memory usage scales with batch size * embedding dimension

=== PRODUCTION EQUIVALENTS ===

  OpenAI:          /embeddings API with batched input
  Pinecone:        Upsert with batch vectors
  Elasticsearch:   Ingest pipeline with inference processor
  Vespa:           Embedding in document processor (ONNX Runtime)
  Weaviate:        Vectorizer modules (text2vec-transformers)
"""

import logging
import time
from typing import Any

from app.events.models import Event, EventMetadata
from app.events.bus import EventBus
from app.events.topics import EMBEDDING_STARTED, EMBEDDING_COMPLETED
from app.config import DistributedIndexingConfig
from app.distributed.indexing.worker_base import WorkerBase

logger = logging.getLogger(__name__)


class EmbeddingWorker(WorkerBase):
    """
    Processes document embedding events.

    Subscribes to: embedding.started
    Produces: embedding.completed

    Handles:
      - Batch embedding (accumulates docs, processes in configurable batches)
      - Incremental embedding (skip already-embedded docs unless force=True)
      - Vector store updates (delegates to EmbeddingPipeline)

    Dependencies:
      - Database (db):              document and chunk storage
      - EmbeddingPipeline:          the Phase 4 embedding orchestrator
      - EventBus:                   for subscribing and emitting events
    """

    topics = [EMBEDDING_STARTED]

    def __init__(
        self,
        worker_id: str,
        db: Any = None,
        embedding_pipeline: Any = None,
        event_bus: EventBus | None = None,
        config: DistributedIndexingConfig | None = None,
    ) -> None:
        super().__init__(
            worker_id=worker_id,
            event_bus=event_bus,
            config=config,
        )
        self.db = db
        self.embedding_pipeline = embedding_pipeline

    # ── Core processing ───────────────────────────────────────────────────

    def process(self, event: Event) -> dict:
        """
        Process an embedding event.

        Loads the document, generates embeddings via the EmbeddingPipeline,
        and emits an EMBEDDING_COMPLETED event.

        The force flag in the payload controls whether already-embedded
        documents are re-embedded:
          - force=False (default): skip documents that already have embeddings
          - force=True: re-embed even if embeddings exist

        Args:
            event: Event with payload containing {"doc_id": int, ...}.
                   Optional payload keys: "force" (bool), "chunk_ids" (list).

        Returns:
            Dict with status, doc_id, and embedding statistics.
        """
        doc_id = event.payload.get("doc_id")
        if doc_id is None:
            raise ValueError("Event payload missing 'doc_id'")

        force = event.payload.get("force", False)

        result = self.embed_document(doc_id, force=force)

        # Emit embedding.completed for downstream consumers
        self._emit_embedding_completed(
            doc_id=doc_id,
            stats=result,
            correlation_id=event.metadata.correlation_id,
            causation_id=event.metadata.event_id,
        )

        return result

    # ── Public convenience methods ────────────────────────────────────────

    def embed_document(self, doc_id: int, force: bool = False) -> dict:
        """
        Embed a single document.

        Delegates to the EmbeddingPipeline's index_document method, which
        handles chunking (if not already chunked), embedding, caching,
        and vector store insertion.

        Args:
            doc_id: The document ID to embed.
            force:  If True, re-embed even if already embedded.

        Returns:
            Dict with status, doc_id, and pipeline statistics.
        """
        if self.embedding_pipeline is None:
            return {
                "status":          "error",
                "doc_id":          doc_id,
                "error":           "No embedding pipeline configured",
                "docs_processed":  0,
                "chunks_created":  0,
                "chunks_embedded": 0,
            }

        t0 = time.perf_counter()

        try:
            stats = self.embedding_pipeline.index_document(doc_id, force=force)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if stats.docs_skipped > 0:
                logger.debug(
                    "EmbeddingWorker %s skipped doc %d (already embedded)",
                    self.worker_id, doc_id,
                )
                return {
                    "status":          "skipped",
                    "doc_id":          doc_id,
                    "reason":          "already_embedded",
                    "docs_processed":  0,
                    "chunks_created":  0,
                    "chunks_embedded": 0,
                    "latency_ms":      round(elapsed_ms, 2),
                }

            logger.info(
                "EmbeddingWorker %s embedded doc %d: %d chunks, %d embedded (%.1f ms)",
                self.worker_id, doc_id, stats.chunks_created,
                stats.chunks_embedded, elapsed_ms,
            )

            return {
                "status":          "ok",
                "doc_id":          doc_id,
                "docs_processed":  stats.docs_processed,
                "chunks_created":  stats.chunks_created,
                "chunks_embedded": stats.chunks_embedded,
                "cache_hits":      stats.cache_hits,
                "latency_ms":      round(elapsed_ms, 2),
            }

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error(
                "EmbeddingWorker %s failed to embed doc %d: %s",
                self.worker_id, doc_id, exc,
            )
            raise

    def embed_batch(self, doc_ids: list[int], force: bool = False) -> dict:
        """
        Embed multiple documents in sequence.

        Processes each document individually through the embedding pipeline.
        Failures for individual documents are collected and reported, but
        do not prevent other documents from being processed.

        Args:
            doc_ids: List of document IDs to embed.
            force:   If True, re-embed even if already embedded.

        Returns:
            Dict with total, success, failed counts and per-doc details.
        """
        t0 = time.perf_counter()
        total = len(doc_ids)
        success = 0
        skipped = 0
        failed: list[int] = []
        total_chunks_created = 0
        total_chunks_embedded = 0

        logger.info(
            "EmbeddingWorker %s starting batch embed of %d documents (force=%s)",
            self.worker_id, total, force,
        )

        for doc_id in doc_ids:
            try:
                result = self.embed_document(doc_id, force=force)
                if result.get("status") == "skipped":
                    skipped += 1
                elif result.get("status") == "ok":
                    success += 1
                    total_chunks_created += result.get("chunks_created", 0)
                    total_chunks_embedded += result.get("chunks_embedded", 0)
                else:
                    failed.append(doc_id)
            except Exception as exc:
                logger.error(
                    "EmbeddingWorker %s batch: failed doc %d: %s",
                    self.worker_id, doc_id, exc,
                )
                failed.append(doc_id)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "EmbeddingWorker %s batch complete: %d total, %d embedded, "
            "%d skipped, %d failed (%.1f ms)",
            self.worker_id, total, success, skipped, len(failed), elapsed_ms,
        )

        return {
            "status":           "ok",
            "total":            total,
            "success":          success,
            "skipped":          skipped,
            "failed":           failed,
            "chunks_created":   total_chunks_created,
            "chunks_embedded":  total_chunks_embedded,
            "latency_ms":       round(elapsed_ms, 2),
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    def _emit_embedding_completed(
        self,
        doc_id: int,
        stats: dict,
        correlation_id: str = "",
        causation_id: str = "",
    ) -> None:
        """
        Emit an EMBEDDING_COMPLETED event for downstream consumers.

        Downstream consumers might include:
          - Search index refresh service
          - Monitoring / analytics
          - Notification system

        Args:
            doc_id:         The document ID that was embedded.
            stats:          Embedding statistics dict.
            correlation_id: Trace correlation from the triggering event.
            causation_id:   The event_id of the event that caused this one.
        """
        if self.event_bus is None:
            return

        metadata = EventMetadata(
            source=f"embedding_worker:{self.worker_id}",
            correlation_id=correlation_id,
            causation_id=causation_id,
        )
        event = Event(
            topic=EMBEDDING_COMPLETED,
            payload={
                "doc_id":          doc_id,
                "status":          stats.get("status", "unknown"),
                "chunks_embedded": stats.get("chunks_embedded", 0),
            },
            metadata=metadata,
        )
        self.event_bus.publish(event)
        logger.debug(
            "EmbeddingWorker %s emitted %s for doc %d",
            self.worker_id, EMBEDDING_COMPLETED, doc_id,
        )
