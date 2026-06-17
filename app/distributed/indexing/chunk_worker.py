"""
Chunk Worker — Phase 8 Batch 2

=== THEORY ===

The ChunkWorker handles the second stage of the distributed indexing
pipeline: document chunking.  It subscribes to CHUNKING_STARTED events
(emitted by the IndexingWorker) and splits documents into smaller chunks
suitable for embedding.

Chunking is separated from embedding because the two stages have very
different computational profiles:

  - Chunking is CPU-bound: string splitting, word counting, offset
    tracking.  Latency is microseconds per document.
  - Embedding is GPU/model-bound: forward pass through a neural network.
    Latency is milliseconds to seconds per batch.

By separating them into distinct workers, we can scale each stage
independently:
  - 1-2 ChunkWorkers can keep up with many IndexingWorkers
  - Multiple EmbeddingWorkers can run on GPU nodes to parallelise
    the bottleneck stage

This follows the Staged Event-Driven Architecture (SEDA) pattern
(Welsh, Culler, Brewer, 2001): decompose processing into stages with
queues between them, allowing each stage to be independently tuned.

=== ARCHITECTURE ===

  EventBus
    │
    ├── "chunking.started"  ──▶  ChunkWorker.process()
    │                               │  1. Load document from DB
    │                               │  2. Chunk via Chunker
    │                               │  3. Store chunks in DB
    │                               │  4. Emit EMBEDDING_STARTED event
    │                               ▼
    └── "embedding.started" ──▶  EmbeddingWorker (next stage)

=== COMPLEXITY ===

  process():         O(L) where L = document length in words
  chunk_document():  O(L) — delegates to Chunker.chunk()

  Space per document: O(C) where C = number of chunks produced
    Fixed-size:   C = ceil(L / chunk_size)
    Sliding:      C = ceil((L - chunk_size) / stride) + 1

=== TRADEOFFS ===

  + Decoupled from indexing and embedding — independent scaling
  + Stores chunks in DB for idempotent reprocessing
  + Lightweight: chunking is fast, so fewer workers needed
  + Configurable chunking strategy via ChunkingConfig
  - Extra event hop adds latency to the overall pipeline
  - Chunk storage requires DB writes (I/O overhead)

=== PRODUCTION EQUIVALENTS ===

  Elasticsearch:  Ingest pipeline processor (split processor)
  LangChain:      RecursiveCharacterTextSplitter
  LlamaIndex:     NodeParser / SentenceSplitter
  Haystack:       PreProcessor (document splitting)
  Vespa:          Document processor for field splitting
"""

import logging
from typing import Any

from app.events.models import Event, EventMetadata
from app.events.bus import EventBus
from app.events.topics import CHUNKING_STARTED, EMBEDDING_STARTED
from app.config import DistributedIndexingConfig
from app.distributed.indexing.worker_base import WorkerBase

logger = logging.getLogger(__name__)


class ChunkWorker(WorkerBase):
    """
    Processes document chunking events.

    Subscribes to: chunking.started
    Produces: embedding.started (triggers the embedding worker)

    A specialised worker that handles the chunking stage separately
    from embedding, allowing different scaling for CPU-bound chunking
    vs GPU-bound embedding.

    Dependencies:
      - Database (db):  document storage and chunk persistence
      - Chunker:        document chunking strategy (app.chunking.chunker)
      - EventBus:       for subscribing and emitting downstream events
    """

    topics = [CHUNKING_STARTED]

    def __init__(
        self,
        worker_id: str,
        db: Any = None,
        chunker: Any = None,
        event_bus: EventBus | None = None,
        config: DistributedIndexingConfig | None = None,
    ) -> None:
        super().__init__(
            worker_id=worker_id,
            event_bus=event_bus,
            config=config,
        )
        self.db = db
        self.chunker = chunker

    # ── Core processing ───────────────────────────────────────────────────

    def process(self, event: Event) -> dict:
        """
        Process a chunking event.

        Loads the document from the database, splits it into chunks using
        the configured Chunker, stores the chunks in the DB, and emits
        an EMBEDDING_STARTED event for the downstream EmbeddingWorker.

        Args:
            event: Event with payload containing {"doc_id": int}.

        Returns:
            Dict with status, doc_id, chunk_count, and chunk_ids.
        """
        doc_id = event.payload.get("doc_id")
        if doc_id is None:
            raise ValueError("Event payload missing 'doc_id'")

        result = self.chunk_document(doc_id)

        # Emit embedding.started for the next pipeline stage
        if result.get("chunk_count", 0) > 0:
            self._emit_embedding_started(
                doc_id=doc_id,
                chunk_ids=result.get("chunk_ids", []),
                correlation_id=event.metadata.correlation_id,
                causation_id=event.metadata.event_id,
            )

        return result

    # ── Public convenience method ─────────────────────────────────────────

    def chunk_document(self, doc_id: int) -> dict:
        """
        Chunk a single document and store the chunks in the database.

        This method can be called directly (outside the event loop) for
        testing or manual processing.

        Args:
            doc_id: The document ID to chunk.

        Returns:
            Dict with status, doc_id, chunk_count, and chunk_ids.
        """
        if self.db is None or self.chunker is None:
            return {
                "status":      "error",
                "doc_id":      doc_id,
                "error":       "Missing dependencies (db or chunker)",
                "chunk_count": 0,
                "chunk_ids":   [],
            }

        doc = self.db.get_document(doc_id)
        if doc is None:
            raise ValueError(f"Document {doc_id} not found for chunking")

        # Chunk the document
        chunks = self.chunker.chunk(doc.content, doc_id)

        # Store each chunk in the database
        chunk_ids: list[str] = []
        for chunk in chunks:
            try:
                self.db.insert_chunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    word_count=chunk.word_count,
                )
                chunk_ids.append(chunk.chunk_id)
            except Exception as exc:
                # Chunk may already exist (idempotent reprocessing)
                logger.debug(
                    "ChunkWorker %s: chunk %s may already exist: %s",
                    self.worker_id, chunk.chunk_id, exc,
                )
                chunk_ids.append(chunk.chunk_id)

        logger.info(
            "ChunkWorker %s chunked doc %d into %d chunks",
            self.worker_id, doc_id, len(chunks),
        )

        return {
            "status":      "ok",
            "doc_id":      doc_id,
            "chunk_count": len(chunks),
            "chunk_ids":   chunk_ids,
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    def _emit_embedding_started(
        self,
        doc_id: int,
        chunk_ids: list[str] | None = None,
        correlation_id: str = "",
        causation_id: str = "",
    ) -> None:
        """
        Emit an EMBEDDING_STARTED event for the downstream EmbeddingWorker.

        Args:
            doc_id:         The document ID whose chunks need embedding.
            chunk_ids:      List of chunk IDs produced (for targeted embedding).
            correlation_id: Trace correlation from the triggering event.
            causation_id:   The event_id of the event that caused this one.
        """
        if self.event_bus is None:
            return

        metadata = EventMetadata(
            source=f"chunk_worker:{self.worker_id}",
            correlation_id=correlation_id,
            causation_id=causation_id,
        )
        event = Event(
            topic=EMBEDDING_STARTED,
            payload={
                "doc_id":    doc_id,
                "chunk_ids": chunk_ids or [],
            },
            metadata=metadata,
        )
        self.event_bus.publish(event)
        logger.debug(
            "ChunkWorker %s emitted %s for doc %d (%d chunks)",
            self.worker_id, EMBEDDING_STARTED, doc_id, len(chunk_ids or []),
        )
