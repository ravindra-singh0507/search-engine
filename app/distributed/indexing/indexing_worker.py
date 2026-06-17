"""
Indexing Worker — Phase 8 Batch 2

=== THEORY ===

The IndexingWorker handles the first stage of the distributed indexing
pipeline: document ingestion.  It subscribes to document lifecycle events
(document.indexed, document.updated) and ensures documents are properly
indexed in the inverted index via the existing Indexer.

After successfully indexing a document, the worker emits a CHUNKING_STARTED
event to trigger the next pipeline stage (ChunkWorker), which in turn
triggers the EmbeddingWorker.

This event-driven decomposition follows the Pipes and Filters pattern
(Hohpe & Woolf, 2003): each processing stage is a self-contained filter
connected by event channels (pipes).  Adding or removing stages does not
require changes to other stages.

=== ARCHITECTURE ===

  EventBus
    │
    ├── "document.indexed"  ──▶  IndexingWorker.process()
    │                               │  1. Extract doc_id from payload
    │                               │  2. Load document from DB
    │                               │  3. Index via Indexer
    │                               │  4. Emit CHUNKING_STARTED event
    │                               ▼
    ├── "document.updated"  ──▶  IndexingWorker.process()
    │                               │  1. Extract doc_id from payload
    │                               │  2. Reindex via Indexer
    │                               │  3. Emit CHUNKING_STARTED event
    │                               ▼
    └── "chunking.started"  ──▶  ChunkWorker (next stage)

=== COMPLEXITY ===

  process():     O(T) where T = tokens in the document (tokenisation cost)
  reindex_all(): O(N * T) where N = total documents
  reindex_doc(): O(T) single document reindex

=== TRADEOFFS ===

  + Decoupled from chunking/embedding — can scale independently
  + Incremental indexing by default (only re-indexes changed docs)
  + Full reindex available via reindex_all() for bulk operations
  + Emits events for downstream stages (loose coupling)
  - Synchronous processing within a single worker instance
  - Relies on DB for document storage (not event-sourced)

=== PRODUCTION EQUIVALENTS ===

  Elasticsearch:    Ingest node with pipeline processors
  Apache Solr:      UpdateRequestProcessor chain
  Vespa:            Document processor chain in content cluster
  Google:           MapReduce indexing with forward/inverted index stages
"""

import logging
import time
from typing import Any

from app.events.models import Event, EventMetadata
from app.events.bus import EventBus
from app.events.topics import (
    DOCUMENT_INDEXED, DOCUMENT_UPDATED,
    CHUNKING_STARTED,
)
from app.config import DistributedIndexingConfig
from app.distributed.indexing.worker_base import WorkerBase

logger = logging.getLogger(__name__)


class IndexingWorker(WorkerBase):
    """
    Processes document indexing events.

    Subscribes to: document.indexed, document.updated
    Produces: chunking.started (triggers the chunk worker)

    Handles:
      - Parallel document indexing (multiple workers can run concurrently)
      - Incremental indexing (only processes new/updated documents)
      - Full reindexing (force mode via reindex_all)
      - Worker recovery (resume from last checkpoint via event replay)

    Dependencies:
      - Database (db):  document storage and retrieval
      - Indexer:        inverted index builder (app.indexer.indexer.Indexer)
      - EventBus:       for subscribing to events and emitting downstream events
    """

    topics = [DOCUMENT_INDEXED, DOCUMENT_UPDATED]

    def __init__(
        self,
        worker_id: str,
        db: Any = None,
        indexer: Any = None,
        event_bus: EventBus | None = None,
        config: DistributedIndexingConfig | None = None,
    ) -> None:
        super().__init__(
            worker_id=worker_id,
            event_bus=event_bus,
            config=config,
        )
        self.db = db
        self.indexer = indexer

    # ── Core processing ───────────────────────────────────────────────────

    def process(self, event: Event) -> dict:
        """
        Process a document indexing event.

        For DOCUMENT_INDEXED events:
          - The document has already been inserted into the DB and indexed
            by the Indexer (the event is informational).  The worker's job
            is to trigger the downstream chunking stage.

        For DOCUMENT_UPDATED events:
          - The document content has changed.  The worker reindexes the
            document and then triggers chunking.

        Args:
            event: Event with payload containing at least {"doc_id": int}.

        Returns:
            Dict with status, doc_id, and action taken.
        """
        doc_id = event.payload.get("doc_id")
        if doc_id is None:
            raise ValueError("Event payload missing 'doc_id'")

        action = "indexed"

        if event.topic == DOCUMENT_UPDATED:
            # Reindex the updated document
            result = self._reindex_document(doc_id)
            action = "reindexed"
        else:
            # Document already indexed — verify it exists
            result = self._verify_document(doc_id)

        # Emit chunking.started for the next pipeline stage
        self._emit_chunking_started(
            doc_id=doc_id,
            correlation_id=event.metadata.correlation_id,
            causation_id=event.metadata.event_id,
        )

        logger.info(
            "IndexingWorker %s %s doc %d",
            self.worker_id, action, doc_id,
        )

        return {
            "status":  "ok",
            "doc_id":  doc_id,
            "action":  action,
            **result,
        }

    # ── Public convenience methods ────────────────────────────────────────

    def reindex_all(self) -> dict:
        """
        Trigger a full reindex of all documents in the database.

        This is a bulk operation intended for administrative use (e.g.,
        after a schema change or tokenizer update).  Each document is
        reindexed and a CHUNKING_STARTED event is emitted.

        Returns:
            Dict with total count, success count, and failed doc_ids.
        """
        if self.db is None:
            return {"status": "error", "error": "No database configured"}

        docs = self.db.get_all_documents()
        total = len(docs)
        success = 0
        failed: list[int] = []

        logger.info(
            "IndexingWorker %s starting full reindex of %d documents",
            self.worker_id, total,
        )

        for doc in docs:
            try:
                self._reindex_document(doc.doc_id)
                self._emit_chunking_started(doc_id=doc.doc_id)
                success += 1
            except Exception as exc:
                logger.error(
                    "IndexingWorker %s failed to reindex doc %d: %s",
                    self.worker_id, doc.doc_id, exc,
                )
                failed.append(doc.doc_id)

        logger.info(
            "IndexingWorker %s full reindex complete: %d/%d succeeded",
            self.worker_id, success, total,
        )

        return {
            "status":  "ok",
            "total":   total,
            "success": success,
            "failed":  failed,
        }

    def reindex_doc(self, doc_id: int) -> dict:
        """
        Reindex a single document by ID.

        Convenience wrapper that reindexes and emits the downstream event.

        Args:
            doc_id: The document ID to reindex.

        Returns:
            Dict with status and reindex details.
        """
        result = self._reindex_document(doc_id)
        self._emit_chunking_started(doc_id=doc_id)

        return {
            "status": "ok",
            "doc_id": doc_id,
            "action": "reindexed",
            **result,
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    def _verify_document(self, doc_id: int) -> dict:
        """
        Verify that a document exists in the database.

        Returns:
            Dict with document metadata if found.

        Raises:
            ValueError: If the document does not exist.
        """
        if self.db is None:
            return {"verified": False, "reason": "no_database"}

        doc = self.db.get_document(doc_id)
        if doc is None:
            raise ValueError(f"Document {doc_id} not found in database")

        return {
            "verified": True,
            "title":    doc.title,
        }

    def _reindex_document(self, doc_id: int) -> dict:
        """
        Reindex a single document using the Indexer.

        Loads the document from the DB, clears old postings, and rebuilds
        the inverted index entries.

        Returns:
            Dict with reindex details (terms_indexed, total_tokens).
        """
        if self.db is None or self.indexer is None:
            return {"reindexed": False, "reason": "missing_dependencies"}

        doc = self.db.get_document(doc_id)
        if doc is None:
            raise ValueError(f"Document {doc_id} not found for reindexing")

        index_result = self.indexer.reindex_document(doc_id, doc.content)

        return {
            "reindexed":     True,
            "terms_indexed": index_result.terms_indexed,
            "total_tokens":  index_result.total_tokens,
        }

    def _emit_chunking_started(
        self,
        doc_id: int,
        correlation_id: str = "",
        causation_id: str = "",
    ) -> None:
        """
        Emit a CHUNKING_STARTED event for the downstream ChunkWorker.

        Args:
            doc_id:         The document ID to chunk.
            correlation_id: Trace correlation from the triggering event.
            causation_id:   The event_id of the event that caused this one.
        """
        if self.event_bus is None:
            return

        metadata = EventMetadata(
            source=f"indexing_worker:{self.worker_id}",
            correlation_id=correlation_id,
            causation_id=causation_id,
        )
        event = Event(
            topic=CHUNKING_STARTED,
            payload={"doc_id": doc_id},
            metadata=metadata,
        )
        self.event_bus.publish(event)
        logger.debug(
            "IndexingWorker %s emitted %s for doc %d",
            self.worker_id, CHUNKING_STARTED, doc_id,
        )
