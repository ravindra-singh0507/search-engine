"""
Distributed Indexing Pipeline — Phase 8 Batch 2

=== THEORY ===

The DistributedIndexingPipeline orchestrates a multi-stage document
processing pipeline using event-driven workers.  Each stage is a
self-contained worker that subscribes to events from the previous
stage and emits events for the next.

Pipeline topology:

  Document ingestion
        │
        ▼
  ┌──────────────┐     "chunking.started"     ┌─────────────┐
  │ IndexingWorker│ ──────────────────────────▶│ ChunkWorker  │
  └──────────────┘                             └─────────────┘
                                                      │
                                          "embedding.started"
                                                      │
                                                      ▼
                                               ┌──────────────┐
                                               │EmbeddingWorker│
                                               └──────────────┘
                                                      │
                                          "embedding.completed"
                                                      │
                                                      ▼
                                               (downstream consumers)

This is the Pipes and Filters architectural pattern (Buschmann et al.,
*Pattern-Oriented Software Architecture*, 1996):
  - Filters: IndexingWorker, ChunkWorker, EmbeddingWorker
  - Pipes:   EventBus topics (chunking.started, embedding.started, etc.)

Benefits of this decomposition:
  1. Independent scaling: add more EmbeddingWorkers for GPU-heavy loads
  2. Fault isolation: a failing chunker doesn't block the indexer
  3. Observability: each stage reports independent metrics
  4. Flexibility: swap chunking strategy without touching embedding

The pipeline also supports the Scatter-Gather pattern for multi-worker
stages: if num_indexing_workers > 1, the EventBus broadcasts to all
workers, and the first one to process wins (idempotent processing).

=== ARCHITECTURE ===

  DistributedIndexingPipeline
    │
    ├── setup()   → create and register workers
    ├── start()   → start all workers (subscribe to events)
    ├── stop()    → stop all workers (unsubscribe, drain)
    ├── process_document(doc_id) → manually trigger the full pipeline
    ├── stats()   → aggregate metrics from all workers
    └── get_workers() → list worker statuses

=== COMPLEXITY ===

  setup():            O(W) where W = total workers created
  start() / stop():   O(W * T) where T = topics per worker
  process_document(): O(1) — emits one event, processing is async
  stats():            O(W) — collects from each worker

=== TRADEOFFS ===

  + Single entry point for pipeline management
  + Configurable worker counts per stage
  + Aggregated stats across all workers
  + Manual trigger via process_document() for testing/debugging
  - All workers run in the same process (event-driven, not multi-process)
  - No partition-based load balancing (broadcast to all workers)
  - No persistent checkpointing (relies on DB state for recovery)

=== PRODUCTION EQUIVALENTS ===

  Apache Flink:       DataStream pipeline with operators
  Apache Beam:        PTransform pipeline (Runner-agnostic)
  Kafka Streams:      KStream/KTable topology
  AWS Step Functions: State machine with parallel branches
  Temporal:           Workflow with activity workers
  Dagster:            Asset-based pipeline with ops and jobs
"""

import logging
from typing import Any

from app.events.models import Event, EventMetadata
from app.events.bus import EventBus
from app.events.topics import DOCUMENT_INDEXED
from app.config import DistributedIndexingConfig
from app.distributed.indexing.indexing_worker import IndexingWorker
from app.distributed.indexing.chunk_worker import ChunkWorker
from app.distributed.indexing.embedding_worker import EmbeddingWorker

logger = logging.getLogger(__name__)


class DistributedIndexingPipeline:
    """
    Orchestrates the distributed indexing pipeline.

    Pipeline stages:
      1. Document ingestion  ->  IndexingWorker(s)
      2. Chunking            ->  ChunkWorker(s)
      3. Embedding           ->  EmbeddingWorker(s)

    Each stage is connected via the EventBus.  The pipeline provides
    lifecycle management (setup/start/stop) and aggregated metrics.

    Usage:
        config = DistributedIndexingConfig(
            num_indexing_workers=2,
            num_embedding_workers=2,
        )
        pipeline = DistributedIndexingPipeline(
            config=config,
            db=db,
            indexer=indexer,
            chunker=chunker,
            embedding_pipeline=embedding_pipeline,
            event_bus=bus,
        )
        pipeline.setup()
        pipeline.start()

        # Process a document through the pipeline
        pipeline.process_document(doc_id=42)

        # Later...
        print(pipeline.stats())
        pipeline.stop()
    """

    def __init__(
        self,
        config: DistributedIndexingConfig | None = None,
        db: Any = None,
        indexer: Any = None,
        chunker: Any = None,
        embedding_pipeline: Any = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.config = config or DistributedIndexingConfig()
        self.db = db
        self.indexer = indexer
        self.chunker = chunker
        self.embedding_pipeline = embedding_pipeline
        self.event_bus = event_bus

        # Worker registries (populated by setup())
        self._indexing_workers:  list[IndexingWorker]  = []
        self._chunk_workers:     list[ChunkWorker]     = []
        self._embedding_workers: list[EmbeddingWorker] = []
        self._is_setup = False
        self._is_running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def setup(self) -> None:
        """
        Create and register all workers according to the config.

        Worker counts are determined by:
          - config.num_indexing_workers  (default 2)
          - config.num_embedding_workers (default 2)
          - 1 chunk worker (chunking is fast, rarely a bottleneck)

        Idempotent: calling setup() again replaces existing workers
        (after stopping them if running).
        """
        if self._is_running:
            self.stop()

        self._indexing_workers.clear()
        self._chunk_workers.clear()
        self._embedding_workers.clear()

        # Create indexing workers
        for i in range(self.config.num_indexing_workers):
            worker = IndexingWorker(
                worker_id=f"indexing-{i}",
                db=self.db,
                indexer=self.indexer,
                event_bus=self.event_bus,
                config=self.config,
            )
            self._indexing_workers.append(worker)

        # Create chunk worker(s) — typically one is sufficient
        chunk_worker = ChunkWorker(
            worker_id="chunk-0",
            db=self.db,
            chunker=self.chunker,
            event_bus=self.event_bus,
            config=self.config,
        )
        self._chunk_workers.append(chunk_worker)

        # Create embedding workers
        for i in range(self.config.num_embedding_workers):
            worker = EmbeddingWorker(
                worker_id=f"embedding-{i}",
                db=self.db,
                embedding_pipeline=self.embedding_pipeline,
                event_bus=self.event_bus,
                config=self.config,
            )
            self._embedding_workers.append(worker)

        self._is_setup = True
        total = (
            len(self._indexing_workers)
            + len(self._chunk_workers)
            + len(self._embedding_workers)
        )
        logger.info(
            "DistributedIndexingPipeline setup: %d workers "
            "(%d indexing, %d chunk, %d embedding)",
            total,
            len(self._indexing_workers),
            len(self._chunk_workers),
            len(self._embedding_workers),
        )

    def start(self) -> None:
        """
        Start all workers (subscribe to their event topics).

        Calls setup() automatically if not already done.
        Idempotent: calling start() on a running pipeline is a no-op.
        """
        if self._is_running:
            logger.debug("Pipeline already running, ignoring start()")
            return

        if not self._is_setup:
            self.setup()

        for worker in self._all_workers():
            worker.start()

        self._is_running = True
        logger.info("DistributedIndexingPipeline started")

    def stop(self) -> None:
        """
        Stop all workers (unsubscribe, finish current work).

        Workers complete any in-progress event before stopping.
        Idempotent: calling stop() on a stopped pipeline is a no-op.
        """
        if not self._is_running:
            logger.debug("Pipeline already stopped, ignoring stop()")
            return

        for worker in self._all_workers():
            worker.stop()

        self._is_running = False
        logger.info("DistributedIndexingPipeline stopped")

    # ── Manual trigger ────────────────────────────────────────────────────

    def process_document(self, doc_id: int) -> dict:
        """
        Manually trigger the pipeline for a single document.

        Emits a DOCUMENT_INDEXED event, which the IndexingWorker picks up
        and propagates through the pipeline stages.

        This is useful for:
          - Testing the full pipeline end-to-end
          - Re-processing a specific document
          - Debugging pipeline issues

        Args:
            doc_id: The document ID to process.

        Returns:
            Dict with the emitted event metadata.
        """
        if self.event_bus is None:
            return {"status": "error", "error": "No event bus configured"}

        if not self._is_running:
            return {"status": "error", "error": "Pipeline not running"}

        metadata = EventMetadata(source="distributed_pipeline")
        event = Event(
            topic=DOCUMENT_INDEXED,
            payload={"doc_id": doc_id},
            metadata=metadata,
        )
        self.event_bus.publish(event)

        logger.info(
            "Pipeline triggered for doc %d (event=%s)",
            doc_id, metadata.event_id[:8],
        )

        return {
            "status":   "ok",
            "doc_id":   doc_id,
            "event_id": metadata.event_id,
        }

    # ── Metrics ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """
        Return aggregated statistics from all workers.

        Groups metrics by worker type (indexing, chunking, embedding)
        and provides pipeline-level totals.

        Returns:
            Dict with per-type and total metrics.
        """
        indexing_stats = [w.stats() for w in self._indexing_workers]
        chunk_stats = [w.stats() for w in self._chunk_workers]
        embedding_stats = [w.stats() for w in self._embedding_workers]

        total_processed = sum(
            s["processed"] for s in indexing_stats + chunk_stats + embedding_stats
        )
        total_failed = sum(
            s["failed"] for s in indexing_stats + chunk_stats + embedding_stats
        )

        return {
            "running":          self._is_running,
            "total_workers":    len(list(self._all_workers())),
            "total_processed":  total_processed,
            "total_failed":     total_failed,
            "indexing_workers":  indexing_stats,
            "chunk_workers":     chunk_stats,
            "embedding_workers": embedding_stats,
        }

    def get_workers(self) -> list[dict]:
        """
        Return status information for all workers.

        Returns:
            List of dicts, one per worker, with worker_id, type,
            running status, and metrics.
        """
        return [w.stats() for w in self._all_workers()]

    # ── Internal helpers ──────────────────────────────────────────────────

    def _all_workers(self):
        """Yield all workers across all stages."""
        yield from self._indexing_workers
        yield from self._chunk_workers
        yield from self._embedding_workers
