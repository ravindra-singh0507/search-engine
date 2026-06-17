"""
Distributed Indexing — Phase 8 Batch 2

Event-driven document processing pipeline with independently scalable
workers for indexing, chunking, and embedding stages.

=== PIPELINE ===

  Document ingestion
    │  (DOCUMENT_INDEXED / DOCUMENT_UPDATED event)
    ▼
  IndexingWorker ──▶ CHUNKING_STARTED ──▶ ChunkWorker
                                              │
                                    EMBEDDING_STARTED
                                              │
                                              ▼
                                       EmbeddingWorker ──▶ EMBEDDING_COMPLETED

=== USAGE ===

  from app.distributed.indexing import DistributedIndexingPipeline

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
  pipeline.process_document(doc_id=42)
  pipeline.stop()
"""

from app.distributed.indexing.worker_base import WorkerBase
from app.distributed.indexing.indexing_worker import IndexingWorker
from app.distributed.indexing.chunk_worker import ChunkWorker
from app.distributed.indexing.embedding_worker import EmbeddingWorker
from app.distributed.indexing.pipeline import DistributedIndexingPipeline

__all__ = [
    "WorkerBase",
    "IndexingWorker",
    "ChunkWorker",
    "EmbeddingWorker",
    "DistributedIndexingPipeline",
]
