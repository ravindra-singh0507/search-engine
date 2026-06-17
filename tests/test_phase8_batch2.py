"""
Phase 8 Batch 2 Test Suite — Distributed AI Infrastructure Platform

Tests cover:
  - Kafka config defaults (KafkaConfig bootstrap_servers, group_id, etc.)
  - KafkaEventBus protocol compliance (structural check without broker)
  - InMemoryFrontier (add, get_batch, dedup, priority, mark_complete, stats)
  - URLDeduplicator (mark_seen, is_seen, not_seen, clear, count)
  - CrawlerCoordinator (add_seeds, register_worker, assign_batch, report_result, stats)
  - WorkerBase (lifecycle, stats via concrete subclass)
  - IndexingWorker (creation, stats)
  - EmbeddingWorker (creation, stats)
  - ChunkWorker (creation, stats)
  - DistributedIndexingPipeline (creation, get_workers, stats)
  - QdrantVectorStore protocol methods (structural check)
  - QdrantConfig defaults
  - VectorStoreFactory (default FAISS, config reading)
  - GatewayModels (GatewayRequest defaults, GatewayResponse construction)
  - GatewayCache (put/get, miss, invalidate, make_key, stats)
  - QueryRouter (default routing, hint override)
  - RetrievalGateway (creation, stats)
  - API endpoints (gateway/stats, cache/stats, crawler/stats, indexing/stats, vector-store/backend)
  - Config compat (EngineConfig zero-arg, batch 2 config defaults)
"""

import json
import os
import pytest
import time
from pathlib import Path
from unittest.mock import MagicMock


# ══════════════════════════════════════════════════════════════════════════════
#  KAFKA CONFIG AND PROTOCOL TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestKafkaConfig:
    """Kafka configuration defaults."""

    def test_bootstrap_servers_default(self):
        from app.config import KafkaConfig
        cfg = KafkaConfig()
        assert cfg.bootstrap_servers == "localhost:9092"

    def test_group_id_default(self):
        from app.config import KafkaConfig
        cfg = KafkaConfig()
        assert cfg.group_id == "search-engine"

    def test_auto_offset_reset_default(self):
        from app.config import KafkaConfig
        cfg = KafkaConfig()
        assert cfg.auto_offset_reset == "earliest"

    def test_num_partitions_default(self):
        from app.config import KafkaConfig
        cfg = KafkaConfig()
        assert cfg.num_partitions == 3

    def test_replication_factor_default(self):
        from app.config import KafkaConfig
        cfg = KafkaConfig()
        assert cfg.replication_factor == 1

    def test_custom_config(self):
        from app.config import KafkaConfig
        cfg = KafkaConfig(
            bootstrap_servers="broker1:9092,broker2:9092",
            group_id="custom-group",
            num_partitions=6,
        )
        assert cfg.bootstrap_servers == "broker1:9092,broker2:9092"
        assert cfg.group_id == "custom-group"
        assert cfg.num_partitions == 6


class TestKafkaEventBusProtocol:
    """Structural check that KafkaEventBus has EventBus-compatible methods."""

    def test_class_has_publish_method(self):
        from app.kafka.bus import KafkaEventBus
        assert hasattr(KafkaEventBus, "publish"), "KafkaEventBus must have publish()"

    def test_class_has_subscribe_method(self):
        from app.kafka.bus import KafkaEventBus
        assert hasattr(KafkaEventBus, "subscribe"), "KafkaEventBus must have subscribe()"

    def test_class_has_unsubscribe_method(self):
        from app.kafka.bus import KafkaEventBus
        assert hasattr(KafkaEventBus, "unsubscribe"), "KafkaEventBus must have unsubscribe()"

    def test_class_has_subscriber_count_method(self):
        from app.kafka.bus import KafkaEventBus
        assert hasattr(KafkaEventBus, "subscriber_count"), (
            "KafkaEventBus must have subscriber_count()"
        )

    def test_constructor_accepts_kafka_config(self):
        """Verify constructor signature accepts KafkaConfig."""
        import inspect
        from app.kafka.bus import KafkaEventBus
        sig = inspect.signature(KafkaEventBus.__init__)
        params = list(sig.parameters.keys())
        assert "config" in params, "KafkaEventBus.__init__ should accept config parameter"


# ══════════════════════════════════════════════════════════════════════════════
#  IN-MEMORY FRONTIER TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestInMemoryFrontier:
    """InMemoryFrontier: heapq-backed URL frontier with set dedup."""

    def test_add_url(self):
        from app.distributed.crawler.frontier import InMemoryFrontier
        frontier = InMemoryFrontier(max_size=100)
        result = frontier.add("https://example.com", priority=0.0, depth=0)
        assert result is True, "First add should succeed"
        assert frontier.size() == 1

    def test_get_batch(self):
        from app.distributed.crawler.frontier import InMemoryFrontier
        frontier = InMemoryFrontier(max_size=100)
        frontier.add("https://example.com/a", priority=1.0, depth=0)
        frontier.add("https://example.com/b", priority=2.0, depth=1)
        batch = frontier.get_batch(batch_size=10)
        assert len(batch) == 2
        assert batch[0]["url"] == "https://example.com/a", "Lower priority should come first"

    def test_duplicate_rejection(self):
        from app.distributed.crawler.frontier import InMemoryFrontier
        frontier = InMemoryFrontier(max_size=100)
        assert frontier.add("https://example.com") is True
        assert frontier.add("https://example.com") is False, "Duplicate should be rejected"
        assert frontier.size() == 1

    def test_priority_ordering(self):
        from app.distributed.crawler.frontier import InMemoryFrontier
        frontier = InMemoryFrontier(max_size=100)
        frontier.add("https://low.com", priority=10.0, depth=0)
        frontier.add("https://high.com", priority=1.0, depth=0)
        frontier.add("https://mid.com", priority=5.0, depth=0)
        batch = frontier.get_batch(batch_size=3)
        urls = [item["url"] for item in batch]
        assert urls[0] == "https://high.com", "Highest priority (lowest score) should be first"
        assert urls[1] == "https://mid.com"
        assert urls[2] == "https://low.com"

    def test_mark_complete(self):
        from app.distributed.crawler.frontier import InMemoryFrontier
        frontier = InMemoryFrontier(max_size=100)
        frontier.add("https://example.com", priority=0.0, depth=0)
        frontier.get_batch(batch_size=1)  # remove from queue
        frontier.mark_complete("https://example.com")
        stats = frontier.stats()
        assert stats["complete"] == 1

    def test_is_empty(self):
        from app.distributed.crawler.frontier import InMemoryFrontier
        frontier = InMemoryFrontier(max_size=100)
        assert frontier.is_empty() is True, "New frontier should be empty"
        frontier.add("https://example.com")
        assert frontier.is_empty() is False, "Frontier with URL should not be empty"

    def test_contains(self):
        from app.distributed.crawler.frontier import InMemoryFrontier
        frontier = InMemoryFrontier(max_size=100)
        assert frontier.contains("https://example.com") is False
        frontier.add("https://example.com")
        assert frontier.contains("https://example.com") is True

    def test_stats(self):
        from app.distributed.crawler.frontier import InMemoryFrontier
        frontier = InMemoryFrontier(max_size=1000)
        frontier.add("https://a.com", priority=0.0, depth=0)
        frontier.add("https://b.com", priority=1.0, depth=1)
        stats = frontier.stats()
        assert stats["pending"] == 2
        assert stats["seen"] == 2
        assert stats["complete"] == 0
        assert stats["failed"] == 0
        assert stats["max_size"] == 1000

    def test_max_size_enforcement(self):
        from app.distributed.crawler.frontier import InMemoryFrontier
        frontier = InMemoryFrontier(max_size=2)
        assert frontier.add("https://a.com") is True
        assert frontier.add("https://b.com") is True
        assert frontier.add("https://c.com") is False, "Should reject when full"
        assert frontier.size() == 2

    def test_mark_failed(self):
        from app.distributed.crawler.frontier import InMemoryFrontier
        frontier = InMemoryFrontier(max_size=100)
        frontier.add("https://fail.com", priority=0.0, depth=0)
        frontier.get_batch(batch_size=1)
        frontier.mark_failed("https://fail.com", "Connection timeout")
        stats = frontier.stats()
        assert stats["failed"] == 1


# ══════════════════════════════════════════════════════════════════════════════
#  URL DEDUPLICATOR TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestURLDeduplicator:
    """URLDeduplicator: track seen URLs for deduplication."""

    def test_mark_seen(self):
        from app.distributed.crawler.dedup import URLDeduplicator
        dedup = URLDeduplicator()  # in-memory fallback
        dedup.mark_seen("https://example.com")
        assert dedup.count() == 1

    def test_is_seen(self):
        from app.distributed.crawler.dedup import URLDeduplicator
        dedup = URLDeduplicator()
        dedup.mark_seen("https://example.com")
        assert dedup.is_seen("https://example.com") is True

    def test_not_seen(self):
        from app.distributed.crawler.dedup import URLDeduplicator
        dedup = URLDeduplicator()
        assert dedup.is_seen("https://unknown.com") is False

    def test_clear(self):
        from app.distributed.crawler.dedup import URLDeduplicator
        dedup = URLDeduplicator()
        dedup.mark_seen("https://a.com")
        dedup.mark_seen("https://b.com")
        assert dedup.count() == 2
        dedup.clear()
        assert dedup.count() == 0

    def test_count(self):
        from app.distributed.crawler.dedup import URLDeduplicator
        dedup = URLDeduplicator()
        assert dedup.count() == 0
        dedup.mark_seen("https://a.com")
        dedup.mark_seen("https://b.com")
        dedup.mark_seen("https://c.com")
        assert dedup.count() == 3

    def test_mark_seen_batch(self):
        from app.distributed.crawler.dedup import URLDeduplicator
        dedup = URLDeduplicator()
        dedup.mark_seen_batch(["https://a.com", "https://b.com", "https://c.com"])
        assert dedup.count() == 3
        assert dedup.is_seen("https://b.com") is True


# ══════════════════════════════════════════════════════════════════════════════
#  CRAWLER COORDINATOR TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestCrawlerCoordinator:
    """CrawlerCoordinator: manages distributed crawl cluster."""

    def _make_coordinator(self):
        from app.config import DistributedCrawlerConfig
        from app.distributed.crawler.frontier import InMemoryFrontier
        from app.distributed.crawler.coordinator import CrawlerCoordinator
        config = DistributedCrawlerConfig(max_workers=4, batch_size=5)
        frontier = InMemoryFrontier(max_size=1000)
        return CrawlerCoordinator(config, frontier)

    def test_add_seeds(self):
        coord = self._make_coordinator()
        count = coord.add_seeds(["https://a.com", "https://b.com"])
        assert count == 2

    def test_add_seeds_dedup(self):
        coord = self._make_coordinator()
        coord.add_seeds(["https://a.com"])
        count = coord.add_seeds(["https://a.com", "https://b.com"])
        assert count == 1, "Duplicate seed should not be counted"

    def test_register_worker(self):
        coord = self._make_coordinator()
        coord.register_worker("worker-1")
        workers = coord.get_workers()
        assert len(workers) == 1
        assert workers[0]["worker_id"] == "worker-1"
        assert workers[0]["status"] == "active"

    def test_assign_batch(self):
        coord = self._make_coordinator()
        coord.add_seeds(["https://a.com", "https://b.com", "https://c.com"])
        coord.register_worker("worker-1")
        batch = coord.assign_batch("worker-1")
        assert len(batch) == 3, "Should return all 3 seed URLs"

    def test_assign_batch_unknown_worker(self):
        coord = self._make_coordinator()
        coord.add_seeds(["https://a.com"])
        batch = coord.assign_batch("unknown-worker")
        assert len(batch) == 0, "Unknown worker should get empty batch"

    def test_report_result(self):
        coord = self._make_coordinator()
        coord.add_seeds(["https://a.com"])
        coord.register_worker("worker-1")
        coord.assign_batch("worker-1")
        coord.report_result("worker-1", "https://a.com", success=True, doc_id=1)
        stats = coord.stats()
        assert stats["total_completed"] == 1

    def test_report_failure(self):
        coord = self._make_coordinator()
        coord.add_seeds(["https://fail.com"])
        coord.register_worker("worker-1")
        coord.assign_batch("worker-1")
        coord.report_result("worker-1", "https://fail.com", success=False)
        stats = coord.stats()
        assert stats["total_failed"] == 1

    def test_stats(self):
        coord = self._make_coordinator()
        stats = coord.stats()
        assert "total_completed" in stats
        assert "total_failed" in stats
        assert "worker_count" in stats
        assert "frontier_size" in stats
        assert "frontier" in stats

    def test_is_complete(self):
        coord = self._make_coordinator()
        assert coord.is_complete() is True, "Empty coordinator should be complete"
        coord.add_seeds(["https://a.com"])
        assert coord.is_complete() is False, "Should not be complete with pending URLs"


# ══════════════════════════════════════════════════════════════════════════════
#  WORKER BASE TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestWorkerBase:
    """WorkerBase abstract class via a concrete test subclass."""

    def _make_worker(self, worker_id="test-worker-1"):
        from app.distributed.indexing.worker_base import WorkerBase
        from app.events.models import Event

        class TestableWorker(WorkerBase):
            topics = ["test.topic"]

            def process(self, event: Event) -> dict:
                return {"status": "ok"}

        return TestableWorker(worker_id=worker_id)

    def test_lifecycle_start_stop(self):
        worker = self._make_worker()
        assert worker.is_running() is False
        worker.start()
        assert worker.is_running() is True
        worker.stop()
        assert worker.is_running() is False

    def test_start_idempotent(self):
        worker = self._make_worker()
        worker.start()
        worker.start()  # should not raise
        assert worker.is_running() is True
        worker.stop()

    def test_stop_idempotent(self):
        worker = self._make_worker()
        worker.stop()  # should not raise on already-stopped
        assert worker.is_running() is False

    def test_stats_initial(self):
        worker = self._make_worker("stats-worker")
        stats = worker.stats()
        assert stats["worker_id"] == "stats-worker"
        assert stats["running"] is False
        assert stats["processed"] == 0
        assert stats["failed"] == 0

    def test_stats_worker_type(self):
        worker = self._make_worker()
        stats = worker.stats()
        assert stats["worker_type"] == "TestableWorker"


# ══════════════════════════════════════════════════════════════════════════════
#  INDEXING WORKER TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestIndexingWorker:
    """IndexingWorker: document indexing pipeline stage."""

    def test_creation(self):
        from app.distributed.indexing.indexing_worker import IndexingWorker
        worker = IndexingWorker(worker_id="idx-0")
        assert worker.worker_id == "idx-0"
        assert worker.is_running() is False

    def test_stats(self):
        from app.distributed.indexing.indexing_worker import IndexingWorker
        worker = IndexingWorker(worker_id="idx-0")
        stats = worker.stats()
        assert stats["worker_id"] == "idx-0"
        assert stats["worker_type"] == "IndexingWorker"
        assert stats["processed"] == 0
        assert stats["failed"] == 0

    def test_topics(self):
        from app.distributed.indexing.indexing_worker import IndexingWorker
        from app.events.topics import DOCUMENT_INDEXED, DOCUMENT_UPDATED
        worker = IndexingWorker(worker_id="idx-0")
        assert DOCUMENT_INDEXED in worker.topics
        assert DOCUMENT_UPDATED in worker.topics


# ══════════════════════════════════════════════════════════════════════════════
#  EMBEDDING WORKER TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestEmbeddingWorker:
    """EmbeddingWorker: document embedding pipeline stage."""

    def test_creation(self):
        from app.distributed.indexing.embedding_worker import EmbeddingWorker
        worker = EmbeddingWorker(worker_id="emb-0")
        assert worker.worker_id == "emb-0"
        assert worker.is_running() is False

    def test_stats(self):
        from app.distributed.indexing.embedding_worker import EmbeddingWorker
        worker = EmbeddingWorker(worker_id="emb-0")
        stats = worker.stats()
        assert stats["worker_id"] == "emb-0"
        assert stats["worker_type"] == "EmbeddingWorker"
        assert stats["processed"] == 0
        assert stats["failed"] == 0

    def test_embed_document_no_pipeline(self):
        from app.distributed.indexing.embedding_worker import EmbeddingWorker
        worker = EmbeddingWorker(worker_id="emb-0", embedding_pipeline=None)
        result = worker.embed_document(doc_id=1)
        assert result["status"] == "error"
        assert "No embedding pipeline" in result["error"]


# ══════════════════════════════════════════════════════════════════════════════
#  CHUNK WORKER TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestChunkWorker:
    """ChunkWorker: document chunking pipeline stage."""

    def test_creation(self):
        from app.distributed.indexing.chunk_worker import ChunkWorker
        worker = ChunkWorker(worker_id="chunk-0")
        assert worker.worker_id == "chunk-0"
        assert worker.is_running() is False

    def test_stats(self):
        from app.distributed.indexing.chunk_worker import ChunkWorker
        worker = ChunkWorker(worker_id="chunk-0")
        stats = worker.stats()
        assert stats["worker_id"] == "chunk-0"
        assert stats["worker_type"] == "ChunkWorker"
        assert stats["processed"] == 0

    def test_chunk_document_no_deps(self):
        from app.distributed.indexing.chunk_worker import ChunkWorker
        worker = ChunkWorker(worker_id="chunk-0", db=None, chunker=None)
        result = worker.chunk_document(doc_id=1)
        assert result["status"] == "error"
        assert result["chunk_count"] == 0


# ══════════════════════════════════════════════════════════════════════════════
#  DISTRIBUTED INDEXING PIPELINE TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestDistributedIndexingPipeline:
    """DistributedIndexingPipeline: multi-stage worker orchestration."""

    def test_creation(self):
        from app.distributed.indexing.pipeline import DistributedIndexingPipeline
        pipeline = DistributedIndexingPipeline()
        assert pipeline._is_setup is False
        assert pipeline._is_running is False

    def test_setup_creates_workers(self):
        from app.distributed.indexing.pipeline import DistributedIndexingPipeline
        from app.config import DistributedIndexingConfig
        config = DistributedIndexingConfig(
            num_indexing_workers=2, num_embedding_workers=2,
        )
        pipeline = DistributedIndexingPipeline(config=config)
        pipeline.setup()
        assert pipeline._is_setup is True
        workers = pipeline.get_workers()
        # 2 indexing + 1 chunk + 2 embedding = 5
        assert len(workers) == 5

    def test_get_workers_empty(self):
        from app.distributed.indexing.pipeline import DistributedIndexingPipeline
        pipeline = DistributedIndexingPipeline()
        workers = pipeline.get_workers()
        assert workers == [], "No workers before setup"

    def test_stats(self):
        from app.distributed.indexing.pipeline import DistributedIndexingPipeline
        from app.config import DistributedIndexingConfig
        config = DistributedIndexingConfig(
            num_indexing_workers=1, num_embedding_workers=1,
        )
        pipeline = DistributedIndexingPipeline(config=config)
        pipeline.setup()
        stats = pipeline.stats()
        assert "running" in stats
        assert "total_workers" in stats
        assert "total_processed" in stats
        assert "total_failed" in stats
        assert stats["total_workers"] == 3  # 1 indexing + 1 chunk + 1 embedding
        assert stats["total_processed"] == 0
        assert stats["total_failed"] == 0

    def test_process_document_not_running(self):
        from app.distributed.indexing.pipeline import DistributedIndexingPipeline
        pipeline = DistributedIndexingPipeline()
        result = pipeline.process_document(doc_id=1)
        assert result["status"] == "error"
        assert "not running" in result["error"].lower() or "No event bus" in result["error"]


# ══════════════════════════════════════════════════════════════════════════════
#  QDRANT VECTOR STORE TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestQdrantVectorStoreProtocol:
    """Structural check that QdrantVectorStore has VectorStore protocol methods."""

    def test_class_has_add_method(self):
        from app.vector_store.qdrant import QdrantVectorStore
        assert hasattr(QdrantVectorStore, "add")

    def test_class_has_search_method(self):
        from app.vector_store.qdrant import QdrantVectorStore
        assert hasattr(QdrantVectorStore, "search")

    def test_class_has_delete_method(self):
        from app.vector_store.qdrant import QdrantVectorStore
        assert hasattr(QdrantVectorStore, "delete")

    def test_class_has_save_method(self):
        from app.vector_store.qdrant import QdrantVectorStore
        assert hasattr(QdrantVectorStore, "save")

    def test_class_has_load_method(self):
        from app.vector_store.qdrant import QdrantVectorStore
        assert hasattr(QdrantVectorStore, "load")

    def test_class_has_total_vectors_property(self):
        from app.vector_store.qdrant import QdrantVectorStore
        assert hasattr(QdrantVectorStore, "total_vectors")

    def test_qdrant_config_defaults(self):
        from app.config import QdrantConfig
        cfg = QdrantConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 6333
        assert cfg.grpc_port == 6334
        assert cfg.collection_name == "search_engine"
        assert cfg.vector_size == 384
        assert cfg.distance == "Cosine"
        assert cfg.hnsw_m == 16
        assert cfg.hnsw_ef_construct == 100


# ══════════════════════════════════════════════════════════════════════════════
#  VECTOR STORE FACTORY TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestVectorStoreFactory:
    """Vector store factory: selects Qdrant or FAISS based on availability."""

    def test_factory_default_faiss(self, tmp_path):
        """Factory should fall back to FaissVectorStore when Qdrant is unavailable."""
        import sys
        from unittest.mock import MagicMock
        from app.config import EngineConfig, VectorStoreConfig
        from app.vector_store.store import FaissVectorStore

        saved = sys.modules.get("app.vector_store.qdrant")
        sys.modules["app.vector_store.qdrant"] = None  # force ImportError
        try:
            import importlib
            import app.vector_store.factory as fmod
            importlib.reload(fmod)
            config = EngineConfig(
                vector_store=VectorStoreConfig(
                    index_path=tmp_path / "idx", dimension=16,
                ),
            )
            store = fmod.create_vector_store(config)
            assert isinstance(store, FaissVectorStore)
        finally:
            if saved is not None:
                sys.modules["app.vector_store.qdrant"] = saved
            elif "app.vector_store.qdrant" in sys.modules:
                del sys.modules["app.vector_store.qdrant"]

    def test_factory_config(self):
        """Factory reads qdrant config from EngineConfig."""
        from app.config import EngineConfig, QdrantConfig
        config = EngineConfig(
            qdrant=QdrantConfig(host="qdrant.example.com", port=6333),
        )
        assert config.qdrant.host == "qdrant.example.com"
        assert config.qdrant.port == 6333


# ══════════════════════════════════════════════════════════════════════════════
#  GATEWAY MODELS TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestGatewayModels:
    """GatewayRequest / GatewayResponse data transfer objects."""

    def test_request_defaults(self):
        from app.gateway.models import GatewayRequest
        req = GatewayRequest(query="test query")
        assert req.query == "test query"
        assert req.mode == "hybrid"
        assert req.top_k == 10
        assert req.fusion == "rrf"
        assert req.rerank is True
        assert req.client_id == ""
        assert req.timeout_sec == 30.0

    def test_request_custom(self):
        from app.gateway.models import GatewayRequest
        req = GatewayRequest(
            query="custom", mode="bm25", top_k=5,
            fusion="combsum", rerank=False, client_id="client-1",
        )
        assert req.mode == "bm25"
        assert req.top_k == 5
        assert req.fusion == "combsum"
        assert req.rerank is False
        assert req.client_id == "client-1"

    def test_response_creation(self):
        from app.gateway.models import GatewayResponse
        resp = GatewayResponse(
            query="test", mode="hybrid",
            results=[{"doc_id": 1}], total_results=1,
            latency_ms=5.0, cache_hit=True,
        )
        assert resp.query == "test"
        assert resp.mode == "hybrid"
        assert resp.total_results == 1
        assert resp.latency_ms == 5.0
        assert resp.cache_hit is True

    def test_response_defaults(self):
        from app.gateway.models import GatewayResponse
        resp = GatewayResponse(
            query="q", mode="bm25", results=[], total_results=0,
            latency_ms=1.0,
        )
        assert resp.cache_hit is False
        assert resp.fusion_strategy == ""
        assert resp.reranked is False
        assert resp.metadata == {}


# ══════════════════════════════════════════════════════════════════════════════
#  GATEWAY CACHE TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestGatewayCache:
    """GatewayCache: two-tier (L1 in-process LRU + optional L2 Redis)."""

    def test_put_get(self):
        from app.gateway.cache import GatewayCache
        cache = GatewayCache(redis_client=None, l1_capacity=100)
        cache.put("key1", {"results": [1, 2, 3]})
        result = cache.get("key1")
        assert result == {"results": [1, 2, 3]}

    def test_cache_miss(self):
        from app.gateway.cache import GatewayCache
        cache = GatewayCache(redis_client=None, l1_capacity=100)
        assert cache.get("nonexistent") is None

    def test_invalidate(self):
        from app.gateway.cache import GatewayCache
        cache = GatewayCache(redis_client=None, l1_capacity=100)
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        count = cache.invalidate("*")
        assert count == 2
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_make_key(self):
        from app.gateway.cache import GatewayCache
        key1 = GatewayCache.make_key("python", "hybrid", 10, "rrf")
        key2 = GatewayCache.make_key("python", "hybrid", 10, "rrf")
        key3 = GatewayCache.make_key("python", "hybrid", 5, "rrf")
        assert key1 == key2, "Same params should produce same key"
        assert key1 != key3, "Different top_k should produce different key"
        assert len(key1) == 32, "Key should be 32 chars (SHA-256 truncated)"

    def test_make_key_case_insensitive(self):
        from app.gateway.cache import GatewayCache
        key_lower = GatewayCache.make_key("Python", "hybrid", 10, "rrf")
        key_upper = GatewayCache.make_key("python", "hybrid", 10, "rrf")
        assert key_lower == key_upper, "Keys should be case-insensitive"

    def test_stats(self):
        from app.gateway.cache import GatewayCache
        cache = GatewayCache(redis_client=None, l1_capacity=100)
        cache.put("a", 1)
        cache.get("a")        # L1 hit
        cache.get("missing")  # L1 miss
        stats = cache.stats()
        assert stats["l1_hits"] == 1
        assert stats["l1_misses"] == 1
        assert stats["l1_size"] == 1
        assert stats["l1_capacity"] == 100
        assert stats["l2_enabled"] is False

    def test_lru_eviction(self):
        from app.gateway.cache import GatewayCache
        cache = GatewayCache(redis_client=None, l1_capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # should evict "a"
        assert cache.get("a") is None, "Oldest entry should be evicted"
        assert cache.get("b") == 2
        assert cache.get("c") == 3


# ══════════════════════════════════════════════════════════════════════════════
#  QUERY ROUTER TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestQueryRouter:
    """QueryRouter: intent-based retrieval backend selection."""

    def test_route_default_no_classifier(self):
        from app.gateway.router import QueryRouter
        router = QueryRouter(classifier=None)
        mode = router.route("how does gradient descent work")
        assert mode == "hybrid", "Without classifier, default should be hybrid"

    def test_route_with_hint(self):
        from app.gateway.router import QueryRouter
        router = QueryRouter(classifier=None)
        mode = router.route("anything", hint="bm25")
        assert mode == "bm25", "Hint should override classification"

    def test_route_hint_case_insensitive(self):
        from app.gateway.router import QueryRouter
        router = QueryRouter(classifier=None)
        mode = router.route("anything", hint="PIPELINE")
        assert mode == "pipeline", "Hint should be case-insensitive"

    def test_route_invalid_hint_falls_back(self):
        from app.gateway.router import QueryRouter
        router = QueryRouter(classifier=None)
        mode = router.route("query", hint="invalid_mode")
        assert mode == "hybrid", "Invalid hint should fall back to default"


# ══════════════════════════════════════════════════════════════════════════════
#  RETRIEVAL GATEWAY TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestRetrievalGateway:
    """RetrievalGateway: central retrieval orchestration service."""

    def test_creation(self):
        from app.gateway.service import RetrievalGateway
        from app.config import GatewayConfig
        gw = RetrievalGateway(config=GatewayConfig())
        assert gw is not None

    def test_stats(self):
        from app.gateway.service import RetrievalGateway
        from app.config import GatewayConfig
        gw = RetrievalGateway(config=GatewayConfig())
        stats = gw.stats()
        assert "total_requests" in stats
        assert "total_errors" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "rate_limited" in stats
        assert "config" in stats
        assert stats["total_requests"] == 0

    def test_search_no_backends(self):
        """Search with no backends should return empty results."""
        from app.gateway.service import RetrievalGateway
        from app.config import GatewayConfig
        gw = RetrievalGateway(config=GatewayConfig())
        resp = gw.search(query="test", mode="hybrid", top_k=5)
        assert resp.query == "test"
        assert resp.results == []
        assert resp.total_results == 0

    def test_invalidate_cache(self):
        from app.gateway.service import RetrievalGateway
        from app.config import GatewayConfig
        gw = RetrievalGateway(config=GatewayConfig())
        count = gw.invalidate_cache("*")
        assert count == 0  # nothing to invalidate


# ══════════════════════════════════════════════════════════════════════════════
#  API ENDPOINT TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestBatch2APIEndpoints:
    """Phase 8 Batch 2 API endpoints using FastAPI TestClient."""

    def _make_client(self, tmp_path):
        from app.config import EngineConfig, DatabaseConfig, VectorStoreConfig
        config = EngineConfig(
            database=DatabaseConfig(db_path=tmp_path / "test.db"),
            vector_store=VectorStoreConfig(
                index_path=tmp_path / "idx", dimension=16,
            ),
        )
        from app.api.routes import create_app
        from fastapi.testclient import TestClient
        return TestClient(create_app(config))

    def test_gateway_stats(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.get("/gateway/stats")
            assert resp.status_code == 200
            data = resp.json()
            # Gateway may or may not be initialized depending on lifespan
            assert isinstance(data, dict)

    def test_gateway_cache_stats(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.get("/gateway/cache/stats")
            assert resp.status_code == 200
            data = resp.json()
            assert "l1_size" in data
            assert "l1_capacity" in data

    def test_distributed_crawler_stats(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.get("/distributed/crawler/stats")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "available"
            assert "config" in data
            assert "max_workers" in data["config"]

    def test_distributed_indexing_stats(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.get("/distributed/indexing/stats")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "available"
            assert "config" in data
            assert "num_indexing_workers" in data["config"]

    def test_vector_store_backend(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.get("/vector-store/backend")
            assert resp.status_code == 200
            data = resp.json()
            assert "backend" in data
            assert data["backend"] == "FaissVectorStore"
            assert "total_vectors" in data


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG COMPATIBILITY TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestBatch2ConfigCompat:
    """Phase 8 Batch 2 config backward compatibility."""

    def test_engine_config_zero_arg(self):
        from app.config import EngineConfig
        config = EngineConfig()
        # All batch 2 configs should have defaults
        assert config.kafka is not None
        assert config.distributed_crawler is not None
        assert config.distributed_indexing is not None
        assert config.qdrant is not None
        assert config.gateway is not None
        # Phase 8 configs still present
        assert config.events is not None
        assert config.redis is not None
        assert config.postgres is not None

    def test_kafka_config_defaults(self):
        from app.config import KafkaConfig
        cfg = KafkaConfig()
        assert cfg.bootstrap_servers == "localhost:9092"
        assert cfg.group_id == "search-engine"
        assert cfg.enable_auto_commit is True
        assert cfg.max_poll_records == 100

    def test_qdrant_config_defaults(self):
        from app.config import QdrantConfig
        cfg = QdrantConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 6333
        assert cfg.collection_name == "search_engine"
        assert cfg.vector_size == 384
        assert cfg.distance == "Cosine"

    def test_gateway_config_defaults(self):
        from app.config import GatewayConfig
        cfg = GatewayConfig()
        assert cfg.cache_ttl == 300
        assert cfg.cache_max_size == 1000
        assert cfg.rate_limit_rpm == 120
        assert cfg.enable_cache is True
        assert cfg.default_fusion == "rrf"
        assert cfg.default_rerank is True

    def test_distributed_crawler_config_defaults(self):
        from app.config import DistributedCrawlerConfig
        cfg = DistributedCrawlerConfig()
        assert cfg.max_workers == 4
        assert cfg.frontier_max_size == 100000
        assert cfg.batch_size == 10
        assert cfg.dedup_backend == "memory"

    def test_distributed_indexing_config_defaults(self):
        from app.config import DistributedIndexingConfig
        cfg = DistributedIndexingConfig()
        assert cfg.num_indexing_workers == 2
        assert cfg.num_embedding_workers == 2
        assert cfg.batch_size == 10
        assert cfg.auto_embed is True
        assert cfg.retry_on_failure is True
