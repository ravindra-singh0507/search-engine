"""
Phase 4 — Test Suite

Tests all Phase 4 components using MockEmbeddingProvider (no ML deps needed).

Coverage:
  - Chunking (FixedSize, SlidingWindow)
  - EmbeddingCache
  - EmbeddingProvider (Mock)
  - FaissVectorStore
  - EmbeddingPipeline (integration)
  - SemanticSearchService (integration)
  - HybridSearchService + RRF (integration)
  - Evaluation metrics
  - RetrievalEvaluator
  - API endpoints (/semantic-search, /hybrid-search, /embeddings/*)
"""

import json
import math
import pytest
from pathlib import Path
from unittest.mock import MagicMock

# ── Unit-test-safe imports (no ML deps) ──────────────────────────────────────

from app.chunking.chunker import (
    FixedSizeChunker, SlidingWindowChunker, make_chunker,
)
from app.embeddings.provider import MockEmbeddingProvider
from app.embeddings.cache import EmbeddingCache
from app.evaluation.metrics import (
    precision_at_k, recall_at_k, reciprocal_rank,
    mean_reciprocal_rank, average_precision, mean_average_precision,
    ndcg_at_k, compute_all_metrics,
)
from app.hybrid_search.hybrid_service import reciprocal_rank_fusion, linear_combination
from app.config import ChunkingConfig, EmbeddingConfig, VectorStoreConfig


# ══════════════════════════════════════════════════════════════════════════════
# Chunking
# ══════════════════════════════════════════════════════════════════════════════

class TestFixedSizeChunker:
    def _cfg(self, size: int = 4, overlap: int = 0, min_words: int = 1):
        return ChunkingConfig(strategy="fixed", chunk_size=size,
                              chunk_overlap=overlap, min_chunk_words=min_words)

    def test_basic_split(self):
        c = FixedSizeChunker(self._cfg(size=3))
        chunks = c.chunk("a b c d e f", doc_id=1)
        assert len(chunks) == 2
        assert chunks[0].text == "a b c"
        assert chunks[1].text == "d e f"

    def test_chunk_ids_are_unique(self):
        c = FixedSizeChunker(self._cfg(size=2))
        chunks = c.chunk("a b c d e f", doc_id=5)
        ids = [ch.chunk_id for ch in chunks]
        assert len(ids) == len(set(ids))

    def test_short_doc_single_chunk(self):
        c = FixedSizeChunker(self._cfg(size=100))
        chunks = c.chunk("hello world", doc_id=3)
        assert len(chunks) == 1
        assert chunks[0].text == "hello world"

    def test_offsets_are_correct(self):
        c = FixedSizeChunker(self._cfg(size=3))
        chunks = c.chunk("a b c d e f", doc_id=1)
        assert chunks[0].start_offset == 0
        assert chunks[0].end_offset   == 3
        assert chunks[1].start_offset == 3
        assert chunks[1].end_offset   == 6

    def test_word_count_stored(self):
        c = FixedSizeChunker(self._cfg(size=3))
        chunks = c.chunk("a b c d", doc_id=1)
        assert all(ch.word_count > 0 for ch in chunks)

    def test_empty_text_returns_empty(self):
        c = FixedSizeChunker(self._cfg())
        assert c.chunk("", doc_id=1) == []


class TestSlidingWindowChunker:
    def _cfg(self, size: int = 4, overlap: int = 2, min_words: int = 1):
        return ChunkingConfig(strategy="sliding_window",
                              chunk_size=size, chunk_overlap=overlap,
                              min_chunk_words=min_words)

    def test_produces_overlapping_chunks(self):
        c = SlidingWindowChunker(self._cfg(size=4, overlap=2))
        chunks = c.chunk("a b c d e f g h", doc_id=1)
        # stride = 4-2=2; starts at 0, 2, 4, 6
        assert len(chunks) >= 2
        # Check overlap: adjacent chunks share words
        w0 = set(chunks[0].text.split())
        w1 = set(chunks[1].text.split())
        assert len(w0 & w1) > 0

    def test_every_phrase_covered(self):
        # "hello world" should appear fully in at least one chunk
        c = SlidingWindowChunker(self._cfg(size=4, overlap=2))
        text = "apple banana cherry date elderberry fig"
        chunks = c.chunk(text, doc_id=2)
        texts = " ".join(ch.text for ch in chunks)
        assert "banana" in texts
        assert "elderberry" in texts

    def test_short_doc_single_chunk(self):
        c = SlidingWindowChunker(self._cfg(size=100, overlap=10))
        chunks = c.chunk("short text here", doc_id=1)
        assert len(chunks) == 1

    def test_make_chunker_factory_sliding(self):
        cfg = ChunkingConfig(strategy="sliding_window")
        chunker = make_chunker(cfg)
        assert isinstance(chunker, SlidingWindowChunker)

    def test_make_chunker_factory_fixed(self):
        cfg = ChunkingConfig(strategy="fixed")
        chunker = make_chunker(cfg)
        assert isinstance(chunker, FixedSizeChunker)


# ══════════════════════════════════════════════════════════════════════════════
# MockEmbeddingProvider
# ══════════════════════════════════════════════════════════════════════════════

class TestMockEmbeddingProvider:
    def _provider(self, dim: int = 16):
        return MockEmbeddingProvider(dim=dim)

    def test_dimension_property(self):
        p = self._provider(32)
        assert p.dimension == 32

    def test_embed_texts_returns_correct_count(self):
        p = self._provider()
        vecs = p.embed_texts(["hello", "world", "test"])
        assert len(vecs) == 3

    def test_embed_returns_unit_vectors(self):
        p = self._provider(16)
        for vec in p.embed_texts(["hello world", "python", "search engine"]):
            norm = math.sqrt(sum(v * v for v in vec))
            assert abs(norm - 1.0) < 1e-5

    def test_deterministic_embeddings(self):
        p = self._provider()
        v1 = p.embed_texts(["same text"])[0]
        v2 = p.embed_texts(["same text"])[0]
        assert v1 == v2

    def test_different_texts_different_vectors(self):
        p = self._provider(32)
        v1 = p.embed_query("python")
        v2 = p.embed_query("database")
        assert v1 != v2

    def test_embed_query_unit_vector(self):
        p = self._provider(8)
        v = p.embed_query("test query")
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-5


# ══════════════════════════════════════════════════════════════════════════════
# EmbeddingCache
# ══════════════════════════════════════════════════════════════════════════════

class TestEmbeddingCache:
    def test_put_and_get(self, tmp_path):
        from app.database.db import Database
        db = Database(tmp_path / "t.db")
        db.connect()
        cache = EmbeddingCache(db)
        cache.put("hello", "model-v1", [0.1, 0.2, 0.3])
        result = cache.get("hello", "model-v1")
        assert result == [0.1, 0.2, 0.3]
        db.close()

    def test_miss_returns_none(self, tmp_path):
        from app.database.db import Database
        db = Database(tmp_path / "t.db")
        db.connect()
        cache = EmbeddingCache(db)
        assert cache.get("unknown", "model-v1") is None
        db.close()

    def test_model_name_is_cache_key_component(self, tmp_path):
        from app.database.db import Database
        db = Database(tmp_path / "t.db")
        db.connect()
        cache = EmbeddingCache(db)
        cache.put("hello", "model-A", [1.0])
        # Different model → different key
        assert cache.get("hello", "model-B") is None
        db.close()

    def test_stats(self, tmp_path):
        from app.database.db import Database
        db = Database(tmp_path / "t.db")
        db.connect()
        cache = EmbeddingCache(db)
        cache.get("x", "m")     # miss
        cache.put("x", "m", [0.5])
        cache.get("x", "m")     # hit
        s = cache.stats()
        assert s["hits"]   == 1
        assert s["misses"] == 1
        assert s["hit_rate"] == 0.5
        db.close()


# ══════════════════════════════════════════════════════════════════════════════
# FaissVectorStore
# ══════════════════════════════════════════════════════════════════════════════

faiss = pytest.importorskip("faiss", reason="faiss-cpu not installed")


class TestFaissVectorStore:
    def _store(self, dim: int = 16):
        from app.vector_store.store import FaissVectorStore
        cfg = VectorStoreConfig(dimension=dim)
        return FaissVectorStore(cfg)

    def _unit_vec(self, dim: int, seed: int) -> list[float]:
        import numpy as np
        rng = np.random.RandomState(seed)
        v   = rng.randn(dim).astype(np.float32)
        v  /= np.linalg.norm(v)
        return v.tolist()

    def test_add_and_search(self):
        store = self._store(16)
        v1 = self._unit_vec(16, 1)
        v2 = self._unit_vec(16, 2)
        store.add(["c1", "c2"], [v1, v2])
        results = store.search(v1, top_k=2)
        assert results[0][0] == "c1"
        assert results[0][1] > 0.99   # cosine ≈ 1.0 for same vector

    def test_total_vectors_count(self):
        store = self._store()
        store.add(["a", "b", "c"], [self._unit_vec(16, i) for i in range(3)])
        assert store.total_vectors == 3

    def test_soft_delete_filters_results(self):
        store = self._store(16)
        v1 = self._unit_vec(16, 1)
        v2 = self._unit_vec(16, 2)
        store.add(["keep", "delete"], [v1, v2])
        store.delete(["delete"])
        results = store.search(v2, top_k=5)
        chunk_ids = [r[0] for r in results]
        assert "delete" not in chunk_ids

    def test_save_and_load(self, tmp_path):
        from app.vector_store.store import FaissVectorStore
        store = FaissVectorStore(VectorStoreConfig(
            index_path=tmp_path / "idx", dimension=16
        ))
        v = self._unit_vec(16, 99)
        store.add(["chunk_99"], [v])
        store.save(tmp_path / "idx")

        store2 = FaissVectorStore(VectorStoreConfig(
            index_path=tmp_path / "idx", dimension=16
        ))
        store2.load(tmp_path / "idx")
        assert store2.total_vectors == 1
        results = store2.search(v, top_k=1)
        assert results[0][0] == "chunk_99"

    def test_skip_duplicate_chunk_ids(self):
        store = self._store()
        v = self._unit_vec(16, 1)
        store.add(["dup"], [v])
        store.add(["dup"], [v])
        assert store.total_vectors == 1

    def test_empty_store_returns_empty(self):
        store = self._store()
        results = store.search(self._unit_vec(16, 1), top_k=5)
        assert results == []


# ══════════════════════════════════════════════════════════════════════════════
# EmbeddingPipeline (integration)
# ══════════════════════════════════════════════════════════════════════════════

class TestEmbeddingPipeline:
    def _setup(self, tmp_path):
        from app.database.db import Database
        from app.indexer.indexer import Indexer
        from app.tokenizer.tokenizer import Tokenizer
        from app.embeddings.pipeline import EmbeddingPipeline
        from app.vector_store.store import FaissVectorStore

        db = Database(tmp_path / "t.db")
        db.connect()
        tok = Tokenizer()
        idx = Indexer(db, tok)
        idx.index_document("Python", "Python is great for programming", source="s1")
        idx.index_document("Java",   "Java is used for enterprise apps", source="s2")

        provider     = MockEmbeddingProvider(dim=16)
        cache        = EmbeddingCache(db)
        vector_store = FaissVectorStore(VectorStoreConfig(dimension=16))
        chunker      = make_chunker(ChunkingConfig(chunk_size=50, min_chunk_words=1))
        pipeline     = EmbeddingPipeline(
            db=db, provider=provider, cache=cache,
            vector_store=vector_store, chunker=chunker,
            emb_config=EmbeddingConfig(cache_embeddings=True, batch_size=4),
            vs_config=VectorStoreConfig(dimension=16,
                                        index_path=tmp_path / "idx"),
        )
        return db, pipeline, vector_store

    def test_index_document_embeds_chunks(self, tmp_path):
        db, pipeline, vs = self._setup(tmp_path)
        stats = pipeline.index_document(1)
        assert stats.docs_processed == 1
        assert stats.chunks_embedded >= 1
        assert vs.total_vectors >= 1
        db.close()

    def test_index_all_embeds_all_docs(self, tmp_path):
        db, pipeline, vs = self._setup(tmp_path)
        stats = pipeline.index_all()
        assert stats.docs_processed == 2
        db.close()

    def test_incremental_skips_embedded_docs(self, tmp_path):
        db, pipeline, vs = self._setup(tmp_path)
        pipeline.index_document(1)
        stats2 = pipeline.index_document(1)   # should skip
        assert stats2.docs_skipped == 1
        db.close()

    def test_cache_used_on_second_call(self, tmp_path):
        db, pipeline, vs = self._setup(tmp_path)
        pipeline.index_document(1)
        cache_hits_before = pipeline.cache.stats()["hits"]
        # Re-embed with force=True — same text → cache hits
        pipeline.index_document(1, force=True)
        assert pipeline.cache.stats()["hits"] > cache_hits_before
        db.close()

    def test_remove_document_cleans_up(self, tmp_path):
        db, pipeline, vs = self._setup(tmp_path)
        pipeline.index_all()
        total_before = vs.total_vectors
        pipeline.remove_document(1)
        assert vs.total_vectors < total_before
        db.close()


# ══════════════════════════════════════════════════════════════════════════════
# SemanticSearchService (integration)
# ══════════════════════════════════════════════════════════════════════════════

class TestSemanticSearchService:
    def _setup(self, tmp_path):
        from app.database.db import Database
        from app.indexer.indexer import Indexer
        from app.tokenizer.tokenizer import Tokenizer
        from app.embeddings.pipeline import EmbeddingPipeline
        from app.vector_store.store import FaissVectorStore
        from app.semantic_search.semantic_service import SemanticSearchService

        db = Database(tmp_path / "t.db")
        db.connect()
        tok = Tokenizer()
        idx = Indexer(db, tok)
        idx.index_document("Python", "Python is great for programming", source="s1")
        idx.index_document("Java",   "Java is used for enterprise apps", source="s2")

        provider     = MockEmbeddingProvider(dim=16)
        cache        = EmbeddingCache(db)
        vector_store = FaissVectorStore(VectorStoreConfig(dimension=16))
        chunker      = make_chunker(ChunkingConfig(chunk_size=50, min_chunk_words=1))
        pipeline     = EmbeddingPipeline(
            db=db, provider=provider, cache=cache,
            vector_store=vector_store, chunker=chunker,
            emb_config=EmbeddingConfig(cache_embeddings=False, batch_size=4),
            vs_config=VectorStoreConfig(dimension=16, index_path=tmp_path/"idx"),
        )
        pipeline.index_all()

        svc = SemanticSearchService(db=db, provider=provider, vector_store=vector_store)
        return db, svc

    def test_search_returns_results(self, tmp_path):
        db, svc = self._setup(tmp_path)
        resp = svc.search("programming language", top_k=5)
        assert resp.total_results >= 1
        assert all(r.semantic_score >= 0 for r in resp.results)
        db.close()

    def test_search_scores_in_valid_range(self, tmp_path):
        db, svc = self._setup(tmp_path)
        resp = svc.search("java enterprise", top_k=5)
        for r in resp.results:
            assert 0.0 <= r.semantic_score <= 1.01
        db.close()

    def test_empty_index_returns_empty(self, tmp_path):
        from app.database.db import Database
        from app.vector_store.store import FaissVectorStore
        from app.semantic_search.semantic_service import SemanticSearchService
        db = Database(tmp_path / "empty.db")
        db.connect()
        vs  = FaissVectorStore(VectorStoreConfig(dimension=16))
        svc = SemanticSearchService(db=db,
                                    provider=MockEmbeddingProvider(dim=16),
                                    vector_store=vs)
        resp = svc.search("anything", top_k=5)
        assert resp.total_results == 0
        db.close()

    def test_explain_returns_dict(self, tmp_path):
        db, svc = self._setup(tmp_path)
        result = svc.explain("python programming", doc_id=1)
        assert "doc_id" in result
        assert "best_semantic_score" in result
        db.close()


# ══════════════════════════════════════════════════════════════════════════════
# Hybrid Search + RRF
# ══════════════════════════════════════════════════════════════════════════════

class TestRRF:
    def test_basic_fusion(self):
        list1 = [(1, 5.0), (2, 3.0), (3, 1.0)]
        list2 = [(2, 0.9), (1, 0.7), (4, 0.5)]
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        # doc 1 is rank 1 in list1, rank 2 in list2 → high score
        # doc 2 is rank 2 in list1, rank 1 in list2 → also high
        scores = dict(fused)
        assert 1 in scores and 2 in scores

    def test_rrf_score_formula(self):
        # doc 1 is rank 1 in both lists: score = 1/61 + 1/61 ≈ 0.0328
        list1 = [(1, 10.0)]
        list2 = [(1, 0.9)]
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        expected = 2.0 / 61.0
        assert abs(fused[0][1] - expected) < 1e-9

    def test_rrf_result_sorted_descending(self):
        list1 = [(1, 5.0), (2, 3.0)]
        list2 = [(3, 0.9), (1, 0.7)]
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        scores = [s for _, s in fused]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_absent_doc_gets_no_contribution(self):
        list1 = [(1, 5.0)]
        list2 = [(2, 0.9)]   # doc 1 not in list2
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        scores = dict(fused)
        assert scores[1] == pytest.approx(1.0 / 61.0)

    def test_linear_combination_normalises(self):
        list1 = [(1, 100.0), (2, 50.0)]
        list2 = [(1, 0.9),   (2, 0.5)]
        fused = linear_combination(list1, list2, 0.5, 0.5)
        scores = dict(fused)
        assert scores[1] > scores[2]

    def test_rrf_doc_only_in_one_list(self):
        list1 = [(1, 5.0), (2, 3.0)]
        list2 = [(1, 0.9)]
        fused  = reciprocal_rank_fusion([list1, list2], k=60)
        scores = dict(fused)
        # doc 2 only in list1 so it gets 1/62, doc 1 gets 1/61+1/61
        assert scores[1] > scores[2]


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation Metrics
# ══════════════════════════════════════════════════════════════════════════════

class TestEvalMetrics:
    def test_precision_at_k_perfect(self):
        assert precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0

    def test_precision_at_k_zero(self):
        assert precision_at_k([4, 5, 6], {1, 2, 3}, k=3) == 0.0

    def test_precision_at_k_partial(self):
        assert precision_at_k([1, 4, 2], {1, 2, 3}, k=3) == pytest.approx(2 / 3)

    def test_recall_at_k_perfect(self):
        assert recall_at_k([1, 2, 3, 4], {1, 2, 3}, k=3) == 1.0

    def test_recall_at_k_partial(self):
        assert recall_at_k([1, 4, 5], {1, 2, 3}, k=3) == pytest.approx(1 / 3)

    def test_recall_at_k_empty_relevant(self):
        assert recall_at_k([1, 2], set(), k=2) == 0.0

    def test_reciprocal_rank_first(self):
        assert reciprocal_rank([3, 1, 2], {1}) == pytest.approx(0.5)

    def test_reciprocal_rank_not_found(self):
        assert reciprocal_rank([4, 5, 6], {1, 2}) == 0.0

    def test_mrr(self):
        retrieved = [[1, 2, 3], [4, 1, 2]]
        relevant  = [{1}, {1}]
        mrr = mean_reciprocal_rank(retrieved, relevant)
        assert mrr == pytest.approx((1.0 + 0.5) / 2)

    def test_average_precision(self):
        # relevant = {1, 3}; retrieved = [1, 2, 3, 4]
        # Hits at ranks 1 and 3: AP = (1/1 + 2/3) / 2 = 0.833
        ap = average_precision([1, 2, 3, 4], {1, 3})
        assert abs(ap - (1.0 + 2.0 / 3) / 2) < 1e-9

    def test_map(self):
        retrieved = [[1, 2, 3], [4, 5, 6]]
        relevant  = [{1}, {4}]
        result    = mean_average_precision(retrieved, relevant)
        assert result == pytest.approx(1.0)   # rank 1 hit in both

    def test_ndcg_at_k_perfect(self):
        rel_scores = {1: 3.0, 2: 2.0, 3: 1.0}
        ideal      = [1, 2, 3]
        assert ndcg_at_k(ideal, rel_scores, k=3) == pytest.approx(1.0)

    def test_ndcg_at_k_no_relevant(self):
        assert ndcg_at_k([1, 2, 3], {}, k=3) == 0.0

    def test_compute_all_metrics(self):
        retrieved = [[1, 2, 3], [4, 5, 1]]
        relevant  = [{1, 2}, {4}]
        metrics   = compute_all_metrics(retrieved, relevant, k_values=[1, 3])
        assert "P@1"    in metrics
        assert "R@3"    in metrics
        assert "MRR"    in metrics
        assert "MAP"    in metrics
        assert "NDCG@1" in metrics
        for v in metrics.values():
            assert 0.0 <= v <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# RetrievalEvaluator
# ══════════════════════════════════════════════════════════════════════════════

class TestRetrievalEvaluator:
    def _eval_query(self):
        from app.evaluation.evaluator import EvalQuery
        return EvalQuery(
            query_id="q1",
            query="python programming",
            relevant_doc_ids=[1, 3],
            relevance_scores={1: 2.0, 3: 1.0},
        )

    def test_evaluator_runs(self):
        from app.evaluation.evaluator import RetrievalEvaluator
        ev = RetrievalEvaluator()
        ev.add_system("test", lambda q, k: [1, 2, 3])
        report = ev.run(dataset=[self._eval_query()], top_k=5)
        assert "test" in report
        assert "MRR"  in report["test"]

    def test_perfect_retrieval_scores_one(self):
        from app.evaluation.evaluator import RetrievalEvaluator, EvalQuery
        from app.config import EvaluationConfig
        ev = RetrievalEvaluator(EvaluationConfig(k_values=[1, 2, 3, 5]))
        ev.add_system("perfect", lambda q, k: [1, 3, 2])
        q  = EvalQuery("q1", "query", [1, 3], {1: 1.0, 3: 1.0})
        r  = ev.run(dataset=[q], top_k=5)
        assert r["perfect"]["P@2"] == pytest.approx(1.0)

    def test_zero_retrieval_scores_zero(self):
        from app.evaluation.evaluator import RetrievalEvaluator, EvalQuery
        ev = RetrievalEvaluator()
        ev.add_system("bad", lambda q, k: [99, 98, 97])
        q  = EvalQuery("q1", "query", [1, 2], {1: 1.0, 2: 1.0})
        r  = ev.run(dataset=[q], top_k=5)
        assert r["bad"]["MRR"] == 0.0

    def test_comparison_table_renders(self):
        from app.evaluation.evaluator import RetrievalEvaluator, EvalQuery
        ev = RetrievalEvaluator()
        ev.add_system("sys_a", lambda q, k: [1])
        ev.add_system("sys_b", lambda q, k: [2])
        q = EvalQuery("q1", "test", [1], {1: 1.0})
        report = ev.run(dataset=[q], top_k=5)
        table  = RetrievalEvaluator.comparison_table(report)
        assert "sys_a" in table
        assert "sys_b" in table

    def test_empty_dataset_returns_empty(self):
        from app.evaluation.evaluator import RetrievalEvaluator
        ev = RetrievalEvaluator()
        ev.add_system("x", lambda q, k: [1])
        assert ev.run(dataset=[], top_k=5) == {}

    def test_load_eval_dataset(self, tmp_path):
        from app.evaluation.evaluator import load_eval_dataset
        data = [{"query_id": "q1", "query": "test",
                 "relevant_doc_ids": [1, 2], "relevance_scores": {"1": 2, "2": 1}}]
        p = tmp_path / "eval.json"
        p.write_text(json.dumps(data))
        qs = load_eval_dataset(p)
        assert len(qs) == 1
        assert qs[0].query == "test"
        assert 1 in qs[0].relevant_doc_ids


# ══════════════════════════════════════════════════════════════════════════════
# API endpoint smoke tests (Phase 4)
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase4APIEndpoints:
    def _client(self, tmp_path):
        from fastapi.testclient import TestClient
        from app.api.routes import create_app
        from app.config import EngineConfig, DatabaseConfig, VectorStoreConfig
        cfg = EngineConfig(
            database=DatabaseConfig(db_path=tmp_path / "t.db"),
            vector_store=VectorStoreConfig(
                index_path=tmp_path / "idx", dimension=16
            ),
        )
        return TestClient(create_app(cfg))

    def test_embeddings_stats(self, tmp_path):
        with self._client(tmp_path) as c:
            r = c.get("/embeddings/stats")
            assert r.status_code == 200
            body = r.json()
            assert "vector_store" in body

    def test_semantic_search_empty_index(self, tmp_path):
        with self._client(tmp_path) as c:
            r = c.get("/semantic-search?q=python")
            assert r.status_code == 200
            assert r.json()["total_results"] == 0

    def test_hybrid_search_empty_index(self, tmp_path):
        with self._client(tmp_path) as c:
            r = c.get("/hybrid-search?q=python")
            assert r.status_code == 200

    def test_vector_store_stats(self, tmp_path):
        with self._client(tmp_path) as c:
            r = c.get("/vector-store/stats")
            assert r.status_code == 200

    def test_embeddings_reindex_starts(self, tmp_path):
        with self._client(tmp_path) as c:
            # Index a document first
            c.post("/index", json={"title": "T", "content": "some content here"})
            # Use sync=true so the thread completes before the context exits
            r = c.post("/embeddings/reindex", json={"force": False, "sync": True})
            assert r.status_code in (200, 409)

    def test_clear_embedding_cache(self, tmp_path):
        with self._client(tmp_path) as c:
            r = c.delete("/embeddings/cache")
            assert r.status_code == 200

    def test_semantic_explain_missing_doc(self, tmp_path):
        with self._client(tmp_path) as c:
            r = c.get("/semantic-search/explain?q=test&doc_id=999")
            assert r.status_code == 200
            assert "error" in r.json()

    def test_hybrid_explain_missing_doc(self, tmp_path):
        with self._client(tmp_path) as c:
            r = c.get("/hybrid-search/explain?q=test&doc_id=999")
            assert r.status_code == 200
            assert "error" in r.json()


# ══════════════════════════════════════════════════════════════════════════════
# BUG REGRESSION TESTS (audit fixes)
# ══════════════════════════════════════════════════════════════════════════════

class TestBugC1CompactUsesIDMap2:
    """compact() must not crash — requires IndexIDMap2, not IndexIDMap."""

    def _unit_vec(self, dim: int, seed: int):
        import numpy as np
        rng = np.random.RandomState(seed)
        v   = rng.randn(dim).astype(np.float32)
        v  /= np.linalg.norm(v)
        return v.tolist()

    def test_compact_after_deletion_does_not_crash(self):
        from app.vector_store.store import FaissVectorStore
        store = FaissVectorStore(VectorStoreConfig(dimension=16))
        vecs = [self._unit_vec(16, i) for i in range(5)]
        ids  = [f"chunk_{i}" for i in range(5)]
        store.add(ids, vecs)

        store.delete(["chunk_1", "chunk_3"])
        assert store.total_vectors == 3

        # This used to crash with RuntimeError on IndexIDMap.
        # With IndexIDMap2 it must succeed.
        store.compact()
        assert store.total_vectors == 3

    def test_compact_all_deleted_resets_index(self):
        from app.vector_store.store import FaissVectorStore
        store = FaissVectorStore(VectorStoreConfig(dimension=16))
        store.add(["a"], [self._unit_vec(16, 1)])
        store.delete(["a"])
        store.compact()
        assert store.total_vectors == 0

    def test_compact_search_still_correct_after(self):
        from app.vector_store.store import FaissVectorStore
        store = FaissVectorStore(VectorStoreConfig(dimension=16))
        v0 = self._unit_vec(16, 0)
        v1 = self._unit_vec(16, 1)
        v2 = self._unit_vec(16, 2)
        store.add(["keep_a", "delete_b", "keep_c"], [v0, v1, v2])
        store.delete(["delete_b"])
        store.compact()

        results = store.search(v0, top_k=5)
        chunk_ids = [r[0] for r in results]
        assert "delete_b" not in chunk_ids
        assert "keep_a" in chunk_ids


class TestBugC2OperatorSpellProtection:
    """Boolean operators must never be spell-corrected."""

    def _checker_with_vocab(self):
        from app.spellcheck.spell_checker import SpellChecker
        from app.config import SpellCheckConfig
        ch = SpellChecker(SpellCheckConfig(max_edit_distance=2))
        # "end" is close to "and"; "ore" to "or" etc. — vocab words that could
        # attract corrections if operators weren't protected.
        ch.build_vocabulary([
            "python", "search", "engine", "end", "ore", "note",
            "java", "backend", "machine", "learning",
        ])
        return ch

    def test_and_operator_not_corrected(self):
        ch = self._checker_with_vocab()
        result = ch.correct_query("python AND java")
        assert "AND" in result.upper()
        # "AND" must survive — not become "end" or anything else
        assert "AND" in result

    def test_or_operator_not_corrected(self):
        ch = self._checker_with_vocab()
        result = ch.correct_query("python OR java")
        assert "OR" in result

    def test_not_operator_not_corrected(self):
        ch = self._checker_with_vocab()
        result = ch.correct_query("python NOT java")
        assert "NOT" in result

    def test_typo_beside_operator_corrected(self):
        ch = self._checker_with_vocab()
        result = ch.correct_query("pythn AND java")
        # "pythn" should be corrected but AND preserved
        assert "AND" in result
        assert "pythn" not in result   # corrected to "python"

    def test_mixed_case_operator_not_corrected(self):
        ch = self._checker_with_vocab()
        result = ch.correct_query("python and java")
        # lowercase "and" is a stop word → not in vocab, but also low confidence
        # The key thing is it should remain unchanged
        tokens = result.split()
        assert "and" in tokens


class TestBugC3ExplainEfficiency:
    """explain() must do ONE search, not one per chunk."""

    def _setup(self, tmp_path):
        from app.database.db import Database
        from app.indexer.indexer import Indexer
        from app.tokenizer.tokenizer import Tokenizer
        from app.embeddings.pipeline import EmbeddingPipeline
        from app.vector_store.store import FaissVectorStore
        from app.semantic_search.semantic_service import SemanticSearchService

        db  = Database(tmp_path / "t.db")
        db.connect()
        tok = Tokenizer()
        idx = Indexer(db, tok)
        idx.index_document("Python Guide",
                           "Python is a powerful language for web development",
                           source="s1")

        provider     = MockEmbeddingProvider(dim=16)
        vs           = FaissVectorStore(VectorStoreConfig(dimension=16))
        pipeline     = EmbeddingPipeline(
            db=db, provider=provider, cache=EmbeddingCache(db),
            vector_store=vs, chunker=make_chunker(ChunkingConfig(chunk_size=50)),
            emb_config=EmbeddingConfig(cache_embeddings=False, batch_size=4),
            vs_config=VectorStoreConfig(dimension=16, index_path=tmp_path/"idx"),
        )
        pipeline.index_all()
        svc = SemanticSearchService(db=db, provider=provider, vector_store=vs)
        return db, svc

    def test_explain_returns_correct_structure(self, tmp_path):
        db, svc = self._setup(tmp_path)
        result = svc.explain("python web", doc_id=1)
        assert "best_semantic_score" in result
        assert "chunks" in result
        assert isinstance(result["chunks"], list)
        db.close()

    def test_explain_single_search_call(self, tmp_path):
        """Verify explain() calls vector_store.search exactly once."""
        from unittest.mock import patch
        db, svc = self._setup(tmp_path)
        call_count = [0]
        original = svc.vector_store.search

        def counting_search(*args, **kwargs):
            call_count[0] += 1
            return original(*args, **kwargs)

        svc.vector_store.search = counting_search
        svc.explain("python web", doc_id=1)
        # Must be exactly 1, not N (one per chunk)
        assert call_count[0] == 1
        db.close()


class TestBugC4NdcgNoDeadCode:
    """ndcg_at_k must not have the dead `ideal` / `ideal_list` variables."""

    def test_ndcg_correct_result(self):
        # Perfect ranking: should score 1.0
        rel = {1: 3.0, 2: 2.0, 3: 1.0}
        assert ndcg_at_k([1, 2, 3], rel, k=3) == pytest.approx(1.0)

    def test_ndcg_imperfect_ranking(self):
        rel = {1: 3.0, 2: 2.0}
        # retrieved=[2,1] vs ideal=[1,2]: DCG([2,1]) < DCG([1,2])
        assert ndcg_at_k([2, 1], rel, k=2) < 1.0

    def test_ndcg_no_relevant_returns_zero(self):
        assert ndcg_at_k([1, 2, 3], {}, k=3) == 0.0

    def test_ndcg_source_has_no_dead_variables(self):
        """Confirm the dead-code lines `ideal = ...` are gone from source."""
        import inspect
        from app.evaluation import metrics as m
        source = inspect.getsource(m.ndcg_at_k)
        # These two dead-code lines must not appear in the function body
        assert "ideal = sorted(relevance_scores" not in source
        assert "ideal_list = list(relevance_scores.keys())" not in source


class TestPerfBatchDFUpdate:
    """_update_document_frequencies must issue a single SQL call."""

    def test_batch_update_called_once_per_index(self, tmp_path):
        from app.database.db import Database
        from app.indexer.indexer import Indexer
        from app.tokenizer.tokenizer import Tokenizer
        from unittest.mock import patch

        db  = Database(tmp_path / "t.db")
        db.connect()
        tok = Tokenizer()
        idx = Indexer(db, tok)

        # Patch the batch method on the Database instance
        call_args = []
        original = db.batch_update_document_frequencies

        def recording(*args, **kwargs):
            call_args.append(args)
            return original(*args, **kwargs)

        db.batch_update_document_frequencies = recording
        idx.index_document("Test", "python web backend machine learning", source="s1")

        # Must be called exactly once (batch), not once per unique term
        assert len(call_args) == 1, (
            f"Expected 1 batch DF update call, got {len(call_args)}. "
            "Performance regression: _update_document_frequencies is N+1 again."
        )
        # All terms passed in the single call
        terms_passed = set(call_args[0][0])
        assert "python" in terms_passed
        assert "web" in terms_passed
        db.close()


class TestSpellCheckerBodySizeLimit:
    """IndexDocumentRequest must reject content > 1 MB."""

    def test_content_over_1mb_rejected(self, tmp_path):
        from fastapi.testclient import TestClient
        from app.api.routes import create_app
        from app.config import EngineConfig, DatabaseConfig
        cfg = EngineConfig(database=DatabaseConfig(db_path=tmp_path / "t.db"))
        with TestClient(create_app(cfg)) as client:
            big_content = "x" * (1_000_001)
            r = client.post("/index", json={"title": "T", "content": big_content})
            assert r.status_code == 422   # Pydantic validation error
