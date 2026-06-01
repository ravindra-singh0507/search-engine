"""Tests for TF-IDF Ranking."""

import math


class TestTFIDF:

    def test_term_frequency(self, ranker):
        tf = ranker.compute_tf(3, 10)
        assert tf == 0.3

    def test_term_frequency_zero_total(self, ranker):
        tf = ranker.compute_tf(3, 0)
        assert tf == 0.0

    def test_inverse_document_frequency(self, ranker):
        # 10 documents, term appears in 2
        idf = ranker.compute_idf(2, 10)
        expected = math.log(10 / 2)
        assert abs(idf - expected) < 1e-9

    def test_idf_zero_df(self, ranker):
        idf = ranker.compute_idf(0, 10)
        assert idf == 0.0

    def test_tfidf_score(self, ranker):
        tf = 0.3
        idf = math.log(5)
        score = ranker.compute_tfidf(tf, idf)
        assert abs(score - tf * idf) < 1e-9

    def test_cosine_similarity_identical(self, ranker):
        vec = {"python": 0.5, "java": 0.3}
        sim = ranker.cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-9

    def test_cosine_similarity_orthogonal(self, ranker):
        vec_a = {"python": 1.0}
        vec_b = {"java": 1.0}
        sim = ranker.cosine_similarity(vec_a, vec_b)
        assert sim == 0.0

    def test_cosine_similarity_partial_overlap(self, ranker):
        vec_a = {"python": 1.0, "java": 0.5}
        vec_b = {"python": 0.8, "ruby": 0.3}
        sim = ranker.cosine_similarity(vec_a, vec_b)
        assert 0.0 < sim < 1.0

    def test_cosine_similarity_empty_vector(self, ranker):
        sim = ranker.cosine_similarity({}, {"python": 1.0})
        assert sim == 0.0

    def test_rank_documents(self, indexed_db, ranker):
        all_docs = indexed_db.get_all_documents()
        doc_ids = {d.doc_id for d in all_docs}
        results = ranker.rank_documents(["python"], doc_ids, top_k=10)
        assert len(results) > 0
        # Results should be sorted by score descending
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_rank_returns_top_k(self, indexed_db, ranker):
        all_docs = indexed_db.get_all_documents()
        doc_ids = {d.doc_id for d in all_docs}
        results = ranker.rank_documents(["python"], doc_ids, top_k=2)
        assert len(results) <= 2

    def test_document_vector(self, indexed_db, ranker):
        total = indexed_db.get_document_count()
        vector = ranker.build_document_vector(1, total)
        assert len(vector) > 0
        assert all(v >= 0 for v in vector.values())

    def test_query_vector(self, indexed_db, ranker):
        total = indexed_db.get_document_count()
        vector = ranker.build_query_vector(["python", "programming"], total)
        assert "python" in vector
        assert "programming" in vector
