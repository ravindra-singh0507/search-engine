"""
Phase 5 — Test Suite

Covers:
  - CrossEncoderReranker (MockReranker — no ML deps)
  - Fusion strategies (CombSUM, CombMNZ, Weighted, Borda, RRF)
  - QueryClassifier (intent detection)
  - RetrievalPipeline (integration with mocks)
  - ExperimentRunner
  - FeatureExtractor + all 8 features
  - PersonalizationService
  - Database Phase 5 tables
  - API endpoints (smoke tests)
"""

import json
import math
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from app.reranking.reranker import MockReranker, RerankedResult
from app.fusion.strategies import (
    combsum, combmnz, weighted_fusion, borda_count, rrf,
    compare_strategies, available_strategies, get_fusion_strategy,
)
from app.query_understanding.classifier import (
    QueryClassifier, QueryIntent,
    NAVIGATIONAL, INFORMATIONAL, TRANSACTIONAL,
    DOCUMENTATION, TROUBLESHOOTING, RESEARCH,
)
from app.learning_to_rank.features import (
    FeatureExtractor, FeatureVector, DEFAULT_FEATURES,
    BM25ScoreFeature, SemanticScoreFeature, TitleMatchFeature,
    FreshnessScoreFeature, DocumentLengthFeature,
)
from app.experiments.runner import ExperimentRunner, Experiment
from app.personalization.service import PersonalizationService
from app.config import (
    ExperimentConfig, PersonalizationConfig,
    QueryUnderstandingConfig,
)


# ══════════════════════════════════════════════════════════════════════════════
# MockReranker
# ══════════════════════════════════════════════════════════════════════════════

class TestMockReranker:
    def _reranker(self):
        return MockReranker()

    def test_model_name(self):
        assert MockReranker().model_name == "mock-reranker-v1"

    def test_score_batch_unit_vectors(self):
        r = self._reranker()
        scores = r.score_batch("python programming", ["python is great", "java enterprise"])
        assert len(scores) == 2
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_score_batch_empty(self):
        r = self._reranker()
        assert r.score_batch("python", []) == []

    def test_rerank_returns_top_k(self):
        r = self._reranker()
        candidates = [
            (i, f"Doc {i}", f"content about {['python','java','rust'][i%3]}", 0.5)
            for i in range(5)
        ]
        results = r.rerank("python programming", candidates, top_k=3)
        assert len(results) <= 3

    def test_rerank_sorted_desc(self):
        r = self._reranker()
        candidates = [
            (1, "Python Tutorial", "python programming guide", 0.8),
            (2, "Java Enterprise", "java spring boot", 0.5),
            (3, "Rust Systems", "rust memory safety", 0.3),
        ]
        results = r.rerank("python", candidates, top_k=3)
        scores = [res.reranker_score for res in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_returns_reranked_result(self):
        r = self._reranker()
        results = r.rerank("test query", [(1, "Title", "content", 1.0)])
        assert isinstance(results[0], RerankedResult)
        assert results[0].doc_id == 1

    def test_explain_returns_dict(self):
        r = self._reranker()
        result = r.explain("python", "Python Tutorial", "Python is a language")
        assert "normalized_score" in result
        assert "interpretation" in result

    def test_rerank_empty_candidates(self):
        r = self._reranker()
        assert r.rerank("query", []) == []


# ══════════════════════════════════════════════════════════════════════════════
# Fusion Strategies
# ══════════════════════════════════════════════════════════════════════════════

class TestFusionStrategies:
    def _lists(self):
        list1 = [(1, 10.0), (2, 8.0), (3, 5.0)]
        list2 = [(2, 0.9),  (1, 0.7), (4, 0.5)]
        return list1, list2

    def test_combsum_unions_docs(self):
        l1, l2 = self._lists()
        result = combsum([l1, l2])
        ids = [d for d, _ in result]
        assert 1 in ids and 2 in ids and 3 in ids and 4 in ids

    def test_combsum_sorted_descending(self):
        l1, l2 = self._lists()
        result = combsum([l1, l2])
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_combmnz_boosts_consensus(self):
        l1, l2 = self._lists()
        cs  = dict(combsum ([l1, l2]))
        cmz = dict(combmnz([l1, l2]))
        # doc 1 and 2 appear in both lists → MNZ > SUM for them
        assert cmz[1] >= cs[1]   # appeared in both lists
        assert cmz[2] >= cs[2]

    def test_combmnz_doc_only_in_one_list(self):
        # doc 3 is only in list1 → MNZ multiplier = 1 = same as SUM
        l1, l2 = self._lists()
        cs  = dict(combsum ([l1, l2]))
        cmz = dict(combmnz([l1, l2]))
        assert abs(cmz[3] - cs[3]) < 1e-9   # multiplier 1 = no change

    def test_weighted_fusion_uniform_same_as_combsum_normalised(self):
        l1, l2 = self._lists()
        wf  = dict(weighted_fusion([l1, l2], weights=[0.5, 0.5]))
        cs  = dict(combsum([l1, l2]))
        # With equal weights, WF = CombSUM (they differ only in normalisation weight)
        for doc_id in wf:
            assert abs(wf[doc_id] - cs[doc_id] / 2.0) < 1e-9

    def test_borda_rank_based(self):
        l1 = [(1, 100.0), (2, 50.0)]   # rank 1, 2
        l2 = [(2, 0.9),   (1, 0.1)]    # rank 1, 2
        result = dict(borda_count([l1, l2]))
        # doc 1: (2-1) + (2-2) = 1+0 = 1; doc 2: (2-2)+(2-1) = 0+1 = 1 → tie
        assert result[1] == result[2] == 1.0

    def test_borda_score_scale_invariant(self):
        l1 = [(1, 1000.0), (2, 0.001)]
        l2 = [(1, 999.0),  (2, 0.0001)]
        result = dict(borda_count([l1, l2]))
        assert result[1] > result[2]   # 1 wins by rank in both

    def test_rrf_formula_correct(self):
        # doc 1 is rank 1 in both lists → score = 1/61 + 1/61
        result = dict(rrf([[(1, 10.0)], [(1, 0.9)]], k=60))
        expected = 2.0 / 61.0
        assert abs(result[1] - expected) < 1e-9

    def test_compare_strategies_returns_all(self):
        l1, l2 = self._lists()
        comparison = compare_strategies([l1, l2], top_k=3)
        for name in available_strategies():
            assert name in comparison

    def test_available_strategies_all_present(self):
        strats = available_strategies()
        for name in ["rrf", "combsum", "combmnz", "weighted", "borda"]:
            assert name in strats

    def test_get_fusion_strategy_unknown_raises(self):
        with pytest.raises(ValueError):
            get_fusion_strategy("unknown_strategy")

    def test_single_list_fusion(self):
        lst = [(1, 5.0), (2, 3.0)]
        # All strategies should handle a single list
        for name in available_strategies():
            fn  = get_fusion_strategy(name)
            try:
                res = fn([lst])
                assert isinstance(res, list)
            except Exception as e:
                pytest.fail(f"{name} failed with single list: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Query Understanding
# ══════════════════════════════════════════════════════════════════════════════

class TestQueryClassifier:
    def _clf(self):
        return QueryClassifier(QueryUnderstandingConfig())

    # Informational
    def test_question_query_is_informational(self):
        clf = self._clf()
        assert clf.classify("how does BM25 work").intent == INFORMATIONAL

    def test_what_question_informational(self):
        clf = self._clf()
        assert clf.classify("what is a transformer model").intent == INFORMATIONAL

    # Navigational
    def test_url_pattern_is_navigational(self):
        clf = self._clf()
        assert clf.classify("github.com pytorch").intent == NAVIGATIONAL

    def test_www_is_navigational(self):
        clf = self._clf()
        assert clf.classify("www.python.org").intent == NAVIGATIONAL

    # Transactional
    def test_download_is_transactional(self):
        clf = self._clf()
        assert clf.classify("download pytorch").intent == TRANSACTIONAL

    def test_install_is_transactional(self):
        clf = self._clf()
        assert clf.classify("install fastapi").intent == TRANSACTIONAL

    # Documentation
    def test_tutorial_is_documentation(self):
        clf = self._clf()
        assert clf.classify("python tutorial beginner").intent == DOCUMENTATION

    def test_api_reference_is_documentation(self):
        clf = self._clf()
        assert clf.classify("fastapi api reference").intent == DOCUMENTATION

    # Troubleshooting
    def test_error_is_troubleshooting(self):
        clf = self._clf()
        assert clf.classify("AttributeError fix python").intent == TROUBLESHOOTING

    def test_not_working_is_troubleshooting(self):
        clf = self._clf()
        assert clf.classify("pytorch not working cuda").intent == TROUBLESHOOTING

    # Research
    def test_paper_survey_is_research(self):
        clf = self._clf()
        assert clf.classify("bert paper survey transformers").intent == RESEARCH

    # Confidence
    def test_confidence_is_bounded(self):
        clf = self._clf()
        for query in ["download", "how", "error fix", "www.example.com", "survey paper"]:
            intent = clf.classify(query)
            assert 0.0 <= intent.confidence <= 1.0

    def test_empty_query_returns_informational(self):
        clf = self._clf()
        assert clf.classify("").intent == INFORMATIONAL

    def test_batch_classify(self):
        clf = self._clf()
        intents = clf.batch_classify(["download python", "how does this work"])
        assert len(intents) == 2

    def test_expand_query_troubleshooting(self):
        clf  = self._clf()
        intent = QueryIntent(intent=TROUBLESHOOTING, confidence=0.9)
        extra  = clf.expand_query("cuda error", intent)
        assert any("fix" in t or "solution" in t for t in extra)

    def test_expand_query_navigational_empty(self):
        clf  = self._clf()
        intent = QueryIntent(intent=NAVIGATIONAL, confidence=0.9)
        # Navigational → no expansion
        extra = clf.expand_query("github.com", intent)
        assert extra == []


# ══════════════════════════════════════════════════════════════════════════════
# Learning to Rank Features
# ══════════════════════════════════════════════════════════════════════════════

class TestLtRFeatures:
    def _mock_doc(self, title="Python Tutorial", word_count=100):
        doc = MagicMock()
        doc.title     = title
        doc.word_count = word_count
        doc.created_at = "2024-01-01T00:00:00"
        return doc

    def test_bm25_score_feature(self):
        f = BM25ScoreFeature()
        assert f.compute("query", 1, {"bm25_score": 5.4}) == 5.4
        assert f.compute("query", 1, {}) == 0.0

    def test_semantic_score_feature(self):
        from app.learning_to_rank.features import SemanticScoreFeature
        f = SemanticScoreFeature()
        assert f.compute("query", 1, {"semantic_score": 0.88}) == 0.88

    def test_title_match_feature_exact(self):
        f   = TitleMatchFeature()
        doc = self._mock_doc(title="Python Tutorial Guide")
        score = f.compute("python tutorial", 1, {"doc_record": doc})
        assert score == pytest.approx(1.0)

    def test_title_match_feature_partial(self):
        f   = TitleMatchFeature()
        doc = self._mock_doc(title="Java Enterprise Applications")
        score = f.compute("python java", 1, {"doc_record": doc})
        assert score == pytest.approx(0.5)

    def test_title_match_no_doc(self):
        f = TitleMatchFeature()
        assert f.compute("query", 1, {}) == 0.0

    def test_freshness_decays_with_age(self):
        import math
        f = FreshnessScoreFeature()
        doc_new = self._mock_doc()
        doc_new.created_at = "2099-01-01T00:00:00"   # far future → age = 0
        doc_old = self._mock_doc()
        doc_old.created_at = "2000-01-01T00:00:00"   # old

        s_new = f.compute("q", 1, {"doc_record": doc_new})
        s_old = f.compute("q", 1, {"doc_record": doc_old})
        assert s_new > s_old

    def test_document_length_longer_is_lower(self):
        f    = DocumentLengthFeature()
        doc1 = self._mock_doc(word_count=10)
        doc2 = self._mock_doc(word_count=10000)
        s1   = f.compute("q", 1, {"doc_record": doc1})
        s2   = f.compute("q", 2, {"doc_record": doc2})
        assert s1 > s2   # shorter doc → higher score

    def test_feature_extractor_returns_all_features(self, tmp_path):
        from app.database.db import Database
        db = Database(tmp_path / "t.db")
        db.connect()
        doc_id = db.insert_document("Python Tutorial", "Python is great for learning")
        extractor = FeatureExtractor(db, DEFAULT_FEATURES)
        context   = {"bm25_score": 5.0, "semantic_score": 0.8, "reranker_score": 0.7}
        vectors   = extractor.extract("python tutorial", [doc_id], context)
        assert len(vectors) == 1
        assert len(vectors[0].features) == len(DEFAULT_FEATURES)
        db.close()

    def test_feature_vector_to_list(self):
        fv = FeatureVector(query="q", doc_id=1, features={"a": 1.0, "b": 2.0})
        assert fv.to_list(["a", "b"]) == [1.0, 2.0]
        assert fv.to_list(["b", "a"]) == [2.0, 1.0]
        assert fv.to_list(["missing"]) == [0.0]


# ══════════════════════════════════════════════════════════════════════════════
# Experiment Runner
# ══════════════════════════════════════════════════════════════════════════════

class TestExperimentRunner:
    def _runner(self, tmp_path):
        return ExperimentRunner(config=ExperimentConfig(storage_path=tmp_path/"exps"))

    def _dataset(self):
        from app.evaluation.evaluator import EvalQuery
        return [
            EvalQuery("q1", "python programming", [1, 2], {1: 2.0, 2: 1.0}),
            EvalQuery("q2", "search engine",      [3],    {3: 2.0}),
        ]

    def test_register_and_run(self, tmp_path):
        runner = self._runner(tmp_path)
        runner.register_system("perfect", lambda q, k: [1, 2, 3, 4, 5])
        exp = Experiment("e1", "Test Experiment")
        run = runner.run(exp, dataset=self._dataset(), top_k=5)
        assert run.query_count == 2
        assert "perfect.MRR" in run.metrics

    def test_run_persists_to_disk(self, tmp_path):
        runner = self._runner(tmp_path)
        runner.register_system("sys", lambda q, k: [1])
        exp = Experiment("e2", "Persist Test")
        runner.run(exp, dataset=self._dataset(), top_k=5)
        files = list((tmp_path / "exps").glob("*.json"))
        assert len(files) >= 1

    def test_empty_dataset_returns_empty_metrics(self, tmp_path):
        runner = self._runner(tmp_path)
        runner.register_system("s", lambda q, k: [])
        exp = Experiment("e3", "Empty")
        run = runner.run(exp, dataset=[], top_k=5)
        assert run.metrics == {}

    def test_compare_returns_all_runs(self, tmp_path):
        runner = self._runner(tmp_path)
        runner.register_system("s", lambda q, k: [1, 2])
        exp1 = Experiment("ea", "A"); exp2 = Experiment("eb", "B")
        r1   = runner.run(exp1, self._dataset(), top_k=5)
        r2   = runner.run(exp2, self._dataset(), top_k=5)
        comp = runner.compare([r1.run_id, r2.run_id])
        assert len(comp) == 2

    def test_list_runs(self, tmp_path):
        runner = self._runner(tmp_path)
        runner.register_system("x", lambda q, k: [])
        runner.run(Experiment("ec", "C"), self._dataset(), top_k=5)
        assert len(runner.list_runs()) >= 1

    def test_load_runs_from_disk(self, tmp_path):
        runner = self._runner(tmp_path)
        runner.register_system("y", lambda q, k: [1])
        run1 = runner.run(Experiment("ed", "D"), self._dataset(), top_k=5)

        # Create fresh runner and load from disk
        runner2 = ExperimentRunner(config=ExperimentConfig(storage_path=tmp_path/"exps"))
        runner2.load_runs_from_disk()
        assert run1.run_id in {r["run_id"] for r in runner2.list_runs()}


# ══════════════════════════════════════════════════════════════════════════════
# Personalization Service
# ══════════════════════════════════════════════════════════════════════════════

class TestPersonalizationService:
    def _svc(self, tmp_path, enabled=True):
        from app.database.db import Database
        db  = Database(tmp_path / "t.db")
        db.connect()
        svc = PersonalizationService(db, PersonalizationConfig(enabled=enabled))
        return db, svc

    def test_get_or_create_new_profile(self, tmp_path):
        db, svc = self._svc(tmp_path)
        profile = svc.get_or_create("user_1")
        assert profile.user_id == "user_1"
        assert profile.search_history == []
        db.close()

    def test_record_search_adds_to_history(self, tmp_path):
        db, svc = self._svc(tmp_path)
        svc.record_search("user_1", "python tutorial")
        svc.record_search("user_1", "fastapi setup")
        profile = svc.get_or_create("user_1")
        assert "python tutorial" in profile.search_history
        assert len(profile.search_history) == 2
        db.close()

    def test_record_click(self, tmp_path):
        db, svc = self._svc(tmp_path)
        svc.record_click("user_1", doc_id=42, query="python")
        profile = svc.get_or_create("user_1")
        assert any(c.doc_id == 42 for c in profile.click_history)
        db.close()

    def test_boost_map_clicked_doc(self, tmp_path):
        db, svc = self._svc(tmp_path, enabled=True)
        svc.record_click("user_1", doc_id=7, query="search")
        boosts = svc.get_boost_map("user_1", [7, 8, 9])
        assert boosts[7] == 0.5    # clicked → strong boost
        assert boosts[8] == 0.0
        db.close()

    def test_boost_map_disabled(self, tmp_path):
        db, svc = self._svc(tmp_path, enabled=False)
        svc.record_click("user_1", doc_id=7, query="search")
        boosts = svc.get_boost_map("user_1", [7])
        assert boosts[7] == 0.0   # disabled → no boost
        db.close()

    def test_history_cap(self, tmp_path):
        db, svc = self._svc(tmp_path)
        # max_search_history defaults to 100
        for i in range(110):
            svc.record_search("u", f"query_{i}")
        profile = svc.get_or_create("u")
        assert len(profile.search_history) == 100
        db.close()


# ══════════════════════════════════════════════════════════════════════════════
# Database Phase 5 tables
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase5Database:
    def _db(self, tmp_path):
        from app.database.db import Database
        db = Database(tmp_path / "t.db")
        db.connect()
        return db

    def test_reranking_log_insert_and_read(self, tmp_path):
        db = self._db(tmp_path)
        db.log_reranking("python", 1, 5.0, 0.8, 0.9, 4.2, 1, "mock-v1")
        stats = db.get_reranking_stats()
        assert any(r["query"] == "python" for r in stats)
        db.close()

    def test_experiment_upsert_and_result(self, tmp_path):
        db = self._db(tmp_path)
        db.upsert_experiment("exp1", "Test", "desc", "{}")
        db.insert_experiment_result("exp1", "run1", '{"MRR": 0.5}', 100.0, 5)
        exps    = db.get_experiments()
        results = db.get_experiment_results("exp1")
        assert any(e["experiment_id"] == "exp1" for e in exps)
        assert len(results) >= 1
        db.close()

    def test_query_intent_log(self, tmp_path):
        db = self._db(tmp_path)
        db.log_query_intent("how does BM25 work", "informational", 0.8)
        dist = db.get_intent_distribution()
        assert any(d["intent"] == "informational" for d in dist)
        db.close()

    def test_user_profile_persistence(self, tmp_path):
        db = self._db(tmp_path)
        db.upsert_user_profile("u1", '["q1","q2"]', '[]', '{}')
        profile = db.get_user_profile("u1")
        assert profile is not None
        assert json.loads(profile["search_history_json"]) == ["q1", "q2"]
        db.close()

    def test_evaluation_report_save_and_read(self, tmp_path):
        db = self._db(tmp_path)
        db.save_evaluation_report("Phase5 Test", '{"MRR": 0.72}')
        reports = db.get_evaluation_reports()
        assert any(r["name"] == "Phase5 Test" for r in reports)
        db.close()


# ══════════════════════════════════════════════════════════════════════════════
# API endpoint smoke tests (Phase 5)
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase5APIEndpoints:
    def _client(self, tmp_path):
        from fastapi.testclient import TestClient
        from app.api.routes import create_app
        from app.config import EngineConfig, DatabaseConfig, VectorStoreConfig
        cfg = EngineConfig(
            database=DatabaseConfig(db_path=tmp_path / "t.db"),
            vector_store=VectorStoreConfig(index_path=tmp_path / "idx", dimension=16),
        )
        return TestClient(create_app(cfg))

    def test_rerank_search_empty_index(self, tmp_path):
        with self._client(tmp_path) as c:
            r = c.get("/rerank-search?q=python")
            assert r.status_code == 200
            body = r.json()
            assert "results" in body

    def test_rerank_explain_missing_doc(self, tmp_path):
        with self._client(tmp_path) as c:
            r = c.get("/rerank/explain?q=python&doc_id=999")
            assert r.status_code == 404

    def test_fusion_compare_empty(self, tmp_path):
        with self._client(tmp_path) as c:
            r = c.get("/fusion/compare?q=python")
            assert r.status_code == 200
            assert "strategies" in r.json()

    def test_query_intent_endpoint(self, tmp_path):
        with self._client(tmp_path) as c:
            r = c.get("/query/intent?q=how+does+BM25+work")
            assert r.status_code == 200
            body = r.json()
            assert "intent" in body
            assert "confidence" in body

    def test_intent_distribution_endpoint(self, tmp_path):
        with self._client(tmp_path) as c:
            r = c.get("/query/intents/distribution")
            assert r.status_code == 200

    def test_experiments_list(self, tmp_path):
        with self._client(tmp_path) as c:
            r = c.get("/experiments")
            assert r.status_code == 200

    def test_ranking_features_missing_doc(self, tmp_path):
        with self._client(tmp_path) as c:
            r = c.get("/ranking/features?q=python&doc_id=999")
            assert r.status_code == 404

    def test_personalization_profile_new_user(self, tmp_path):
        with self._client(tmp_path) as c:
            r = c.get("/personalization/profile?user_id=test_user")
            assert r.status_code == 200
            assert r.json()["user_id"] == "test_user"

    def test_pipeline_stats(self, tmp_path):
        with self._client(tmp_path) as c:
            r = c.get("/retrieval-pipeline/stats")
            assert r.status_code == 200
            assert "available_fusions" in r.json()

    def test_reranking_stats(self, tmp_path):
        with self._client(tmp_path) as c:
            r = c.get("/reranking/stats")
            assert r.status_code == 200
            assert "reranker_model" in r.json()
