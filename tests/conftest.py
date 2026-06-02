"""
conftest.py — Shared fixtures for Phase 2 and Phase 3 tests.

Phase 3 SearchService has many dependencies.  `search_service_v3`
assembles them all from scratch so tests don't have to.
"""

import pytest
from pathlib import Path

from app.database.db import Database
from app.tokenizer.tokenizer import Tokenizer, TokenizerConfig
from app.indexer.indexer import Indexer
from app.parser.query_parser import QueryParser, BooleanRetriever
from app.ranking.tfidf import TFIDFRanker
from app.bm25.bm25 import BM25Ranker
from app.ranking.relevance_tuning import RelevanceTuner
from app.snippets.snippet_generator import SnippetGenerator
from app.autocomplete.trie import AutocompleteService
from app.spellcheck.spell_checker import SpellChecker
from app.query_expansion.expander import QueryExpander
from app.cache.lru_cache import QueryCache
from app.analytics.analytics import AnalyticsService
from app.observability.metrics import MetricsCollector
from app.search.search_service import SearchService
from app.config import (
    BM25Config, AutocompleteConfig, SpellCheckConfig,
    SnippetConfig, CacheConfig, ObservabilityConfig, RankingWeights,
)


# ── Core fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    database.connect()
    yield database
    database.close()


@pytest.fixture
def tokenizer():
    return Tokenizer(TokenizerConfig())


@pytest.fixture
def indexer(db, tokenizer):
    return Indexer(db, tokenizer)


@pytest.fixture
def query_parser(tokenizer):
    return QueryParser(tokenizer)


@pytest.fixture
def boolean_retriever(db):
    return BooleanRetriever(db)


@pytest.fixture
def ranker(db, tokenizer):
    return TFIDFRanker(db, tokenizer)


# ── Phase 3 component fixtures ─────────────────────────────────────────────────

@pytest.fixture
def bm25_ranker(db, tokenizer):
    return BM25Ranker(db, tokenizer, BM25Config())


@pytest.fixture
def relevance_tuner(db, bm25_ranker):
    return RelevanceTuner(db, bm25_ranker, RankingWeights())


@pytest.fixture
def search_service(db, tokenizer, tmp_path):
    """Phase 3 SearchService with all dependencies wired."""
    bm25    = BM25Ranker(db, tokenizer, BM25Config())
    tuner   = RelevanceTuner(db, bm25, RankingWeights())
    snippet = SnippetGenerator(SnippetConfig())
    ac      = AutocompleteService(AutocompleteConfig(persist_path=tmp_path / "trie.json"))
    spell   = SpellChecker(SpellCheckConfig())
    expander = QueryExpander()
    cache   = QueryCache(capacity=32, ttl_seconds=60)
    analytics = AnalyticsService(db)
    metrics = MetricsCollector(ObservabilityConfig())
    return SearchService(
        db=db, tokenizer=tokenizer, bm25_ranker=bm25,
        relevance_tuner=tuner, snippet_generator=snippet,
        autocomplete=ac, spell_checker=spell,
        query_expander=expander, query_cache=cache,
        analytics=analytics, metrics=metrics,
    )


@pytest.fixture
def indexed_db(indexer):
    """Pre-loaded DB for search tests."""
    indexer.index_document(
        "Python Basics",
        "Python is a popular programming language. Python is great for web development "
        "and data science. Python has simple syntax.",
        source="test1",
    )
    indexer.index_document(
        "Java Programming",
        "Java is a statically typed programming language. Java is used for enterprise "
        "applications and Android development.",
        source="test2",
    )
    indexer.index_document(
        "Web Development",
        "Web development involves creating websites and web applications. "
        "Python and JavaScript are popular for web development.",
        source="test3",
    )
    indexer.index_document(
        "Data Science",
        "Data science uses statistics and programming to extract insights from data. "
        "Python is the most popular language for data science.",
        source="test4",
    )
    return indexer.db
