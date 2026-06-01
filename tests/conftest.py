"""
conftest.py — Shared test fixtures.

Creates a fresh in-memory database and pre-wired components for each test.
"""

import pytest
from pathlib import Path

from app.database.db import Database
from app.tokenizer.tokenizer import Tokenizer, TokenizerConfig
from app.indexer.indexer import Indexer
from app.search.search_service import SearchService
from app.parser.query_parser import QueryParser, BooleanRetriever
from app.ranking.tfidf import TFIDFRanker


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


@pytest.fixture
def search_service(db, tokenizer):
    return SearchService(db, tokenizer)


@pytest.fixture
def indexed_db(indexer):
    """A database pre-loaded with sample documents for search testing."""
    indexer.index_document(
        "Python Basics",
        "Python is a popular programming language. Python is great for web development "
        "and data science. Python has simple syntax.",
        source="test1"
    )
    indexer.index_document(
        "Java Programming",
        "Java is a statically typed programming language. Java is used for enterprise "
        "applications and Android development.",
        source="test2"
    )
    indexer.index_document(
        "Web Development",
        "Web development involves creating websites and web applications. "
        "Python and JavaScript are popular for web development.",
        source="test3"
    )
    indexer.index_document(
        "Data Science",
        "Data science uses statistics and programming to extract insights from data. "
        "Python is the most popular language for data science.",
        source="test4"
    )
    return indexer.db
