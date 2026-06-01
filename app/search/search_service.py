"""
Search Service

Orchestrates the full search pipeline:
1. Parse the query (extract terms and operators)
2. Retrieve candidate documents (Boolean retrieval)
3. Rank candidates (TF-IDF cosine similarity)
4. Return top-k results

This is the service layer — it composes the lower-level components
(parser, retriever, ranker) into a coherent search operation.
"""

import logging
import time
from dataclasses import dataclass

from app.database.db import Database
from app.tokenizer.tokenizer import Tokenizer
from app.parser.query_parser import QueryParser, BooleanRetriever, ParsedQuery
from app.ranking.tfidf import TFIDFRanker, ScoredDocument

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    query: str
    parsed_query: ParsedQuery
    total_matches: int
    results: list[ScoredDocument]
    search_time_ms: float


class SearchService:
    """
    Main search interface. Takes a raw query string, returns ranked results.
    """

    def __init__(self, db: Database, tokenizer: Tokenizer):
        self.db = db
        self.tokenizer = tokenizer
        self.query_parser = QueryParser(tokenizer)
        self.boolean_retriever = BooleanRetriever(db)
        self.ranker = TFIDFRanker(db, tokenizer)

    def search(self, query: str, top_k: int = 10) -> SearchResult:
        """
        Execute a search query end-to-end.

        Pipeline:
        1. Parse query → structured representation with terms and operators
        2. Boolean retrieval → set of candidate document IDs
        3. TF-IDF ranking → scored and sorted results
        """
        start = time.perf_counter()

        parsed = self.query_parser.parse(query)
        if not parsed.terms:
            return SearchResult(
                query=query, parsed_query=parsed,
                total_matches=0, results=[],
                search_time_ms=0.0
            )

        candidate_ids = self.boolean_retriever.retrieve(parsed)
        logger.debug("Boolean retrieval found %d candidates", len(candidate_ids))

        if not candidate_ids:
            elapsed = (time.perf_counter() - start) * 1000
            return SearchResult(
                query=query, parsed_query=parsed,
                total_matches=0, results=[],
                search_time_ms=round(elapsed, 2)
            )

        ranked = self.ranker.rank_documents(parsed.terms, candidate_ids, top_k)

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "Search '%s': %d matches, returned %d in %.2fms",
            query, len(candidate_ids), len(ranked), elapsed
        )

        return SearchResult(
            query=query, parsed_query=parsed,
            total_matches=len(candidate_ids), results=ranked,
            search_time_ms=round(elapsed, 2)
        )
