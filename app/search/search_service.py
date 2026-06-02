"""
Search Service — Phase 3

Orchestrates the full Phase 3 search pipeline:

  raw query
    ↓  spell correction (optional)
    ↓  query expansion (synonyms)
    ↓  advanced query parsing → AST
    ↓  AST evaluation → candidate doc IDs (Boolean retrieval)
    ↓  BM25 + relevance tuning → ranked results
    ↓  snippet generation
    ↓  cache write
    ↓  analytics logging
    → SearchResult
"""

import logging
import time
from dataclasses import dataclass

from app.database.db import Database
from app.tokenizer.tokenizer import Tokenizer
from app.parser.query_parser import QueryParser, BooleanRetriever, ParsedQuery
from app.parser.advanced_query_parser import AdvancedQueryParser, ASTEvaluator
from app.bm25.bm25 import BM25Ranker
from app.ranking.relevance_tuning import RelevanceTuner, RankedDocument
from app.snippets.snippet_generator import SnippetGenerator
from app.autocomplete.trie import AutocompleteService
from app.spellcheck.spell_checker import SpellChecker
from app.query_expansion.expander import QueryExpander
from app.cache.lru_cache import QueryCache
from app.analytics.analytics import AnalyticsService, SearchEvent
from app.observability.metrics import MetricsCollector
from app.config import EngineConfig

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    query: str
    corrected_query: str | None
    expanded_terms: list[str]
    total_matches: int
    results: list[RankedDocument]
    search_time_ms: float
    cache_hit: bool
    log_id: int | None = None    # for click correlation


class SearchService:
    """
    Main search entry-point.  All Phase 3 components are injected.
    """

    def __init__(
        self,
        db: Database,
        tokenizer: Tokenizer,
        bm25_ranker: BM25Ranker,
        relevance_tuner: RelevanceTuner,
        snippet_generator: SnippetGenerator,
        autocomplete: AutocompleteService,
        spell_checker: SpellChecker,
        query_expander: QueryExpander,
        query_cache: QueryCache,
        analytics: AnalyticsService,
        metrics: MetricsCollector,
        config: EngineConfig | None = None,
    ):
        self.db = db
        self.tokenizer = tokenizer
        self.bm25 = bm25_ranker
        self.relevance_tuner = relevance_tuner
        self.snippet_gen = snippet_generator
        self.autocomplete = autocomplete
        self.spell_checker = spell_checker
        self.query_expander = query_expander
        self.cache = query_cache
        self.analytics = analytics
        self.metrics = metrics
        self.config = config or EngineConfig()

        # Parsers
        self._simple_parser     = QueryParser(tokenizer)
        self._advanced_parser   = AdvancedQueryParser(tokenizer)
        self._ast_evaluator     = ASTEvaluator(db)
        self._boolean_retriever = BooleanRetriever(db)

    def invalidate_all_caches(self) -> None:
        """
        Called after indexing or deletion to keep all caches consistent:
          - BM25 corpus stats (avgdl, doc count)
          - Query result cache
          - Wildcard vocabulary cache in ASTEvaluator
        """
        self.bm25.invalidate_cache()
        self.cache.invalidate_all()
        self._ast_evaluator.invalidate_vocab_cache()

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_advanced_parser: bool = True,
        use_spell_correction: bool = True,
        use_query_expansion: bool = True,
        session_id: str | None = None,
    ) -> SearchResult:
        start = time.perf_counter()

        # ── Cache check ────────────────────────────────────────────────────
        cached = self.cache.get(query, top_k)
        if cached is not None:
            self.metrics.record_cache_hit()
            elapsed = (time.perf_counter() - start) * 1000
            self.metrics.record_search(elapsed)
            # Re-assign a fresh log_id for this session so click events are
            # attributed to the current user, not the original cached session
            # (BUG-002 cache log_id poisoning fix).
            fresh_log_id = self.analytics.record_search(SearchEvent(
                query=query, results_count=cached.total_matches,
                latency_ms=elapsed, session_id=session_id,
            ))
            import dataclasses
            return dataclasses.replace(cached, log_id=fresh_log_id, cache_hit=True)

        self.metrics.record_cache_miss()

        # ── Spell correction ───────────────────────────────────────────────
        corrected_query: str | None = None
        working_query = query
        if use_spell_correction:
            auto_corrected = self.spell_checker.correct_query(query)
            if auto_corrected.lower() != query.lower():
                corrected_query = auto_corrected
                working_query = auto_corrected
                logger.debug("Spell corrected: %r → %r", query, corrected_query)

        # ── Parse & retrieve ───────────────────────────────────────────────
        if use_advanced_parser:
            ast = self._advanced_parser.parse(working_query)
            candidate_ids = self._ast_evaluator.evaluate(ast)
            # Extract terms for ranking
            parsed = self._simple_parser.parse(working_query)
            query_terms = parsed.terms
        else:
            parsed = self._simple_parser.parse(working_query)
            query_terms = parsed.terms
            candidate_ids = self._boolean_retriever.retrieve(parsed)

        if not query_terms:
            elapsed = (time.perf_counter() - start) * 1000
            result = SearchResult(
                query=query, corrected_query=corrected_query,
                expanded_terms=[], total_matches=0,
                results=[], search_time_ms=round(elapsed, 2),
                cache_hit=False,
            )
            log_id = self.analytics.record_search(SearchEvent(
                query=query, results_count=0,
                latency_ms=elapsed, session_id=session_id,
            ))
            result.log_id = log_id
            self.metrics.record_search(elapsed)
            return result

        # ── Query expansion ────────────────────────────────────────────────
        expanded_terms = query_terms
        if use_query_expansion:
            expanded_terms = self.query_expander.expand(query_terms)
            if len(expanded_terms) > len(query_terms):
                # Re-retrieve with expanded terms → more candidates
                expanded_parsed = self._simple_parser.parse(" OR ".join(expanded_terms))
                extra_ids = self._boolean_retriever.retrieve(expanded_parsed)
                candidate_ids |= extra_ids

        # ── Rank ───────────────────────────────────────────────────────────
        rank_start = time.perf_counter()
        ranked = self.relevance_tuner.rank(
            query_terms=query_terms,
            candidate_doc_ids=candidate_ids,
            top_k=top_k,
        )
        rank_ms = (time.perf_counter() - rank_start) * 1000
        self.metrics.record_ranking(rank_ms)

        # ── Snippets ───────────────────────────────────────────────────────
        for doc in ranked:
            full_doc = self.db.get_document(doc.doc_id)
            if full_doc:
                doc.snippet = self.snippet_gen.generate(
                    full_doc.content, query_terms
                )

        # ── Index query in autocomplete trie ──────────────────────────────
        self.autocomplete.index_query(query)

        elapsed = (time.perf_counter() - start) * 1000
        self.metrics.record_search(elapsed)

        result = SearchResult(
            query=query,
            corrected_query=corrected_query,
            expanded_terms=expanded_terms if len(expanded_terms) > len(query_terms) else [],
            total_matches=len(candidate_ids),
            results=ranked,
            search_time_ms=round(elapsed, 2),
            cache_hit=False,
        )

        # ── Log analytics ──────────────────────────────────────────────────
        log_id = self.analytics.record_search(SearchEvent(
            query=query,
            results_count=len(ranked),
            latency_ms=elapsed,
            session_id=session_id,
        ))
        result.log_id = log_id

        # ── Cache write ────────────────────────────────────────────────────
        self.cache.put(query, top_k, result)

        logger.info(
            "Search %r: %d candidates → %d results in %.1f ms (cache miss)",
            query, len(candidate_ids), len(ranked), elapsed,
        )
        return result
