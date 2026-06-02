"""
Phase 3 — Comprehensive Test Suite (updated for bug fixes)
"""

import pytest
import threading
from pathlib import Path
from unittest.mock import MagicMock

from app.bm25.bm25 import BM25Ranker
from app.autocomplete.trie import Trie, AutocompleteService
from app.spellcheck.levenshtein import levenshtein_distance
from app.spellcheck.bk_tree import BKTree
from app.spellcheck.spell_checker import SpellChecker
from app.snippets.snippet_generator import SnippetGenerator
from app.query_expansion.expander import QueryExpander
from app.parser.advanced_query_parser import (
    AdvancedQueryParser, ASTEvaluator, TermNode, AndNode, OrNode,
    NotNode, PhraseNode, FieldNode, WildcardNode,
)
from app.parser.query_parser import QueryParser, BooleanRetriever, Operator
from app.cache.lru_cache import LRUCache, QueryCache
from app.observability.metrics import MetricsCollector, Counter, Histogram
from app.config import (
    BM25Config, SnippetConfig, SpellCheckConfig,
    AutocompleteConfig, ObservabilityConfig,
)


# ══════════════════════════════════════════════════════════════════════════════
# BM25
# ══════════════════════════════════════════════════════════════════════════════

class TestBM25Math:
    def _ranker(self):
        db = MagicMock()
        db.get_document_count.return_value = 100
        db.get_average_document_length.return_value = 150.0
        from app.tokenizer.tokenizer import Tokenizer
        tok = Tokenizer()
        return BM25Ranker(db, tok, BM25Config(k1=1.5, b=0.75))

    def test_idf_always_positive(self):
        r = self._ranker()
        # Even when df == N (every doc contains the term)
        assert r._compute_idf(100, 100) >= 0.0

    def test_idf_rare_term_beats_common(self):
        r = self._ranker()
        idf_rare   = r._compute_idf(1,  100)
        idf_common = r._compute_idf(80, 100)
        assert idf_rare > idf_common

    def test_tf_saturates(self):
        r = self._ranker()
        avgdl = 150.0
        tf_5  = r._compute_tf_bm25(5,  150, avgdl)
        tf_50 = r._compute_tf_bm25(50, 150, avgdl)
        tf_500 = r._compute_tf_bm25(500, 150, avgdl)
        # Should grow but diminish
        assert tf_5 < tf_50 < tf_500
        # Upper bound ≈ k1+1 = 2.5
        assert tf_500 < r.config.k1 + 1 + 0.01

    def test_tf_zero_gives_zero(self):
        r = self._ranker()
        assert r._compute_tf_bm25(0, 150, 150.0) == 0.0

    def test_length_normalisation(self):
        """A shorter doc should score higher than a longer doc with same raw TF."""
        r = self._ranker()
        avgdl = 150.0
        tf_short = r._compute_tf_bm25(3, 50,  avgdl)   # short doc
        tf_long  = r._compute_tf_bm25(3, 500, avgdl)   # long doc
        assert tf_short > tf_long


# ══════════════════════════════════════════════════════════════════════════════
# Trie & Autocomplete
# ══════════════════════════════════════════════════════════════════════════════

class TestTrie:
    def test_insert_and_contains(self):
        t = Trie()
        t.insert("python", 10)
        assert t.contains("python")
        assert not t.contains("pyth")

    def test_prefix_search_returns_matches(self):
        t = Trie()
        t.insert("python", 10)
        t.insert("python3", 5)
        t.insert("java", 8)
        results = t.search_prefix("py")
        words = [r[0] for r in results]
        assert "python"  in words
        assert "python3" in words
        assert "java"    not in words

    def test_prefix_search_sorted_by_frequency(self):
        t = Trie()
        t.insert("python", 100)
        t.insert("python3", 5)
        results = t.search_prefix("py", top_k=2)
        assert results[0][0] == "python"   # higher freq first

    def test_top_k_respected(self):
        t = Trie()
        for i in range(20):
            t.insert(f"word{i}", i)
        results = t.search_prefix("word", top_k=5)
        assert len(results) <= 5

    def test_remove(self):
        t = Trie()
        t.insert("hello")
        assert t.contains("hello")
        t.remove("hello")
        assert not t.contains("hello")

    def test_size_tracks_correctly(self):
        t = Trie()
        t.insert("a")
        t.insert("b")
        t.insert("a")   # duplicate — not counted twice
        assert t.size == 2

    def test_serialise_roundtrip(self):
        t = Trie()
        t.insert("alpha", 3)
        t.insert("beta", 7)
        data = t.to_dict()
        t2 = Trie.from_dict(data)
        assert t2.contains("alpha")
        assert t2.contains("beta")

    def test_empty_prefix_returns_empty(self):
        t = Trie()
        t.insert("hello")
        assert t.search_prefix("") == []


class TestAutocompleteService:
    def test_suggest_after_seed(self, tmp_path):
        cfg = AutocompleteConfig(persist_path=tmp_path / "trie.json")
        svc = AutocompleteService(cfg)
        svc.seed_from_vocabulary(["python", "java", "javascript"])
        results = svc.suggest("java")
        words = [r["suggestion"] for r in results]
        assert "java" in words or "javascript" in words

    def test_index_query_adds_to_trie(self, tmp_path):
        cfg = AutocompleteService(AutocompleteConfig(persist_path=tmp_path / "trie.json"))
        cfg.index_query("machine learning")
        results = cfg.suggest("mach")
        assert any("machine" in r["suggestion"] for r in results)

    def test_persist_and_reload(self, tmp_path):
        path = tmp_path / "trie.json"
        cfg  = AutocompleteConfig(persist_path=path)
        svc  = AutocompleteService(cfg)
        svc.seed_from_vocabulary(["hello", "world"])
        svc.save()

        svc2 = AutocompleteService(cfg)
        svc2.load()
        assert svc2.vocabulary_size >= 2


# ══════════════════════════════════════════════════════════════════════════════
# Levenshtein
# ══════════════════════════════════════════════════════════════════════════════

class TestLevenshtein:
    def test_identical_strings(self):
        assert levenshtein_distance("hello", "hello") == 0

    def test_empty_strings(self):
        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("abc", "") == 3
        assert levenshtein_distance("", "abc") == 3

    def test_single_substitution(self):
        assert levenshtein_distance("cat", "bat") == 1

    def test_insertion(self):
        assert levenshtein_distance("cat", "cats") == 1

    def test_deletion(self):
        assert levenshtein_distance("cats", "cat") == 1

    def test_classic_example(self):
        assert levenshtein_distance("kitten", "sitting") == 3

    def test_symmetric(self):
        assert levenshtein_distance("abc", "xyz") == levenshtein_distance("xyz", "abc")

    def test_max_distance_early_exit(self):
        # Distance is 5; with max=2 should return 3 (max+1)
        d = levenshtein_distance("abcde", "vwxyz", max_distance=2)
        assert d > 2


# ══════════════════════════════════════════════════════════════════════════════
# BK-Tree
# ══════════════════════════════════════════════════════════════════════════════

class TestBKTree:
    def test_insert_and_search_exact(self):
        tree = BKTree()
        tree.insert("python")
        results = tree.search("python", max_distance=0)
        assert any(w == "python" for w, _ in results)

    def test_search_typo(self):
        tree = BKTree()
        for w in ["python", "java", "golang"]:
            tree.insert(w)
        results = tree.search("pythn", max_distance=1)
        words = [w for w, _ in results]
        assert "python" in words

    def test_search_respects_max_distance(self):
        tree = BKTree()
        tree.bulk_insert(["cat", "hat", "bat", "dog", "elephant"])
        # "cat" ↔ "bat" = 1, "hat" = 1; "elephant" >> 2
        results = tree.search("cat", max_distance=1)
        words = [w for w, _ in results]
        assert "cat"      in words
        assert "elephant" not in words

    def test_sorted_by_distance(self):
        tree = BKTree()
        tree.bulk_insert(["python", "pythn", "pyton"])
        results = tree.search("python", max_distance=2)
        dists = [d for _, d in results]
        assert dists == sorted(dists)

    def test_duplicate_insert_ignored(self):
        tree = BKTree()
        tree.insert("hello")
        tree.insert("hello")
        assert tree.size == 1


# ══════════════════════════════════════════════════════════════════════════════
# SpellChecker
# ══════════════════════════════════════════════════════════════════════════════

class TestSpellChecker:
    def _checker(self):
        ch = SpellChecker(SpellCheckConfig(max_edit_distance=2, min_word_length=3))
        ch.build_vocabulary(["python", "machine", "learning", "search", "engine"])
        return ch

    def test_known_word_returns_no_suggestions(self):
        ch = self._checker()
        assert ch.correct("python") == []

    def test_typo_corrected(self):
        ch = self._checker()
        suggestions = ch.correct("pythn")
        assert suggestions, "Expected at least one suggestion"
        assert suggestions[0].suggestion == "python"

    def test_confidence_is_bounded(self):
        ch = self._checker()
        for s in ch.correct("machne"):
            assert 0.0 < s.confidence <= 1.0

    def test_correct_query_replaces_tokens(self):
        ch = self._checker()
        corrected = ch.correct_query("pythn lerning")
        assert "python" in corrected or "learning" in corrected

    def test_is_known(self):
        ch = self._checker()
        assert ch.is_known("python")
        assert not ch.is_known("pythn")


# ══════════════════════════════════════════════════════════════════════════════
# SnippetGenerator
# ══════════════════════════════════════════════════════════════════════════════

class TestSnippetGenerator:
    def _gen(self):
        return SnippetGenerator(SnippetConfig(max_length=300, context_words=5))

    def test_highlights_term(self):
        gen = self._gen()
        snippet = gen.generate("Python is a great language for programming", ["python"])
        assert "**Python**" in snippet or "**python**" in snippet

    def test_no_terms_returns_truncated(self):
        gen = self._gen()
        snippet = gen.generate("Hello world this is a test", [])
        assert "Hello" in snippet

    def test_multiple_terms_highlighted(self):
        gen = self._gen()
        text = "Python and machine learning go well together for building models"
        snippet = gen.generate(text, ["python", "machine"])
        assert "**" in snippet

    def test_long_doc_truncated(self):
        gen = self._gen()
        long_text = "word " * 500
        snippet = gen.generate(long_text, ["word"])
        assert len(snippet) <= 350   # some slack for …

    def test_empty_content(self):
        gen = self._gen()
        assert gen.generate("", ["python"]) == ""

    def test_term_not_in_text(self):
        gen = self._gen()
        snippet = gen.generate("Hello world", ["zzz"])
        assert "Hello" in snippet   # falls back to truncation


# ══════════════════════════════════════════════════════════════════════════════
# QueryExpander
# ══════════════════════════════════════════════════════════════════════════════

class TestQueryExpander:
    def _expander(self):
        path = Path(__file__).parent.parent / "app" / "query_expansion" / "synonyms.json"
        return QueryExpander(path if path.exists() else None)

    def test_expand_known_term(self):
        exp = self._expander()
        expanded = exp.expand(["car"])
        assert "car" in expanded
        if exp.dictionary_size > 0:
            assert len(expanded) > 1    # synonyms added

    def test_expand_unknown_term(self):
        exp = self._expander()
        expanded = exp.expand(["zzzzz"])
        assert expanded == ["zzzzz"]

    def test_no_duplicates(self):
        exp = self._expander()
        expanded = exp.expand(["car", "car"])
        assert len(expanded) == len(set(expanded))

    def test_add_synonym_runtime(self):
        exp = QueryExpander()
        exp.add_synonym("foo", "bar")
        assert "bar" in exp.expand(["foo"])


# ══════════════════════════════════════════════════════════════════════════════
# Advanced Query Parser
# ══════════════════════════════════════════════════════════════════════════════

class TestAdvancedQueryParser:
    def _parser(self):
        from app.tokenizer.tokenizer import Tokenizer
        return AdvancedQueryParser(Tokenizer())

    def test_simple_term(self):
        p = self._parser()
        ast = p.parse("python")
        assert isinstance(ast, TermNode)

    def test_and_expression(self):
        p = self._parser()
        ast = p.parse("python AND backend")
        assert isinstance(ast, AndNode)

    def test_or_expression(self):
        p = self._parser()
        ast = p.parse("python OR java")
        assert isinstance(ast, OrNode)

    def test_not_expression(self):
        p = self._parser()
        ast = p.parse("python NOT java")
        # python followed by NOT java → AndNode(python, NotNode(java))
        assert ast is not None

    def test_parentheses_grouping(self):
        p = self._parser()
        ast = p.parse("(python OR java) AND backend")
        assert isinstance(ast, AndNode)
        assert isinstance(ast.left, OrNode)

    def test_phrase_node(self):
        p = self._parser()
        ast = p.parse('"machine learning"')
        assert isinstance(ast, PhraseNode)

    def test_field_node(self):
        p = self._parser()
        ast = p.parse("title:python")
        assert isinstance(ast, FieldNode)
        assert ast.field == "title"

    def test_wildcard_node(self):
        p = self._parser()
        ast = p.parse("py*")
        assert isinstance(ast, WildcardNode)
        assert ast.prefix == "py"

    def test_empty_query_returns_none(self):
        p = self._parser()
        assert p.parse("") is None

    def test_implicit_and(self):
        p = self._parser()
        ast = p.parse("python backend")
        assert isinstance(ast, AndNode)


# ══════════════════════════════════════════════════════════════════════════════
# LRU Cache
# ══════════════════════════════════════════════════════════════════════════════

class TestLRUCache:
    def test_put_and_get(self):
        c = LRUCache(capacity=3)
        c.put("a", 1)
        assert c.get("a") == 1

    def test_miss_returns_none(self):
        c = LRUCache(capacity=3)
        assert c.get("missing") is None

    def test_eviction_when_full(self):
        c = LRUCache(capacity=2)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)   # should evict "a" (LRU)
        assert c.get("a") is None
        assert c.get("b") == 2
        assert c.get("c") == 3

    def test_get_updates_recency(self):
        c = LRUCache(capacity=2)
        c.put("a", 1)
        c.put("b", 2)
        c.get("a")       # access a → a is now MRU
        c.put("c", 3)    # should evict "b" (LRU now)
        assert c.get("b") is None
        assert c.get("a") == 1

    def test_ttl_expiry(self):
        import time
        c = LRUCache(capacity=10, ttl_seconds=0.05)
        c.put("key", "value")
        assert c.get("key") == "value"
        time.sleep(0.1)
        assert c.get("key") is None

    def test_hit_rate(self):
        c = LRUCache(capacity=10)
        c.put("x", 1)
        c.get("x")
        c.get("missing")
        assert c.hit_rate == 0.5

    def test_clear(self):
        c = LRUCache(capacity=10)
        c.put("a", 1)
        c.clear()
        assert c.size == 0
        assert c.get("a") is None


class TestQueryCache:
    def test_cache_and_retrieve(self):
        qc = QueryCache(capacity=10)
        qc.put("python", 10, {"results": []})
        assert qc.get("python", 10) is not None

    def test_different_top_k_cached_separately(self):
        qc = QueryCache(capacity=10)
        qc.put("python", 10, "result_10")
        qc.put("python", 5,  "result_5")
        assert qc.get("python", 10) == "result_10"
        assert qc.get("python", 5)  == "result_5"


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

class TestMetrics:
    def test_counter_increments(self):
        c = Counter("test")
        c.inc()
        c.inc(3)
        assert c.value == 4

    def test_histogram_records(self):
        h = Histogram("test_hist", (10, 50, 100))
        h.observe(5.0)
        h.observe(60.0)
        assert h.count == 2
        assert h.mean == 32.5

    def test_metrics_collector_search(self):
        m = MetricsCollector(ObservabilityConfig(slow_query_threshold_ms=100))
        m.record_search(50.0)
        m.record_search(200.0)   # slow
        assert m.search_requests.value == 2
        assert m.slow_queries.value == 1

    def test_prometheus_text_format(self):
        m = MetricsCollector()
        m.record_search(20.0)
        text = m.to_prometheus_text()
        assert "search_requests_total" in text
        assert "search_latency_ms_count" in text

    def test_snapshot_is_dict(self):
        m = MetricsCollector()
        snap = m.snapshot()
        assert "search_requests_total" in snap
        assert "cache_hit_rate" in snap

    def test_histogram_thread_safety(self):
        """Concurrent observe() and snapshot() must not raise or produce NaN."""
        h = Histogram("concurrent_test", (10, 50, 100))
        errors = []

        def writer():
            for _ in range(500):
                h.observe(42.0)

        def reader():
            for _ in range(100):
                try:
                    h.snapshot()
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        threads += [threading.Thread(target=reader) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Thread-safety errors: {errors}"


# ══════════════════════════════════════════════════════════════════════════════
# BUG REGRESSION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestBUG001_ParserThreadSafety:
    """AdvancedQueryParser must be safe for concurrent use."""

    def test_concurrent_parse_returns_correct_results(self):
        from app.tokenizer.tokenizer import Tokenizer
        parser  = AdvancedQueryParser(Tokenizer())
        results = []
        errors  = []

        def parse_worker(q):
            try:
                ast = parser.parse(q)
                results.append((q, type(ast).__name__))
            except Exception as e:
                errors.append(e)

        queries = [
            "python AND java",
            "python OR java",
            "(python OR java) AND backend",
            "title:python",
            '"machine learning"',
        ]
        # Create FRESH Thread objects each time (Thread can only be started once)
        threads = [
            threading.Thread(target=parse_worker, args=(q,))
            for _ in range(10)
            for q in queries
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent parse errors: {errors}"
        # Build type map — verify expected AST types
        type_map = {q: typ for q, typ in results}
        assert type_map.get("python AND java")              == "AndNode"
        assert type_map.get("python OR java")               == "OrNode"
        assert type_map.get("(python OR java) AND backend") == "AndNode"
        assert type_map.get("title:python")                 == "FieldNode"
        assert type_map.get('"machine learning"')           == "PhraseNode"


class TestBUG005_PhraseSearch:
    """Phrase search must use positional matching, not just AND."""

    def _make_db_with_docs(self, tmp_path):
        from app.database.db import Database
        from app.indexer.indexer import Indexer
        from app.tokenizer.tokenizer import Tokenizer
        db = Database(tmp_path / "test.db")
        db.connect()
        idx = Indexer(db, Tokenizer())
        # Doc 1: machine and learning appear consecutively
        idx.index_document("Doc1", "machine learning is powerful", source="d1")
        # Doc 2: machine and learning appear but NOT consecutively
        idx.index_document("Doc2", "machine is used for deep learning", source="d2")
        # Doc 3: only one of the terms
        idx.index_document("Doc3", "machine vision applications", source="d3")
        return db

    def test_phrase_matches_adjacent_terms_only(self, tmp_path):
        db  = self._make_db_with_docs(tmp_path)
        ev  = ASTEvaluator(db)
        # "machine learning" should match Doc1 only
        node = PhraseNode(terms=["machine", "learning"])
        result = ev.evaluate(node)
        docs = {db.get_document(d).title for d in result}
        assert "Doc1" in docs
        assert "Doc2" not in docs
        db.close()

    def test_phrase_parser_produces_phrase_node(self):
        from app.tokenizer.tokenizer import Tokenizer
        p = AdvancedQueryParser(Tokenizer())
        ast = p.parse('"machine learning"')
        assert isinstance(ast, PhraseNode)
        assert ast.terms == ["machine", "learning"]

    def test_single_term_phrase_works(self, tmp_path):
        db = self._make_db_with_docs(tmp_path)
        ev = ASTEvaluator(db)
        node = PhraseNode(terms=["machine"])
        result = ev.evaluate(node)
        assert len(result) >= 1
        db.close()


class TestBUG006_RobotsTextSectionBleed:
    """robots.txt parser must not bleed rules between sections."""

    def _parser(self):
        from app.crawler.robots import RobotsParser
        return RobotsParser(user_agent="MyBot")

    def test_rules_dont_bleed_between_user_agent_sections(self):
        robots_txt = (
            "User-agent: *\n"
            "Disallow: /private/\n"
            "\n"
            "User-agent: Googlebot\n"
            "Disallow: /google-only/\n"
        )
        p = self._parser()
        data = p._parse_content(robots_txt)
        # MyBot matches *, so /private/ should be disallowed
        paths = [r.path for r in data.rules]
        assert "/private/" in paths
        # /google-only/ belongs to Googlebot section — MyBot should NOT see it
        assert "/google-only/" not in paths

    def test_our_agent_specific_rules_are_picked_up(self):
        robots_txt = (
            "User-agent: SomeOtherBot\n"
            "Disallow: /other/\n"
            "\n"
            "User-agent: MyBot\n"
            "Disallow: /mybot-path/\n"
        )
        p = self._parser()
        data = p._parse_content(robots_txt)
        paths = [r.path for r in data.rules]
        assert "/mybot-path/" in paths
        assert "/other/" not in paths


class TestBUG004_BM25BatchFetch:
    """BM25 must use batch fetch — verify no N+1 queries via call counting."""

    def test_batch_method_called_once(self, tmp_path):
        from app.database.db import Database
        from app.indexer.indexer import Indexer
        from app.tokenizer.tokenizer import Tokenizer
        db = Database(tmp_path / "t.db")
        db.connect()
        idx = Indexer(db, Tokenizer())
        idx.index_document("D1", "python backend web framework", source="s1")
        idx.index_document("D2", "java enterprise backend framework", source="s2")

        call_count = [0]
        original = db.get_postings_for_terms_batch
        def counted(*args, **kwargs):
            call_count[0] += 1
            return original(*args, **kwargs)
        db.get_postings_for_terms_batch = counted

        ranker = BM25Ranker(db, Tokenizer(), BM25Config())
        candidates = {1, 2}
        ranker.rank_documents(["python", "backend", "web"], candidates, top_k=5)

        # Batch method called ONCE regardless of candidate or term count
        assert call_count[0] == 1
        db.close()


class TestBUG_BooleanPrecedence:
    """AND must bind tighter than OR in Boolean retrieval."""

    def test_and_before_or_precedence(self, tmp_path):
        from app.database.db import Database
        from app.indexer.indexer import Indexer
        from app.tokenizer.tokenizer import Tokenizer
        db = Database(tmp_path / "t.db")
        db.connect()
        idx = Indexer(db, Tokenizer())
        # Doc1: python only
        idx.index_document("Python", "python language", source="s1")
        # Doc2: java AND backend
        idx.index_document("Java Backend", "java backend service", source="s2")
        # Doc3: java only (no backend)
        idx.index_document("Java", "java language", source="s3")

        tok = Tokenizer()
        parser   = QueryParser(tok)
        retriever = BooleanRetriever(db)

        # "python OR java AND backend" with correct precedence (AND first):
        # → python OR (java AND backend) → Doc1 ∪ Doc2
        # With wrong left-to-right: (python OR java) AND backend → Doc2 only
        parsed = parser.parse("python OR java AND backend")
        result = retriever.retrieve(parsed)

        # Doc1 (python) must be in results
        doc_titles = {db.get_document(d).title for d in result}
        assert "Python" in doc_titles, "python should match via OR"
        assert "Java Backend" in doc_titles, "java AND backend should match"
        db.close()


class TestTokenizerFixes:
    """Verify underscore and numeric token filtering."""

    def test_underscores_split_into_parts(self):
        from app.tokenizer.tokenizer import Tokenizer
        tok = Tokenizer()
        result = tok.tokenize("some_variable_name")
        # Should produce tokens from splitting on underscore
        combined = " ".join(result.tokens)
        assert "some_variable_name" not in combined  # NOT kept as one token

    def test_pure_numbers_filtered(self):
        from app.tokenizer.tokenizer import Tokenizer
        tok = Tokenizer()
        result = tok.tokenize("in 2024 the version 3 was released")
        # Pure-number tokens should not appear
        for token in result.tokens:
            assert not token.isdigit(), f"Pure digit token leaked: {token}"

    def test_words_with_numbers_kept(self):
        from app.tokenizer.tokenizer import Tokenizer
        tok = Tokenizer()
        result = tok.tokenize("python3 version2 h2o")
        # Tokens with at least one letter are kept
        tokens_set = set(result.tokens)
        assert any(t in tokens_set for t in ["python3", "version2", "h2o"])


class TestSpellCheckerClearVocabulary:
    """Deleted terms must not appear as spell suggestions after rebuild."""

    def test_clear_and_rebuild_removes_deleted_terms(self):
        checker = SpellChecker(SpellCheckConfig(max_edit_distance=2))
        checker.build_vocabulary(["python", "pytho", "machine", "learning"])
        assert checker.is_known("python")

        # Simulate deleting "python" from index → rebuild without it
        checker.clear_vocabulary()
        checker.build_vocabulary(["machine", "learning"])

        assert not checker.is_known("python")
        # "pytho" is close to "machine"/"learning"? No — but we verify
        # "python" is NOT suggested (was cleared)
        suggestions = checker.correct("pythn")
        for s in suggestions:
            assert s.suggestion != "python"


class TestAnalyticsAtomicity:
    """search_logs and query_stats must be written atomically."""

    def test_log_search_creates_both_records(self, tmp_path):
        from app.database.db import Database
        db = Database(tmp_path / "t.db")
        db.connect()
        log_id = db.log_search("test query", 5, 12.3)
        assert log_id > 0

        # search_logs entry exists
        row = db.conn.execute(
            "SELECT * FROM search_logs WHERE log_id = ?", (log_id,)
        ).fetchone()
        assert row is not None
        assert row["query"] == "test query"

        # query_stats entry exists
        stats = db.conn.execute(
            "SELECT * FROM query_stats WHERE query = 'test query'"
        ).fetchone()
        assert stats is not None
        assert stats["total_searches"] == 1
        db.close()


class TestPathTraversalProtection:
    """Index directory endpoint must reject paths outside project root."""

    def test_traversal_path_rejected(self, tmp_path):
        from fastapi.testclient import TestClient
        from app.api.routes import create_app
        from app.config import EngineConfig, DatabaseConfig
        cfg = EngineConfig(database=DatabaseConfig(db_path=tmp_path / "t.db"))
        with TestClient(create_app(cfg)) as client:
            resp = client.post("/index/directory", json={"directory": "../../etc"})
            assert resp.status_code == 400

    def test_valid_path_not_rejected_as_traversal(self, tmp_path):
        from fastapi.testclient import TestClient
        from app.api.routes import create_app
        from app.config import EngineConfig, DatabaseConfig
        cfg = EngineConfig(database=DatabaseConfig(db_path=tmp_path / "t.db"))
        with TestClient(create_app(cfg)) as client:
            # documents/ is a valid relative path inside the project
            resp = client.post("/index/directory", json={"directory": "documents"})
            assert resp.status_code in (200, 404)  # 404 if dir doesn't exist, not 400


class TestSSRFProtection:
    """Crawl endpoint must reject private/loopback URLs."""

    def _make_client(self, tmp_path):
        from fastapi.testclient import TestClient
        from app.api.routes import create_app
        from app.config import EngineConfig, DatabaseConfig
        cfg = EngineConfig(database=DatabaseConfig(db_path=tmp_path / "t.db"))
        return TestClient(create_app(cfg))

    def test_localhost_rejected(self, tmp_path):
        with self._make_client(tmp_path) as client:
            resp = client.post("/crawl", json={"seed_urls": ["http://localhost/admin"]})
            assert resp.status_code == 400

    def test_aws_metadata_rejected(self, tmp_path):
        with self._make_client(tmp_path) as client:
            resp = client.post("/crawl", json={"seed_urls": ["http://169.254.169.254/"]})
            assert resp.status_code == 400

    def test_private_ip_rejected(self, tmp_path):
        with self._make_client(tmp_path) as client:
            resp = client.post("/crawl", json={"seed_urls": ["http://192.168.1.1/"]})
            assert resp.status_code == 400

    def test_public_url_accepted(self, tmp_path):
        with self._make_client(tmp_path) as client:
            resp = client.post("/crawl", json={
                "seed_urls": ["https://example.com"],
                "max_pages": 1,
            })
            assert resp.status_code == 200

