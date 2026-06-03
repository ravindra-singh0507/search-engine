"""
FastAPI Routes — Phase 3 (security-hardened)

Security fixes applied:
  1. Path traversal — /index/directory validates the resolved path is
     inside the project working directory.
  2. SSRF — /crawl blocks private/loopback IP ranges and non-HTTP schemes
     using ipaddress stdlib (no extra dependency).
  3. API key auth — optional via SEARCH_API_KEY env var; if set, all
     mutating endpoints require X-API-Key header (constant-time compare).
  4. Rate limiting — simple in-process sliding-window limiter (60 req/min
     per client IP on /search).  Fine for single-node; swap for Redis
     token bucket in multi-process deployment.
"""

import ipaddress
import logging
import os
import secrets
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response, Depends
from pydantic import BaseModel, Field

from app.database.db import Database
from app.tokenizer.tokenizer import Tokenizer, TokenizerConfig
from app.indexer.indexer import Indexer
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
from app.crawler.crawler import WebCrawler
from app.config import EngineConfig

# Phase 4 imports
from app.embeddings.provider import LocalEmbeddingProvider, MockEmbeddingProvider
from app.embeddings.cache import EmbeddingCache
from app.embeddings.pipeline import EmbeddingPipeline
from app.chunking.chunker import make_chunker
from app.vector_store.store import FaissVectorStore
from app.semantic_search.semantic_service import SemanticSearchService
from app.hybrid_search.hybrid_service import HybridSearchService
from app.evaluation.evaluator import RetrievalEvaluator, load_eval_dataset

logger = logging.getLogger(__name__)

# ── Pydantic schemas ───────────────────────────────────────────────────────────

class IndexDocumentRequest(BaseModel):
    title:    str = Field(..., min_length=1, max_length=500)
    content:  str = Field(..., min_length=1, max_length=1_000_000)   # 1 MB text cap (SEC-3)
    source:   str = Field(default="api", max_length=1000)
    doc_type: str = Field(default="text", max_length=50)


class IndexDirectoryRequest(BaseModel):
    directory: str = Field(default="documents")


class CrawlRequest(BaseModel):
    seed_urls:      list[str] = Field(..., min_length=1)
    max_depth:      int       = Field(default=2, ge=1, le=10)
    max_pages:      int       = Field(default=50, ge=1, le=1000)
    stay_on_domain: bool      = Field(default=True)


class ClickEventRequest(BaseModel):
    log_id:   int = Field(..., description="log_id from the search response")
    doc_id:   int
    position: int = Field(..., ge=0)


class EmbedRequest(BaseModel):
    doc_ids: list[int] | None = Field(
        default=None,
        description="Specific doc IDs to embed. Omit to embed all unembedded docs.",
    )
    force: bool = Field(default=False)
    sync:  bool = Field(
        default=False,
        description="Run synchronously (blocks until done). Use in tests or small corpora.",
    )


# ── Security helpers ───────────────────────────────────────────────────────────

_API_KEY: str = os.environ.get("SEARCH_API_KEY", "")

# Private / loopback ranges blocked for SSRF protection
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),  # AWS/cloud metadata
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]


def _is_safe_crawl_url(url: str) -> bool:
    """Return False if the URL targets a private/loopback address."""
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        if hostname.lower() in ("localhost",):
            return False
        try:
            addr = ipaddress.ip_address(hostname)
            return not any(addr in net for net in _BLOCKED_NETWORKS)
        except ValueError:
            # Not an IP literal — hostname. Allow (DNS resolution happens
            # inside requests; we can't resolve at validation time safely).
            return True
    except Exception:
        return False


def _verify_api_key(request: Request) -> None:
    """FastAPI dependency — enforces X-API-Key if SEARCH_API_KEY is set."""
    if not _API_KEY:
        return  # auth disabled
    key = request.headers.get("X-API-Key", "")
    if not secrets.compare_digest(key, _API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Rate limiter (sliding window, in-process) ─────────────────────────────────

class _RateLimiter:
    def __init__(self, max_requests: int = 60, window: float = 60.0):
        self._max    = max_requests
        self._window = window
        self._times: dict[str, list[float]] = defaultdict(list)
        self._lock   = threading.Lock()

    def is_allowed(self, client_id: str) -> bool:
        now = time.monotonic()
        with self._lock:
            ts = self._times[client_id]
            cutoff = now - self._window
            # Evict expired timestamps
            self._times[client_id] = [t for t in ts if t > cutoff]
            if len(self._times[client_id]) >= self._max:
                return False
            self._times[client_id].append(now)
            return True


_search_limiter = _RateLimiter(max_requests=60, window=60.0)


def _rate_limit(request: Request) -> None:
    """FastAPI dependency — rate-limits /search by client IP."""
    client_ip = request.client.host if request.client else "unknown"
    if not _search_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Retry after 60 s.")


# ── Application factory ────────────────────────────────────────────────────────

def create_app(config: EngineConfig | None = None) -> FastAPI:
    config = config or EngineConfig()

    db        = Database(config.database.db_path)
    tokenizer = Tokenizer(TokenizerConfig(
        min_token_length=config.tokenizer.min_token_length,
        max_token_length=config.tokenizer.max_token_length,
        custom_stop_words=config.tokenizer.custom_stop_words,
    ))
    indexer      = Indexer(db, tokenizer)
    bm25         = BM25Ranker(db, tokenizer, config.bm25)
    tuner        = RelevanceTuner(db, bm25, config.ranking_weights)
    snippets     = SnippetGenerator(config.snippet)
    autocomplete = AutocompleteService(config.autocomplete)
    spell_checker = SpellChecker(config.spellcheck)
    expander     = QueryExpander(config.synonyms_path)
    cache        = QueryCache(
        capacity=config.cache.query_cache_size,
        ttl_seconds=config.cache.result_cache_ttl_seconds,
    )
    analytics = AnalyticsService(db)
    metrics   = MetricsCollector(config.observability)
    search    = SearchService(
        db=db, tokenizer=tokenizer, bm25_ranker=bm25,
        relevance_tuner=tuner, snippet_generator=snippets,
        autocomplete=autocomplete, spell_checker=spell_checker,
        query_expander=expander, query_cache=cache,
        analytics=analytics, metrics=metrics, config=config,
    )
    crawler = WebCrawler(
        db=db, indexer=indexer,
        user_agent=config.crawler.user_agent,
        request_delay=config.crawler.request_delay,
        timeout=config.crawler.timeout,
        respect_robots=config.crawler.respect_robots_txt,
    )

    # ── Phase 4: Semantic Retrieval components ─────────────────────────────
    try:
        import sentence_transformers as _st   # noqa: F401
        emb_provider = LocalEmbeddingProvider(config.embedding)
        logger.info("Using LocalEmbeddingProvider: %s", config.embedding.model_name)
    except ImportError:
        logger.warning(
            "sentence-transformers not installed; using MockEmbeddingProvider. "
            "Install with: pip install sentence-transformers"
        )
        emb_provider = MockEmbeddingProvider(dim=config.vector_store.dimension)

    emb_cache    = EmbeddingCache(db)
    chunker      = make_chunker(config.chunking)
    vector_store = FaissVectorStore(config.vector_store)
    emb_pipeline = EmbeddingPipeline(
        db=db, provider=emb_provider, cache=emb_cache,
        vector_store=vector_store, chunker=chunker,
        emb_config=config.embedding, vs_config=config.vector_store,
    )
    semantic_svc = SemanticSearchService(
        db=db, provider=emb_provider,
        vector_store=vector_store, snippet_gen=snippets,
    )
    hybrid_svc = HybridSearchService(
        db=db, keyword_search=search, semantic_search=semantic_svc,
        snippet_gen=snippets, config=config.hybrid_search,
    )

    # Track background embedding job
    _embed_job_running = {"running": False}

    # Project root for path-traversal validation
    _PROJECT_ROOT = Path.cwd().resolve()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        db.connect()
        autocomplete.load()
        vocab = db.get_all_terms()
        autocomplete.seed_from_vocabulary(vocab)
        spell_checker.build_vocabulary(vocab)
        # Phase 4: load saved FAISS index from disk
        vs_path = config.vector_store.index_path
        if (vs_path / "index.faiss").exists():
            vector_store.load(vs_path)
            logger.info("FAISS index loaded — %d vectors", vector_store.total_vectors)
        logger.info(
            "Engine started — vocab=%d terms, autocomplete=%d words, vectors=%d",
            len(vocab), autocomplete.vocabulary_size, vector_store.total_vectors,
        )
        yield
        autocomplete.save()
        # Phase 4: persist FAISS index
        if vector_store.total_vectors > 0:
            vector_store.save(vs_path)
        db.close()
        logger.info("Engine stopped")

    app = FastAPI(
        title="Search Engine — Phase 4 Semantic Retrieval Platform",
        description=(
            "Full semantic retrieval: BM25 + FAISS vector search + RRF hybrid, "
            "chunking, embedding pipeline, evaluation metrics, explainability."
        ),
        version="4.0.0",
        lifespan=lifespan,
    )

    # ── Indexing endpoints ─────────────────────────────────────────────────

    @app.post("/index", summary="Index a single document", tags=["Indexing"],
              dependencies=[Depends(_verify_api_key)])
    def index_document(req: IndexDocumentRequest):
        t0 = time.perf_counter()
        result = indexer.index_document(
            title=req.title, content=req.content,
            source=req.source, doc_type=req.doc_type,
        )
        ms = (time.perf_counter() - t0) * 1000
        metrics.record_index(ms)
        search.invalidate_all_caches()
        new_terms = db.get_all_terms()
        autocomplete.seed_from_vocabulary(new_terms)
        spell_checker.build_vocabulary(new_terms)
        return {
            "status": "indexed",
            "doc_id": result.doc_id,
            "title": result.title,
            "terms_indexed": result.terms_indexed,
            "total_tokens": result.total_tokens,
            "latency_ms": round(ms, 2),
        }

    @app.post("/index/directory",
              summary="Index all .txt files in a directory", tags=["Indexing"],
              dependencies=[Depends(_verify_api_key)])
    def index_directory(req: IndexDirectoryRequest):
        # ── Path traversal protection ──────────────────────────────────────
        requested = Path(req.directory).resolve()
        try:
            requested.relative_to(_PROJECT_ROOT)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="directory must be within the project root",
            )

        if not requested.exists():
            raise HTTPException(status_code=404, detail=f"Directory not found: {req.directory}")

        t0 = time.perf_counter()
        results = indexer.index_directory(requested)
        ms = (time.perf_counter() - t0) * 1000
        metrics.record_index(ms)
        if results:
            search.invalidate_all_caches()
            new_terms = db.get_all_terms()
            autocomplete.seed_from_vocabulary(new_terms)
            spell_checker.build_vocabulary(new_terms)
        return {
            "status": "indexed",
            "documents_indexed": len(results),
            "latency_ms": round(ms, 2),
            "details": [
                {"doc_id": r.doc_id, "title": r.title, "terms_indexed": r.terms_indexed}
                for r in results
            ],
        }

    # ── Search endpoint ────────────────────────────────────────────────────

    @app.get("/search", summary="Search the index", tags=["Search"],
             dependencies=[Depends(_rate_limit)])
    def search_docs(
        q:          str  = Query(..., min_length=1, description="Search query"),
        top_k:      int  = Query(default=10, ge=1, le=100),
        spell:      bool = Query(default=True),
        expand:     bool = Query(default=True),
        advanced:   bool = Query(default=True),
        session_id: str  = Query(default=None),
    ):
        result = search.search(
            query=q, top_k=top_k,
            use_advanced_parser=advanced,
            use_spell_correction=spell,
            use_query_expansion=expand,
            session_id=session_id,
        )
        return {
            "query":            result.query,
            "corrected_query":  result.corrected_query,
            "expanded_terms":   result.expanded_terms,
            "total_matches":    result.total_matches,
            "search_time_ms":   result.search_time_ms,
            "cache_hit":        result.cache_hit,
            "log_id":           result.log_id,
            "results": [
                {
                    "rank":          i + 1,
                    "doc_id":        r.doc_id,
                    "score":         r.score,
                    "bm25_score":    r.bm25_score,
                    "title_score":   r.title_score,
                    "recency_score": r.recency_score,
                    "click_score":   r.click_score,
                    "title":         r.title,
                    "snippet":       r.snippet,
                    "term_scores":   r.term_scores,
                }
                for i, r in enumerate(result.results)
            ],
        }

    # ── Click tracking ─────────────────────────────────────────────────────

    @app.post("/search/click", summary="Record a result click", tags=["Search"])
    def record_click(req: ClickEventRequest):
        analytics.record_click(
            log_id=req.log_id, doc_id=req.doc_id, position=req.position
        )
        return {"status": "recorded"}

    # ── Document endpoints ─────────────────────────────────────────────────

    @app.get("/document/{doc_id}", tags=["Documents"])
    def get_document(doc_id: int):
        doc = db.get_document(doc_id)
        if doc is None:
            raise HTTPException(404, detail=f"Document {doc_id} not found")
        return doc.__dict__

    @app.delete("/document/{doc_id}", tags=["Documents"],
                dependencies=[Depends(_verify_api_key)])
    def delete_document(doc_id: int):
        if not db.delete_document(doc_id):
            raise HTTPException(404, detail=f"Document {doc_id} not found")
        search.invalidate_all_caches()
        # Rebuild spell checker from scratch so deleted terms are removed
        spell_checker.clear_vocabulary()
        spell_checker.build_vocabulary(db.get_all_terms())
        return {"status": "deleted", "doc_id": doc_id}

    # ── Autocomplete ───────────────────────────────────────────────────────

    @app.get("/autocomplete", tags=["Autocomplete"])
    def autocomplete_suggest(
        q:     str = Query(..., min_length=1),
        top_k: int = Query(default=10, ge=1, le=50),
    ):
        suggestions = autocomplete.suggest(q, top_k=top_k)
        return {
            "prefix": q,
            "suggestions": suggestions,
            "vocabulary_size": autocomplete.vocabulary_size,
        }

    # ── Spell check ────────────────────────────────────────────────────────

    @app.get("/spellcheck", tags=["Spell Check"])
    def spell_check(
        q:     str = Query(..., min_length=1),
        top_n: int = Query(default=5, ge=1, le=20),
    ):
        suggestions = spell_checker.correct(q, top_n=top_n)
        return {
            "input": q,
            "is_known": spell_checker.is_known(q),
            "suggestions": [s.__dict__ for s in suggestions],
        }

    @app.get("/spellcheck/query", tags=["Spell Check"])
    def spell_check_query(q: str = Query(...)):
        corrected = spell_checker.correct_query(q)
        return {"original": q, "corrected": corrected, "changed": corrected != q.lower()}

    # ── BM25 explain ──────────────────────────────────────────────────────

    @app.get("/explain", tags=["Search"])
    def explain_score(
        q:      str = Query(...),
        doc_id: int = Query(...),
    ):
        parsed = search._simple_parser.parse(q)
        return bm25.explain(parsed.terms, doc_id)

    # ── Analytics ─────────────────────────────────────────────────────────

    @app.get("/analytics/top-queries", tags=["Analytics"])
    def top_queries(limit: int = Query(default=20, ge=1, le=100)):
        return {"queries": analytics.top_queries(limit)}

    @app.get("/analytics/search-volume", tags=["Analytics"])
    def search_volume(hours: int = Query(default=24, ge=1, le=168)):
        return {"hours": hours, "volume": analytics.search_volume(hours)}

    @app.get("/analytics/failures", tags=["Analytics"])
    def failed_searches(limit: int = Query(default=20, ge=1, le=100)):
        return {"failed_queries": analytics.failed_queries(limit)}

    @app.get("/analytics/click-through-rate", tags=["Analytics"])
    def click_through_rate():
        return analytics.click_through_rate()

    @app.get("/analytics/dashboard", tags=["Analytics"])
    def dashboard():
        return analytics.dashboard()

    # ── Metrics ────────────────────────────────────────────────────────────

    @app.get("/metrics", tags=["Observability"])
    def prometheus_metrics():
        return Response(
            content=metrics.to_prometheus_text(),
            media_type="text/plain; version=0.0.4",
        )

    @app.get("/metrics/snapshot", tags=["Observability"])
    def metrics_snapshot():
        snap = metrics.snapshot()
        snap["cache"] = cache.stats()
        return snap

    # ── Stats ──────────────────────────────────────────────────────────────

    @app.get("/stats", tags=["Stats"])
    def get_stats(include_index: bool = Query(default=False)):
        stats = db.get_stats()
        if include_index:
            stats["index_snapshot"] = indexer.get_inverted_index_snapshot()
        return stats

    # ── Crawler ────────────────────────────────────────────────────────────

    @app.post("/crawl", tags=["Crawler"],
              dependencies=[Depends(_verify_api_key)])
    def start_crawl(req: CrawlRequest):
        if crawler.current_job and crawler.current_job.is_running:
            raise HTTPException(409, detail="A crawl is already running")

        # ── SSRF protection ────────────────────────────────────────────────
        unsafe = [u for u in req.seed_urls if not _is_safe_crawl_url(u)]
        if unsafe:
            raise HTTPException(
                status_code=400,
                detail=f"Unsafe seed URL(s) rejected: {unsafe}",
            )

        def run_crawl():
            try:
                crawler.crawl(
                    seed_urls=req.seed_urls,
                    max_depth=req.max_depth,
                    max_pages=req.max_pages,
                    stay_on_domain=req.stay_on_domain,
                )
                if db.conn is None:   # engine already shut down (e.g. in tests)
                    return
                search.invalidate_all_caches()
                new_terms = db.get_all_terms()
                autocomplete.seed_from_vocabulary(new_terms)
                spell_checker.build_vocabulary(new_terms)
            except Exception as exc:
                logger.error("Crawl thread error: %s", exc)

        threading.Thread(target=run_crawl, daemon=True).start()
        return {
            "status": "started",
            "seed_urls": req.seed_urls,
            "max_depth": req.max_depth,
            "max_pages": req.max_pages,
        }

    @app.get("/crawl/status", tags=["Crawler"])
    def crawl_status():
        return crawler.get_status()

    @app.get("/crawl/stats", tags=["Crawler"])
    def crawl_stats():
        return db.get_crawl_stats()

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║                  PHASE 4 — SEMANTIC RETRIEVAL                       ║
    # ╚══════════════════════════════════════════════════════════════════════╝

    # ── Embedding pipeline ─────────────────────────────────────────────────

    @app.post("/embeddings/reindex",
              summary="Chunk + embed documents into the vector store",
              tags=["Embeddings"],
              dependencies=[Depends(_verify_api_key)])
    def embeddings_reindex(req: EmbedRequest):
        """
        Embed documents and store their vectors in FAISS.

        If doc_ids is omitted, all unembedded documents are processed.
        Set force=true to re-embed even previously embedded documents.
        Runs in a background thread; poll /embeddings/stats for progress.
        """
        if _embed_job_running["running"]:
            raise HTTPException(409, detail="An embedding job is already running")

        doc_ids = req.doc_ids

        def run_embed():
            _embed_job_running["running"] = True
            try:
                t0 = time.perf_counter()
                if doc_ids:
                    from app.embeddings.pipeline import PipelineStats
                    agg = PipelineStats()
                    for did in doc_ids:
                        if db.conn is None:
                            break
                        s = emb_pipeline.index_document(did, force=req.force)
                        agg.docs_processed  += s.docs_processed
                        agg.chunks_embedded += s.chunks_embedded
                else:
                    result = emb_pipeline.index_all(force=req.force)
                if db.conn is None:
                    return
                ms = (time.perf_counter() - t0) * 1000
                metrics.record_embedding(ms)
                metrics.record_vector_index(ms)
            except Exception as exc:
                logger.error("Embedding job failed: %s", exc)
            finally:
                _embed_job_running["running"] = False

        threading.Thread(target=run_embed, daemon=True).start() if not req.sync else run_embed()
        return {
            "status":  "done" if req.sync else "started",
            "doc_ids": doc_ids,
            "force":   req.force,
            "model":   emb_provider.model_name,
        }

    @app.get("/embeddings/stats", summary="Embedding and vector store statistics",
             tags=["Embeddings"])
    def embeddings_stats():
        model = emb_provider.model_name
        return {
            "model_name":      model,
            "dimension":       emb_provider.dimension,
            "vector_store":    vector_store.stats(),
            "semantic_stats":  db.get_semantic_stats(model),
            "embedding_cache": emb_cache.stats(),
            "jobs_running":    _embed_job_running["running"],
            "recent_jobs":     db.get_embedding_jobs(limit=5),
            "index_metadata":  db.get_vector_index_metadata(),
        }

    @app.delete("/embeddings/cache", summary="Clear embedding cache",
                tags=["Embeddings"], dependencies=[Depends(_verify_api_key)])
    def clear_embedding_cache():
        emb_cache.clear()
        return {"status": "cleared"}

    # ── Semantic search ────────────────────────────────────────────────────

    @app.get("/semantic-search", summary="Semantic vector search",
             tags=["Semantic Search"], dependencies=[Depends(_rate_limit)])
    def semantic_search(
        q:     str = Query(..., min_length=1),
        top_k: int = Query(default=10, ge=1, le=100),
    ):
        """
        Embed the query and find semantically similar documents via FAISS ANN.

        Returns documents ranked by cosine similarity even if they don't
        contain the exact query terms.
        """
        t0     = time.perf_counter()
        result = semantic_svc.search(q, top_k=top_k)
        ms     = (time.perf_counter() - t0) * 1000
        metrics.record_semantic_search(ms)
        return {
            "query":          result.query,
            "model":          result.model_name,
            "search_time_ms": result.search_time_ms,
            "total_results":  result.total_results,
            "results": [
                {
                    "rank":           r.rank,
                    "doc_id":         r.doc_id,
                    "chunk_id":       r.chunk_id,
                    "title":          r.title,
                    "snippet":        r.snippet,
                    "chunk_text":     r.chunk_text,
                    "semantic_score": r.semantic_score,
                }
                for r in result.results
            ],
        }

    @app.get("/semantic-search/explain",
             summary="Explain semantic score for a document",
             tags=["Semantic Search"])
    def semantic_explain(
        q:      str = Query(...),
        doc_id: int = Query(...),
    ):
        return semantic_svc.explain(q, doc_id)

    # ── Hybrid search ──────────────────────────────────────────────────────

    @app.get("/hybrid-search",
             summary="Hybrid BM25 + semantic search (RRF fusion)",
             tags=["Hybrid Search"], dependencies=[Depends(_rate_limit)])
    def hybrid_search(
        q:     str = Query(..., min_length=1),
        top_k: int = Query(default=10, ge=1, le=100),
    ):
        """
        Combine BM25 keyword retrieval with semantic vector retrieval using
        Reciprocal Rank Fusion (RRF).  Best of both worlds: keyword precision
        + semantic recall.
        """
        t0     = time.perf_counter()
        result = hybrid_svc.search(q, top_k=top_k)
        ms     = (time.perf_counter() - t0) * 1000
        metrics.record_hybrid_search(ms)
        return {
            "query":            result.query,
            "fusion_strategy":  result.fusion_strategy,
            "search_time_ms":   result.search_time_ms,
            "bm25_results":     result.bm25_count,
            "semantic_results": result.semantic_count,
            "total_results":    result.total_results,
            "results": [
                {
                    "rank":            r.rank,
                    "doc_id":          r.doc_id,
                    "title":           r.title,
                    "snippet":         r.snippet,
                    "fusion_score":    r.fusion_score,
                    "bm25_score":      r.bm25_score,
                    "bm25_rank":       r.bm25_rank,
                    "semantic_score":  r.semantic_score,
                    "semantic_rank":   r.semantic_rank,
                }
                for r in result.results
            ],
        }

    @app.get("/hybrid-search/explain",
             summary="Score breakdown for hybrid search result",
             tags=["Hybrid Search"])
    def hybrid_explain(
        q:      str = Query(...),
        doc_id: int = Query(...),
    ):
        return hybrid_svc.explain(q, doc_id)

    # ── Evaluation ─────────────────────────────────────────────────────────

    @app.get("/evaluation",
             summary="Run retrieval evaluation against the test dataset",
             tags=["Evaluation"],
             dependencies=[Depends(_verify_api_key), Depends(_rate_limit)])
    def run_evaluation(
        top_k: int = Query(default=10, ge=1, le=100),
        systems: str = Query(
            default="bm25,semantic,hybrid",
            description="Comma-separated list of systems to evaluate",
        ),
    ):
        """
        Evaluate registered retrieval systems using the eval dataset
        at config.evaluation.eval_dataset_path.

        Returns P@K, R@K, MRR, MAP, NDCG@K for each system.
        """
        dataset = load_eval_dataset(config.evaluation.eval_dataset_path)
        if not dataset:
            raise HTTPException(
                404,
                detail=f"Eval dataset not found at {config.evaluation.eval_dataset_path}. "
                       "Create data/eval_dataset.json with query/relevant_doc_ids pairs.",
            )

        evaluator = RetrievalEvaluator(config.evaluation)

        requested = [s.strip() for s in systems.split(",")]

        if "bm25" in requested:
            def bm25_fn(q: str, k: int) -> list[int]:
                r = search.search(q, top_k=k)
                return [x.doc_id for x in r.results]
            evaluator.add_system("bm25", bm25_fn)

        if "semantic" in requested:
            def sem_fn(q: str, k: int) -> list[int]:
                r = semantic_svc.search(q, top_k=k)
                return [x.doc_id for x in r.results]
            evaluator.add_system("semantic", sem_fn)

        if "hybrid" in requested:
            def hybrid_fn(q: str, k: int) -> list[int]:
                r = hybrid_svc.search(q, top_k=k)
                return [x.doc_id for x in r.results]
            evaluator.add_system("hybrid", hybrid_fn)

        report = evaluator.run(dataset=dataset, top_k=top_k)
        return {
            "eval_queries": len(dataset),
            "systems":      list(report.keys()),
            "results":      {k: {m: v for m, v in v.items() if m != "per_query"}
                             for k, v in report.items()},
        }

    @app.get("/evaluation/detail",
             summary="Evaluation with per-query breakdown",
             tags=["Evaluation"])
    def evaluation_detail(
        system: str = Query(default="hybrid"),
        top_k:  int = Query(default=10),
    ):
        dataset = load_eval_dataset(config.evaluation.eval_dataset_path)
        if not dataset:
            raise HTTPException(404, detail="Eval dataset not found")

        evaluator = RetrievalEvaluator(config.evaluation)

        def _get_fn(name: str):
            if name == "bm25":
                return lambda q, k: [x.doc_id for x in search.search(q, k).results]
            if name == "semantic":
                return lambda q, k: [x.doc_id for x in semantic_svc.search(q, k).results]
            return lambda q, k: [x.doc_id for x in hybrid_svc.search(q, k).results]

        evaluator.add_system(system, _get_fn(system))
        report = evaluator.run(dataset=dataset, top_k=top_k)
        return report.get(system, {})

    # ── Vector store stats ─────────────────────────────────────────────────

    @app.get("/vector-store/stats", tags=["Embeddings"])
    def vector_store_stats():
        return {
            "faiss":    vector_store.stats(),
            "metadata": db.get_vector_index_metadata(),
        }

    return app
