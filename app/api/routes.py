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

# Phase 5 imports
from app.reranking.reranker import CrossEncoderReranker, MockReranker
from app.fusion.strategies import compare_strategies, available_strategies, get_fusion_strategy
from app.query_understanding.classifier import QueryClassifier
from app.retrieval_pipeline.pipeline import RetrievalPipeline
from app.experiments.runner import ExperimentRunner, Experiment
from app.learning_to_rank.features import FeatureExtractor, DEFAULT_FEATURES
from app.personalization.service import PersonalizationService

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

    db        = Database(
        config.database.db_path,
        backend=config.database.backend,
        postgres_config=config.postgres,
    )
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

    # ── Phase 5: Advanced Retrieval components ─────────────────────────────
    try:
        import sentence_transformers as _st2   # noqa: F401
        reranker = CrossEncoderReranker(config.reranking)
        logger.info("Using CrossEncoderReranker: %s", config.reranking.model_name)
    except ImportError:
        logger.warning("CrossEncoderReranker unavailable; using MockReranker")
        reranker = MockReranker()

    pipeline = RetrievalPipeline(
        db=db, keyword_search=search, semantic_search=semantic_svc,
        reranker=reranker, snippet_gen=snippets, config=config.pipeline,
    )
    query_classifier  = QueryClassifier(config.query_understanding)
    feature_extractor = FeatureExtractor(db, DEFAULT_FEATURES)
    experiment_runner = ExperimentRunner(db=db, config=config.experiment)
    personalization   = PersonalizationService(db, config.personalization)

    # Register default experiment systems
    experiment_runner.register_system(
        "bm25", lambda q, k: [r.doc_id for r in search.search(q, k).results]
    )
    experiment_runner.register_system(
        "semantic", lambda q, k: [r.doc_id for r in semantic_svc.search(q, k).results]
    )
    experiment_runner.register_system(
        "hybrid", lambda q, k: [r.doc_id for r in hybrid_svc.search(q, k).results]
    )
    experiment_runner.load_runs_from_disk()

    # Track background embedding job
    _embed_job_running = {"running": False}

    # Project root for path-traversal validation
    _PROJECT_ROOT = Path.cwd().resolve()

    # ── Phase 8: Event Bus ────────────────────────────────────────────────
    from app.events.bus import InMemoryEventBus
    from app.events.producer import EventProducer
    from app.events.store import InMemoryEventStore
    from app.events.retry import DeadLetterQueue

    event_bus    = InMemoryEventBus()
    event_store  = InMemoryEventStore(max_size=config.events.max_store_events)
    event_dlq    = DeadLetterQueue(max_size=1000)
    event_prod   = EventProducer(event_bus, source="search-engine")

    # Wire event store: subscribe to all events for persistence
    def _store_event(event):
        event_store.append(event)
    event_bus.subscribe("*", _store_event)

    # ── Phase 8: Redis ────────────────────────────────────────────────────
    try:
        from app.redis.client import RealRedisClient
        redis_client = RealRedisClient(config.redis)
        redis_client.ping()
        logger.info("Redis connected at %s:%d", config.redis.host, config.redis.port)
    except Exception as exc:
        from app.redis.client import InMemoryRedisClient
        redis_client = InMemoryRedisClient()
        logger.info("Redis unavailable (%s), using in-memory fallback", exc)

    # ── Phase 8 Batch 2: Gateway ──────────────────────────────────────────
    from app.gateway.cache import GatewayCache
    from app.gateway.router import QueryRouter
    from app.gateway.service import RetrievalGateway

    gateway_cache = GatewayCache(
        redis_client=redis_client,
        l1_capacity=config.gateway.cache_max_size,
        l2_ttl=config.gateway.cache_ttl,
    )
    query_router = QueryRouter()
    retrieval_gw = None  # wired after pipeline is constructed (in lifespan)

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
        # Phase 8 Batch 2: wire gateway after all services are ready
        nonlocal retrieval_gw
        retrieval_gw = RetrievalGateway(
            config=config.gateway,
            search_service=search, semantic_service=semantic_svc,
            hybrid_service=hybrid_svc, pipeline=pipeline,
            redis_client=redis_client, metrics=metrics,
        )
        yield
        autocomplete.save()
        # Phase 4: persist FAISS index
        if vector_store.total_vectors > 0:
            vector_store.save(vs_path)
        db.close()
        logger.info("Engine stopped")

    app = FastAPI(
        title="Search Engine — Phase 8 Distributed AI Infrastructure Platform",
        description=(
            "Full semantic retrieval: BM25 + FAISS + RRF hybrid + cross-encoder reranking, "
            "RAG pipeline with citations, grounding verification, conversation memory, "
            "streaming responses, confidence scoring, agentic research workflows, "
            "event-driven architecture, Redis caching, PostgreSQL-ready."
        ),
        version="8.0.0",
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

        from app.events.topics import DOCUMENT_INDEXED
        event_prod.emit(DOCUMENT_INDEXED, {
            "doc_id": result.doc_id, "title": result.title,
            "terms_indexed": result.terms_indexed, "latency_ms": round(ms, 2),
        })

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

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║                    PHASE 5 — ADVANCED RETRIEVAL                     ║
    # ╚══════════════════════════════════════════════════════════════════════╝

    # ── Multi-stage pipeline ────────────────────────────────────────────────

    @app.get("/rerank-search",
             summary="Full 4-stage pipeline: BM25 + Semantic + Fusion + Reranker",
             tags=["Pipeline"], dependencies=[Depends(_rate_limit)])
    def rerank_search(
        q:       str  = Query(..., min_length=1),
        top_k:   int  = Query(default=10, ge=1, le=50),
        fusion:  str  = Query(default="rrf", description="rrf|combsum|combmnz|weighted|borda"),
        rerank:  bool = Query(default=True, description="Enable cross-encoder reranking"),
    ):
        """
        Full multi-stage pipeline:
        1. BM25 retrieval (100 candidates)
        2. Semantic / FAISS retrieval (100 candidates)
        3. Fusion (RRF by default)
        4. Cross-encoder reranking (top-50 candidates)
        5. Final weighted score → top_k results
        """
        t0 = time.perf_counter()
        override = {"fusion_strategy": fusion, "use_reranker": rerank, "final_top_k": top_k}
        result   = pipeline.search(q, top_k=top_k, override=override)
        ms       = (time.perf_counter() - t0) * 1000
        metrics.record_pipeline_search(ms)

        # Log reranking if it ran
        if rerank and config.reranking.enabled:
            for r in result.results:
                db.log_reranking(
                    query=q, doc_id=r.doc_id,
                    bm25_score=r.bm25_score, semantic_score=r.semantic_score,
                    reranker_score=r.reranker_score, final_score=r.final_score,
                    final_rank=r.final_rank or 0, model_name=reranker.model_name,
                )

        return {
            "query":           result.query,
            "total_latency_ms": result.total_latency_ms,
            "stage_latencies": result.stage_latencies,
            "retrieval_count": result.retrieval_count,
            "reranked_count":  result.reranked_count,
            "pipeline_config": result.pipeline_config,
            "results": [
                {
                    "rank":            r.final_rank,
                    "doc_id":          r.doc_id,
                    "title":           r.title,
                    "snippet":         r.snippet,
                    "final_score":     r.final_score,
                    "bm25_score":      r.bm25_score,
                    "semantic_score":  r.semantic_score,
                    "reranker_score":  r.reranker_score,
                    "fusion_score":    r.fusion_score,
                }
                for r in result.results
            ],
        }

    @app.get("/rerank/explain",
             summary="Score breakdown for a specific document in reranked results",
             tags=["Pipeline"])
    def rerank_explain(
        q:      str = Query(...),
        doc_id: int = Query(...),
    ):
        """
        Full score explanation combining BM25, semantic, and reranker signals.
        """
        doc = db.get_document(doc_id)
        if doc is None:
            raise HTTPException(404, detail=f"Document {doc_id} not found")

        # BM25 score
        kw_res   = search.search(q, top_k=100)
        bm25_s   = next((r.score for r in kw_res.results if r.doc_id == doc_id), 0.0)
        bm25_rnk = next((i+1 for i, r in enumerate(kw_res.results) if r.doc_id == doc_id), None)

        # Semantic score
        sem_res  = semantic_svc.search(q, top_k=100)
        sem_s    = next((r.semantic_score for r in sem_res.results if r.doc_id == doc_id), 0.0)
        sem_rnk  = next((i+1 for i, r in enumerate(sem_res.results) if r.doc_id == doc_id), None)

        # Reranker score
        rerank_out = reranker.explain(q, doc.title, doc.content[:800])

        reason_parts = []
        if bm25_s > 0:
            reason_parts.append(f"keyword relevance (BM25={bm25_s:.3f})")
        if sem_s > 0:
            reason_parts.append(f"semantic similarity (cos={sem_s:.3f})")
        if rerank_out["normalized_score"] > 0:
            reason_parts.append(
                f"contextual relevance ({rerank_out['interpretation']})"
            )
        reason = "High " + " and ".join(reason_parts) if reason_parts else "No strong signal"

        return {
            "doc_id":          doc_id,
            "title":           doc.title,
            "query":           q,
            "bm25_score":      round(bm25_s, 6),
            "bm25_rank":       bm25_rnk,
            "semantic_score":  round(sem_s, 6),
            "semantic_rank":   sem_rnk,
            "reranker_score":  rerank_out["normalized_score"],
            "reranker_model":  reranker.model_name,
            "final_score":     round(
                0.4 * max(bm25_s, sem_s) + 0.6 * rerank_out["normalized_score"], 6
            ),
            "reason": reason,
        }

    # ── Fusion comparison ───────────────────────────────────────────────────

    @app.get("/fusion/compare",
             summary="Compare all fusion strategies on the same query",
             tags=["Pipeline"], dependencies=[Depends(_rate_limit)])
    def fusion_compare(
        q:     str = Query(..., min_length=1),
        top_k: int = Query(default=10, ge=1, le=50),
    ):
        """
        Run BM25 + semantic retrieval once, then apply every fusion strategy
        and return the top_k results for each side by side.
        Useful for understanding how different fusion methods rank the same candidates.
        """
        kw_res  = search.search(q, top_k=100)
        sem_res = semantic_svc.search(q, top_k=100)
        bm25_list = [(r.doc_id, r.score) for r in kw_res.results]
        sem_list  = [(r.doc_id, r.semantic_score) for r in sem_res.results]

        comparison = compare_strategies([bm25_list, sem_list], top_k=top_k)

        # Enrich with titles
        def enrich(lst):
            out = []
            for rank, (doc_id, score) in enumerate(lst, 1):
                d = db.get_document(doc_id)
                out.append({
                    "rank": rank, "doc_id": doc_id,
                    "title": d.title if d else "?",
                    "score": round(score, 6),
                })
            return out

        return {
            "query":      q,
            "strategies": {name: enrich(lst) for name, lst in comparison.items()},
            "available":  available_strategies(),
        }

    # ── Query understanding ─────────────────────────────────────────────────

    @app.get("/query/intent",
             summary="Classify the intent of a search query",
             tags=["Query Understanding"])
    def query_intent(q: str = Query(..., min_length=1)):
        """Classify the intent of a query: navigational, informational, etc."""
        t0     = time.perf_counter()
        intent = query_classifier.classify(q)
        ms     = (time.perf_counter() - t0) * 1000
        metrics.record_query_classification(ms)

        # Persist to DB
        import json as _json
        db.log_query_intent(
            q, intent.intent, intent.confidence,
            _json.dumps({"tokens": intent.tokens, "expansion_hints": intent.expansion_hints}),
        )

        return {
            "query":           q,
            "intent":          intent.intent,
            "confidence":      intent.confidence,
            "is_question":     intent.is_question,
            "has_error_terms": intent.has_error_terms,
            "is_url_like":     intent.is_url_like,
            "expansion_hints": intent.expansion_hints,
            "latency_ms":      round(ms, 3),
        }

    @app.get("/query/intents/distribution", tags=["Query Understanding"])
    def intent_distribution():
        """Aggregated intent statistics across all classified queries."""
        return {"distribution": db.get_intent_distribution()}

    # ── Experiments ─────────────────────────────────────────────────────────

    @app.get("/experiments", summary="List retrieval experiments",
             tags=["Experiments"])
    def list_experiments():
        return {
            "experiments": db.get_experiments(),
            "runs":        experiment_runner.list_runs(),
        }

    @app.post("/experiments/run",
              summary="Run an experiment comparing retrieval systems",
              tags=["Experiments"],
              dependencies=[Depends(_verify_api_key), Depends(_rate_limit)])
    def run_experiment(
        name:        str = Query(...),
        description: str = Query(default=""),
        systems:     str = Query(default="bm25,semantic,hybrid"),
        top_k:       int = Query(default=10, ge=1, le=100),
    ):
        """
        Run a retrieval experiment over the eval dataset.
        Results are stored in the experiments table.
        """
        import uuid, json as _json
        exp_id = str(uuid.uuid4())[:8]
        exp    = Experiment(
            experiment_id = exp_id,
            name          = name,
            description   = description,
            config        = {"systems": [s.strip() for s in systems.split(",")]},
        )
        dataset = load_eval_dataset(config.evaluation.eval_dataset_path)
        run     = experiment_runner.run(exp, dataset=dataset, top_k=top_k)

        return {
            "experiment_id": exp.experiment_id,
            "run_id":        run.run_id,
            "query_count":   run.query_count,
            "latency_ms":    run.latency_ms,
            "metrics":       run.metrics,
        }

    @app.get("/experiments/results", tags=["Experiments"])
    def experiment_results(experiment_id: str = Query(...)):
        return {
            "experiment_id": experiment_id,
            "results":       db.get_experiment_results(experiment_id),
        }

    # ── Ranking features ────────────────────────────────────────────────────

    @app.get("/ranking/features",
             summary="Extract LtR feature vector for a query-doc pair",
             tags=["Learning to Rank"])
    def ranking_features(
        q:      str = Query(...),
        doc_id: int = Query(...),
    ):
        """
        Extract all Learning-to-Rank features for a (query, document) pair.
        These features are the foundation for training a LambdaMART / LTR model.
        """
        # Validate document exists first
        if db.get_document(doc_id) is None:
            raise HTTPException(404, detail=f"Document {doc_id} not found")

        kw_res  = search.search(q, top_k=200)
        sem_res = semantic_svc.search(q, top_k=200)

        context = {
            "bm25_score":     next((r.score for r in kw_res.results if r.doc_id == doc_id), 0.0),
            "semantic_score": next((r.semantic_score for r in sem_res.results if r.doc_id == doc_id), 0.0),
        }
        vectors = feature_extractor.extract(q, [doc_id], context)
        if not vectors:
            raise HTTPException(404, detail=f"Document {doc_id} not found")

        return {
            "query":    q,
            "doc_id":   doc_id,
            "features": vectors[0].features,
            "feature_names": feature_extractor.feature_names,
        }

    # ── Personalization ─────────────────────────────────────────────────────

    @app.get("/personalization/profile", tags=["Personalization"])
    def get_user_profile(user_id: str = Query(..., min_length=1)):
        profile = personalization.get_or_create(user_id)
        return {
            "user_id":          profile.user_id,
            "search_count":     len(profile.search_history),
            "click_count":      len(profile.click_history),
            "recent_searches":  profile.search_history[-10:],
            "personalization_enabled": config.personalization.enabled,
        }

    @app.post("/personalization/click", tags=["Personalization"])
    def record_personalized_click(
        user_id: str = Query(...),
        doc_id:  int = Query(...),
        q:       str = Query(default=""),
    ):
        personalization.record_click(user_id, doc_id, q)
        return {"status": "recorded"}

    # ── Reranking stats ─────────────────────────────────────────────────────

    @app.get("/reranking/stats", tags=["Pipeline"])
    def reranking_stats():
        return {
            "reranker_model":  reranker.model_name,
            "recent_queries":  db.get_reranking_stats(limit=20),
            "pipeline_config": {
                "bm25_candidates":     config.pipeline.bm25_candidates,
                "semantic_candidates": config.pipeline.semantic_candidates,
                "fusion_strategy":     config.pipeline.fusion_strategy,
                "rerank_top_k":        config.pipeline.rerank_top_k,
                "final_top_k":         config.pipeline.final_top_k,
                "reranker_enabled":    config.reranking.enabled,
            },
        }

    @app.get("/retrieval-pipeline/stats", tags=["Pipeline"])
    def pipeline_stats():
        return {
            "pipeline_searches_total": metrics.pipeline_searches.value,
            "reranking_operations":    metrics.reranking_operations.value,
            "pipeline_latency":        metrics.pipeline_latency.snapshot(),
            "reranking_latency":       metrics.reranking_latency.snapshot(),
            "fusion_latency":          metrics.fusion_latency.snapshot(),
            "available_fusions":       available_strategies(),
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 6 — RAG / Knowledge Assistant Platform
    # ═══════════════════════════════════════════════════════════════════════════

    from app.context_builder.builder import ContextBuilder
    from app.prompts.templates import get_registry as get_prompt_registry
    from app.llm.provider import create_llm_provider
    from app.citations.engine import CitationEngine
    from app.grounding.verifier import GroundingVerifier
    from app.confidence.engine import ConfidenceEngine
    from app.memory.memory import MemoryService
    from app.rag.pipeline import RAGPipeline, RAGRequest
    from app.rag_evaluation.evaluator import RAGEvaluator, RAGEvalCase
    from fastapi.responses import StreamingResponse

    ctx_builder  = ContextBuilder(config.rag.context)
    prompt_reg   = get_prompt_registry()
    llm_provider = create_llm_provider(config.rag.llm)
    cite_engine  = CitationEngine(config.rag.citation)
    ground_ver   = GroundingVerifier(config.rag.grounding)
    conf_engine  = ConfidenceEngine()
    memory_svc   = MemoryService(db, config.rag.memory)
    rag_eval     = RAGEvaluator()

    rag_pipeline = RAGPipeline(
        retriever          = hybrid_svc,
        context_builder    = ctx_builder,
        prompt_registry    = prompt_reg,
        llm                = llm_provider,
        citation_engine    = cite_engine,
        grounding_verifier = ground_ver,
        confidence_engine  = conf_engine,
        memory             = memory_svc,
        metrics            = metrics,
        config             = config.rag,
    )

    # ── Pydantic schemas for Phase 6 ──────────────────────────────────────────

    class ChatRequest(BaseModel):
        message:    str   = Field(..., min_length=1, max_length=10_000)
        session_id: str | None = Field(default=None)
        top_k:      int   = Field(default=5, ge=1, le=20)
        template:   str   = Field(default="qa")
        multi_step: bool  = Field(default=False)
        user_id:    str | None = Field(default=None)

    class RAGQueryRequest(BaseModel):
        query:      str   = Field(..., min_length=1, max_length=10_000)
        session_id: str | None = Field(default=None)
        top_k:      int   = Field(default=5, ge=1, le=20)
        template:   str   = Field(default="qa")
        multi_step: bool  = Field(default=False)

    # ── POST /chat ─────────────────────────────────────────────────────────────

    @app.post("/chat", summary="Conversational knowledge assistant", tags=["RAG"],
              dependencies=[Depends(_rate_limit)])
    def chat(req: ChatRequest):
        """
        Multi-turn conversational RAG endpoint.
        Creates a session automatically if session_id is not provided.
        Returns the grounded answer with citations and confidence score.
        """
        session = memory_svc.get_or_create(req.session_id, req.user_id or "")
        rag_req = RAGRequest(
            query=req.message, session_id=session.session_id,
            top_k=req.top_k, template=req.template,
            multi_step=req.multi_step,
        )
        resp = rag_pipeline.query(rag_req)
        metrics.record_chat_session()

        return {
            "session_id":        session.session_id,
            "query":             resp.query,
            "answer":            resp.formatted_answer,
            "citations":         [
                {"index": c.index, "title": c.title,
                 "snippet": c.snippet, "score": c.relevance_score}
                for c in resp.citations
            ],
            "grounding": {
                "score":            resp.grounding.grounding_score,
                "support_score":    resp.grounding.support_score,
                "risk":             resp.grounding.hallucination_risk,
            },
            "confidence": {
                "overall":    resp.confidence.overall_confidence,
                "tier":       resp.confidence.tier,
                "retrieval":  resp.confidence.retrieval_confidence,
                "grounding":  resp.confidence.grounding_confidence,
            },
            "context":     resp.context_metadata,
            "latency_ms":  resp.total_latency_ms,
            "tokens_used": resp.tokens_used,
            "subqueries":  resp.subqueries,
        }

    # ── POST /chat/stream ──────────────────────────────────────────────────────

    @app.post("/chat/stream",
              summary="Streaming conversational assistant (SSE)",
              tags=["RAG"], dependencies=[Depends(_rate_limit)])
    def chat_stream(req: ChatRequest):
        """
        Server-Sent Events streaming endpoint.
        Each event is a JSON object:
          {"type":"token","content":"..."}
          {"type":"done","grounding_score":0.7,"confidence":"high","citations":[...]}
        """
        metrics.record_streaming()
        session = memory_svc.get_or_create(req.session_id, req.user_id or "")
        rag_req = RAGRequest(
            query=req.message, session_id=session.session_id,
            top_k=req.top_k, template=req.template, stream=True,
        )

        def _generate():
            yield f"data: {__import__('json').dumps({'type':'session','session_id':session.session_id})}\n\n"
            for chunk in rag_pipeline.stream(rag_req):
                yield chunk

        return StreamingResponse(_generate(), media_type="text/event-stream")

    # ── POST /rag/query ────────────────────────────────────────────────────────

    @app.post("/rag/query",
              summary="Single-turn RAG query with full diagnostics",
              tags=["RAG"], dependencies=[Depends(_rate_limit)])
    def rag_query(req: RAGQueryRequest):
        """
        Full RAG pipeline with detailed per-stage diagnostics.
        Returns answer, citations, grounding report, and stage latencies.
        """
        rag_req = RAGRequest(
            query=req.query, session_id=req.session_id,
            top_k=req.top_k, template=req.template,
            multi_step=req.multi_step,
        )
        resp = rag_pipeline.query(rag_req)
        return {
            "query":              resp.query,
            "answer":             resp.answer,
            "formatted_answer":   resp.formatted_answer,
            "citations":          [
                {"index": c.index, "doc_id": c.doc_id, "title": c.title,
                 "snippet": c.snippet, "url": c.url,
                 "relevance_score": c.relevance_score}
                for c in resp.citations
            ],
            "grounding": {
                "score":             resp.grounding.grounding_score,
                "support_score":     resp.grounding.support_score,
                "risk":              resp.grounding.hallucination_risk,
                "supported_claims":  resp.grounding.supported_claims[:5],
                "unsupported_claims": resp.grounding.unsupported_claims[:5],
            },
            "confidence": {
                "overall":    resp.confidence.overall_confidence,
                "tier":       resp.confidence.tier,
                "retrieval":  resp.confidence.retrieval_confidence,
                "context":    resp.confidence.context_confidence,
                "grounding":  resp.confidence.grounding_confidence,
                "citation":   resp.confidence.citation_confidence,
            },
            "context_metadata": resp.context_metadata,
            "retrieval_count":  resp.retrieval_count,
            "stage_latencies":  resp.stage_latencies,
            "total_latency_ms": resp.total_latency_ms,
            "tokens_used":      resp.tokens_used,
            "subqueries":       resp.subqueries,
        }

    # ── POST /research/query ───────────────────────────────────────────────────

    @app.post("/research/query",
              summary="Deep research assistant (multi-step)",
              tags=["RAG"], dependencies=[Depends(_rate_limit)])
    def research_query(req: RAGQueryRequest):
        """
        Research mode: decomposes complex queries into sub-queries,
        retrieves independently, synthesizes a comprehensive answer.
        """
        rag_req = RAGRequest(
            query=req.query, session_id=req.session_id,
            top_k=req.top_k, template="research", multi_step=True,
        )
        resp = rag_pipeline.query(rag_req)
        return {
            "query":            resp.query,
            "subqueries":       resp.subqueries,
            "answer":           resp.formatted_answer,
            "citations":        [
                {"index": c.index, "title": c.title, "snippet": c.snippet}
                for c in resp.citations
            ],
            "grounding_score":  resp.grounding.grounding_score,
            "confidence_tier":  resp.confidence.tier,
            "sources_used":     resp.context_metadata.get("sources", []),
            "total_latency_ms": resp.total_latency_ms,
        }

    # ── GET /memory ────────────────────────────────────────────────────────────

    @app.get("/memory", summary="Get conversation history for a session", tags=["Memory"])
    def get_memory(session_id: str = Query(...)):
        session = memory_svc.get_session(session_id)
        if not session:
            raise HTTPException(404, detail=f"Session {session_id!r} not found")
        return {
            "session_id":    session.session_id,
            "message_count": session.message_count,
            "created_at":    session.created_at,
            "updated_at":    session.updated_at,
            "messages":      [
                {"role": m.role, "content": m.content[:500],
                 "timestamp": m.timestamp}
                for m in session.messages
            ],
        }

    @app.delete("/memory", summary="Delete a conversation session", tags=["Memory"])
    def delete_memory(session_id: str = Query(...)):
        ok = memory_svc.delete_session(session_id)
        if not ok:
            raise HTTPException(404, detail=f"Session {session_id!r} not found")
        return {"status": "deleted", "session_id": session_id}

    @app.get("/memory/sessions", summary="List all conversation sessions", tags=["Memory"])
    def list_sessions(limit: int = Query(default=20, ge=1, le=100)):
        return {"sessions": memory_svc.get_all_sessions(limit)}

    # ── GET /citations ─────────────────────────────────────────────────────────

    @app.get("/citations", summary="Get citations for a session", tags=["RAG"])
    def get_citations(session_id: str = Query(...)):
        rows = db.get_citations_for_session(session_id)
        return {"session_id": session_id, "citations": rows}

    # ── GET /grounding ─────────────────────────────────────────────────────────

    @app.get("/grounding", summary="Grounding statistics", tags=["RAG"])
    def grounding_stats():
        return db.get_grounding_stats()

    @app.post("/grounding/verify", summary="Verify answer grounding against context", tags=["RAG"])
    def verify_grounding(
        answer:  str = Query(..., description="The answer text to verify"),
        context: str = Query(..., description="The context to verify against"),
    ):
        from app.context_builder.builder import Context, ContextChunk, ContextMetadata
        # Build a minimal Context from the raw text
        chunk = ContextChunk(
            chunk_id="manual_0", doc_id=0, text=context,
            score=1.0, source_title="Provided Context",
        )
        meta = ContextMetadata(1, chunk.token_count, 1, 0.0, 1.0)
        ctx  = Context(text=context, chunks=[chunk], metadata=meta)
        report = ground_ver.verify(answer, ctx)
        return {
            "grounding_score":    report.grounding_score,
            "support_score":      report.support_score,
            "hallucination_risk": report.hallucination_risk,
            "unsupported_claims": report.unsupported_claims,
        }

    # ── GET /confidence ────────────────────────────────────────────────────────

    @app.get("/confidence", summary="Confidence statistics", tags=["RAG"])
    def confidence_stats():
        return db.get_confidence_stats()

    # ── GET /prompts ───────────────────────────────────────────────────────────

    @app.get("/prompts", summary="List available prompt templates", tags=["RAG"])
    def list_prompts():
        return {"templates": prompt_reg.list_templates()}

    @app.get("/prompts/{name}", summary="Get a specific prompt template", tags=["RAG"])
    def get_prompt(name: str):
        try:
            t = prompt_reg.get(name)
        except KeyError as e:
            raise HTTPException(404, detail=str(e))
        return {
            "name":    t.name,
            "version": t.version,
            "system":  t.system[:300] + "…" if len(t.system) > 300 else t.system,
            "tags":    t.tags,
        }

    # ── POST /rag/evaluate ────────────────────────────────────────────────────

    @app.post("/rag/evaluate",
              summary="Evaluate a RAG response",
              tags=["RAG Evaluation"])
    def evaluate_rag(
        query:    str = Query(...),
        answer:   str = Query(...),
        context:  str = Query(...),
        ground_truth: str = Query(default=""),
    ):
        from app.context_builder.builder import Context, ContextChunk, ContextMetadata
        chunk  = ContextChunk("c0", 0, context, 1.0, "Provided")
        meta   = ContextMetadata(1, chunk.token_count, 1, 0.0, 1.0)
        ctx    = Context(text=context, chunks=[chunk], metadata=meta)
        from app.grounding.verifier import GroundingVerifier
        gr     = GroundingVerifier().verify(answer, ctx)
        case   = RAGEvalCase(
            query_id="api", query=query, answer=answer,
            context=ctx, grounding=gr, ground_truth=ground_truth,
        )
        result = rag_eval.evaluate(case)
        return result.to_dict()

    @app.get("/rag/eval-stats", summary="Aggregate RAG evaluation stats", tags=["RAG Evaluation"])
    def rag_eval_stats():
        return db.get_rag_eval_stats()

    # ── RAG observability ─────────────────────────────────────────────────────

    @app.get("/rag/stats", summary="RAG pipeline statistics", tags=["RAG"])
    def rag_stats():
        return {
            "rag_queries_total":       metrics.rag_queries.value,
            "rag_tokens_total":        metrics.rag_tokens_used.value,
            "chat_sessions_total":     metrics.chat_sessions.value,
            "streaming_requests":      metrics.streaming_requests.value,
            "grounding_checks":        metrics.grounding_checks.value,
            "high_risk_responses":     metrics.high_risk_responses.value,
            "citations_generated":     metrics.citations_generated.value,
            "llm_provider":            llm_provider.model_name,
            "rag_latency":             metrics.rag_total_latency.snapshot(),
            "llm_latency":             metrics.rag_llm_latency.snapshot(),
            "grounding_db_stats":      db.get_grounding_stats(),
            "confidence_db_stats":     db.get_confidence_stats(),
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 7 — Agentic Research Endpoints
    # ══════════════════════════════════════════════════════════════════════════

    from app.agents.base import (
        Agent, AgentContext, AgentTask, AgentType, RetryPolicy,
    )
    from app.agents.planner import PlannerAgent
    from app.agents.retrieval import RetrievalAgent
    from app.agents.critic import CriticAgent
    from app.agents.citation_validator import CitationValidationAgent
    from app.agents.synthesis import SynthesisAgent
    from app.orchestration.engine import (
        AgentOrchestrator, WorkflowEngine, WorkflowStep,
    )
    from app.workflows.templates import get_workflow_registry
    from app.tools.framework import create_default_registry as create_tool_registry
    from app.mcp.registry import MCPRegistry
    from app.reports.generator import ReportGenerator, ReportFormat
    from app.research_memory.memory import ResearchSession

    # Build agent instances
    agent_retry = RetryPolicy(
        max_attempts  = config.research.agent.max_retries,
        base_delay_sec = config.research.agent.base_delay_sec,
        max_delay_sec  = config.research.agent.max_delay_sec,
    )
    planner_agent     = PlannerAgent(retry_policy=agent_retry)
    retrieval_agent   = RetrievalAgent(retry_policy=agent_retry)
    critic_agent      = CriticAgent(retry_policy=agent_retry)
    citval_agent      = CitationValidationAgent(retry_policy=agent_retry)
    synthesis_agent   = SynthesisAgent(retry_policy=agent_retry)

    agent_map = {
        AgentType.PLANNER:           planner_agent,
        AgentType.RETRIEVAL:         retrieval_agent,
        AgentType.CRITIC:            critic_agent,
        AgentType.CITATION_VALIDATOR: citval_agent,
        AgentType.SYNTHESIS:         synthesis_agent,
    }

    agent_context = AgentContext(
        retriever    = hybrid_svc,
        db           = db,
        rag_pipeline = rag_pipeline,
        memory       = memory_svc,
        config       = config.research,
        event_bus    = event_bus,
        redis        = redis_client,
    )

    orchestrator    = AgentOrchestrator(agent_map, agent_context, metrics)
    workflow_engine = WorkflowEngine(
        agent_map, agent_context, metrics,
        parallel=config.research.orchestrator.parallel,
    )
    workflow_registry = get_workflow_registry()
    tool_registry     = create_tool_registry()
    mcp_registry      = MCPRegistry(tool_registry)
    report_generator  = ReportGenerator()
    research_sessions: dict[str, ResearchSession] = {}

    # ── POST /research ────────────────────────────────────────────────────────

    class ResearchRequest(BaseModel):
        goal:         str
        workflow:     str  = "investigation"
        params:       dict = {}
        session_id:   str  = ""
        user_id:      str  = ""
        parallel:     bool = False

    @app.post("/research", summary="Run an agentic research workflow", tags=["Research"])
    def run_research(req: ResearchRequest):
        import uuid as _uuid
        session_id = req.session_id or str(_uuid.uuid4())

        session = ResearchSession(
            session_id=session_id, goal=req.goal, user_id=req.user_id,
        )
        research_sessions[session_id] = session
        db.insert_research_session(session_id, req.user_id, req.goal)

        # Use planner to generate steps or workflow template
        template = workflow_registry.get(req.workflow)
        if template:
            steps = template.generate(req.goal, req.params)
        else:
            plan_task = AgentTask(goal=req.goal, task_type="plan",
                                  params={"max_topics": config.research.workflow.max_topics})
            plan_result = planner_agent.run(plan_task, agent_context)
            if not plan_result.is_success():
                raise HTTPException(500, detail=f"Planning failed: {plan_result.error}")

            from app.orchestration.engine import WorkflowStep as WS
            steps = []
            for s in plan_result.output.get("steps", []):
                steps.append(WS(
                    step_id    = s["step_id"],
                    agent_type = AgentType(s["agent_type"]),
                    goal       = s["goal"],
                    depends_on = s.get("depends_on", []),
                    optional   = s.get("optional", False),
                ))

        engine = WorkflowEngine(
            agent_map, agent_context, metrics,
            parallel=req.parallel,
        )
        run = engine.run(steps, goal=req.goal, workflow_name=req.workflow)

        # Persist
        db.insert_workflow_run(
            run.run_id, run.workflow_name, run.goal,
            run.status.value, len(run.steps),
        )
        db.update_workflow_run(
            run.run_id, run.status.value,
            run.success_count(), run.failure_count(), run.total_latency_ms,
        )

        # Extract final report if synthesis succeeded
        final_report = None
        for step_id, result in run.results.items():
            if result.agent_type == AgentType.SYNTHESIS and result.is_success():
                final_report = result.output
                session.add_agent_result(result)

        # Record all evidence
        for step_id, result in run.results.items():
            if result.evidence:
                session.add_evidence(result.evidence)
            session.add_agent_result(result)

        metrics.record_workflow_run(run.total_latency_ms, run.status.value == "completed")

        return {
            "session_id":       session_id,
            "run_id":           run.run_id,
            "status":           run.status.value,
            "total_steps":      len(run.steps),
            "success_count":    run.success_count(),
            "failure_count":    run.failure_count(),
            "total_latency_ms": run.total_latency_ms,
            "report":           final_report,
            "step_results":     {
                sid: r.to_dict() for sid, r in run.results.items()
            },
        }

    # ── POST /research/plan ──────────────────────────────────────────────────

    class PlanRequest(BaseModel):
        goal:       str
        max_topics: int = 6

    @app.post("/research/plan", summary="Generate a research plan without executing", tags=["Research"])
    def create_plan(req: PlanRequest):
        task = AgentTask(goal=req.goal, task_type="plan",
                         params={"max_topics": req.max_topics})
        result = planner_agent.run(task, agent_context)
        if not result.is_success():
            raise HTTPException(500, detail=f"Planning failed: {result.error}")
        return {
            "plan":       result.output,
            "confidence": result.confidence,
            "latency_ms": result.latency_ms,
        }

    # ── POST /research/retrieve ──────────────────────────────────────────────

    class RetrieveRequest(BaseModel):
        query:   str
        top_k:   int  = 5
        use_rag: bool = False

    @app.post("/research/retrieve", summary="Run a single retrieval agent", tags=["Research"])
    def agent_retrieve(req: RetrieveRequest):
        task = AgentTask(
            goal=req.query, task_type="retrieve",
            params={"query": req.query, "top_k": req.top_k, "use_rag": req.use_rag},
        )
        result = retrieval_agent.run(task, agent_context)
        return result.to_dict()

    # ── GET /research/workflows ──────────────────────────────────────────────

    @app.get("/research/workflows", summary="List available workflow templates", tags=["Research"])
    def list_workflows():
        return {
            "workflows": [
                {"name": t.name, "description": t.description}
                for t in workflow_registry.values()
            ]
        }

    # ── GET /research/sessions ───────────────────────────────────────────────

    @app.get("/research/sessions", summary="List research sessions", tags=["Research"])
    def list_research_sessions(user_id: str = Query(default="")):
        return db.get_research_sessions(user_id or None)

    # ── GET /research/sessions/{session_id} ──────────────────────────────────

    @app.get("/research/sessions/{session_id}", summary="Get research session detail", tags=["Research"])
    def get_research_session(session_id: str):
        session = research_sessions.get(session_id)
        if session:
            return session.to_snapshot()
        rows = db.get_research_sessions()
        for row in rows:
            if row.get("session_id") == session_id:
                return row
        raise HTTPException(404, detail="Session not found")

    # ── GET /research/reports ────────────────────────────────────────────────

    @app.get("/research/reports", summary="List research reports", tags=["Research"])
    def list_research_reports(session_id: str = Query(default="")):
        return db.get_research_reports(session_id or None)

    # ── POST /research/reports/generate ──────────────────────────────────────

    class ReportRequest(BaseModel):
        synthesis_output: dict
        format:           str = "markdown"

    @app.post("/research/reports/generate", summary="Generate a report from synthesis output", tags=["Research"])
    def generate_report(req: ReportRequest):
        try:
            fmt = ReportFormat(req.format)
        except ValueError:
            raise HTTPException(400, detail=f"Unknown format: {req.format}. Use: markdown, html, json")
        report_text = report_generator.generate(req.synthesis_output, fmt)
        return {"format": req.format, "report": report_text}

    # ── GET /research/workflow-runs ──────────────────────────────────────────

    @app.get("/research/workflow-runs", summary="List workflow run history", tags=["Research"])
    def list_workflow_runs():
        return db.get_workflow_runs()

    # ── GET /research/evidence ───────────────────────────────────────────────

    @app.get("/research/evidence/{session_id}", summary="Get evidence for a session", tags=["Research"])
    def get_evidence(session_id: str):
        return db.get_evidence_by_session(session_id)

    # ── GET /tools ───────────────────────────────────────────────────────────

    @app.get("/tools", summary="List available tools", tags=["Tools"])
    def list_tools():
        return {"tools": tool_registry.all_schemas()}

    # ── POST /tools/execute ──────────────────────────────────────────────────

    class ToolRequest(BaseModel):
        tool_name: str
        params:    dict = {}

    @app.post("/tools/execute", summary="Execute a tool by name", tags=["Tools"])
    def execute_tool(req: ToolRequest):
        from app.tools.framework import ToolExecutor
        executor = ToolExecutor(tool_registry)
        result = executor.execute(req.tool_name, req.params, agent_context)
        return result.to_dict()

    # ── MCP endpoints ────────────────────────────────────────────────────────

    @app.get("/mcp/tools", summary="List MCP-compatible tools", tags=["MCP"])
    def mcp_list_tools():
        return {"tools": mcp_registry.list_tools()}

    @app.post("/mcp/tools/call", summary="Call an MCP tool", tags=["MCP"])
    def mcp_call_tool(name: str = Query(...), arguments: dict = {}):
        return mcp_registry.call_tool(name, arguments, agent_context)

    # ── Agent metrics ────────────────────────────────────────────────────────

    @app.get("/research/metrics", summary="Agent and workflow metrics", tags=["Research"])
    def research_metrics():
        return {
            "agent_executions":    metrics.agent_executions.value,
            "agent_successes":     metrics.agent_successes.value,
            "agent_failures":      metrics.agent_failures.value,
            "workflow_runs":       metrics.workflow_runs.value,
            "workflow_completions": metrics.workflow_completions.value,
            "agent_latency":       metrics.agent_latency.snapshot(),
            "workflow_latency":    metrics.workflow_latency.snapshot(),
            "planner_latency":     metrics.planner_latency.snapshot(),
            "retrieval_agent_latency": metrics.retrieval_agent_latency.snapshot(),
            "critic_latency":      metrics.critic_latency.snapshot(),
            "synthesis_latency":   metrics.synthesis_latency.snapshot(),
            "db_metrics":          db.get_agent_metrics_summary(),
        }

    # ── GET /research/agents ─────────────────────────────────────────────────

    @app.get("/research/agents", summary="List available agent types", tags=["Research"])
    def list_agents():
        return {
            "agents": [
                {"type": at.value, "description": f"{at.value} agent"}
                for at in AgentType
            ]
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 8 — Distributed Infrastructure Endpoints
    # ══════════════════════════════════════════════════════════════════════════

    # ── GET /events ──────────────────────────────────────────────────────────

    @app.get("/events", summary="List recent events", tags=["Events"])
    def list_events(
        topic: str = Query(default=None),
        limit: int = Query(default=50, ge=1, le=500),
    ):
        events = event_store.get_events(topic=topic, limit=limit)
        return {
            "total":  event_store.count(),
            "events": [e.to_dict() for e in events],
        }

    # ── GET /events/dlq (must be before /events/{event_id} to avoid shadowing)

    @app.get("/events/dlq", summary="List dead-letter queue", tags=["Events"])
    def list_dlq(limit: int = Query(default=50, ge=1, le=500)):
        return {
            "count":        event_dlq.count(),
            "dead_letters": event_dlq.get_all(limit=limit),
        }

    # ── POST /events/dlq/{event_id}/retry ────────────────────────────────────

    @app.post("/events/dlq/{event_id}/retry",
              summary="Retry a dead-lettered event", tags=["Events"])
    def retry_dlq_event(event_id: str):
        ok = event_dlq.retry(event_id, event_bus)
        if not ok:
            raise HTTPException(404, detail=f"Event {event_id!r} not in DLQ")
        return {"status": "retried", "event_id": event_id}

    # ── GET /events/{event_id} ───────────────────────────────────────────────

    @app.get("/events/{event_id}", summary="Get event by ID", tags=["Events"])
    def get_event(event_id: str):
        event = event_store.get_event(event_id)
        if event is None:
            raise HTTPException(404, detail=f"Event {event_id!r} not found")
        return event.to_dict()

    # ── GET /health ──────────────────────────────────────────────────────────

    @app.get("/health", summary="Health check", tags=["Infrastructure"])
    def health_check():
        checks = {
            "status":   "healthy",
            "database": "connected" if (db.conn is not None or db.is_postgres) else "disconnected",
            "events":   "enabled" if config.events.enabled else "disabled",
        }
        try:
            redis_client.ping()
            checks["redis"] = "connected"
        except Exception:
            checks["redis"] = "disconnected"
        return checks

    # ── GET /infrastructure/stats ────────────────────────────────────────────

    @app.get("/infrastructure/stats",
             summary="Infrastructure component status", tags=["Infrastructure"])
    def infrastructure_stats():
        return {
            "database_backend": "postgres" if db.is_postgres else "sqlite",
            "event_bus":        config.events.backend,
            "event_store_size": event_store.count(),
            "dlq_size":         event_dlq.count(),
            "redis_type":       type(redis_client).__name__,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 8 BATCH 2 — Distributed Services Endpoints
    # ══════════════════════════════════════════════════════════════════════════

    from app.gateway.models import GatewayRequest

    # ── POST /gateway/search ─────────────────────────────────────────────────

    @app.post("/gateway/search",
              summary="Retrieval gateway — unified search with caching + routing",
              tags=["Gateway"])
    def gateway_search(
        q:       str  = Query(..., min_length=1),
        mode:    str  = Query(default="hybrid", description="bm25|semantic|hybrid|pipeline"),
        top_k:   int  = Query(default=10, ge=1, le=50),
        fusion:  str  = Query(default="rrf"),
        rerank:  bool = Query(default=True),
    ):
        if retrieval_gw is None:
            raise HTTPException(503, detail="Gateway not yet initialized")
        gw_req = GatewayRequest(
            query=q, mode=mode, top_k=top_k,
            fusion=fusion, rerank=rerank,
        )
        resp = retrieval_gw.search(gw_req)
        return {
            "query":          resp.query,
            "mode":           resp.mode,
            "results":        resp.results,
            "total_results":  resp.total_results,
            "latency_ms":     resp.latency_ms,
            "cache_hit":      resp.cache_hit,
            "fusion_strategy": resp.fusion_strategy,
            "reranked":       resp.reranked,
        }

    # ── GET /gateway/stats ───────────────────────────────────────────────────

    @app.get("/gateway/stats", summary="Gateway statistics", tags=["Gateway"])
    def gateway_stats():
        if retrieval_gw is None:
            return {"status": "not_initialized"}
        return retrieval_gw.stats()

    # ── GET /gateway/cache/stats ─────────────────────────────────────────────

    @app.get("/gateway/cache/stats", summary="Gateway cache statistics", tags=["Gateway"])
    def gateway_cache_stats():
        return gateway_cache.stats()

    # ── DELETE /gateway/cache ────────────────────────────────────────────────

    @app.delete("/gateway/cache", summary="Invalidate gateway cache", tags=["Gateway"])
    def gateway_cache_invalidate():
        count = gateway_cache.invalidate()
        return {"status": "cleared", "invalidated": count}

    # ── GET /distributed/crawler/stats ───────────────────────────────────────

    @app.get("/distributed/crawler/stats",
             summary="Distributed crawler status", tags=["Distributed"])
    def distributed_crawler_stats():
        return {
            "status": "available",
            "config": {
                "max_workers": config.distributed_crawler.max_workers,
                "frontier_max_size": config.distributed_crawler.frontier_max_size,
                "batch_size": config.distributed_crawler.batch_size,
            },
        }

    # ── GET /distributed/indexing/stats ──────────────────────────────────────

    @app.get("/distributed/indexing/stats",
             summary="Distributed indexing status", tags=["Distributed"])
    def distributed_indexing_stats():
        return {
            "status": "available",
            "config": {
                "num_indexing_workers": config.distributed_indexing.num_indexing_workers,
                "num_embedding_workers": config.distributed_indexing.num_embedding_workers,
                "batch_size": config.distributed_indexing.batch_size,
                "auto_embed": config.distributed_indexing.auto_embed,
            },
        }

    # ── GET /vector-store/backend ────────────────────────────────────────────

    @app.get("/vector-store/backend",
             summary="Vector store backend info", tags=["Embeddings"])
    def vector_store_backend():
        return {
            "backend": type(vector_store).__name__,
            "total_vectors": vector_store.total_vectors,
            "qdrant_configured": config.qdrant.host != "",
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 8 BATCH 3 — Platform Services Endpoints
    # ══════════════════════════════════════════════════════════════════════════

    from app.services.registry import ServiceRegistry
    from app.services.health import HealthCheck
    from app.tenancy.manager import TenantManager
    from app.tenancy.context import TenantContext

    service_registry = ServiceRegistry(config.service_registry, redis_client=redis_client)
    health_checker   = HealthCheck()
    tenant_manager   = TenantManager(config.tenancy, redis_client=redis_client)

    health_checker.add_check("database", lambda: db.conn is not None or db.is_postgres)
    health_checker.add_check("events", lambda: config.events.enabled)

    # ── GET /services ────────────────────────────────────────────────────────

    @app.get("/services", summary="List registered services", tags=["Services"])
    def list_services():
        return service_registry.get_all_services()

    # ── POST /services/register ──────────────────────────────────────────────

    @app.post("/services/register", summary="Register a service instance", tags=["Services"])
    def register_service(
        name: str = Query(...), host: str = Query(...), port: int = Query(...),
    ):
        instance_id = service_registry.register(name, host, port)
        return {"instance_id": instance_id, "service": name}

    # ── GET /services/health ─────────────────────────────────────────────────

    @app.get("/services/health", summary="Detailed health check", tags=["Services"])
    def detailed_health():
        return health_checker.readiness()

    # ── GET /tenants ─────────────────────────────────────────────────────────

    @app.get("/tenants", summary="List tenants", tags=["Tenancy"])
    def list_tenants():
        return {"tenants": [t.__dict__ for t in tenant_manager.list_tenants()]}

    # ── POST /tenants ────────────────────────────────────────────────────────

    @app.post("/tenants", summary="Create a tenant", tags=["Tenancy"])
    def create_tenant(
        tenant_id: str = Query(...), name: str = Query(...),
    ):
        t = tenant_manager.create_tenant(tenant_id, name)
        return t.__dict__

    # ── GET /tenants/{tenant_id} ─────────────────────────────────────────────

    @app.get("/tenants/{tenant_id}", summary="Get tenant details", tags=["Tenancy"])
    def get_tenant(tenant_id: str):
        t = tenant_manager.get_tenant(tenant_id)
        if t is None:
            raise HTTPException(404, detail=f"Tenant {tenant_id!r} not found")
        return t.__dict__

    # ── GET /tenants/{tenant_id}/usage ───────────────────────────────────────

    @app.get("/tenants/{tenant_id}/usage", summary="Tenant usage stats", tags=["Tenancy"])
    def get_tenant_usage(tenant_id: str):
        usage = tenant_manager.get_usage(tenant_id)
        return usage.__dict__

    # ── GET /agents/distributed/stats ────────────────────────────────────────

    @app.get("/agents/distributed/stats",
             summary="Distributed agent execution stats", tags=["Agents"])
    def distributed_agent_stats():
        return {
            "config": {
                "max_workers": config.agent_execution.max_workers,
                "max_queue_size": config.agent_execution.max_queue_size,
                "scheduling_strategy": config.agent_execution.scheduling_strategy,
            },
            "status": "available",
        }

    # ── GET /workflows/distributed/stats ─────────────────────────────────────

    @app.get("/workflows/distributed/stats",
             summary="Distributed workflow engine stats", tags=["Workflows"])
    def distributed_workflow_stats():
        return {
            "config": {
                "max_concurrent": config.distributed_workflow.max_concurrent_workflows,
                "checkpoint_enabled": config.distributed_workflow.checkpoint_enabled,
                "state_backend": config.distributed_workflow.state_backend,
            },
            "status": "available",
        }

    return app
