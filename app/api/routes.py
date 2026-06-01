"""
FastAPI API Endpoints

All HTTP endpoints for the search engine:
- Document indexing
- Search
- Document retrieval
- Statistics
- Web crawling
"""

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.database.db import Database
from app.tokenizer.tokenizer import Tokenizer, TokenizerConfig
from app.indexer.indexer import Indexer
from app.search.search_service import SearchService
from app.crawler.crawler import WebCrawler
from app.config import EngineConfig

logger = logging.getLogger(__name__)

# ── Pydantic Models ──


class IndexDocumentRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    source: str = Field(default="api", max_length=1000)
    doc_type: str = Field(default="text", max_length=50)


class IndexDirectoryRequest(BaseModel):
    directory: str = Field(default="documents")


class CrawlRequest(BaseModel):
    seed_urls: list[str] = Field(..., min_length=1)
    max_depth: int = Field(default=2, ge=1, le=10)
    max_pages: int = Field(default=50, ge=1, le=1000)
    stay_on_domain: bool = Field(default=True)


class SearchResponse(BaseModel):
    query: str
    total_matches: int
    search_time_ms: float
    results: list[dict]


class DocumentResponse(BaseModel):
    doc_id: int
    title: str
    content: str
    source: str
    doc_type: str
    word_count: int
    created_at: str


class StatsResponse(BaseModel):
    total_documents: int
    total_terms: int
    total_postings: int
    total_crawled_pages: int
    index_snapshot: Optional[dict] = None


# ── Application Setup ──

def create_app(config: EngineConfig | None = None) -> FastAPI:
    """Application factory: creates and configures the FastAPI app with all dependencies."""
    config = config or EngineConfig()

    db = Database(config.database.db_path)
    tokenizer = Tokenizer(TokenizerConfig(
        min_token_length=config.tokenizer.min_token_length,
        max_token_length=config.tokenizer.max_token_length,
        custom_stop_words=config.tokenizer.custom_stop_words,
    ))
    indexer = Indexer(db, tokenizer)
    search_service = SearchService(db, tokenizer)
    crawler = WebCrawler(
        db=db, indexer=indexer,
        user_agent=config.crawler.user_agent,
        request_delay=config.crawler.request_delay,
        timeout=config.crawler.timeout,
        respect_robots=config.crawler.respect_robots_txt,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        db.connect()
        logger.info("Search engine started")
        yield
        db.close()
        logger.info("Search engine stopped")

    app = FastAPI(
        title="Search Engine",
        description="A search engine built from scratch — inverted index, TF-IDF ranking, "
                    "Boolean retrieval, and BFS web crawler. No frameworks, no shortcuts.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── Indexing Endpoints ──

    @app.post("/index", summary="Index a single document")
    def index_document(req: IndexDocumentRequest):
        """
        Index a document by providing its title and content.
        The content will be tokenized and added to the inverted index.
        """
        result = indexer.index_document(
            title=req.title, content=req.content,
            source=req.source, doc_type=req.doc_type
        )
        return {
            "status": "indexed",
            "doc_id": result.doc_id,
            "title": result.title,
            "terms_indexed": result.terms_indexed,
            "total_tokens": result.total_tokens,
        }

    @app.post("/index/directory", summary="Index all text files in a directory")
    def index_directory(req: IndexDirectoryRequest):
        """
        Index all .txt files in the specified directory.
        Files already indexed (by source path) are skipped.
        """
        directory = Path(req.directory)
        if not directory.exists():
            raise HTTPException(status_code=404, detail=f"Directory not found: {req.directory}")

        results = indexer.index_directory(directory)
        return {
            "status": "indexed",
            "documents_indexed": len(results),
            "details": [
                {
                    "doc_id": r.doc_id,
                    "title": r.title,
                    "terms_indexed": r.terms_indexed,
                }
                for r in results
            ],
        }

    # ── Search Endpoints ──

    @app.get("/search", summary="Search indexed documents")
    def search(
        q: str = Query(..., min_length=1, description="Search query"),
        top_k: int = Query(default=10, ge=1, le=100, description="Number of results"),
    ):
        """
        Search the index with a query string.

        Supports:
        - Simple queries: `python`
        - Multi-term (implicit AND): `python backend`
        - Boolean: `python AND backend`, `python OR java`, `python NOT java`

        Results are ranked by TF-IDF cosine similarity.
        """
        result = search_service.search(q, top_k=top_k)
        return SearchResponse(
            query=result.query,
            total_matches=result.total_matches,
            search_time_ms=result.search_time_ms,
            results=[
                {
                    "rank": i + 1,
                    "doc_id": r.doc_id,
                    "score": round(r.score, 6),
                    "title": r.title,
                    "snippet": r.snippet,
                    "term_scores": {k: round(v, 6) for k, v in r.term_scores.items()},
                }
                for i, r in enumerate(result.results)
            ],
        )

    # ── Document Endpoints ──

    @app.get("/document/{doc_id}", summary="Get a document by ID")
    def get_document(doc_id: int):
        """Retrieve the full content of a document by its ID."""
        doc = db.get_document(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        return DocumentResponse(
            doc_id=doc.doc_id, title=doc.title, content=doc.content,
            source=doc.source, doc_type=doc.doc_type,
            word_count=doc.word_count, created_at=doc.created_at,
        )

    @app.delete("/document/{doc_id}", summary="Delete a document")
    def delete_document(doc_id: int):
        """Delete a document and its postings from the index."""
        deleted = db.delete_document(doc_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        return {"status": "deleted", "doc_id": doc_id}

    # ── Stats Endpoint ──

    @app.get("/stats", summary="Get engine statistics")
    def get_stats(include_index: bool = Query(default=False)):
        """
        Get statistics about the search engine: document count, term count, etc.
        Set include_index=true to also return the full inverted index snapshot.
        """
        stats = db.get_stats()
        response = StatsResponse(
            total_documents=stats["total_documents"],
            total_terms=stats["total_terms"],
            total_postings=stats["total_postings"],
            total_crawled_pages=stats["total_crawled_pages"],
        )
        if include_index:
            response.index_snapshot = indexer.get_inverted_index_snapshot()
        return response

    # ── Crawler Endpoints ──

    @app.post("/crawl", summary="Start a web crawl")
    def start_crawl(req: CrawlRequest):
        """
        Start a BFS web crawl from the given seed URLs.
        Crawled pages are automatically indexed.
        The crawl runs in a background thread.
        """
        if crawler.current_job and crawler.current_job.is_running:
            raise HTTPException(status_code=409, detail="A crawl is already running")

        def run_crawl():
            crawler.crawl(
                seed_urls=req.seed_urls,
                max_depth=req.max_depth,
                max_pages=req.max_pages,
                stay_on_domain=req.stay_on_domain,
            )

        thread = threading.Thread(target=run_crawl, daemon=True)
        thread.start()

        return {
            "status": "started",
            "seed_urls": req.seed_urls,
            "max_depth": req.max_depth,
            "max_pages": req.max_pages,
        }

    @app.get("/crawl/status", summary="Get crawl status")
    def get_crawl_status():
        """Get the current status of the web crawler."""
        return crawler.get_status()

    @app.get("/crawl/stats", summary="Get crawl statistics")
    def get_crawl_stats():
        """Get statistics about all crawled pages."""
        return db.get_crawl_stats()

    return app
