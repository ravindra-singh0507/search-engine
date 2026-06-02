"""
Semantic Search Service

=== PIPELINE ===

  Query string
    ↓  embed_query(query)          — dense vector (384 dims)
    ↓  vector_store.search(vec, k) — ANN search → [(chunk_id, score)]
    ↓  chunk_id → doc_id mapping   — lookup via document_chunks table
    ↓  best-chunk-per-doc          — deduplicate: keep highest-scored chunk
    ↓  sort by score               — rank
    → SemanticSearchResult

=== WHY CHUNK → DOC DEDUP ===

If a document has 5 chunks and 3 of them are similar to the query,
all 3 would appear in the top-k raw FAISS results.  We want top-k
*distinct documents*, not top-k chunks.  We keep the highest-scoring
chunk per document as the "representative passage".

=== LATENCY PROFILE ===

  embed_query:    10-50 ms  (CPU, bge-small)
  FAISS search:   0.5-5 ms  (IndexFlatIP, 10k vectors)
  DB chunk lookup: 1-3 ms
  Total:          12-60 ms  on CPU

=== COMPARISON WITH BM25 ===

  BM25 Strengths:              Semantic Strengths:
  - Exact keyword match        - Synonym handling
  - Rare term boost            - Paraphrase understanding
  - Very fast (<1 ms)          - Cross-lingual (multilingual models)
  - No pre-indexing needed     - Natural language questions
  - Explainable                - Better on short queries

=== PRODUCTION EQUIVALENTS ===

  Elasticsearch/OpenSearch: kNN vector search (Lucene HNSW)
  Pinecone:                 Managed vector DB, serverless
  Qdrant:                   Open-source, rich filter support
  Weaviate:                 Schema-based, hybrid BM25+vector built-in
"""

import logging
import time
from dataclasses import dataclass

from app.database.db import Database
from app.embeddings.provider import EmbeddingProvider
from app.vector_store.store import FaissVectorStore
from app.snippets.snippet_generator import SnippetGenerator

logger = logging.getLogger(__name__)


@dataclass
class SemanticResult:
    rank:           int
    doc_id:         int
    chunk_id:       str
    title:          str
    snippet:        str          # highlighted passage from the top matching chunk
    chunk_text:     str          # raw text of the best matching chunk
    semantic_score: float        # cosine similarity in [0, 1]


@dataclass
class SemanticSearchResponse:
    query:          str
    search_time_ms: float
    model_name:     str
    total_results:  int
    results:        list[SemanticResult]


class SemanticSearchService:
    """
    End-to-end semantic search over the FAISS vector index.
    """

    def __init__(
        self,
        db:             Database,
        provider:       EmbeddingProvider,
        vector_store:   FaissVectorStore,
        snippet_gen:    SnippetGenerator | None = None,
    ):
        self.db           = db
        self.provider     = provider
        self.vector_store = vector_store
        self.snippet_gen  = snippet_gen

    # ── Public API ────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> SemanticSearchResponse:
        """
        Embed the query and retrieve the top_k most semantically similar documents.
        """
        start = time.perf_counter()

        if self.vector_store.total_vectors == 0:
            return SemanticSearchResponse(
                query=query, search_time_ms=0.0,
                model_name=self.provider.model_name,
                total_results=0, results=[],
            )

        # Step 1: embed query
        query_vec = self.provider.embed_query(query)

        # Step 2: ANN search — fetch more than top_k to allow doc dedup
        raw_results = self.vector_store.search(query_vec, top_k=top_k * 3)

        # Step 3: deduplicate to best chunk per document
        best: dict[int, tuple[str, float]] = {}   # doc_id → (chunk_id, score)
        for chunk_id, score in raw_results:
            chunk = self.db.get_chunk(chunk_id)
            if chunk is None:
                continue
            existing = best.get(chunk.doc_id)
            if existing is None or score > existing[1]:
                best[chunk.doc_id] = (chunk_id, score)

        # Step 4: sort by score, take top_k
        sorted_docs = sorted(best.items(), key=lambda x: x[1][1], reverse=True)[:top_k]

        # Step 5: build results
        results: list[SemanticResult] = []
        for rank, (doc_id, (chunk_id, score)) in enumerate(sorted_docs, 1):
            doc   = self.db.get_document(doc_id)
            chunk = self.db.get_chunk(chunk_id)
            if doc is None or chunk is None:
                continue

            query_terms = query.lower().split()
            if self.snippet_gen:
                snippet = self.snippet_gen.generate(chunk.text, query_terms)
            else:
                snippet = chunk.text[:300]

            results.append(SemanticResult(
                rank=rank, doc_id=doc_id, chunk_id=chunk_id,
                title=doc.title, snippet=snippet,
                chunk_text=chunk.text,
                semantic_score=round(score, 6),
            ))

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "Semantic search %r: %d results in %.1f ms (model=%s)",
            query, len(results), elapsed, self.provider.model_name,
        )
        return SemanticSearchResponse(
            query=query, search_time_ms=round(elapsed, 2),
            model_name=self.provider.model_name,
            total_results=len(results), results=results,
        )

    def explain(self, query: str, doc_id: int) -> dict:
        """
        Return a detailed explanation of why doc_id scored as it did.
        """
        doc = self.db.get_document(doc_id)
        if doc is None:
            return {"error": f"Document {doc_id} not found"}

        query_vec = self.provider.embed_query(query)
        chunks    = self.db.get_chunks_for_doc(doc_id)

        chunk_scores = []
        for chunk in chunks:
            # Retrieve the stored vector by doing a single-item search
            chunk_search = self.vector_store.search(query_vec, top_k=200)
            # Find this chunk's score if it appears
            score = next((s for cid, s in chunk_search if cid == chunk.chunk_id), 0.0)
            chunk_scores.append({"chunk_id": chunk.chunk_id,
                                  "score": round(score, 6),
                                  "text_preview": chunk.text[:100]})

        best_score = max((c["score"] for c in chunk_scores), default=0.0)
        return {
            "doc_id":       doc_id,
            "title":        doc.title,
            "model":        self.provider.model_name,
            "query":        query,
            "best_semantic_score": round(best_score, 6),
            "chunks":       chunk_scores,
            "total_chunks": len(chunks),
        }
