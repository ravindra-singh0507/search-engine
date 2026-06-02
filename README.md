# Search Engine — Built from Scratch

A production-grade search engine built entirely from scratch in Python — no Elasticsearch, no Solr, no pre-built IR libraries.

**252 tests · 4 phases · BM25 + FAISS + Hybrid RRF**

---

## What This Project Covers

### Phase 1 — Local Document Search
- **Tokenizer** — Lowercasing, punctuation removal, stop words, position tracking
- **Inverted Index** — Term → posting list (doc_id, tf, positions, field)
- **Boolean Retrieval** — AND / OR / NOT with correct AND-before-OR precedence
- **TF-IDF Ranking** — Normalised TF × log(N/df), cosine similarity
- **SQLite Storage** — documents, terms, postings tables with WAL mode
- **REST API** — FastAPI with Pydantic validation, lifespan, OpenAPI docs

### Phase 2 — Web Crawler
- **BFS Crawling** — Breadth-first traversal with deque frontier and visited set
- **URL Normalization** — Lowercase, strip fragments, sort query params, remove tracking params
- **robots.txt** — Per-domain cache, Disallow / Allow / Crawl-delay, section-bleed fix
- **Content Extraction** — BeautifulSoup title + body text + outgoing links
- **Incremental Indexing** — Source-path deduplication; re-crawls are no-ops
- **Crawl Controls** — Depth limit, page limit, stay-on-domain, politeness delay

### Phase 3 — Advanced Search Features
- **BM25 Ranking** — Saturating TF, Robertson–Spärck Jones IDF, batch N+1 fix, field-aware
- **Field-Aware Indexing** — Separate `title` / `body` posting rows; title boost in ranking
- **Advanced Query Parser** — Recursive-descent AST: AND/OR/NOT, parentheses, phrase search (positional), `title:python`, `py*` wildcard, thread-safe `_ParseState`
- **Trie Autocomplete** — Prefix search, frequency ranking, JSON persistence
- **BK-Tree Spell Correction** — Levenshtein distance + BK-tree + confidence scores
- **Query Expansion** — JSON synonym dictionary, OR-joined expansion
- **Snippet Generation** — Hit-centred windows, overlap merge, `**bold**` highlight, content-length cap
- **Search Analytics** — `search_logs`, `click_logs`, `query_stats` — CTR, top queries, zero-result tracking
- **Relevance Tuning** — Weighted BM25 + title_boost + recency decay + click boost
- **LRU Cache** — Thread-safe `OrderedDict`, TTL, per-session `log_id` (cache-poisoning fix)
- **Prometheus Metrics** — Counters + histograms, `/metrics` endpoint, slow-query detection
- **Security** — Path traversal protection, SSRF blocking, optional API key, rate limiting
- **React Dashboard** — Vite + TypeScript + Recharts: Search, Analytics, Metrics, Crawler pages

### Phase 4 — Semantic Retrieval Platform
- **Embedding Infrastructure** — `EmbeddingProvider` Protocol; `LocalEmbeddingProvider` (sentence-transformers, BAAI/bge-small-en-v1.5); `MockEmbeddingProvider` for tests
- **Document Chunking** — `FixedSizeChunker` and `SlidingWindowChunker` (word-level, configurable size and overlap)
- **Embedding Pipeline** — Chunk → cache-check → batch embed → FAISS insert → DB record; incremental (only new docs)
- **FAISS Vector Store** — `IndexFlatIP` + L2-normalised vectors = exact cosine similarity; soft deletion; save/load
- **Semantic Search** — Embed query → ANN search → dedup to best chunk per doc → ranked results
- **Hybrid Retrieval** — BM25 + semantic, fused with **Reciprocal Rank Fusion** (RRF, k=60)
- **Retrieval Explainability** — Full score breakdown: BM25 score/rank, cosine score/rank, RRF score, reason string
- **Embedding Cache** — SHA-256 content-addressed, SQLite-persisted, cross-restart
- **Incremental Vector Updates** — Add / remove from FAISS on document lifecycle events
- **Evaluation Framework** — P@K, R@K, MRR, MAP, NDCG@K; `RetrievalEvaluator`; ASCII comparison table
- **Extended Observability** — Embedding latency, semantic latency, hybrid latency, embedding cache hit rate
- **Frontend (Phase 4)** — Semantic Search page + Hybrid Search page with mode toggle (Hybrid/BM25/Semantic)

---

## Project Structure

```
search-engine-project/
├── app/
│   ├── api/
│   │   └── routes.py                  # All FastAPI endpoints (Phases 1–4)
│   ├── analytics/
│   │   └── analytics.py               # Search event logging + CTR
│   ├── autocomplete/
│   │   └── trie.py                    # Trie + AutocompleteService
│   ├── benchmarks/
│   │   └── benchmarker.py             # Latency + throughput benchmarks
│   ├── bm25/
│   │   └── bm25.py                    # BM25 ranker (batch posting fetch)
│   ├── cache/
│   │   └── lru_cache.py               # LRU cache + QueryCache
│   ├── chunking/
│   │   └── chunker.py                 # FixedSize + SlidingWindow chunkers
│   ├── crawler/
│   │   ├── crawler.py                 # BFS web crawler
│   │   ├── robots.py                  # robots.txt parser
│   │   └── url_normalize.py           # URL canonicalization
│   ├── database/
│   │   └── db.py                      # SQLite layer (10 tables, Phase 1–4)
│   ├── embeddings/
│   │   ├── cache.py                   # SHA-256 embedding cache
│   │   ├── pipeline.py                # Chunk → embed → store pipeline
│   │   └── provider.py                # EmbeddingProvider protocol + Local/Mock
│   ├── evaluation/
│   │   ├── evaluator.py               # RetrievalEvaluator
│   │   └── metrics.py                 # P@K, R@K, MRR, MAP, NDCG@K
│   ├── hybrid_search/
│   │   └── hybrid_service.py          # RRF + linear fusion
│   ├── indexer/
│   │   └── indexer.py                 # Field-aware inverted index builder
│   ├── observability/
│   │   └── metrics.py                 # Prometheus-style counters + histograms
│   ├── parser/
│   │   ├── advanced_query_parser.py   # AST parser (phrase, field, wildcard)
│   │   └── query_parser.py            # Simple Boolean parser
│   ├── query_expansion/
│   │   ├── expander.py                # Synonym expander
│   │   └── synonyms.json              # Synonym dictionary
│   ├── ranking/
│   │   ├── relevance_tuning.py        # Multi-signal ranker
│   │   └── tfidf.py                   # TF-IDF (kept for comparison)
│   ├── search/
│   │   └── search_service.py          # Full pipeline orchestrator
│   ├── semantic_search/
│   │   └── semantic_service.py        # Dense vector retrieval
│   ├── snippets/
│   │   └── snippet_generator.py       # Hit-centred snippet + highlight
│   ├── spellcheck/
│   │   ├── bk_tree.py                 # BK-tree
│   │   ├── levenshtein.py             # Edit distance (rolling DP)
│   │   └── spell_checker.py           # SpellChecker service
│   ├── tokenizer/
│   │   └── tokenizer.py               # Text tokenizer
│   ├── vector_store/
│   │   └── store.py                   # VectorStore protocol + FaissVectorStore
│   └── config.py                      # All configuration dataclasses
├── data/
│   ├── search_engine.db               # SQLite database
│   ├── eval_dataset.json              # Sample evaluation dataset
│   ├── faiss_index/                   # FAISS index files (auto-created)
│   └── trie.json                      # Autocomplete trie (auto-created)
├── documents/                         # Sample .txt files for local indexing
├── frontend/
│   ├── src/
│   │   ├── api/client.ts              # Typed API client
│   │   ├── pages/
│   │   │   ├── SearchPage.tsx         # BM25 search + autocomplete
│   │   │   ├── SemanticSearchPage.tsx # Dense vector search
│   │   │   ├── HybridSearchPage.tsx   # Hybrid + mode toggle
│   │   │   ├── AnalyticsPage.tsx      # Search analytics dashboard
│   │   │   ├── MetricsPage.tsx        # Performance metrics + charts
│   │   │   └── CrawlerPage.tsx        # Crawler control
│   │   ├── App.tsx
│   │   ├── index.css
│   │   └── main.tsx
│   ├── package.json
│   └── vite.config.ts
├── tests/
│   ├── conftest.py                    # Shared pytest fixtures
│   ├── test_tokenizer.py
│   ├── test_indexer.py
│   ├── test_query_parser.py
│   ├── test_ranking.py
│   ├── test_search.py
│   ├── test_api.py
│   ├── test_crawler.py
│   ├── test_url_normalize.py
│   ├── test_phase3.py                 # Phase 3 component tests
│   └── test_phase4.py                 # Phase 4 component tests
├── main.py                            # Application entry point
└── requirements.txt
```

---

## Database Schema

```
documents         doc_id, title, content, source, doc_type, word_count, created_at
terms             term_id, term, document_frequency
postings          term_id, doc_id, term_frequency, positions, field (title|body)
crawled_pages     page_id, url, title, content, html, status_code, depth, doc_id
search_logs       log_id, query, results_count, latency_ms, timestamp, session_id
click_logs        click_id, log_id, doc_id, position, timestamp
query_stats       query_id, query, total_searches, avg_latency_ms, zero_result_searches
document_chunks   chunk_id, doc_id, chunk_index, text, start_offset, end_offset, word_count
document_embeddings embedding_id, chunk_id, doc_id, model_name, vector_dim
embedding_jobs    job_id, doc_id, status, model_name, chunks_total, chunks_processed, error
vector_index_metadata  model_name, dimension, total_vectors, index_path, updated_at
embedding_cache   content_hash, model_name, vector_json, created_at
```

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Phase 4 dependencies** (sentence-transformers + FAISS) are included in `requirements.txt`.  
`sentence-transformers` requires PyTorch (~200 MB download on first install).  
If you only want Phases 1–3, the engine works without them — it automatically falls back to a `MockEmbeddingProvider`.

### 3. Start the backend

```bash
python main.py
```

The API runs at `http://localhost:8000`.  
Interactive docs at `http://localhost:8000/docs`.

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `ENV` | `development` | Set to `production` to disable uvicorn reload |
| `SEARCH_API_KEY` | *(unset)* | If set, mutating endpoints require `X-API-Key` header |

### 4. Start the frontend (optional)

```bash
cd frontend
npm install
npm run dev        # → http://localhost:3000
```

The Vite dev server proxies `/api/*` to `http://localhost:8000`.

---

## Usage

### Index documents

```bash
# Index all .txt files in documents/
curl -X POST http://localhost:8000/index/directory \
  -H "Content-Type: application/json" \
  -d '{"directory": "documents"}'

# Index a single document
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"title": "Flask Guide", "content": "Flask is a lightweight Python web framework."}'
```

### Build the semantic index (Phase 4)

```bash
# Chunk + embed all documents and store in FAISS (synchronous)
curl -X POST http://localhost:8000/embeddings/reindex \
  -H "Content-Type: application/json" \
  -d '{"sync": true}'

# Check embedding status
curl http://localhost:8000/embeddings/stats
```

### Search

```bash
# BM25 keyword search
curl "http://localhost:8000/search?q=python"

# Boolean operators
curl "http://localhost:8000/search?q=python+AND+web"
curl "http://localhost:8000/search?q=python+OR+java"
curl "http://localhost:8000/search?q=python+NOT+java"

# Field search
curl "http://localhost:8000/search?q=title:python"

# Phrase search (positional)
curl "http://localhost:8000/search?q=%22machine+learning%22"

# Wildcard / prefix
curl "http://localhost:8000/search?q=py*"

# Semantic search (dense vector)
curl "http://localhost:8000/semantic-search?q=fast+data+retrieval"

# Hybrid search (BM25 + semantic + RRF)
curl "http://localhost:8000/hybrid-search?q=neural+ranking+models"
```

### Explain scores

```bash
# BM25 term-by-term breakdown
curl "http://localhost:8000/explain?q=python&doc_id=1"

# Full hybrid score breakdown
curl "http://localhost:8000/hybrid-search/explain?q=python&doc_id=1"
```

### Autocomplete and spell check

```bash
curl "http://localhost:8000/autocomplete?q=py"
curl "http://localhost:8000/spellcheck?q=pythn"
curl "http://localhost:8000/spellcheck/query?q=machne+lerning"
```

### Analytics

```bash
curl http://localhost:8000/analytics/dashboard
curl http://localhost:8000/analytics/top-queries
curl http://localhost:8000/analytics/failures
curl "http://localhost:8000/analytics/search-volume?hours=24"
```

### Evaluation

```bash
# Run BM25 vs semantic vs hybrid on data/eval_dataset.json
curl "http://localhost:8000/evaluation?systems=bm25,semantic,hybrid"
```

### Crawl websites

```bash
curl -X POST http://localhost:8000/crawl \
  -H "Content-Type: application/json" \
  -d '{
    "seed_urls": ["https://docs.python.org/3/tutorial/index.html"],
    "max_depth": 2,
    "max_pages": 20,
    "stay_on_domain": true
  }'

curl http://localhost:8000/crawl/status
```

### Metrics

```bash
# Prometheus exposition format
curl http://localhost:8000/metrics

# JSON snapshot
curl http://localhost:8000/metrics/snapshot
```

---

## API Reference

### Indexing

| Method | Endpoint | Description |
|---|---|---|
| POST | `/index` | Index a single document |
| POST | `/index/directory` | Index all `.txt` files in a directory |

### Search

| Method | Endpoint | Description |
|---|---|---|
| GET | `/search?q=...` | BM25 + relevance-tuned keyword search |
| GET | `/semantic-search?q=...` | Dense vector search via FAISS |
| GET | `/hybrid-search?q=...` | BM25 + semantic fused with RRF |
| GET | `/explain?q=...&doc_id=...` | BM25 per-term score breakdown |
| GET | `/semantic-search/explain?q=...&doc_id=...` | Per-chunk cosine scores |
| GET | `/hybrid-search/explain?q=...&doc_id=...` | Full hybrid score explanation |
| POST | `/search/click` | Record a result click (analytics) |

### Documents

| Method | Endpoint | Description |
|---|---|---|
| GET | `/document/{id}` | Get document by ID |
| DELETE | `/document/{id}` | Delete document, index, and embeddings |

### Autocomplete & Spell Check

| Method | Endpoint | Description |
|---|---|---|
| GET | `/autocomplete?q=...` | Trie prefix suggestions |
| GET | `/spellcheck?q=...` | Per-word correction suggestions |
| GET | `/spellcheck/query?q=...` | Auto-correct a full query |

### Embeddings (Phase 4)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/embeddings/reindex` | Chunk + embed documents into FAISS |
| GET | `/embeddings/stats` | FAISS stats, cache, job queue |
| DELETE | `/embeddings/cache` | Clear embedding cache |
| GET | `/vector-store/stats` | FAISS index statistics |

### Evaluation (Phase 4)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/evaluation` | P@K, R@K, MRR, MAP, NDCG@K across systems |
| GET | `/evaluation/detail` | Per-query evaluation breakdown |

### Analytics

| Method | Endpoint | Description |
|---|---|---|
| GET | `/analytics/dashboard` | All metrics in one response |
| GET | `/analytics/top-queries` | Most frequently searched queries |
| GET | `/analytics/search-volume` | Hourly search counts |
| GET | `/analytics/failures` | Zero-result queries |
| GET | `/analytics/click-through-rate` | CTR statistics |

### Observability

| Method | Endpoint | Description |
|---|---|---|
| GET | `/metrics` | Prometheus text format |
| GET | `/metrics/snapshot` | JSON metrics snapshot |
| GET | `/stats` | Index statistics |

### Crawler

| Method | Endpoint | Description |
|---|---|---|
| POST | `/crawl` | Start a BFS web crawl |
| GET | `/crawl/status` | Current crawl status |
| GET | `/crawl/stats` | Pages crawled / indexed |

---

## Running Tests

```bash
pytest tests/ -v
```

**252 tests** covering every component across all four phases.

```bash
# Run just Phase 4 tests
pytest tests/test_phase4.py -v

# Run Phase 3 regression tests
pytest tests/test_phase3.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=term-missing
```

---

## Key Concepts Implemented

### Inverted Index
Maps each term to a posting list: `{doc_id, tf, positions[], field}`. Enables O(1) term lookup vs O(N) document scan. Phase 3 adds a `field` column (`title` / `body`) so the ranker can boost title matches independently.

### TF-IDF
`tf(t,d) = count/total · idf(t) = log(N/df)`. Cosine similarity normalises for document length. Kept for benchmarking; superseded by BM25 as the default ranker.

### BM25
Improves TF-IDF with: (1) saturating TF — `tf*(t,d) = f(k₁+1)/(f+k₁(1-b+b|d|/avgdl))` — so high-frequency terms don't dominate; (2) Robertson–Spärck Jones IDF that is always ≥ 0. Batch posting fetch eliminates the N+1 query bottleneck.

### Positional Phrase Search
Token positions are stored at index time. `"machine learning"` requires `pos(learning) = pos(machine) + 1` in at least one candidate document. Stop words increment the position counter, so `"machine to learning"` correctly fails to match `"machine learning"`.

### Boolean Operator Precedence
AND binds tighter than OR (standard algebra). `"python OR java AND backend"` evaluates as `python OR (java AND backend)` via an OR-split before AND-group evaluation.

### Trie Autocomplete
Prefix tree with frequency-ranked DFS traversal. O(P + K log K) where P = prefix length, K = completions. Serialised to `data/trie.json` on shutdown and reloaded on startup.

### BK-Tree Spell Correction
Metric tree using Levenshtein distance with triangle-inequality pruning. Query cost is empirically O(V^0.36) on English vocabularies. Rolling two-row DP with early termination at `max_distance`.

### Reciprocal Rank Fusion
`RRF(d) = Σᵢ 1/(60 + rankᵢ(d))`. Only cares about rank order, not score magnitude — no normalisation required between BM25 scores and cosine similarities. Empirically outperforms weighted linear combination (Cormack et al., 2009).

### FAISS Vector Store
`IndexFlatIP` with L2-normalised vectors = exact cosine similarity = inner product. Soft deletion via a `_deleted: set[int]`. `compact()` rebuilds without deleted vectors. Persisted as `index.faiss` + `id_map.json`.

### Retrieval Evaluation
Ground-truth pairs in `data/eval_dataset.json`. Metrics: Precision@K, Recall@K, MRR, MAP, NDCG@K. `RetrievalEvaluator` accepts any `(query, top_k) → [doc_ids]` function and compares systems side by side.

---

## Architecture Overview

```
                        ┌─────────────────────────────────┐
  User Query            │         FastAPI Routes           │
       │                │   /search  /semantic  /hybrid    │
       ▼                └──────────────┬──────────────────┘
  ┌────────────┐                       │
  │   Spell    │   ┌───────────────────▼──────────────────┐
  │ Correction │   │            SearchService              │
  │  (BK-tree) │   │  spell → expand → parse → retrieve   │
  └─────┬──────┘   │  → BM25 rank → snippet → cache       │
        │          └───────────────────┬──────────────────┘
        ▼                              │
  ┌────────────┐         ┌─────────────┴──────────────────┐
  │  Query     │         │                                  │
  │ Expansion  │   BM25  │  Inverted Index (SQLite)        │
  │ (synonyms) │  ◄──────┤  postings(term_id, doc_id,      │
  └────────────┘         │  tf, positions, field)          │
                         └─────────────┬──────────────────┘
                                       │
                         ┌─────────────▼──────────────────┐
                         │      EmbeddingPipeline          │
                         │  doc → chunk → embed → FAISS    │
                         └─────────────┬──────────────────┘
                                       │
              ┌────────────────────────▼─────────────────────┐
              │            HybridSearchService                │
              │                                               │
              │   BM25 results + Semantic results             │
              │          │                │                   │
              │   ┌──────▼──────┐  ┌─────▼──────┐           │
              │   │  BM25Ranker │  │  FAISS ANN  │           │
              │   │ (batch SQL) │  │ (cosine sim)│           │
              │   └──────┬──────┘  └─────┬───────┘           │
              │          └──────┬────────┘                   │
              │            RRF Fusion                         │
              │       score = Σ 1/(60 + rankᵢ)               │
              └──────────────────────────────────────────────┘
                                 │
                         Final ranked results
```

---

## Benchmarking

```bash
# Via Python
python -c "
from app.api.routes import create_app
from app.config import EngineConfig
from app.benchmarks.benchmarker import Benchmarker
# ... (requires a running DB with indexed documents)
"

# Via API — compare BM25 vs semantic vs hybrid
curl 'http://localhost:8000/evaluation?systems=bm25,semantic,hybrid&top_k=10'
```

---

## What's Next (Phase 5 Ideas)

- **Cross-Encoder Re-ranking** — Use a `cross-encoder/ms-marco-MiniLM-L-6-v2` to re-rank the top-50 hybrid results
- **RAG Integration** — Feed the top-k hybrid passages into an LLM for answer generation
- **Streaming search** — Server-Sent Events for progressive result delivery
- **Multi-language support** — `paraphrase-multilingual-MiniLM-L12-v2` for cross-lingual retrieval
- **Personalization** — Per-user click signal incorporated into ranking weights
- **Approximate ANN** — Switch FAISS `IndexFlatIP` → `IndexHNSWFlat` for sub-linear query time at >100k vectors
- **Distributed indexing** — Shard the inverted index and vector store across multiple nodes
