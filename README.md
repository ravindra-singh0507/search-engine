# Search Engine & Distributed AI Infrastructure Platform

A production-grade **information retrieval, RAG, and agentic research platform** built entirely from scratch in Python across 8 phases — from a basic tokenizer to a distributed AI infrastructure with multi-tenancy, event-driven architecture, and Kubernetes deployment.

Every algorithm is implemented from first principles — no Elasticsearch, no LangChain, no vector database services.

**924 tests | 8 phases | 35 tables | 113 API endpoints | 51K lines | React dashboard | Docker + K8s**

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server
python main.py
# API docs: http://localhost:8000/docs

# Run all tests
python -m pytest tests/ -q        # 924 pass

# Start the frontend
cd frontend && npm install && npm run dev
```

### Docker (recommended for full stack)

```bash
docker-compose up                        # app + postgres + redis
docker-compose --profile kafka up        # + Kafka event streaming
docker-compose --profile vector up       # + Qdrant vector DB
docker-compose --profile monitoring up   # + Prometheus + Grafana
```

### Kubernetes

```bash
kubectl apply -f k8s/
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                Phase 8 — Distributed Infrastructure                  │
│  EventBus (Kafka/InMemory) · Redis · PostgreSQL · Qdrant · Docker   │
│  Security (JWT/RBAC) · Tenancy · CI/CD · K8s · Resilience · Cost    │
├─────────────────────────────────────────────────────────────────────┤
│                Phase 7 — Agentic Research                            │
│  PlannerAgent · RetrievalAgent · CriticAgent · SynthesisAgent       │
│  WorkflowEngine (DAG) · EvidenceStore · ToolFramework · MCP         │
├─────────────────────────────────────────────────────────────────────┤
│                Phase 6 — RAG Platform                                │
│  ContextBuilder → PromptRegistry → LLM → Citations → Grounding     │
│  ConfidenceEngine · MemoryService · RAGEvaluator · Streaming        │
├─────────────────────────────────────────────────────────────────────┤
│                Phase 5 — Advanced Retrieval                          │
│  RetrievalPipeline (4-stage) · CrossEncoderReranker · Fusion        │
│  QueryClassifier · ExperimentRunner · LearningToRank                │
├─────────────────────────────────────────────────────────────────────┤
│                Phase 4 — Semantic Retrieval                          │
│  EmbeddingPipeline · FAISS/Qdrant VectorStore · HybridSearch        │
│  Fusion (RRF, CombSUM, CombMNZ, Weighted, Borda) · Evaluation      │
├─────────────────────────────────────────────────────────────────────┤
│                Phase 3 — Search Quality                              │
│  BM25 · AdvancedParser (AST) · SpellCheck (BK-Tree) · Autocomplete │
│  Analytics · Prometheus · Relevance Tuning · LRU Cache              │
├─────────────────────────────────────────────────────────────────────┤
│                Phase 2 — Web Crawling                                │
│  WebCrawler (BFS) · RobotsParser · URL Normalizer                   │
├─────────────────────────────────────────────────────────────────────┤
│                Phase 1 — Search Fundamentals                         │
│  Tokenizer · Inverted Index · TF-IDF · BooleanRetriever · SQLite    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phases

### Phase 1 — Search Fundamentals
Tokenizer, inverted index, TF-IDF ranking, boolean retrieval (AND/OR/NOT), SQLite storage, FastAPI.

### Phase 2 — Web Crawling
BFS crawler, robots.txt, URL normalization, content extraction, SSRF protection.

### Phase 3 — Search Quality
BM25 (Robertson-Spark Jones), AST query parser, BK-tree spell correction, trie autocomplete, query expansion, snippet generation, analytics, Prometheus metrics, React dashboard.

### Phase 4 — Semantic Retrieval
Sentence-transformer embeddings, FAISS vector store, hybrid search (BM25 + semantic + RRF), 5 fusion strategies, embedding cache, retrieval evaluation (P@K, NDCG, MRR, MAP).

### Phase 5 — Advanced Retrieval
Cross-encoder reranking (ms-marco), 4-stage retrieval pipeline, query intent classification, learning-to-rank features, A/B experiment framework, personalization.

### Phase 6 — RAG Platform
LLM abstraction (Mock/Ollama/OpenAI/Anthropic/Gemini), context builder (MMR), 6 prompt templates, citation engine, grounding verification, confidence scoring, conversation memory, SSE streaming, prompt injection detection, RAG evaluation (7 metrics).

### Phase 7 — Agentic Research
5 specialist agents, DAG workflow engine (topological sort), evidence store + graph, research memory, tool framework (MCP-compatible), 6 workflow templates, report generation (Markdown/HTML/JSON), research evaluation.

### Phase 8 — Distributed AI Infrastructure
Event-driven architecture, Kafka integration, PostgreSQL migration, Redis (cache/locks/rate-limit/sessions), Docker containerization, distributed crawling (coordinator/frontier/workers), distributed indexing (event-driven pipeline), Qdrant vector search, retrieval gateway (L1+L2 cache), microservice foundations, distributed agent execution (queue/scheduler/worker pool), distributed workflow engine (checkpoints/recovery), multi-tenancy (isolation/middleware/RBAC), JWT security (auth/API keys/audit), observability (tracing/structured logging), resilience (circuit breakers/retry/health probes), cost tracking, CI/CD (GitHub Actions), Kubernetes (HPA/PDB/Ingress), load testing (Locust), performance optimization (batch processing/distributed cache).

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_BACKEND` | `sqlite` | `sqlite` or `postgres` |
| `EVENT_BACKEND` | `memory` | `memory` or `kafka` |
| `VECTOR_BACKEND` | `faiss` | `faiss` or `qdrant` |
| `CRAWLER_MODE` | `single` | `single` or `distributed` |
| `AGENT_MODE` | `local` | `local` or `distributed` |
| `SECURITY_ENABLED` | `false` | Enable JWT/RBAC/audit |
| `TENANCY_ENABLED` | `false` | Enable multi-tenant isolation |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `REDIS_HOST` | `localhost` | Redis host |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka brokers |
| `JWT_SECRET` | dev secret | JWT signing key |

---

## API Reference (113 endpoints)

### Search & Retrieval
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/search?q=...` | BM25 keyword search |
| GET | `/semantic-search?q=...` | Vector similarity search |
| GET | `/hybrid-search?q=...` | BM25 + semantic + RRF |
| GET | `/rerank-search?q=...` | Full 4-stage pipeline |
| POST | `/gateway/search` | Retrieval gateway (cached + routed) |

### RAG & Chat
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Conversational RAG |
| POST | `/chat/stream` | SSE streaming |
| POST | `/rag/query` | Single-turn with diagnostics |
| POST | `/research/query` | Multi-step research |

### Agents & Workflows
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/research` | Full agentic workflow |
| POST | `/research/plan` | Generate plan only |
| GET | `/research/workflows` | List templates |
| GET | `/tools` | Available tools |

### Infrastructure
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| GET | `/metrics` | Prometheus metrics |
| GET | `/events` | Event stream |
| GET | `/resilience/health-probes` | Dependency checks |
| GET | `/cost/summary` | Cost breakdown |
| GET | `/observability/traces` | Distributed traces |

### Security & Tenancy
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/security/token` | Issue JWT token |
| GET | `/security/rbac` | Roles & permissions |
| POST | `/tenants` | Create tenant |
| GET | `/tenants/{id}/usage` | Tenant usage stats |

---

## Project Structure

```
app/
  ── Phase 1-3: Search ──────────────────────────
  tokenizer/           Tokenization + stopwords
  indexer/             Inverted index
  bm25/                BM25 ranking
  parser/              Boolean + advanced AST parser
  search/              Search orchestration
  spellcheck/          BK-tree spell correction
  autocomplete/        Trie autocomplete
  analytics/           Search analytics
  cache/               LRU cache
  observability/       Prometheus + tracing + logging
  
  ── Phase 2: Crawling ──────────────────────────
  crawler/             BFS crawler + robots.txt
  
  ── Phase 4-5: Retrieval ───────────────────────
  embeddings/          Sentence-transformers + cache
  chunking/            Document chunkers
  vector_store/        FAISS + Qdrant + factory
  semantic_search/     ANN search
  hybrid_search/       BM25 + semantic fusion
  fusion/              5 fusion strategies
  reranking/           Cross-encoder reranker
  retrieval_pipeline/  4-stage pipeline
  evaluation/          IR metrics
  
  ── Phase 6: RAG ──────────────────────────────
  rag/                 8-stage RAG pipeline
  llm/                 5 LLM providers
  context_builder/     MMR context selection
  prompts/             6 prompt templates
  citations/           Citation engine
  grounding/           Hallucination detection
  confidence/          Confidence scoring
  memory/              Conversation memory
  
  ── Phase 7: Agents ───────────────────────────
  agents/              5 agent types
  orchestration/       DAG workflow engine
  workflows/           6 workflow templates
  tools/               MCP-compatible tools
  evidence/            Evidence store + graph
  research_memory/     Research sessions
  reports/             Report generation
  
  ── Phase 8: Infrastructure ───────────────────
  events/              Event bus + store + DLQ
  redis/               Cache + locks + rate limiter
  kafka/               Producer + consumer + topics
  distributed/
    crawler/           Coordinator + frontier + workers
    indexing/          Indexing + embedding workers
    agents/            Queue + scheduler + pool
    workflows/         Checkpoint + recovery
  gateway/             Query router + L1+L2 cache
  services/            Registry + health + discovery
  tenancy/             Manager + context + isolation
  security/            JWT + API keys + RBAC + audit
  resilience/          Circuit breakers + retry
  cost/                Cost tracking + budgets
  performance/         Distributed cache + batching
  database/            Backend abstraction (SQLite/PG)

  api/routes.py        113 API endpoints
  config.py            42 config dataclasses

frontend/src/pages/    8 React pages
k8s/                   12 Kubernetes manifests
.github/workflows/     CI/CD pipelines
load_tests/            Locust + benchmarks
docker-compose.yml     9 Docker services
docs/OPERATIONS.md     Operations handbook
```

---

## Testing

```bash
python -m pytest tests/ -q                          # All 924 tests
python -m pytest tests/test_phase8.py -v            # Phase 8 Batch 1
python -m pytest tests/test_phase8_enterprise.py -v # Enterprise readiness
python -m pytest tests/ -k "not API" -q             # Fast unit tests only (~3s)
```

---

## Security

| Layer | Implementation |
|-------|---------------|
| Authentication | JWT (HMAC-SHA256) + API keys (SHA-256 hashed) |
| Authorization | RBAC (5 roles: admin, operator, agent_user, reader, service) |
| Endpoint protection | Permission matrix (34 rules, 5 access levels) |
| Audit logging | Append-only JSONL with event types |
| Tenant isolation | App-layer wrappers (DB, vectors, cache) |
| Rate limiting | Redis sorted-set sliding window |
| Prompt injection | Regex detection + sanitization |
| Path traversal | Resolved path validation |
| SSRF | IP blocklist (RFC 1918 + loopback) |

---

## Key Algorithms

| Algorithm | Phase | Complexity |
|-----------|-------|-----------|
| Inverted Index | 1 | O(1) term lookup |
| BM25 (Robertson-Spark Jones) | 3 | O(T) per document |
| BK-Tree (Levenshtein) | 3 | O(V^0.36) empirical |
| FAISS (L2-normalized cosine) | 4 | O(N×D) exact |
| Reciprocal Rank Fusion | 4 | O(K) per document |
| Cross-Encoder Reranking | 5 | O(K×L) per batch |
| MMR Diversification | 6 | O(K²) over chunks |
| Kahn's Topological Sort | 7 | O(V+E) for DAG |
| Circuit Breaker (state machine) | 8 | O(1) per call |
| Sliding Window Rate Limiter | 8 | O(1) via sorted set |

---

## Production Equivalents

| Our Component | Industry Equivalent |
|---------------|-------------------|
| BM25 + Inverted Index | Elasticsearch, Lucene |
| FAISS / Qdrant Vector Store | Pinecone, Weaviate, Milvus |
| Hybrid Search + RRF | Vespa, Elastic kNN |
| Cross-Encoder Reranking | Cohere Rerank, Jina |
| RAG Pipeline | LangChain, LlamaIndex |
| Agent Framework | CrewAI, LangGraph, AutoGen |
| Workflow Engine | Temporal, Airflow, Prefect |
| Event Bus | Apache Kafka, Google Pub/Sub |
| Service Registry | Consul, Eureka |
| Circuit Breaker | Netflix Hystrix, Resilience4j |
| Distributed Cache | Netflix EVCache, Memcached |
| Cost Tracking | AWS Cost Explorer, OpenAI Usage |

---

## License

This project is for educational and portfolio purposes — built to learn information retrieval, NLP, distributed systems, and AI infrastructure from first principles.
