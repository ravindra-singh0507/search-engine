# Search Engine & Agentic Research Platform

A production-grade **information retrieval and agentic research platform** built entirely from scratch in Python across 7 phases — from a basic tokenizer to a multi-agent autonomous research system.

Every algorithm is implemented from first principles — no Elasticsearch, no LangChain, no vector database services.

**465 tests | 7 phases | 34 tables | 65+ API endpoints | React dashboard**

---

## Quick Start

```bash
# Install dependencies
pip install fastapi uvicorn requests beautifulsoup4 pytest pytest-asyncio httpx numpy pydantic
# Optional (real embeddings + reranking):
pip install sentence-transformers faiss-cpu

# Run the API server
uvicorn app.api.routes:create_app --factory --host 0.0.0.0 --port 8000 --reload

# Run all tests
python -m pytest tests/ -q        # 465 pass, 1 skip (faiss-cpu optional)

# Start the frontend
cd frontend && npm install && npm run dev
```

### Index a document, search, chat, and research

```bash
# Index
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"title": "Python Guide", "content": "Python is a high-level programming language..."}'

# BM25 search
curl "http://localhost:8000/search?q=python"

# Hybrid search (BM25 + semantic + RRF)
curl "http://localhost:8000/hybrid-search?q=python+web+framework"

# RAG chat (Phase 6)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Python?"}'

# Agentic research (Phase 7)
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"goal": "Compare FastAPI vs Flask", "workflow": "comparison", "params": {"entities": ["FastAPI", "Flask"]}}'
```

---

## Architecture

```
+-----------------------------------------------------------------+
|                   Phase 7 -- Agentic Research                    |
|  PlannerAgent . RetrievalAgent . CriticAgent . SynthesisAgent   |
|  WorkflowEngine (DAG) . EvidenceStore . ToolFramework           |
|  MCPRegistry . ResearchMemory . ReportGenerator                 |
+-----------------------------------------------------------------+
|                      Phase 6 -- RAG Layer                        |
|  ContextBuilder -> PromptRegistry -> LLMProvider -> Citations    |
|  GroundingVerifier -> ConfidenceEngine -> MemoryService          |
|  RAGPipeline -> RAGEvaluator -> StreamingResponses              |
+-----------------------------------------------------------------+
|                Phase 5 -- Advanced Retrieval                     |
|  RetrievalPipeline (4-stage) . CrossEncoderReranker             |
|  FusionStrategies . QueryClassifier . ExperimentRunner          |
+-----------------------------------------------------------------+
|                Phase 4 -- Semantic Retrieval                     |
|  EmbeddingPipeline . FaissVectorStore . SemanticSearch          |
|  HybridSearch (BM25 + FAISS + RRF) . RetrievalEvaluator        |
+-----------------------------------------------------------------+
|                Phase 3 -- Search Quality                         |
|  BM25Ranker . AdvancedQueryParser (AST) . SpellChecker (BK-Tree)|
|  Trie Autocomplete . QueryExpander . RelevanceTuner             |
|  SnippetGenerator . LRU Cache . MetricsCollector (Prometheus)   |
+-----------------------------------------------------------------+
|                Phase 2 -- Web Crawling                           |
|  WebCrawler (BFS) . RobotsParser . URL Normalizer               |
+-----------------------------------------------------------------+
|                Phase 1 -- Search Fundamentals                    |
|  Tokenizer . Inverted Index . TF-IDF . BooleanRetriever         |
|  SQLite Storage . FastAPI                                       |
+-----------------------------------------------------------------+
```

---

## Phases

### Phase 1 -- Search Fundamentals
- **Tokenizer** -- Lowercasing, punctuation removal, stop words, position tracking
- **Inverted Index** -- Term -> posting list (doc_id, tf, positions, field)
- **Boolean Retrieval** -- AND / OR / NOT with correct AND-before-OR precedence
- **TF-IDF Ranking** -- Normalised TF x log(N/df), cosine similarity
- **SQLite Storage** -- documents, terms, postings tables with WAL mode
- **REST API** -- FastAPI with Pydantic validation, lifespan, OpenAPI docs

### Phase 2 -- Web Crawling
- **BFS Crawling** -- Breadth-first traversal with deque frontier and visited set
- **URL Normalization** -- Lowercase, strip fragments, sort query params, remove tracking params
- **robots.txt** -- Per-domain cache, Disallow / Allow / Crawl-delay
- **Content Extraction** -- BeautifulSoup title + body text + outgoing links
- **Incremental Indexing** -- Source-path deduplication; re-crawls are no-ops
- **SSRF Protection** -- Blocks private/loopback IP ranges

### Phase 3 -- Search Quality
- **BM25 Ranking** -- Saturating TF, Robertson-Spark Jones IDF, batch N+1 fix, field-aware
- **Advanced Query Parser** -- Recursive-descent AST: AND/OR/NOT, parentheses, phrase search (positional), `title:python`, `py*` wildcard
- **Trie Autocomplete** -- Prefix search, frequency ranking, JSON persistence
- **BK-Tree Spell Correction** -- Levenshtein distance + triangle-inequality pruning
- **Query Expansion** -- JSON synonym dictionary, OR-joined expansion
- **Snippet Generation** -- Hit-centred windows, overlap merge, bold highlight
- **Search Analytics** -- search_logs, click_logs, query_stats, CTR tracking
- **Relevance Tuning** -- Weighted BM25 + title_boost + recency decay + click boost
- **LRU Cache** -- Thread-safe, TTL, session-aware
- **Prometheus Metrics** -- Counters + histograms, `/metrics` endpoint
- **Security** -- Path traversal protection, SSRF blocking, API key auth, rate limiting
- **React Dashboard** -- Search, Analytics, Metrics, Crawler pages

### Phase 4 -- Semantic Retrieval
- **Embedding Infrastructure** -- EmbeddingProvider protocol, BGE-small-en-v1.5, mock for tests
- **Document Chunking** -- FixedSize and SlidingWindow chunkers (word-level, configurable)
- **FAISS Vector Store** -- IndexFlatIP + L2-normalised = exact cosine, soft deletion, save/load
- **Hybrid Retrieval** -- BM25 + semantic, fused with Reciprocal Rank Fusion (RRF, k=60)
- **Fusion Strategies** -- RRF, CombSUM, CombMNZ, Weighted, Borda Count
- **Embedding Cache** -- SHA-256 content-addressed, SQLite-persisted
- **Retrieval Evaluation** -- P@K, R@K, MRR, MAP, NDCG@K; system comparison table

### Phase 5 -- Advanced Retrieval
- **Cross-Encoder Reranking** -- ms-marco-MiniLM-L-6-v2 with sigmoid normalisation
- **4-Stage Pipeline** -- BM25+Semantic -> Fusion -> Rerank -> Final (concurrent)
- **Query Understanding** -- Rule-based intent classification (6 intents)
- **Learning-to-Rank** -- 8-feature extraction (BM25, semantic, title match, recency, etc.)
- **Experiment Framework** -- A/B retrieval experiments with metric comparison
- **Personalization** -- User profile infrastructure with click/search history

### Phase 6 -- RAG Platform
- **LLM Abstraction** -- Mock (tests), Ollama (local), OpenAI, Anthropic, Gemini
- **Context Builder** -- MMR diversification, token budgeting, Jaccard deduplication
- **Prompt Templates** -- 6 versioned templates (qa, research, summarization, documentation, comparison, troubleshooting)
- **Citation Engine** -- Sentence-level [N] attribution with source snippets
- **Grounding Verification** -- Bigram Jaccard at sentence level, risk tiers (low/medium/high)
- **Confidence Scoring** -- 4-component weighted score: retrieval(0.25) x context(0.15) x grounding(0.40) x citation(0.20)
- **Conversation Memory** -- SQLite-persisted sessions, auto-summarization at N=30
- **SSE Streaming** -- Token-by-token with post-stream citation/grounding metadata
- **Prompt Injection Detection** -- Regex patterns with [REDACTED] substitution
- **RAG Evaluation** -- 7 metrics: faithfulness, groundedness, answer relevance, context precision/recall, citation accuracy, response completeness
- **Knowledge Assistant UI** -- Chat interface with citations panel, grounding/confidence badges

### Phase 7 -- Agentic Research Platform
- **Agent Framework** -- Agent base class with lifecycle state machine, retry with exponential backoff, timeout enforcement, per-agent memory
- **5 Specialist Agents** -- PlannerAgent (goal decomposition), RetrievalAgent (evidence gathering), CriticAgent (evidence quality review), CitationValidationAgent (source verification), SynthesisAgent (report generation)
- **Workflow Orchestration** -- DAG-based execution with Kahn's topological sort, sequential and parallel modes, dependency failure propagation
- **Evidence Engine** -- EvidenceStore (CRUD + filtering), EvidenceGraph (supports/contradicts/extends relations), EvidenceExtractor, EvidenceValidator (quality gate with deduplication)
- **Research Memory** -- TaskMemory (agent execution history), EvidenceMemory (deduplicated, score-pruned), ResearchSessionMemory (event log + summarization)
- **Tool Framework** -- Abstract Tool interface, ToolRegistry, ToolExecutor; 5 built-in tools (search, retrieval, database, memory, evaluation)
- **MCP Architecture** -- MCP-compatible tool definitions with JSON Schema, list_tools/call_tool interface matching the MCP specification
- **Workflow Templates** -- 6 templates: comparison, investigation, documentation, summarization, tech_evaluation, root_cause
- **Report Generation** -- Markdown, HTML (inline CSS), and structured JSON output
- **Research Evaluation** -- 7 metrics: task completion, research completeness, citation accuracy, evidence coverage, grounding quality, hallucination rate, report quality
- **Agent Dashboard** -- Research execution UI with workflow selector, step-by-step visualization, metrics panel, tools explorer
- **Database** -- 8 new tables: agent_tasks, agent_runs, workflow_runs, evidence_records, research_sessions, citation_validation_reports, research_reports, agent_metrics
- **Observability** -- Per-agent-type latency histograms, success/failure counters, workflow latency tracking

---

## Project Structure

```
app/
  agents/                     Phase 7: Agent framework + 5 agents
    base.py                     Agent, AgentTask, AgentResult, AgentContext, Lifecycle
    planner.py                  PlannerAgent -- goal decomposition
    retrieval.py                RetrievalAgent -- evidence gathering
    critic.py                   CriticAgent -- evidence quality review
    citation_validator.py       CitationValidationAgent
    synthesis.py                SynthesisAgent -- report generation
  orchestration/              Phase 7: Workflow engine
    engine.py                   ExecutionGraph (DAG), WorkflowEngine, TaskScheduler
  evidence/                   Phase 7: Evidence tracking
    engine.py                   EvidenceStore, EvidenceGraph, Extractor, Validator
  research_memory/            Phase 7: Research session memory
    memory.py                   TaskMemory, EvidenceMemory, SessionMemory
  tools/                      Phase 7: Tool framework
    framework.py                Tool, ToolRegistry, ToolExecutor, 5 built-in tools
  mcp/                        Phase 7: MCP-compatible architecture
    registry.py                 MCPRegistry, MCPToolDefinition
  workflows/                  Phase 7: Workflow templates
    templates.py                6 templates + registry
  reports/                    Phase 7: Report generation
    generator.py                Markdown, HTML, JSON formats
  research_evaluation/        Phase 7: Research quality metrics
    evaluator.py                7-metric evaluation framework
  rag/pipeline.py             Phase 6: RAG orchestration pipeline
  context_builder/builder.py  Phase 6: Context construction with MMR
  prompts/templates.py        Phase 6: 6 prompt templates
  llm/provider.py             Phase 6: LLM abstraction (5 providers)
  citations/engine.py         Phase 6: Sentence-level citation engine
  grounding/verifier.py       Phase 6: Grounding verification
  confidence/engine.py        Phase 6: 4-component confidence scoring
  memory/memory.py            Phase 6: Conversation memory
  rag_evaluation/evaluator.py Phase 6: RAG evaluation (7 metrics)
  retrieval_pipeline/         Phase 5: Multi-stage pipeline
  reranking/                  Phase 5: Cross-encoder reranker
  query_understanding/        Phase 5: Query intent classifier
  fusion/                     Phase 4-5: Fusion strategies
  hybrid_search/              Phase 4: BM25 + semantic fusion
  semantic_search/            Phase 4: Vector similarity search
  vector_store/               Phase 4: FAISS index
  embeddings/                 Phase 4: Sentence-transformer embeddings
  bm25/                       Phase 3: BM25 ranking
  parser/                     Phase 3: AST query parser
  spellcheck/                 Phase 3: BK-tree spell correction
  autocomplete/               Phase 3: Trie autocomplete
  analytics/                  Phase 3: Search analytics
  observability/metrics.py    Phase 3-7: Prometheus metrics
  database/db.py              Phase 1-7: SQLite (34 tables)
  api/routes.py               Phase 1-7: FastAPI (65+ endpoints)
  config.py                   All configuration dataclasses
  tokenizer/                  Phase 1: Tokenization
  indexer/                    Phase 1: Inverted index
  ranking/                    Phase 1: TF-IDF
  search/                     Phase 1: Search service
  crawler/                    Phase 2: Web crawler

frontend/src/pages/
  SearchPage.tsx              BM25 search + autocomplete
  SemanticSearchPage.tsx      Dense vector search
  HybridSearchPage.tsx        Hybrid search + mode toggle
  AnalyticsPage.tsx           Search analytics dashboard
  MetricsPage.tsx             Performance metrics + charts
  CrawlerPage.tsx             Crawler control
  KnowledgeAssistantPage.tsx  Phase 6: Chat UI
  AgentDashboardPage.tsx      Phase 7: Research dashboard

tests/
  test_phase7.py              89 tests -- agents, orchestration, evidence, tools, API
  test_phase6.py              122 tests -- RAG, LLM, citations, grounding, streaming
  test_phase5.py              92 tests -- reranking, pipeline, query understanding
  test_phase4.py              46 tests -- embeddings, vector store, hybrid search
  test_phase3.py              44 tests -- BM25, parser, spell check, autocomplete
  + others                    72 tests -- tokenizer, indexer, crawler, URL normalization
```

---

## Database Schema (34 tables)

| Phase | Tables |
|---|---|
| 1-2 | documents, terms, postings, crawled_pages |
| 3 | search_logs, click_logs, query_stats |
| 4 | document_chunks, document_embeddings, embedding_jobs, vector_index_metadata, embedding_cache |
| 5 | reranking_logs, retrieval_experiments, experiment_results, ranking_features, query_intents, evaluation_reports, personalization_profiles |
| 6 | conversation_sessions, conversation_messages, citations, grounding_reports, rag_evaluations, answer_confidence, memory_snapshots |
| 7 | agent_tasks, agent_runs, workflow_runs, evidence_records, research_sessions, citation_validation_reports, research_reports, agent_metrics |

---

## API Reference

### Phase 1-3: Indexing and Search

| Method | Endpoint | Description |
|---|---|---|
| POST | `/index` | Index a single document |
| POST | `/index/directory` | Index all .txt files in a directory |
| GET | `/search?q=...` | BM25 + relevance-tuned keyword search |
| GET | `/explain?q=...&doc_id=...` | BM25 per-term score breakdown |
| GET | `/autocomplete?q=...` | Trie prefix suggestions |
| GET | `/spellcheck?q=...` | Per-word correction suggestions |
| GET | `/analytics/dashboard` | All analytics in one response |
| GET | `/metrics` | Prometheus exposition format |

### Phase 4: Semantic and Hybrid Search

| Method | Endpoint | Description |
|---|---|---|
| GET | `/semantic-search?q=...` | Dense vector search via FAISS |
| GET | `/hybrid-search?q=...` | BM25 + semantic fused with RRF |
| POST | `/embeddings/reindex` | Chunk + embed documents into FAISS |
| GET | `/evaluation` | P@K, NDCG, MRR across retrieval systems |

### Phase 5: Advanced Retrieval

| Method | Endpoint | Description |
|---|---|---|
| GET | `/pipeline-search?q=...` | 4-stage retrieval pipeline |
| GET | `/query/classify?q=...` | Query intent classification |
| POST | `/experiments` | Run retrieval experiments |

### Phase 6: RAG and Chat

| Method | Endpoint | Description |
|---|---|---|
| POST | `/chat` | RAG chat (retrieve -> generate -> cite -> ground) |
| POST | `/chat/stream` | SSE streaming chat |
| GET | `/prompts` | List prompt templates |
| POST | `/rag/evaluate` | Evaluate a RAG response |
| GET | `/rag/stats` | RAG pipeline statistics |

### Phase 7: Agentic Research

| Method | Endpoint | Description |
|---|---|---|
| POST | `/research` | Run a full agentic research workflow |
| POST | `/research/plan` | Generate a research plan without executing |
| POST | `/research/retrieve` | Run a single retrieval agent |
| GET | `/research/workflows` | List available workflow templates |
| GET | `/research/agents` | List available agent types |
| GET | `/research/sessions` | List research sessions |
| GET | `/research/sessions/{id}` | Get session details |
| GET | `/research/reports` | List research reports |
| POST | `/research/reports/generate` | Generate report from synthesis output |
| GET | `/research/workflow-runs` | Workflow run history |
| GET | `/research/evidence/{id}` | Evidence for a session |
| GET | `/research/metrics` | Agent and workflow metrics |
| GET | `/tools` | List available tools |
| POST | `/tools/execute` | Execute a tool by name |
| GET | `/mcp/tools` | List MCP-compatible tools |
| POST | `/mcp/tools/call` | Call an MCP tool |

---

## Configuration

All configuration is via Python dataclasses in `app/config.py`:

```python
from app.config import EngineConfig
config = EngineConfig()

# LLM (Phase 6) -- mock by default, no API key needed
config.rag.llm.provider = "mock"        # mock | ollama | openai | anthropic | gemini
config.rag.llm.model_name = "mock-llm-v1"

# Agent settings (Phase 7)
config.research.agent.max_retries = 3
config.research.agent.default_timeout = 120.0

# Workflow orchestration
config.research.orchestrator.parallel = True  # parallel agent execution
config.research.workflow.max_topics = 6
```

### Environment Variables

| Variable | Purpose |
|---|---|
| `SEARCH_API_KEY` | Optional API authentication key |
| `OPENAI_API_KEY` | OpenAI LLM provider |
| `ANTHROPIC_API_KEY` | Anthropic LLM provider |
| `GEMINI_API_KEY` | Google Gemini provider |

---

## Testing

```bash
python -m pytest tests/ -q                      # All: 465 pass, 1 skip
python -m pytest tests/test_phase7.py -v         # Phase 7: 89 tests
python -m pytest tests/test_phase6.py -v         # Phase 6: 122 tests
python -m pytest tests/test_phase7.py::TestPlannerAgent -v  # Single class
```

---

## Security Controls

| Control | Implementation |
|---|---|
| Path traversal | `/index/directory` validates resolved path |
| SSRF | `/crawl` blocks private/loopback IPs |
| API key auth | Optional via `SEARCH_API_KEY` env var |
| Rate limiting | 60 req/min per IP on `/search` |
| Body size limit | 1MB max on document content |
| Prompt injection | Regex detection with `[REDACTED]` substitution |
| Agent sandboxing | Per-agent memory isolation, tool permission model |
| Workflow validation | DAG cycle detection, step timeout enforcement |

---

## Key Algorithms Implemented

| Algorithm | Phase | Complexity |
|---|---|---|
| Inverted Index | 1 | O(1) term lookup |
| TF-IDF | 1 | O(T) per document |
| BM25 (Robertson-Spark Jones) | 3 | O(T) per document |
| BK-Tree (Levenshtein) | 3 | O(V^0.36) empirical |
| Trie Autocomplete | 3 | O(P + K log K) |
| Recursive-Descent AST Parser | 3 | O(Q) per query |
| L2-Normalized FAISS (cosine) | 4 | O(N x D) exact search |
| Reciprocal Rank Fusion | 4 | O(K) per document |
| Cross-Encoder Reranking | 5 | O(K x L) per batch |
| MMR Diversification | 6 | O(K^2) over chunks |
| Bigram Jaccard Grounding | 6 | O(S x C) sentences x chunks |
| Kahn's Topological Sort (DAG) | 7 | O(V + E) |

---

## Production Equivalents

| Our Component | Industry Equivalent |
|---|---|
| BM25 + Inverted Index | Elasticsearch, Apache Lucene |
| FAISS Vector Store | Pinecone, Weaviate, Milvus |
| Hybrid Search + RRF | Vespa, Elasticsearch with kNN |
| Cross-Encoder Reranking | Cohere Rerank, Jina Reranker |
| RAG Pipeline | LangChain RetrievalQA, LlamaIndex |
| Conversation Memory | LangChain ConversationBufferMemory |
| Agent Framework | CrewAI, LangGraph, AutoGen |
| Workflow Orchestration | Prefect, Airflow, LangGraph StateGraph |
| Tool Framework | OpenAI Function Calling, Anthropic Tool Use |
| MCP Registry | Anthropic MCP SDK |
| Research Pipeline | OpenAI Deep Research, Perplexity Pro |

---

## License

This project is for educational purposes -- built to learn information retrieval, NLP, and AI systems from first principles.
