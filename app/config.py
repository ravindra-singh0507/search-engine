"""
Search Engine Configuration — Phase 8

Phase 6 adds: LLMConfig, ContextConfig, MemoryConfig, CitationConfig,
GroundingConfig, RAGConfig — for the RAG / Knowledge Assistant layer.

Phase 7 adds: AgentConfig, OrchestratorConfig, WorkflowConfig,
ResearchConfig — for the Agentic Retrieval platform.

Phase 8 adds: EventConfig, PostgresConfig, RedisConfig — for the
Distributed AI Infrastructure platform.
"""

from dataclasses import dataclass, field
from pathlib import Path


# ── Phase 1-3 configs ─────────────────────────────────────────────────────────

@dataclass
class DatabaseConfig:
    db_path: Path = Path("data/search_engine.db")
    backend: str = "sqlite"


@dataclass
class TokenizerConfig:
    min_token_length: int = 2
    max_token_length: int = 50
    custom_stop_words: list[str] = field(default_factory=list)


@dataclass
class IndexerConfig:
    batch_size: int = 100


@dataclass
class BM25Config:
    k1: float = 1.5
    b: float = 0.75


@dataclass
class CrawlerConfig:
    max_depth: int = 3
    max_pages: int = 100
    request_delay: float = 1.0
    timeout: int = 10
    user_agent: str = "SearchEngineBot/1.0"
    respect_robots_txt: bool = True


@dataclass
class SearchConfig:
    default_top_k: int = 10
    max_top_k: int = 100


@dataclass
class SnippetConfig:
    max_length: int = 300
    context_words: int = 15
    max_fragments: int = 3


@dataclass
class AutocompleteConfig:
    max_suggestions: int = 10
    persist_path: Path = Path("data/trie.json")


@dataclass
class SpellCheckConfig:
    max_edit_distance: int = 2
    min_word_length: int = 3


@dataclass
class CacheConfig:
    query_cache_size: int = 512
    result_cache_ttl_seconds: int = 300


@dataclass
class ObservabilityConfig:
    enable_metrics: bool = True
    slow_query_threshold_ms: float = 200.0


@dataclass
class RankingWeights:
    bm25: float = 1.0
    title_boost: float = 1.8
    recency_boost: float = 0.05
    click_boost: float = 0.3


# ── Phase 4 configs ────────────────────────────────────────────────────────────

@dataclass
class EmbeddingConfig:
    model_name: str  = "BAAI/bge-small-en-v1.5"
    batch_size: int  = 32
    device: str      = "cpu"
    auto_embed: bool = False
    cache_embeddings: bool = True


@dataclass
class ChunkingConfig:
    strategy: str      = "sliding_window"
    chunk_size: int    = 256
    chunk_overlap: int = 32
    min_chunk_words: int = 10


@dataclass
class VectorStoreConfig:
    index_path: Path = Path("data/faiss_index")
    dimension: int   = 384
    similarity: str  = "cosine"


@dataclass
class HybridSearchConfig:
    fusion_strategy: str   = "rrf"
    rrf_k: int             = 60
    bm25_weight: float     = 0.5
    semantic_weight: float = 0.5


@dataclass
class EvaluationConfig:
    eval_dataset_path: Path = Path("data/eval_dataset.json")
    k_values: list[int]     = field(default_factory=lambda: [1, 3, 5, 10])


# ── Phase 5 configs ────────────────────────────────────────────────────────────

@dataclass
class RerankingConfig:
    """
    Controls the cross-encoder re-ranking stage.

    model_name:   Any HuggingFace cross-encoder compatible with sentence-transformers.
                  ms-marco-MiniLM-L-6-v2 is fast (6 layers) and strong on MSMARCO.
                  bge-reranker-base is competitive for multi-domain use.

    top_k_rerank: How many candidates to pass to the reranker.  Higher = better
                  quality but more latency.  50 is a good default; production
                  systems typically use 100-200.

    enabled:      Set False to skip reranking (useful for A/B comparisons).
    """
    model_name:   str  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_rerank: int  = 50
    batch_size:   int  = 32
    device:       str  = "cpu"
    enabled:      bool = True


@dataclass
class PipelineConfig:
    """
    Controls the multi-stage retrieval pipeline.

    Stage 1: BM25 + Semantic retrieve bm25_candidates / semantic_candidates each.
    Stage 2: Fuse with fusion_strategy.
    Stage 3: Rerank top rerank_top_k candidates.
    Stage 4: Return final_top_k.
    """
    bm25_candidates:     int  = 100
    semantic_candidates: int  = 100
    fusion_strategy:     str  = "rrf"     # rrf | combsum | combmnz | weighted | borda
    rerank_top_k:        int  = 50
    final_top_k:         int  = 10
    use_reranker:        bool = True
    use_semantic:        bool = True


@dataclass
class QueryUnderstandingConfig:
    """Rule-based query intent classification."""
    enabled:              bool  = True
    confidence_threshold: float = 0.3   # below this → "informational" fallback


@dataclass
class ExperimentConfig:
    """Controls retrieval experiment storage and limits."""
    storage_path:    Path = Path("data/experiments")
    max_experiments: int  = 100


@dataclass
class PersonalizationConfig:
    """
    User profile infrastructure.  Disabled by default until click data accumulates.
    """
    enabled:           bool = False
    max_search_history: int = 100
    max_click_history:  int = 500
    decay_days:         int = 30


# ── Phase 6 configs ────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """
    Controls which LLM backend the RAG pipeline uses.

    provider:    "mock" (tests), "ollama" (local), "openai", "anthropic", "gemini"
    model_name:  Provider-specific model ID.
    base_url:    API base for self-hosted models (Ollama default: localhost:11434).
    api_key_env: Name of the environment variable holding the API key.
                 Leave blank for Ollama or when the key is set another way.
    max_tokens:  Maximum completion tokens the LLM may generate.
    temperature: 0.0 = deterministic; higher = more creative.
    timeout:     HTTP request timeout in seconds.
    max_retries: How many times to retry on 5xx / timeout errors.
    """
    provider:    str   = "mock"
    model_name:  str   = "mock-llm-v1"
    base_url:    str   = "http://localhost:11434"
    api_key_env: str   = ""
    max_tokens:  int   = 2048
    temperature: float = 0.1
    timeout:     int   = 60
    max_retries: int   = 3


@dataclass
class ContextConfig:
    """
    Controls how retrieved chunks are selected and assembled into LLM context.

    max_tokens:       Hard token budget for the context block.
    max_chunks:       Maximum number of chunks regardless of token budget.
    min_score:        Drop chunks below this retrieval score.
    dedup_threshold:  Jaccard similarity above which two chunks are considered
                      duplicates; only the higher-scoring one is kept.
    use_mmr:          Maximal Marginal Relevance diversification (True = avoid
                      repeating the same information from different chunks).
    diversity_lambda: MMR trade-off: 1.0 = pure relevance, 0.0 = pure diversity.
    """
    max_tokens:       int   = 3000
    max_chunks:       int   = 10
    min_score:        float = 0.05
    dedup_threshold:  float = 0.80
    use_mmr:          bool  = True
    diversity_lambda: float = 0.6


@dataclass
class MemoryConfig:
    """
    Controls conversation history management.

    max_messages:   Hard cap on messages stored per session.
    context_window: How many recent messages to include in the prompt.
    summarize_at:   When the session exceeds this many messages, older turns
                    are summarized and replaced with a summary message so the
                    context window stays bounded.
    persist:        If True, sessions are written to SQLite.
    """
    max_messages:   int  = 50
    context_window: int  = 8
    summarize_at:   int  = 30
    persist:        bool = True


@dataclass
class CitationConfig:
    """
    Controls how source attributions are attached to generated answers.

    style:           "numbered" → [1], [2]; "inline" → (Source: title).
    include_snippet: Include a brief excerpt from the source in the reference.
    max_snippet_len: Maximum characters for the reference snippet.
    """
    style:           str  = "numbered"   # "numbered" | "inline"
    include_snippet: bool = True
    max_snippet_len: int  = 200


@dataclass
class GroundingConfig:
    """
    Controls hallucination / grounding verification.

    threshold:       Answers with grounding_score below this are flagged as
                     potentially ungrounded.
    method:          "overlap" = token overlap (fast, no extra deps);
                     "nli" = Natural Language Inference (requires model).
    min_support_len: Minimum tokens an answer sentence must share with the
                     context to count as 'supported'.
    """
    threshold:       float = 0.25
    method:          str   = "overlap"
    min_support_len: int   = 4


@dataclass
class RAGConfig:
    """
    Top-level config for the RAG / Knowledge Assistant layer.

    enable_multi_step:   Decompose complex queries into sub-queries.
    max_subqueries:      Maximum sub-queries for multi-step retrieval.
    stream_chunk_size:   Characters per SSE token chunk.
    response_cache_size: LRU cache for identical (query, session) pairs.
    """
    llm:                 LLMConfig      = field(default_factory=LLMConfig)
    context:             ContextConfig  = field(default_factory=ContextConfig)
    memory:              MemoryConfig   = field(default_factory=MemoryConfig)
    citation:            CitationConfig = field(default_factory=CitationConfig)
    grounding:           GroundingConfig = field(default_factory=GroundingConfig)
    enable_multi_step:   bool = True
    max_subqueries:      int  = 3
    stream_chunk_size:   int  = 8
    response_cache_size: int  = 128


# ── Phase 7 configs ────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    """
    Controls agent execution behaviour.

    max_retries:     retry count per agent task (including first attempt).
    base_delay_sec:  initial delay between retries (doubles each retry).
    max_delay_sec:   cap on retry delay.
    default_timeout: wall-clock limit per agent task in seconds.
    max_memory:      max entries in per-agent memory.
    """
    max_retries:     int   = 3
    base_delay_sec:  float = 0.5
    max_delay_sec:   float = 10.0
    default_timeout: float = 120.0
    max_memory:      int   = 100


@dataclass
class OrchestratorConfig:
    """
    Controls the workflow orchestration engine.

    parallel:         run independent steps concurrently (threading).
    max_steps:        hard limit on steps per workflow (safety cap).
    step_timeout_sec: default timeout per step if not overridden.
    """
    parallel:         bool  = False
    max_steps:        int   = 20
    step_timeout_sec: float = 120.0


@dataclass
class WorkflowConfig:
    """
    Controls workflow template defaults.

    default_template: fallback template when none specified.
    max_topics:       cap on sub-topics the planner can generate.
    """
    default_template: str = "investigation"
    max_topics:       int = 6


@dataclass
class ResearchConfig:
    """
    Top-level config for the Phase 7 Agentic Retrieval layer.

    agent:         agent execution settings
    orchestrator:  workflow engine settings
    workflow:      template defaults
    max_sessions:  max concurrent research sessions
    evidence_max:  max evidence records per session
    """
    agent:         AgentConfig        = field(default_factory=AgentConfig)
    orchestrator:  OrchestratorConfig = field(default_factory=OrchestratorConfig)
    workflow:      WorkflowConfig     = field(default_factory=WorkflowConfig)
    max_sessions:  int = 50
    evidence_max:  int = 500


# ── Phase 8 configs ────────────────────────────────────────────────────────────

@dataclass
class PostgresConfig:
    """
    PostgreSQL connection settings.

    Used when DatabaseConfig.backend = "postgres".
    Connection pooling via psycopg2.pool.ThreadedConnectionPool.
    Graceful fallback to SQLite if PostgreSQL is unavailable.

    === PRODUCTION EQUIVALENTS ===

      Google Cloud SQL / Amazon RDS — managed PostgreSQL
      Uber:    Schemaless on MySQL/PostgreSQL shards
      Netflix: CockroachDB / PostgreSQL with connection pooling via PgBouncer
    """
    host:      str = "localhost"
    port:      int = 5432
    database:  str = "search_engine"
    user:      str = "search_engine"
    password:  str = ""
    pool_size: int = 10
    pool_min:  int = 2
    ssl_mode:  str = "prefer"


@dataclass
class EventConfig:
    """
    Controls the event-driven architecture.

    backend:           "memory" (in-process) or "kafka" (distributed).
    store_backend:     "memory" or "database" (persists events to DB).
    max_store_events:  Bounded event store size (FIFO eviction).
    retry_max_retries: Default retry count for failed event delivery.
    retry_base_delay:  Base delay for exponential backoff.
    dlq_enabled:       Dead-letter queue for permanently failed events.

    === THEORY ===

    Event-driven architecture decouples producers from consumers via
    asynchronous message passing.  This enables:
      - Loose coupling between services
      - Horizontal scaling (add consumers independently)
      - Audit trail (event store)
      - Replay capability (re-process from event store)

    === PRODUCTION EQUIVALENTS ===

      Google:  Pub/Sub
      Netflix: Apache Kafka + Hermes
      Uber:    Apache Kafka + uReplicator
      OpenAI:  Internal event bus for job orchestration
    """
    enabled:           bool  = True
    backend:           str   = "memory"
    store_backend:     str   = "memory"
    max_store_events:  int   = 10000
    retry_max_retries: int   = 3
    retry_base_delay:  float = 1.0
    dlq_enabled:       bool  = True


@dataclass
class RedisConfig:
    """
    Redis connection settings.

    Used for distributed caching, session storage, distributed locks,
    and rate limiting.  Graceful fallback to in-memory implementations
    when Redis is unavailable.

    === THEORY ===

    Redis is an in-memory data structure store supporting strings, hashes,
    lists, sets, sorted sets, and streams.  Sub-millisecond latency for
    reads/writes.  Used as:
      - L2 cache (behind in-process LRU L1 cache)
      - Session store (server-side sessions)
      - Distributed lock (SETNX + TTL)
      - Rate limiter (sliding window via sorted sets)

    === PRODUCTION EQUIVALENTS ===

      Google:    Memorystore (managed Redis)
      Netflix:   EVCache (Redis-based)
      Uber:      Redis + custom sharding
      OpenSearch: Redis for query caching
    """
    host:           str   = "localhost"
    port:           int   = 6379
    db:             int   = 0
    password:       str   = ""
    ssl:            bool  = False
    prefix:         str   = "se:"
    socket_timeout: float = 5.0


# ── Phase 8 Batch 2 configs ────────────────────────────────────────────────────

@dataclass
class KafkaConfig:
    """
    Apache Kafka connection and topic settings.

    === THEORY ===

    Kafka is a distributed event streaming platform using a partitioned,
    replicated commit log.  Each topic is split into partitions for parallel
    consumption.  Consumer groups enable horizontal scaling: each partition
    is consumed by exactly one consumer in the group.

    === PRODUCTION EQUIVALENTS ===

      Google:   Pub/Sub
      Netflix:  Kafka + custom Hermes layer
      Uber:     Kafka + uReplicator for cross-DC replication
      LinkedIn: Kafka (originally developed there)
    """
    bootstrap_servers: str  = "localhost:9092"
    group_id:          str  = "search-engine"
    auto_offset_reset: str  = "earliest"
    enable_auto_commit: bool = True
    max_poll_records:   int  = 100
    session_timeout_ms: int  = 30000
    request_timeout_ms: int  = 60000
    retry_topic_suffix: str  = ".retry"
    dlq_topic_suffix:   str  = ".dlq"
    num_partitions:     int  = 3
    replication_factor: int  = 1


@dataclass
class DistributedCrawlerConfig:
    """
    Configuration for the distributed crawler cluster.

    === THEORY ===

    A distributed crawler uses a URL frontier (priority queue of URLs to
    fetch) shared across multiple worker processes.  A coordinator assigns
    work, enforces politeness (per-domain rate limits), and tracks state.

    Key challenges:
      - URL deduplication (don't crawl the same URL twice)
      - Politeness (respect robots.txt + per-host rate limits)
      - Fault tolerance (reassign work if a worker dies)
      - Priority (crawl important pages first)

    === PRODUCTION EQUIVALENTS ===

      Google:     Googlebot with distributed scheduler
      Bing:       MSNBot with URL frontier service
      CommonCrawl: Nutch/Heritrix-based distributed crawlers
    """
    max_workers:          int   = 4
    frontier_max_size:    int   = 100000
    batch_size:           int   = 10
    worker_timeout_sec:   float = 300.0
    dedup_backend:        str   = "memory"
    rate_limit_per_domain: float = 1.0
    max_retries:          int   = 2
    checkpoint_interval:  int   = 100


@dataclass
class DistributedIndexingConfig:
    """
    Configuration for distributed indexing workers.

    === THEORY ===

    Distributed indexing separates document processing into independent
    stages (chunking, embedding, indexing) that run as event-driven workers.
    Each stage consumes events from the bus and produces events for the
    next stage, forming a processing pipeline.

    === PRODUCTION EQUIVALENTS ===

      Elasticsearch: Index sharding across nodes
      Google:        MapReduce indexing pipeline
      Vespa:         Content nodes with document processors
    """
    num_indexing_workers:  int  = 2
    num_embedding_workers: int  = 2
    batch_size:            int  = 10
    embedding_batch_size:  int  = 32
    worker_timeout_sec:    float = 120.0
    auto_embed:            bool = True
    retry_on_failure:      bool = True
    max_retries:           int  = 3


@dataclass
class QdrantConfig:
    """
    Qdrant vector database connection settings.

    Qdrant is used as the distributed vector store, replacing or
    augmenting the local FAISS index for production deployments.
    Falls back to FAISS when Qdrant is unavailable.

    === THEORY ===

    Qdrant uses HNSW (Hierarchical Navigable Small World) graphs for
    approximate nearest neighbour search.  Unlike FAISS IndexFlatIP
    (exact, O(N·D)), HNSW achieves O(log N) query time with tunable
    recall via ef_construct and m parameters.

    === PRODUCTION EQUIVALENTS ===

      Pinecone:   Managed vector DB
      Weaviate:   Open-source vector DB with hybrid search
      Milvus:     Open-source vector DB with IVF + HNSW
      Vespa:      Hybrid search with ANN built-in
    """
    host:             str  = "localhost"
    port:             int  = 6333
    grpc_port:        int  = 6334
    collection_name:  str  = "search_engine"
    prefer_grpc:      bool = False
    api_key:          str  = ""
    vector_size:      int  = 384
    distance:         str  = "Cosine"
    on_disk:          bool = False
    hnsw_m:           int  = 16
    hnsw_ef_construct: int = 100


@dataclass
class GatewayConfig:
    """
    Retrieval gateway settings.

    The gateway sits between clients and retrieval backends, handling
    query routing, caching, rate limiting, and result fusion.

    === PRODUCTION EQUIVALENTS ===

      Google:     GFE (Google Front End) + serving infrastructure
      Netflix:    Zuul API gateway
      Uber:       Edge gateway with request routing
      Elastic:    Coordinating nodes in Elasticsearch
    """
    cache_ttl:          int   = 300
    cache_max_size:     int   = 1000
    rate_limit_rpm:     int   = 120
    timeout_sec:        float = 30.0
    max_concurrent:     int   = 50
    enable_cache:       bool  = True
    default_fusion:     str   = "rrf"
    default_rerank:     bool  = True


# ── Phase 8 Batch 3 configs ────────────────────────────────────────────────────

@dataclass
class ServiceRegistryConfig:
    """
    Microservice registry and discovery configuration.

    === THEORY ===

    Service discovery enables services to find each other without hardcoded
    addresses.  The registry tracks service instances, health status, and
    metadata.  Clients query the registry to route requests.

    === PRODUCTION EQUIVALENTS ===

      Netflix:    Eureka
      Kubernetes: kube-dns + Service resources
      Consul:     HashiCorp Consul
      Uber:       Hyperbahn (TChannel service mesh)
    """
    enabled:             bool  = True
    heartbeat_interval:  float = 10.0
    health_check_interval: float = 15.0
    stale_threshold_sec: float = 30.0
    max_instances:       int   = 100


@dataclass
class AgentExecutionConfig:
    """
    Distributed agent execution infrastructure.

    Controls the agent worker pool, task queue, and scheduling.

    === THEORY ===

    Distributed agent execution separates task submission from execution.
    Tasks are enqueued in a priority queue and dispatched to a pool of
    workers.  This enables horizontal scaling: add workers to increase
    agent throughput without changing the submission interface.

    === PRODUCTION EQUIVALENTS ===

      Celery:     Distributed task queue with worker pools
      Ray:        Distributed computing framework with actors
      Temporal:   Durable workflow + activity workers
      OpenAI:     Internal agent execution cluster
    """
    max_workers:        int   = 8
    max_queue_size:     int   = 1000
    default_priority:   int   = 5
    worker_timeout_sec: float = 120.0
    max_retries:        int   = 3
    scheduling_strategy: str  = "priority"
    enable_preemption:  bool  = False
    checkpoint_interval: int  = 10


@dataclass
class DistributedWorkflowConfig:
    """
    Distributed workflow execution engine configuration.

    === THEORY ===

    Distributed workflows extend the Phase 7 WorkflowEngine with:
      - State persistence (survives crashes)
      - Checkpointing (resume from last successful step)
      - Distributed step execution (steps run on different workers)
      - Scheduling (cron-like or event-triggered workflows)
      - Execution tracking (audit trail of all step executions)

    === PRODUCTION EQUIVALENTS ===

      Temporal:   Durable workflow execution
      Airflow:    DAG-based workflow scheduling
      Prefect:    Flow orchestration with state management
      Step Functions: AWS state machine execution
    """
    max_concurrent_workflows: int   = 20
    checkpoint_enabled:       bool  = True
    checkpoint_backend:       str   = "memory"
    max_step_retries:         int   = 3
    step_timeout_sec:         float = 300.0
    schedule_enabled:         bool  = False
    execution_ttl_hours:      int   = 24
    state_backend:            str   = "memory"


@dataclass
class TenancyConfig:
    """
    Multi-tenancy configuration.

    === THEORY ===

    Multi-tenancy allows a single platform instance to serve multiple
    isolated customers (tenants).  Each tenant has:
      - Isolated data (documents, indexes, embeddings)
      - Isolated memory (conversations, sessions)
      - Isolated agents (workflows, research sessions)
      - Independent rate limits and quotas
      - Separate analytics and audit trails

    Isolation strategies:
      - Logical: shared DB with tenant_id column (this implementation)
      - Physical: separate DB/schemas per tenant (future)

    === PRODUCTION EQUIVALENTS ===

      Elastic Cloud:  cluster-per-tenant or index-per-tenant
      Salesforce:     shared schema with org_id partitioning
      OpenAI:         organization-level isolation
      Slack:          workspace-level data isolation
    """
    enabled:              bool  = False
    default_tenant:       str   = "default"
    max_tenants:          int   = 100
    isolation_level:      str   = "logical"
    max_docs_per_tenant:  int   = 100000
    max_sessions_per_tenant: int = 1000
    max_agents_per_tenant: int  = 50
    enable_billing_hooks: bool  = False


# ── Phase 8 Batch 4 configs ────────────────────────────────────────────────────

@dataclass
class SecurityConfig:
    """
    Security platform configuration: JWT, API keys, RBAC, audit logging.

    === THEORY ===

    Defense-in-depth security layers:
      1. Authentication: verify who the caller is (JWT / API key)
      2. Authorization: verify what they can do (RBAC)
      3. Audit logging: record what they did (immutable log)
      4. Secrets management: keep credentials out of code

    JWT (RFC 7519) encodes claims in a signed token.  The server verifies
    the signature — no DB lookup needed (stateless auth).  API keys are
    simpler for machine-to-machine use.

    === PRODUCTION EQUIVALENTS ===

      Google: IAM + service accounts
      AWS:    IAM policies + Cognito
      Netflix: ZUUL gateway with JWT validation
      OpenAI: Organization-level API keys + scopes
    """
    enabled:             bool  = False
    jwt_secret_env:      str   = "JWT_SECRET"
    jwt_algorithm:       str   = "HS256"
    jwt_expiry_hours:    int   = 24
    api_key_header:      str   = "X-API-Key"
    audit_log_enabled:   bool  = True
    audit_log_path:      str   = "data/audit.log"
    bcrypt_rounds:       int   = 12
    max_api_keys:        int   = 10


@dataclass
class ObservabilityConfig2:
    """
    Extended observability: OpenTelemetry tracing, structured logging, dashboards.

    Extends the existing ObservabilityConfig (Phase 3) with distributed
    tracing and log aggregation.

    === THEORY ===

    The three pillars of observability:
      1. Metrics: numerical measurements over time (existing, Phase 3)
      2. Traces: distributed request flow across services (new, Phase 8)
      3. Logs: structured event records (extended, Phase 8)

    OpenTelemetry (OTLP) is the open standard for all three, supported
    by Jaeger, Zipkin, Grafana Tempo, AWS X-Ray, and Datadog.

    === PRODUCTION EQUIVALENTS ===

      Google:    Cloud Trace + Cloud Logging
      Netflix:   Atlas (metrics) + Edgar (traces)
      Uber:      Jaeger (created at Uber)
      Elastic:   APM with distributed tracing
    """
    tracing_enabled:     bool  = False
    otlp_endpoint:       str   = "http://localhost:4317"
    service_name:        str   = "search-engine"
    log_format:          str   = "json"
    log_level:           str   = "INFO"
    slow_query_ms:       float = 200.0
    trace_sample_rate:   float = 1.0


@dataclass
class ResilienceConfig:
    """
    Resilience patterns: circuit breakers, retries, backoff, health probes.

    === THEORY ===

    Resilience engineering prevents cascading failures in distributed systems.

    Circuit Breaker (Michael Nygard, "Release It!"):
      CLOSED → OPEN → HALF_OPEN → CLOSED
      Trips when failure rate exceeds threshold. Fast-fails while open.
      Tries one probe request in HALF_OPEN to check recovery.

    Retry with backoff prevents thundering herd when a service recovers.
    Jitter adds randomness to avoid synchronized retries.

    === PRODUCTION EQUIVALENTS ===

      Netflix: Hystrix (circuit breaker — now Resilience4j)
      AWS:     SDK built-in retry + backoff
      Google:  gRPC deadline propagation + circuit breakers
    """
    circuit_breaker_enabled:   bool  = True
    failure_threshold:         int   = 5
    recovery_timeout_sec:      float = 30.0
    half_open_max_calls:       int   = 3
    retry_max_attempts:        int   = 3
    retry_base_delay_sec:      float = 0.5
    retry_max_delay_sec:       float = 30.0
    retry_jitter:              bool  = True
    health_probe_interval_sec: float = 15.0
    graceful_shutdown_sec:     float = 30.0


@dataclass
class CostConfig:
    """
    Cost observability: track LLM, embedding, storage, and agent costs.

    === THEORY ===

    AI systems have significant variable costs tied to usage:
      - LLM calls: priced per token (input + output)
      - Embeddings: priced per token
      - Vector storage: priced per GB/month
      - Agent execution: priced by compute time

    Cost tracking enables budgeting, alerting, and per-tenant billing.

    === PRODUCTION EQUIVALENTS ===

      OpenAI:  Usage dashboard + billing API
      AWS:     Cost Explorer + budgets
      GCP:     Cloud Billing + budget alerts
      Datadog: Cost Management
    """
    enabled:            bool  = True
    track_llm:          bool  = True
    track_embeddings:   bool  = True
    track_storage:      bool  = True
    track_agents:       bool  = True
    budget_alert_usd:   float = 10.0
    cost_log_path:      str   = "data/costs.jsonl"
    retention_days:     int   = 90


# ── Top-level config ───────────────────────────────────────────────────────────

@dataclass
class EngineConfig:
    # Phase 1-3
    database:        DatabaseConfig       = field(default_factory=DatabaseConfig)
    tokenizer:       TokenizerConfig      = field(default_factory=TokenizerConfig)
    indexer:         IndexerConfig        = field(default_factory=IndexerConfig)
    bm25:            BM25Config           = field(default_factory=BM25Config)
    crawler:         CrawlerConfig        = field(default_factory=CrawlerConfig)
    search:          SearchConfig         = field(default_factory=SearchConfig)
    snippet:         SnippetConfig        = field(default_factory=SnippetConfig)
    autocomplete:    AutocompleteConfig   = field(default_factory=AutocompleteConfig)
    spellcheck:      SpellCheckConfig     = field(default_factory=SpellCheckConfig)
    cache:           CacheConfig          = field(default_factory=CacheConfig)
    observability:   ObservabilityConfig  = field(default_factory=ObservabilityConfig)
    ranking_weights: RankingWeights       = field(default_factory=RankingWeights)
    documents_dir:   Path                 = Path("documents")
    synonyms_path:   Path                 = Path("app/query_expansion/synonyms.json")
    # Phase 4
    embedding:       EmbeddingConfig      = field(default_factory=EmbeddingConfig)
    chunking:        ChunkingConfig       = field(default_factory=ChunkingConfig)
    vector_store:    VectorStoreConfig    = field(default_factory=VectorStoreConfig)
    hybrid_search:   HybridSearchConfig   = field(default_factory=HybridSearchConfig)
    evaluation:      EvaluationConfig     = field(default_factory=EvaluationConfig)
    # Phase 5
    reranking:            RerankingConfig           = field(default_factory=RerankingConfig)
    pipeline:             PipelineConfig            = field(default_factory=PipelineConfig)
    query_understanding:  QueryUnderstandingConfig  = field(default_factory=QueryUnderstandingConfig)
    experiment:           ExperimentConfig          = field(default_factory=ExperimentConfig)
    personalization:      PersonalizationConfig     = field(default_factory=PersonalizationConfig)
    # Phase 6
    rag:                  RAGConfig                 = field(default_factory=RAGConfig)
    # Phase 7
    research:             ResearchConfig            = field(default_factory=ResearchConfig)
    # Phase 8
    postgres:             PostgresConfig            = field(default_factory=PostgresConfig)
    events:               EventConfig               = field(default_factory=EventConfig)
    redis:                RedisConfig               = field(default_factory=RedisConfig)
    # Phase 8 Batch 2
    kafka:                KafkaConfig               = field(default_factory=KafkaConfig)
    distributed_crawler:  DistributedCrawlerConfig  = field(default_factory=DistributedCrawlerConfig)
    distributed_indexing: DistributedIndexingConfig  = field(default_factory=DistributedIndexingConfig)
    qdrant:               QdrantConfig              = field(default_factory=QdrantConfig)
    gateway:              GatewayConfig             = field(default_factory=GatewayConfig)
    # Phase 8 Batch 3
    service_registry:     ServiceRegistryConfig     = field(default_factory=ServiceRegistryConfig)
    agent_execution:      AgentExecutionConfig      = field(default_factory=AgentExecutionConfig)
    distributed_workflow: DistributedWorkflowConfig = field(default_factory=DistributedWorkflowConfig)
    tenancy:              TenancyConfig             = field(default_factory=TenancyConfig)
    # Phase 8 Batch 4
    security:             SecurityConfig            = field(default_factory=SecurityConfig)
    observability2:       ObservabilityConfig2      = field(default_factory=ObservabilityConfig2)
    resilience:           ResilienceConfig          = field(default_factory=ResilienceConfig)
    cost:                 CostConfig                = field(default_factory=CostConfig)
