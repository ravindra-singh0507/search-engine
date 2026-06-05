"""
Search Engine Configuration — Phase 7

Phase 6 adds: LLMConfig, ContextConfig, MemoryConfig, CitationConfig,
GroundingConfig, RAGConfig — for the RAG / Knowledge Assistant layer.

Phase 7 adds: AgentConfig, OrchestratorConfig, WorkflowConfig,
ResearchConfig — for the Agentic Retrieval platform.
"""

from dataclasses import dataclass, field
from pathlib import Path


# ── Phase 1-3 configs ─────────────────────────────────────────────────────────

@dataclass
class DatabaseConfig:
    db_path: Path = Path("data/search_engine.db")


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
