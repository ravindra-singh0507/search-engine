"""
Search Engine Configuration — Phase 5

Phase 5 adds: RerankingConfig, PipelineConfig, QueryUnderstandingConfig,
ExperimentConfig, PersonalizationConfig.
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
