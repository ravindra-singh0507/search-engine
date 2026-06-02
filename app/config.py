"""
Search Engine Configuration — Phase 4

All configuration dataclasses in one place.
Phase 4 adds: EmbeddingConfig, ChunkingConfig, VectorStoreConfig,
HybridSearchConfig, EvaluationConfig.
"""

from dataclasses import dataclass, field
from pathlib import Path


# ── Phase 1-3 configs (unchanged) ─────────────────────────────────────────────

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
    """
    Controls the local embedding model.

    model_name:  Any HuggingFace model that sentence-transformers supports.
                 BAAI/bge-small-en-v1.5  → 384 dims, fast, excellent retrieval
                 intfloat/e5-small-v2    → 384 dims, good cross-lingual support

    batch_size:  How many chunks to embed in one forward pass.  Increase for
                 GPU; decrease if you run OOM on CPU.

    auto_embed:  When True, every newly indexed document is automatically
                 chunked + embedded in a background thread.
    """
    model_name: str  = "BAAI/bge-small-en-v1.5"
    batch_size: int  = 32
    device: str      = "cpu"
    auto_embed: bool = False     # enable once embeddings are indexed once
    cache_embeddings: bool = True


@dataclass
class ChunkingConfig:
    """
    Controls how documents are split into passages before embedding.

    strategy:       "fixed"          — non-overlapping equal-size chunks
                    "sliding_window" — overlapping chunks (better recall)

    chunk_size:     Chunk size in *words*.  The BGE model's token limit is
                    512; a 256-word chunk maps to ~320 tokens on average
                    (well within the limit).

    chunk_overlap:  Words shared between adjacent chunks (sliding window only).
                    Typically 10-20% of chunk_size.
    """
    strategy: str     = "sliding_window"
    chunk_size: int   = 256    # words
    chunk_overlap: int = 32    # words (sliding window only)
    min_chunk_words: int = 10  # discard tiny chunks


@dataclass
class VectorStoreConfig:
    """
    Controls the FAISS vector index.

    index_path:  Directory on disk where the FAISS index and ID-map are saved.
    dimension:   Must match the embedding model's output dimension.
    similarity:  "cosine" (default) uses L2-normalised vectors + inner product.
    """
    index_path: Path = Path("data/faiss_index")
    dimension: int   = 384          # matches BAAI/bge-small-en-v1.5
    similarity: str  = "cosine"     # "cosine" | "dot_product"


@dataclass
class HybridSearchConfig:
    """
    Controls how BM25 and semantic scores are fused.

    fusion_strategy:  "rrf"    — Reciprocal Rank Fusion (Cormack 2009)
                      "linear" — weighted linear combination of normalised scores

    rrf_k:            RRF constant k = 60 is the empirical optimum from the
                      original paper.  Larger k = less aggressive rank boosting.

    bm25_weight / semantic_weight:  Used by "linear" strategy only.
    """
    fusion_strategy: str  = "rrf"
    rrf_k: int            = 60
    bm25_weight: float    = 0.5
    semantic_weight: float = 0.5


@dataclass
class EvaluationConfig:
    """Controls the retrieval evaluation framework."""
    eval_dataset_path: Path    = Path("data/eval_dataset.json")
    k_values: list[int]        = field(default_factory=lambda: [1, 3, 5, 10])


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
