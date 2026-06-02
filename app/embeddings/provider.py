"""
Embedding Provider

=== THEORY ===

A dense embedding converts text into a fixed-length vector in a semantic
space where semantically similar texts are close together (measured by
cosine similarity or dot product).

Modern retrieval embeddings are trained with contrastive learning objectives
(e.g. Multiple Negatives Ranking Loss) on pairs of (query, relevant passage).
They learn to map query-like text and document-like text into the same
semantic neighbourhood.

=== WHY EMBEDDING-BASED RETRIEVAL ===

BM25 fails when:
  - The query uses different words than the document
    ("car" vs "automobile", "ML" vs "machine learning")
  - The query is a natural-language question and the doc is a declarative text
    "What framework should I use?" → "Flask is a lightweight web framework"
  - The query is short but the document is paraphrasing

Dense retrieval (bi-encoder) encodes query and document independently,
then finds nearest neighbours in vector space.  This captures semantics
that exact-match methods miss.

=== ARCHITECTURE — EmbeddingProvider Protocol ===

We define a Protocol (structural subtyping) rather than an abstract base
class so that any object with embed_texts/embed_query/dimension/model_name
automatically satisfies the interface — no inheritance needed.

This enables dependency injection of different providers without changing
any retrieval code:

  provider: EmbeddingProvider = LocalEmbeddingProvider()
  provider: EmbeddingProvider = OpenAIEmbeddingProvider()   # future
  provider: EmbeddingProvider = CohereEmbeddingProvider()   # future

=== MODEL CHOICES ===

  BAAI/bge-small-en-v1.5  384 dims, 33M params, ~133MB, MTEB rank ~40
  intfloat/e5-small-v2    384 dims, 33M params, ~133MB, MTEB rank ~38
  BAAI/bge-base-en-v1.5   768 dims, 110M params, ~430MB, better quality
  all-MiniLM-L6-v2        384 dims, fast, widely used baseline

BGE models expect an instruction prefix for queries (not documents):
  "Represent this sentence for searching relevant passages: <query>"
We handle this transparently inside LocalEmbeddingProvider.

=== COMPLEXITY ===

  Embed N texts of average length L tokens:
    Time:  O(N · L · D²)  — D = model hidden dimension, dominated by attention
    Space: O(B · L · D)   — B = batch size (streaming: B can be small)

  Inference on CPU (bge-small): ~10-50 ms per text depending on length
  Inference on GPU:             ~0.5-2 ms per text

=== AT GOOGLE / PRODUCTION SCALE ===

  - Google uses proprietary models (Gecko, LaMDA embeddings)
  - Pinecone / Weaviate / Qdrant all support OpenAI, Cohere, custom models
  - Sentence-Transformers is the standard open-source option
  - In production: model served by a dedicated embedding microservice
    (e.g. Triton Inference Server, vLLM) called via REST / gRPC
"""

import hashlib
import logging
from typing import Protocol, runtime_checkable

import numpy as np

from app.config import EmbeddingConfig

logger = logging.getLogger(__name__)


# ── Protocol ──────────────────────────────────────────────────────────────────

@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Structural interface for any embedding backend.

    Implement this protocol to add a new provider (OpenAI, Cohere, etc.)
    without touching any retrieval or vector-store code.
    """

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document-side texts. Returns list of float vectors."""
        ...

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query.
        Some models apply a different instruction prefix for queries vs documents.
        """
        ...


# ── Local provider (sentence-transformers) ────────────────────────────────────

class LocalEmbeddingProvider:
    """
    Wraps sentence-transformers SentenceTransformer.

    Model is lazy-loaded on first call so server startup is fast even
    when the model is large.

    BGE-family models use an asymmetric encoding:
      Documents: encoded as-is
      Queries:   prefixed with "Represent this sentence for searching relevant passages: "
    This asymmetry improves retrieval quality because query and document
    distributions differ.
    """

    # Instruction prefix used by BGE models for asymmetric encoding
    _BGE_QUERY_INSTRUCTION = (
        "Represent this sentence for searching relevant passages: "
    )
    _E5_QUERY_PREFIX    = "query: "
    _E5_PASSAGE_PREFIX  = "passage: "

    def __init__(self, config: EmbeddingConfig | None = None):
        self._config  = config or EmbeddingConfig()
        self._model   = None        # lazy-loaded
        self._dim: int | None = None

    # ── EmbeddingProvider interface ───────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._config.model_name

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._load_model()
        return self._dim  # type: ignore[return-value]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed document-side texts in batches."""
        if not texts:
            return []
        self._load_model()
        prepared = self._prepare_docs(texts)
        vectors = self._model.encode(  # type: ignore[union-attr]
            prepared,
            batch_size=self._config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, query: str) -> list[float]:
        """Embed a query with the appropriate instruction prefix."""
        self._load_model()
        prepared = self._prepare_query(query)
        vector = self._model.encode(  # type: ignore[union-attr]
            [prepared],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vector[0].tolist()

    # ── Internals ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for LocalEmbeddingProvider. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        logger.info("Loading embedding model: %s (device=%s)",
                    self._config.model_name, self._config.device)
        self._model = SentenceTransformer(
            self._config.model_name, device=self._config.device
        )
        # Probe dimension with a dummy encode
        probe = self._model.encode(["hello"], convert_to_numpy=True)
        self._dim = int(probe.shape[1])
        logger.info("Embedding model loaded — dimension=%d", self._dim)

    def _prepare_docs(self, texts: list[str]) -> list[str]:
        name = self._config.model_name.lower()
        if "e5" in name:
            return [f"{self._E5_PASSAGE_PREFIX}{t}" for t in texts]
        return texts   # BGE and others: no prefix on documents

    def _prepare_query(self, query: str) -> str:
        name = self._config.model_name.lower()
        if "bge" in name:
            return f"{self._BGE_QUERY_INSTRUCTION}{query}"
        if "e5" in name:
            return f"{self._E5_QUERY_PREFIX}{query}"
        return query


# ── Mock provider (for tests — no ML deps required) ──────────────────────────

class MockEmbeddingProvider:
    """
    Deterministic random embedding provider for unit tests.

    Uses SHA-256 hash of the text as a PRNG seed so the same text always
    produces the same vector.  Vectors are L2-normalised (unit sphere).
    Not semantically meaningful — only useful for testing the retrieval
    pipeline structure.
    """

    def __init__(self, dim: int = 64, model: str = "mock-v1"):
        self._dim   = dim
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_vector(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._hash_vector(query)

    def _hash_vector(self, text: str) -> list[float]:
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        rng  = np.random.RandomState(seed % (2**31))
        v    = rng.randn(self._dim).astype(np.float32)
        v   /= np.linalg.norm(v) + 1e-10
        return v.tolist()
