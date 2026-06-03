"""
Cross-Encoder Re-ranking

=== THEORY ===

Bi-encoders (Phase 4: FAISS + sentence-transformers) encode query and
document INDEPENDENTLY, then compute similarity in vector space.  This is
fast (O(1) lookup after pre-computing doc embeddings) but coarse — the model
never sees query and document together, so it misses fine-grained interactions.

Cross-encoders process (query, document) PAIRS jointly through the full
transformer attention mechanism.  This is expensive — O(N · inference) where
N = candidate count — but dramatically more accurate because the model can
attend to relationships BETWEEN the query and document tokens.

=== WHY USE BOTH ===

  Bi-encoder (Stage 1):  Fast candidate retrieval — e.g. top-100 from FAISS
  Cross-encoder (Stage 2): Accurate re-ranking of those top-100 candidates

This two-stage design is the standard production pattern:
  - Elasticsearch / OpenSearch: BM25 → Learning-to-rank (cross-encoder features)
  - Google: Multiple retrieval signals → BERT-based re-ranker
  - Bing: DSSM (bi-encoder) → multi-stage BERT re-rankers
  - Vespa: ANN → phased ranking with neural re-rankers

=== SCORE NORMALISATION ===

ms-marco models output raw logits (typically −10 to +10).  We apply sigmoid:

  score_norm = 1 / (1 + exp(−logit))

Giving a [0, 1] score where 1.0 = perfect relevance.

=== COMPLEXITY ===

  Batch score N candidates:  O(N · L · D²)  — L = avg passage length, D = dim
  Typical wall-clock on CPU: 10-50 ms per candidate for MiniLM-L-6 (6 layers)
  → top-50 rerank: 500ms–2.5s on CPU; 30-150ms on GPU

  Production strategy: Use GPU, cap rerank at 20-50, parallelize with async

=== RERANKER INTERFACE ===

Implement this Protocol to add new reranking backends (GPT reranker, Cohere
Rerank API, custom models) without changing the pipeline.
"""

import math
import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.config import RerankingConfig

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RerankedResult:
    doc_id:         int
    title:          str
    snippet:        str
    reranker_score: float       # normalised [0, 1] or raw if sigmoid=False
    reranker_rank:  int
    original_score: float       # score before reranking (fusion / hybrid)
    original_rank:  int


# ── Protocol ──────────────────────────────────────────────────────────────────

@runtime_checkable
class Reranker(Protocol):
    """
    Structural interface for any re-ranking backend.
    Implement to add GPT reranker, Cohere Rerank API, etc.
    """

    @property
    def model_name(self) -> str: ...

    def score_batch(self, query: str, texts: list[str]) -> list[float]:
        """Score (query, text) pairs. Returns float list, same order as texts."""
        ...

    def rerank(
        self,
        query:    str,
        candidates: list[tuple[int, str, str, float]],   # (doc_id, title, text, orig_score)
        top_k:    int = 10,
    ) -> list[RerankedResult]:
        """Re-rank candidates, return top_k sorted by reranker score desc."""
        ...


# ── Cross-encoder reranker ────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Re-ranker backed by a sentence-transformers CrossEncoder model.

    Lazy-loaded: the model is downloaded and cached on the FIRST call to
    score_batch() or rerank().  Server startup is not delayed.
    """

    def __init__(self, config: RerankingConfig | None = None):
        self._config = config or RerankingConfig()
        self._model  = None   # lazy

    # ── Reranker interface ────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._config.model_name

    def score_batch(self, query: str, texts: list[str]) -> list[float]:
        """
        Score a batch of (query, text) pairs.
        Returns sigmoid-normalised scores in [0, 1].
        """
        if not texts:
            return []
        self._load_model()
        pairs  = [[query, t] for t in texts]
        batch  = self._config.batch_size
        scores: list[float] = []

        for start in range(0, len(pairs), batch):
            chunk   = pairs[start: start + batch]
            raw     = self._model.predict(chunk, show_progress_bar=False)
            scores += [self._sigmoid(float(r)) for r in raw]

        return scores

    def rerank(
        self,
        query:      str,
        candidates: list[tuple[int, str, str, float]],
        top_k:      int = 10,
    ) -> list[RerankedResult]:
        """
        Re-rank candidates.

        Parameters
        ----------
        candidates : list of (doc_id, title, passage_text, original_score)
        top_k      : how many to return

        Returns
        -------
        list[RerankedResult] sorted by reranker_score descending
        """
        if not candidates:
            return []

        texts  = [f"{title} {text}" for _, title, text, _ in candidates]
        scores = self.score_batch(query, texts)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for new_rank, ((doc_id, title, text, orig_score), score) in enumerate(ranked[:top_k], 1):
            orig_rank = next(
                (i + 1 for i, (did, *_) in enumerate(candidates) if did == doc_id), 0
            )
            snippet = text[:300] + " …" if len(text) > 300 else text
            results.append(RerankedResult(
                doc_id         = doc_id,
                title          = title,
                snippet        = snippet,
                reranker_score = round(score, 6),
                reranker_rank  = new_rank,
                original_score = round(orig_score, 6),
                original_rank  = orig_rank,
            ))

        logger.debug(
            "CrossEncoder reranked %d → top %d for %r",
            len(candidates), len(results), query,
        )
        return results

    def explain(self, query: str, title: str, text: str) -> dict:
        """
        Return a detailed breakdown for a single (query, document) pair.
        """
        passage = f"{title} {text}"
        scores  = self.score_batch(query, [passage])
        norm    = scores[0] if scores else 0.0

        self._load_model()
        raw_logit = self._model.predict([[query, passage]], show_progress_bar=False)
        logit = float(raw_logit[0])

        return {
            "query":           query,
            "title":           title,
            "model":           self.model_name,
            "raw_logit":       round(logit, 4),
            "normalized_score": round(norm, 6),
            "interpretation":  self._interpret(norm),
        }

    # ── Internals ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install with: pip install sentence-transformers"
            ) from exc
        logger.info("Loading cross-encoder: %s (device=%s)",
                    self._config.model_name, self._config.device)
        self._model = CrossEncoder(
            self._config.model_name, device=self._config.device
        )
        logger.info("Cross-encoder loaded")

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid normalisation: maps logits to [0, 1]."""
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _interpret(score: float) -> str:
        if score >= 0.8:
            return "highly relevant"
        if score >= 0.6:
            return "relevant"
        if score >= 0.4:
            return "somewhat relevant"
        if score >= 0.2:
            return "marginally relevant"
        return "not relevant"


# ── Mock reranker (for tests — no ML deps) ────────────────────────────────────

class MockReranker:
    """
    Deterministic reranker for unit tests.
    Score = Jaccard similarity between query terms and document terms.
    No ML dependencies.
    """

    def __init__(self, model: str = "mock-reranker-v1"):
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def score_batch(self, query: str, texts: list[str]) -> list[float]:
        q_terms = set(query.lower().split())
        scores  = []
        for text in texts:
            t_terms = set(text.lower().split())
            if not q_terms or not t_terms:
                scores.append(0.0)
                continue
            inter = len(q_terms & t_terms)
            union = len(q_terms | t_terms)
            scores.append(inter / union if union else 0.0)
        return scores

    def rerank(
        self,
        query:      str,
        candidates: list[tuple[int, str, str, float]],
        top_k:      int = 10,
    ) -> list[RerankedResult]:
        if not candidates:
            return []
        texts  = [f"{title} {text}" for _, title, text, _ in candidates]
        scores = self.score_batch(query, texts)
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1], reverse=True,
        )
        results = []
        for new_rank, ((doc_id, title, text, orig_score), score) in enumerate(ranked[:top_k], 1):
            orig_rank = next(
                (i + 1 for i, (did, *_) in enumerate(candidates) if did == doc_id), 0
            )
            results.append(RerankedResult(
                doc_id=doc_id, title=title,
                snippet=text[:300],
                reranker_score=round(score, 6),
                reranker_rank=new_rank,
                original_score=round(orig_score, 6),
                original_rank=orig_rank,
            ))
        return results

    def explain(self, query: str, title: str, text: str) -> dict:
        score = self.score_batch(query, [f"{title} {text}"])[0]
        return {
            "query": query, "title": title, "model": self._model,
            "raw_logit": score, "normalized_score": round(score, 6),
            "interpretation": CrossEncoderReranker._interpret(score),
        }
