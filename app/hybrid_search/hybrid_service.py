"""
Hybrid Search Engine — BM25 + Semantic + Reciprocal Rank Fusion

=== THEORY ===

Neither BM25 nor semantic search is universally better.  Hybrid retrieval
combines both to get the best of both worlds.

RECIPROCAL RANK FUSION (Cormack, Clarke & Buettcher, 2009):

  RRF(d) = Σ_i  1 / (k + rank_i(d))

  where:
    d          = candidate document
    i          = each retrieval system (BM25, semantic)
    rank_i(d)  = 1-indexed rank of d in system i (∞ if not retrieved)
    k          = 60 (empirically optimal constant from the paper)

  WHY k=60?
    Documents that are top-1 in all systems should get the highest scores.
    k=60 means rank 1 contributes 1/61 ≈ 0.016, rank 60 contributes
    1/120 ≈ 0.008 — a 2× spread for a 60-rank difference.
    k→0 makes rank 1 ≫ rank 2; k→∞ makes all ranks equal.

  WHY RRF OVER LINEAR COMBINATION?
    - BM25 scores (4.2, 7.8, ...) and cosine similarities (0.82, 0.91, ...)
      are on completely different scales.  Normalisation is error-prone.
    - RRF only cares about RANK ORDER, not score magnitude.
    - RRF is robust to outliers in individual systems.
    - Shown in the original paper to outperform normalised linear combination
      on diverse retrieval benchmarks.

=== PIPELINE ===

  Query
    ↓  BM25 search   → [(doc_id, bm25_score), ...]  (ranked list 1)
    ↓  Semantic search → [(doc_id, sem_score), ...]  (ranked list 2)
    ↓  RRF fusion    → [(doc_id, rrf_score), ...]   (merged ranking)
    ↓  Enrich with titles, snippets
    → HybridSearchResponse

=== ALTERNATIVE FUSION STRATEGIES ===

  Linear (weighted sum of normalised scores):
    score(d) = α · norm_bm25(d) + β · norm_sem(d)
    Pro: Direct score interpretation
    Con: Scale-sensitive; requires careful normalisation

  CombSUM / CombMNZ (TREC Combination Methods):
    CombSUM(d) = Σ_i score_i(d)
    CombMNZ(d) = CombSUM(d) * count(systems where d appears)
    Pro: Rewards documents that appear in multiple systems
    Con: Also scale-sensitive

=== COMPLEXITY ===

  BM25 retrieval:    O(Q · log T + M · log M)   (standard BM25)
  Semantic search:   O(D + V · D)               (embed + FAISS scan)
  RRF fusion:        O(R1 + R2)                 (R1, R2 = result set sizes)
  Total:             dominated by semantic search embedding

=== PRODUCTION EQUIVALENTS ===

  Elasticsearch:   Hybrid search with linear_combination scorer (8.x)
  OpenSearch:      Hybrid query (score normalization built-in)
  Weaviate:        Alpha parameter hybrid BM25+vector (native)
  Cohere reranker: Use a cross-encoder to re-rank the fused list (best quality)
"""

import logging
import time
from dataclasses import dataclass

from app.database.db import Database
from app.search.search_service import SearchService
from app.semantic_search.semantic_service import SemanticSearchService
from app.snippets.snippet_generator import SnippetGenerator
from app.config import HybridSearchConfig

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class HybridResult:
    rank:           int
    doc_id:         int
    title:          str
    snippet:        str
    fusion_score:   float
    bm25_score:     float          # 0 if not in BM25 results
    semantic_score: float          # 0 if not in semantic results
    bm25_rank:      int | None     # None if absent from that list
    semantic_rank:  int | None


@dataclass
class HybridSearchResponse:
    query:            str
    search_time_ms:   float
    fusion_strategy:  str
    bm25_count:       int
    semantic_count:   int
    total_results:    int
    results:          list[HybridResult]


# ── RRF implementation ────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: each item is [(doc_id, score), ...] sorted best-first.
        k:            RRF constant (60 is the empirical optimum).

    Returns:
        Merged [(doc_id, rrf_score), ...] sorted by rrf_score descending.
    """
    rrf_scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def linear_combination(
    bm25_list:     list[tuple[int, float]],
    semantic_list: list[tuple[int, float]],
    bm25_weight:   float = 0.5,
    sem_weight:    float = 0.5,
) -> list[tuple[int, float]]:
    """
    Fuse via min-max normalised linear combination.

    score(d) = bm25_weight * norm_bm25(d) + sem_weight * norm_sem(d)
    """
    def _normalise(lst: list[tuple[int, float]]) -> dict[int, float]:
        if not lst:
            return {}
        scores  = [s for _, s in lst]
        lo, hi  = min(scores), max(scores)
        span    = hi - lo or 1.0
        return {doc_id: (s - lo) / span for doc_id, s in lst}

    n_bm25 = _normalise(bm25_list)
    n_sem  = _normalise(semantic_list)
    all_ids = set(n_bm25) | set(n_sem)
    combined = {
        doc_id: bm25_weight * n_bm25.get(doc_id, 0.0)
              + sem_weight  * n_sem.get(doc_id, 0.0)
        for doc_id in all_ids
    }
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)


# ── Hybrid search service ─────────────────────────────────────────────────────

class HybridSearchService:
    """
    Combines BM25 and semantic retrieval with configurable fusion.
    """

    def __init__(
        self,
        db:              Database,
        keyword_search:  SearchService,
        semantic_search: SemanticSearchService,
        snippet_gen:     SnippetGenerator | None = None,
        config:          HybridSearchConfig | None = None,
    ):
        self.db           = db
        self.keyword      = keyword_search
        self.semantic     = semantic_search
        self.snippet_gen  = snippet_gen
        self.config       = config or HybridSearchConfig()

    def search(self, query: str, top_k: int = 10) -> HybridSearchResponse:
        """
        Run BM25 + semantic retrieval, fuse with RRF, return top_k results.
        """
        start = time.perf_counter()

        # ── BM25 retrieval ─────────────────────────────────────────────
        kw_result  = self.keyword.search(query, top_k=top_k * 2)
        bm25_list: list[tuple[int, float]] = [
            (r.doc_id, r.score) for r in kw_result.results
        ]

        # ── Semantic retrieval ─────────────────────────────────────────
        sem_result = self.semantic.search(query, top_k=top_k * 2)
        sem_list:  list[tuple[int, float]] = [
            (r.doc_id, r.semantic_score) for r in sem_result.results
        ]

        # ── Fusion ─────────────────────────────────────────────────────
        if self.config.fusion_strategy == "linear":
            fused = linear_combination(
                bm25_list, sem_list,
                self.config.bm25_weight, self.config.semantic_weight,
            )
        else:   # default: RRF
            fused = reciprocal_rank_fusion(
                [bm25_list, sem_list], k=self.config.rrf_k
            )

        fused = fused[:top_k]

        # ── Build score lookup maps ────────────────────────────────────
        bm25_score_map:  dict[int, float] = dict(bm25_list)
        bm25_rank_map:   dict[int, int]   = {
            doc_id: rank for rank, (doc_id, _) in enumerate(bm25_list, 1)
        }
        sem_score_map:   dict[int, float] = dict(sem_list)
        sem_rank_map:    dict[int, int]   = {
            doc_id: rank for rank, (doc_id, _) in enumerate(sem_list, 1)
        }

        # ── Assemble results ───────────────────────────────────────────
        results: list[HybridResult] = []
        for rank, (doc_id, fusion_score) in enumerate(fused, 1):
            doc = self.db.get_document(doc_id)
            if doc is None:
                continue

            query_terms = query.lower().split()
            if self.snippet_gen:
                snippet = self.snippet_gen.generate(doc.content, query_terms)
            else:
                snippet = doc.content[:300]

            results.append(HybridResult(
                rank          = rank,
                doc_id        = doc_id,
                title         = doc.title,
                snippet       = snippet,
                fusion_score  = round(fusion_score, 6),
                bm25_score    = round(bm25_score_map.get(doc_id, 0.0), 6),
                semantic_score= round(sem_score_map.get(doc_id, 0.0), 6),
                bm25_rank     = bm25_rank_map.get(doc_id),
                semantic_rank = sem_rank_map.get(doc_id),
            ))

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "Hybrid search %r: bm25=%d, sem=%d, fused=%d in %.1f ms",
            query, len(bm25_list), len(sem_list), len(results), elapsed,
        )
        return HybridSearchResponse(
            query           = query,
            search_time_ms  = round(elapsed, 2),
            fusion_strategy = self.config.fusion_strategy,
            bm25_count      = len(bm25_list),
            semantic_count  = len(sem_list),
            total_results   = len(results),
            results         = results,
        )

    def explain(self, query: str, doc_id: int) -> dict:
        """
        Return the full score breakdown for a specific document.
        """
        doc = self.db.get_document(doc_id)
        if doc is None:
            return {"error": f"Document {doc_id} not found"}

        # BM25 score
        kw  = self.keyword.search(query, top_k=100)
        bm25_score = next(
            (r.score for r in kw.results if r.doc_id == doc_id), 0.0
        )
        bm25_rank  = next(
            (i + 1 for i, r in enumerate(kw.results) if r.doc_id == doc_id),
            None,
        )

        # Semantic score
        sem = self.semantic.search(query, top_k=100)
        sem_score = next(
            (r.semantic_score for r in sem.results if r.doc_id == doc_id), 0.0
        )
        sem_rank  = next(
            (i + 1 for i, r in enumerate(sem.results) if r.doc_id == doc_id),
            None,
        )

        # RRF contribution
        rrf_score = 0.0
        if bm25_rank:
            rrf_score += 1.0 / (self.config.rrf_k + bm25_rank)
        if sem_rank:
            rrf_score += 1.0 / (self.config.rrf_k + sem_rank)

        # Human-readable reason
        reasons = []
        if bm25_score > 0:
            reasons.append(f"keyword relevance (BM25={bm25_score:.3f}, rank={bm25_rank})")
        if sem_score > 0:
            reasons.append(f"semantic similarity (cos={sem_score:.3f}, rank={sem_rank})")
        reason = "High " + " and ".join(reasons) if reasons else "Not matched"

        return {
            "doc_id":         doc_id,
            "title":          doc.title,
            "query":          query,
            "bm25_score":     round(bm25_score, 6),
            "bm25_rank":      bm25_rank,
            "semantic_score": round(sem_score, 6),
            "semantic_rank":  sem_rank,
            "fusion_score":   round(rrf_score, 6),
            "fusion_strategy":self.config.fusion_strategy,
            "reason":         reason,
        }
