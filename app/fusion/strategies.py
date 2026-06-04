"""
Advanced Fusion Strategies

=== THEORY ===

Fusion combines multiple ranked lists into a single ranking.  Each strategy
makes different assumptions about score distributions and system quality.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CombSUM (Fox & Shaw 1994)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  CombSUM(d) = Σ_i  norm_i(score_i(d))

  Normalises each list to [0,1] with min-max, then sums.
  Documents absent from a list get score 0 for that list.
  Rewards documents that score well in multiple systems.

  Pro:  Simple, intuitive
  Con:  Scale-sensitive; outlier scores can dominate after normalisation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CombMNZ (Fox & Shaw 1994)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  CombMNZ(d) = CombSUM(d) × |{i : d ∈ result_i}|

  Multiplies CombSUM by the number of systems that retrieved the document.
  Strong bonus for documents that appear in ALL systems.

  Pro:  Rewards consensus; usually outperforms CombSUM on diverse systems
  Con:  Still score-sensitive; consensus penalty for domain-specific docs

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Weighted Fusion
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  WF(d) = Σ_i  w_i × norm_i(score_i(d))   where Σw_i = 1

  Explicit weights allow tuning: e.g. weight BM25 higher for keyword queries,
  semantic higher for natural-language questions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Borda Count
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Borda(d) = Σ_i  (|list_i| − rank_i(d))

  Purely rank-based; ignores score magnitudes entirely.
  Very robust to scale differences between heterogeneous systems.

  Pro:  Scale-invariant; works when systems have incomparable scores
  Con:  Loses information about score gaps (rank 1 vs rank 2 = same Δpoints)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RRF (Cormack 2009) — already in hybrid_search, re-exported here
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  RRF(d) = Σ_i  1 / (k + rank_i(d))

=== COMPLEXITY ===

  All strategies: O(N_total) = O(R₁ + R₂ + … + Rₙ) for N lists

=== PRODUCTION EQUIVALENTS ===

  Elasticsearch:  Score combination with `function_score` + `bool` query
  Vespa:          `rank_profile` with multiple rank features + arithmetic
  Weaviate:       `alpha` parameter in hybrid search (RRF or linear combo)
  Qdrant:         Fusion with `rrf` or `dbsf` (distribution-based score fusion)
"""

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

RankedList = list[tuple[int, float]]   # [(doc_id, score), ...] sorted desc


# ── Protocol ──────────────────────────────────────────────────────────────────

@runtime_checkable
class FusionStrategy(Protocol):
    """Structural interface for any fusion strategy."""
    name: str
    def fuse(self, ranked_lists: list[RankedList]) -> RankedList: ...


# ── Normalisation helper ──────────────────────────────────────────────────────

def _minmax_normalise(lst: RankedList) -> dict[int, float]:
    """Map scores to [0, 1] using min-max normalisation."""
    if not lst:
        return {}
    scores = [s for _, s in lst]
    lo, hi = min(scores), max(scores)
    span   = hi - lo or 1.0
    return {doc_id: (s - lo) / span for doc_id, s in lst}


# ── CombSUM ───────────────────────────────────────────────────────────────────

def combsum(ranked_lists: list[RankedList]) -> RankedList:
    """
    CombSUM: sum of min-max normalised scores across all systems.
    Documents absent from a list contribute 0 for that list.
    """
    combined: dict[int, float] = {}
    for lst in ranked_lists:
        normed = _minmax_normalise(lst)
        for doc_id, score in normed.items():
            combined[doc_id] = combined.get(doc_id, 0.0) + score
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)


# ── CombMNZ ───────────────────────────────────────────────────────────────────

def combmnz(ranked_lists: list[RankedList]) -> RankedList:
    """
    CombMNZ: CombSUM × number of systems that retrieved the document.
    Rewards consensus — documents that appear in many lists are boosted.
    """
    combined: dict[int, float] = {}
    hit_count: dict[int, int]   = {}

    for lst in ranked_lists:
        normed = _minmax_normalise(lst)
        for doc_id, score in normed.items():
            combined[doc_id]  = combined.get(doc_id, 0.0) + score
            hit_count[doc_id] = hit_count.get(doc_id, 0)  + 1

    mnz = {
        doc_id: score * hit_count[doc_id]
        for doc_id, score in combined.items()
    }
    return sorted(mnz.items(), key=lambda x: x[1], reverse=True)


# ── Weighted Fusion ───────────────────────────────────────────────────────────

def weighted_fusion(
    ranked_lists: list[RankedList],
    weights: list[float] | None = None,
) -> RankedList:
    """
    Weighted sum of min-max normalised scores.

    Parameters
    ----------
    weights : per-list weights; uniform if None.  Normalised to sum to 1.
    """
    n = len(ranked_lists)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        total    = sum(weights) or 1.0
        weights  = [w / total for w in weights]

    combined: dict[int, float] = {}
    for lst, w in zip(ranked_lists, weights):
        normed = _minmax_normalise(lst)
        for doc_id, score in normed.items():
            combined[doc_id] = combined.get(doc_id, 0.0) + w * score

    return sorted(combined.items(), key=lambda x: x[1], reverse=True)


# ── Borda Count ───────────────────────────────────────────────────────────────

def borda_count(ranked_lists: list[RankedList]) -> RankedList:
    """
    Borda Count: each position in a ranked list awards (list_size - rank) points.
    rank is 1-indexed.  Documents not in a list get 0 points for it.
    Pure rank-based — completely scale-invariant.
    """
    points: dict[int, float] = {}
    for lst in ranked_lists:
        n = len(lst)
        for rank, (doc_id, _) in enumerate(lst, start=1):
            points[doc_id] = points.get(doc_id, 0.0) + (n - rank)
    return sorted(points.items(), key=lambda x: x[1], reverse=True)


# ── RRF (re-exported from hybrid_search for unified interface) ─────────────────

def rrf(ranked_lists: list[RankedList], k: int = 60) -> RankedList:
    """
    Reciprocal Rank Fusion: RRF(d) = Σ_i 1 / (k + rank_i(d))
    Imported from hybrid_search.hybrid_service and re-exported here.
    """
    from app.hybrid_search.hybrid_service import reciprocal_rank_fusion
    return reciprocal_rank_fusion(ranked_lists, k=k)


# ── Strategy registry ─────────────────────────────────────────────────────────

_STRATEGIES: dict[str, callable] = {
    "rrf":      rrf,
    "combsum":  combsum,
    "combmnz":  combmnz,
    "weighted": weighted_fusion,
    "borda":    borda_count,
}


def get_fusion_strategy(name: str) -> callable:
    """
    Look up a fusion function by name.
    Raises ValueError for unknown strategy names.
    """
    fn = _STRATEGIES.get(name.lower())
    if fn is None:
        raise ValueError(
            f"Unknown fusion strategy {name!r}. "
            f"Available: {sorted(_STRATEGIES)}"
        )
    return fn


def available_strategies() -> list[str]:
    return sorted(_STRATEGIES.keys())


def compare_strategies(
    ranked_lists: list[RankedList],
    top_k: int = 10,
) -> dict[str, list[tuple[int, float]]]:
    """
    Run all fusion strategies on the same input and return top_k results for each.
    Useful for benchmarking which strategy works best for a given query.
    """
    return {
        name: fn(ranked_lists)[:top_k]
        for name, fn in _STRATEGIES.items()
    }
