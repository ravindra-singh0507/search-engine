"""
Multi-Stage Retrieval Pipeline

=== ARCHITECTURE ===

  Query
    │
    ├── Stage 1: Retrieval ──── BM25 (top bm25_candidates)
    │                    └───── Semantic / FAISS (top semantic_candidates)
    │
    ├── Stage 2: Fusion ─────── RRF / CombSUM / CombMNZ / Weighted / Borda
    │                            → candidate set
    │
    ├── Stage 3: Re-ranking ─── CrossEncoder on top-N candidates
    │
    └── Stage 4: Final Rank ─── Weighted combination of signals
                                 → top_k results

=== DESIGN PRINCIPLES ===

  Pluggable retrievers: any object with a `.search(query, top_k)` method
  Pluggable rankers: any Reranker Protocol implementor
  Pluggable fusion: any fusion function from app.fusion.strategies
  Configurable per query: pass overrides to search()

=== LATENCY BUDGET ===

  BM25 retrieval:   < 5 ms
  Semantic FAISS:   10-50 ms (CPU, 10k vectors)
  Concurrent ↑↑:    max(BM25, FAISS) ≈ 10-50 ms
  Fusion:           < 1 ms
  Cross-encoder:    500-2500 ms (top-50, CPU, MiniLM-L6)
  Total (CPU):      550-2600 ms
  Total (GPU):      50-200 ms

For latency-sensitive applications: cap rerank at 20, use GPU,
or cache reranked results per popular query.

=== PRODUCTION EQUIVALENTS ===

  Vespa:    Multi-phase ranking with phased re-rankers
  Elastic:  Learning-to-rank with xgboost re-ranker plugin
  Google:   Hundreds of retrieval signals → BERT re-ranker → final MQR
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from app.config import PipelineConfig
from app.fusion.strategies import get_fusion_strategy
from app.reranking.reranker import Reranker, RerankedResult

logger = logging.getLogger(__name__)


# ── Candidate dataclass ────────────────────────────────────────────────────────

@dataclass
class PipelineCandidate:
    """A document candidate flowing through the retrieval pipeline."""
    doc_id:         int
    title:          str
    content:        str          # used by cross-encoder
    snippet:        str          # shown to user
    bm25_score:     float = 0.0
    semantic_score: float = 0.0
    fusion_score:   float = 0.0
    reranker_score: float = 0.0
    final_score:    float = 0.0
    bm25_rank:      int | None = None
    semantic_rank:  int | None = None
    fusion_rank:    int | None = None
    final_rank:     int | None = None


@dataclass
class PipelineResult:
    query:             str
    results:           list[PipelineCandidate]
    stage_latencies:   dict[str, float]     # stage_name → ms
    total_latency_ms:  float
    pipeline_config:   dict
    retrieval_count:   int    # candidates after fusion, before reranking
    reranked_count:    int    # how many were actually reranked


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RetrievalPipeline:
    """
    Multi-stage retrieval pipeline.

    Components are injected; each has a well-defined interface so they can
    be swapped without modifying the pipeline class.
    """

    def __init__(
        self,
        db,                          # Database — for doc lookup
        keyword_search,              # SearchService — BM25 retrieval
        semantic_search,             # SemanticSearchService — FAISS retrieval
        reranker:     Reranker,
        snippet_gen   = None,
        config: PipelineConfig | None = None,
    ):
        self.db              = db
        self.keyword         = keyword_search
        self.semantic        = semantic_search
        self.reranker        = reranker
        self.snippet_gen     = snippet_gen
        self.config          = config or PipelineConfig()

    def search(
        self,
        query:    str,
        top_k:    int | None = None,
        override: dict | None = None,
    ) -> PipelineResult:
        """
        Execute the full pipeline and return a PipelineResult.

        Parameters
        ----------
        query    : raw search query
        top_k    : number of final results (overrides config if given)
        override : dict of config overrides, e.g. {"fusion_strategy": "combsum"}
        """
        cfg = self._effective_config(override)
        k   = top_k or cfg["final_top_k"]
        latencies: dict[str, float] = {}
        start_total = time.perf_counter()

        # ── Stage 1: Retrieve ──────────────────────────────────────────────
        t0 = time.perf_counter()
        bm25_list, semantic_list = self._retrieve(query, cfg)
        latencies["retrieval_ms"] = (time.perf_counter() - t0) * 1000

        # ── Stage 2: Fuse ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        fused = self._fuse(bm25_list, semantic_list, cfg)
        latencies["fusion_ms"] = (time.perf_counter() - t0) * 1000

        # ── Build candidate objects ────────────────────────────────────────
        candidates = self._build_candidates(
            fused, bm25_list, semantic_list, query
        )
        retrieval_count = len(candidates)

        # ── Stage 3: Rerank ────────────────────────────────────────────────
        t0 = time.perf_counter()
        candidates, reranked_count = self._rerank(query, candidates, cfg)
        latencies["reranking_ms"] = (time.perf_counter() - t0) * 1000

        # ── Stage 4: Final ranking ─────────────────────────────────────────
        t0 = time.perf_counter()
        final = self._final_rank(candidates, k)
        latencies["final_rank_ms"] = (time.perf_counter() - t0) * 1000

        total_ms = (time.perf_counter() - start_total) * 1000
        logger.info(
            "Pipeline %r: %d retrieved → %d reranked → %d final in %.1f ms",
            query, retrieval_count, reranked_count, len(final), total_ms,
        )

        return PipelineResult(
            query=query, results=final,
            stage_latencies=latencies,
            total_latency_ms=round(total_ms, 2),
            pipeline_config=cfg,
            retrieval_count=retrieval_count,
            reranked_count=reranked_count,
        )

    # ── Stage implementations ─────────────────────────────────────────────

    def _retrieve(
        self, query: str, cfg: dict
    ) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        """Run BM25 and semantic retrieval concurrently."""
        with ThreadPoolExecutor(max_workers=2) as ex:
            bm25_fut = ex.submit(
                self.keyword.search, query, cfg["bm25_candidates"]
            )
            sem_fut = ex.submit(
                self.semantic.search, query, cfg["semantic_candidates"]
            ) if cfg["use_semantic"] else None

            bm25_res = bm25_fut.result()
            bm25_list = [(r.doc_id, r.score) for r in bm25_res.results]

            if sem_fut:
                sem_res  = sem_fut.result()
                sem_list = [(r.doc_id, r.semantic_score) for r in sem_res.results]
            else:
                sem_list = []

        return bm25_list, sem_list

    def _fuse(
        self,
        bm25_list: list[tuple[int, float]],
        sem_list:  list[tuple[int, float]],
        cfg: dict,
    ) -> list[tuple[int, float]]:
        """Apply the configured fusion strategy."""
        lists = [l for l in [bm25_list, sem_list] if l]
        if not lists:
            return []
        fn    = get_fusion_strategy(cfg["fusion_strategy"])
        # For weighted fusion pass weights if strategy supports it
        if cfg["fusion_strategy"] == "weighted" and len(lists) == 2:
            return fn(lists, weights=[cfg.get("bm25_w", 0.5), cfg.get("sem_w", 0.5)])
        return fn(lists)

    def _build_candidates(
        self,
        fused:     list[tuple[int, float]],
        bm25_list: list[tuple[int, float]],
        sem_list:  list[tuple[int, float]],
        query:     str,
    ) -> list[PipelineCandidate]:
        """Enrich fused results with doc content and per-system scores."""
        bm25_map  = {did: (s, r+1) for r, (did, s) in enumerate(bm25_list)}
        sem_map   = {did: (s, r+1) for r, (did, s) in enumerate(sem_list)}
        candidates: list[PipelineCandidate] = []
        q_terms   = query.lower().split()

        for rank, (doc_id, fusion_score) in enumerate(fused, 1):
            doc = self.db.get_document(doc_id)
            if doc is None:
                continue

            if self.snippet_gen:
                snippet = self.snippet_gen.generate(doc.content, q_terms)
            else:
                snippet = doc.content[:300]

            bm25_s, bm25_r = bm25_map.get(doc_id, (0.0, None))
            sem_s,  sem_r  = sem_map.get(doc_id,  (0.0, None))

            candidates.append(PipelineCandidate(
                doc_id        = doc_id,
                title         = doc.title,
                content       = doc.content,
                snippet       = snippet,
                bm25_score    = round(bm25_s, 6),
                semantic_score= round(sem_s,  6),
                fusion_score  = round(fusion_score, 6),
                bm25_rank     = bm25_r,
                semantic_rank = sem_r,
                fusion_rank   = rank,
            ))

        return candidates

    def _rerank(
        self,
        query:      str,
        candidates: list[PipelineCandidate],
        cfg:        dict,
    ) -> tuple[list[PipelineCandidate], int]:
        """Send top-N candidates to the cross-encoder."""
        use_reranker = cfg.get("use_reranker", True) and self.config.use_reranker
        if not use_reranker or not candidates:
            return candidates, 0

        top_n  = min(cfg["rerank_top_k"], len(candidates))
        to_rerank = candidates[:top_n]

        rerank_input = [
            (c.doc_id, c.title, c.content[:1000], c.fusion_score)
            for c in to_rerank
        ]
        reranked: list[RerankedResult] = self.reranker.rerank(
            query, rerank_input, top_k=top_n
        )

        # Update reranker scores on the matching candidates
        rerank_map = {r.doc_id: r.reranker_score for r in reranked}
        for c in candidates:
            c.reranker_score = rerank_map.get(c.doc_id, 0.0)

        return candidates, len(reranked)

    def _final_rank(
        self, candidates: list[PipelineCandidate], top_k: int
    ) -> list[PipelineCandidate]:
        """
        Compute final score and sort.

        final_score = 0.4 × norm(fusion) + 0.6 × norm(reranker)
        If no reranker scores (all 0.0), fall back to fusion score only.
        """
        if not candidates:
            return []

        max_fusion   = max(c.fusion_score    for c in candidates) or 1.0
        max_reranker = max(c.reranker_score  for c in candidates) or 0.0

        for c in candidates:
            norm_f = c.fusion_score   / max_fusion
            norm_r = c.reranker_score / max_reranker if max_reranker else 0.0
            if max_reranker > 0:
                c.final_score = round(0.4 * norm_f + 0.6 * norm_r, 6)
            else:
                c.final_score = round(norm_f, 6)

        candidates.sort(key=lambda c: c.final_score, reverse=True)
        for i, c in enumerate(candidates[:top_k], 1):
            c.final_rank = i

        return candidates[:top_k]

    def _effective_config(self, override: dict | None) -> dict:
        """Merge pipeline config with per-request overrides."""
        base = {
            "bm25_candidates":  self.config.bm25_candidates,
            "semantic_candidates": self.config.semantic_candidates,
            "fusion_strategy":  self.config.fusion_strategy,
            "rerank_top_k":     self.config.rerank_top_k,
            "final_top_k":      self.config.final_top_k,
            "use_reranker":     self.config.use_reranker,
            "use_semantic":     self.config.use_semantic,
        }
        if override:
            base.update(override)
        return base
