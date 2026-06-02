"""
Relevance Tuning Framework

Combines multiple ranking signals into a single score using
configurable weights.  This is the bridge between pure BM25
retrieval and a multi-signal production ranker.

=== SCORING FORMULA ===

  final_score(d, Q) =
       weights.bm25         * bm25_score(d, Q)
     + weights.title_boost  * title_match_factor(d, Q)
     + weights.recency_boost * recency_factor(d)
     + weights.click_boost  * click_factor(d)

=== SIGNALS ===

  bm25_score          — BM25 text relevance score (primary)
  title_match_factor  — fraction of query terms found in title postings
                         multiplied by the title_boost weight
  recency_factor      — document age in days, exponentially decayed:
                         e^(-decay * age_days)
  click_factor        — normalised click count from analytics data
                         (0.0 if analytics is disabled)

=== TUNING PRINCIPLES ===

  - Start with bm25=1.0 and all boosts at 0.
  - Enable title_boost first — usually the highest ROI.
  - Add recency only when freshness matters for your corpus.
  - Add click_boost only after you have enough click data (cold-start
    problem: new docs have 0 clicks and would never surface otherwise).

=== AT GOOGLE SCALE ===

Google's LambdaMART / GBDT learning-to-rank model has thousands of
features.  The weights here are the human-interpretable equivalent of
one layer of that stack: easy to reason about, easy to A/B test.
"""

import math
import logging
from dataclasses import dataclass

from app.bm25.bm25 import BM25Ranker, BM25ScoredDocument
from app.database.db import Database
from app.config import RankingWeights

logger = logging.getLogger(__name__)


@dataclass
class RankedDocument:
    doc_id: int
    score: float
    bm25_score: float
    title_score: float
    recency_score: float
    click_score: float
    title: str
    snippet: str
    term_scores: dict[str, float]


class RelevanceTuner:
    """
    Wraps BM25Ranker and adds configurable signal boosting.
    """

    def __init__(
        self,
        db: Database,
        bm25_ranker: BM25Ranker,
        weights: RankingWeights | None = None,
    ):
        self.db = db
        self.bm25 = bm25_ranker
        self.weights = weights or RankingWeights()

    def rank(
        self,
        query_terms: list[str],
        candidate_doc_ids: set[int],
        top_k: int = 10,
        click_counts: dict[int, int] | None = None,
    ) -> list[RankedDocument]:
        """
        Score candidate documents using all configured signals.

        Parameters
        ----------
        query_terms :     tokenised query terms
        candidate_doc_ids: set of doc IDs from Boolean retrieval
        top_k :           number of results to return
        click_counts :    optional {doc_id: total_clicks} from analytics
        """
        # Step 1: base BM25 scores
        bm25_results: dict[int, BM25ScoredDocument] = {
            r.doc_id: r
            for r in self.bm25.rank_documents(query_terms, candidate_doc_ids, top_k=len(candidate_doc_ids))
        }

        if not bm25_results:
            return []

        # Normalise BM25 to [0, 1]
        max_bm25 = max(r.score for r in bm25_results.values()) or 1.0

        # Pre-compute title match data (one DB query per term)
        title_hit_terms = self._find_title_hits(query_terms, candidate_doc_ids)

        # Pre-compute max click count for normalisation
        max_clicks = max(click_counts.values()) if click_counts else 1

        # Compute recency reference time ONCE, not per document
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        ranked: list[RankedDocument] = []

        for doc_id, bm25_res in bm25_results.items():
            doc = self.db.get_document(doc_id)
            if doc is None:
                continue

            # — BM25 normalised
            bm25_norm = bm25_res.score / max_bm25

            # — Title boost: fraction of query terms in title
            hits_in_title = len(title_hit_terms.get(doc_id, set()) & set(query_terms))
            title_frac = hits_in_title / len(query_terms) if query_terms else 0.0

            # — Recency (exponential decay over days) — uses pre-computed `now`
            recency = self._recency_factor(doc.created_at, now)

            # — Click boost
            clicks = (click_counts or {}).get(doc_id, 0)
            click_norm = clicks / max_clicks if max_clicks > 0 else 0.0

            w = self.weights
            final = (
                w.bm25         * bm25_norm
                + w.title_boost  * title_frac
                + w.recency_boost * recency
                + w.click_boost  * click_norm
            )

            ranked.append(RankedDocument(
                doc_id=doc_id,
                score=round(final, 6),
                bm25_score=round(bm25_norm, 6),
                title_score=round(title_frac, 6),
                recency_score=round(recency, 6),
                click_score=round(click_norm, 6),
                title=bm25_res.title,
                snippet=bm25_res.snippet,
                term_scores=bm25_res.term_scores,
            ))

        ranked.sort(key=lambda x: x.score, reverse=True)
        logger.debug(
            "RelevanceTuner ranked %d docs → top %d", len(ranked), top_k
        )
        return ranked[:top_k]

    # ── Helpers ───────────────────────────────────────────────────────────

    def _find_title_hits(
        self,
        query_terms: list[str],
        doc_ids: set[int],
    ) -> dict[int, set[str]]:
        """Return {doc_id: set_of_query_terms_found_in_title}."""
        result: dict[int, set[str]] = {d: set() for d in doc_ids}
        for term in set(query_terms):
            postings = self.db.get_postings_for_term(term, field="title")
            for p in postings:
                if p.doc_id in result:
                    result[p.doc_id].add(term)
        return result

    def _recency_factor(self, created_at: str, now=None) -> float:
        """
        Exponential decay: score = e^(-0.01 * age_days)
        `now` is pre-computed by the caller to avoid per-document syscalls.
        """
        if not created_at:
            return 0.5
        try:
            from datetime import datetime, timezone
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            if now is None:
                now = datetime.now(timezone.utc)
            age_days = max(0.0, (now - created).total_seconds() / 86400)
            return math.exp(-0.01 * age_days)
        except Exception:
            return 0.5
