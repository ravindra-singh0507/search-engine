"""
Answer Confidence Engine

=== THEORY ===

Confidence estimation answers: "How certain should the user be in this answer?"

In RAG systems, confidence is a composite signal derived from multiple stages:

  1. Retrieval confidence  — how relevant were the retrieved documents?
                             High mean score → confident retrieval.
  2. Context confidence    — how rich and diverse is the context?
                             More sources + low redundancy → higher confidence.
  3. Grounding confidence  — how much of the answer is supported by sources?
                             Directly from GroundingReport.grounding_score.
  4. Citation confidence   — what fraction of sentences have attributions?
                             More citations → answer is traceable.
  5. Overall               — weighted combination of the above.

=== CONFIDENCE TIERS ===

  HIGH    (≥ 0.65) — answer is well-supported and traceable; show without warning
  MEDIUM  (≥ 0.40) — moderate support; user should verify important claims
  LOW     (< 0.40) — weak support; flag prominently; suggest re-querying

=== WHY NOT JUST USE GROUNDING SCORE? ===

Grounding score alone can be fooled:
  - A very focused answer on one retrieved paragraph gets high grounding score
    but may miss the full picture.
  - An answer from an empty context gets 0 grounding; this is caught by
    retrieval confidence.
  - An answer with no citations gets low citation confidence even if grounding
    score is decent.

The composite signal is more robust.

=== DATA STRUCTURES ===

  ConfidenceScores — all component scores + overall + tier

=== COMPLEXITY ===

  All arithmetic operations: O(C) where C = number of candidates
  Total: < 0.1 ms

=== PRODUCTION EQUIVALENTS ===

  OpenAI:    No native confidence; model logprobs can be used as proxy
  Ragas:     answer_correctness, faithfulness as separate evals
  TruLens:   Groundedness, Question Answering Relevance
  Perplexity: confidence shown as "high" / "medium" / "low" per source card
"""

import logging
from dataclasses import dataclass

from app.grounding.verifier import GroundingReport
from app.citations.engine import Citation
from app.context_builder.builder import Context

logger = logging.getLogger(__name__)


# ── Data structure ────────────────────────────────────────────────────────────

@dataclass
class ConfidenceScores:
    retrieval_confidence: float    # mean retrieval score of context chunks
    context_confidence:   float    # richness: diversity × (1 − redundancy)
    grounding_confidence: float    # from GroundingReport.grounding_score
    citation_confidence:  float    # fraction of answer sentences with citations
    overall_confidence:   float    # weighted combination
    tier:                 str      # "high" | "medium" | "low"


# ── Confidence engine ─────────────────────────────────────────────────────────

class ConfidenceEngine:
    """
    Aggregates retrieval, context, grounding, and citation signals into a
    single confidence estimate.

    Weights:
        retrieval:  0.25
        context:    0.15
        grounding:  0.40   ← most important
        citation:   0.20

    These are configurable via kwargs to score().
    """

    _DEFAULT_WEIGHTS = (0.25, 0.15, 0.40, 0.20)

    def score(
        self,
        context:          Context,
        grounding_report: GroundingReport,
        citations:        list[Citation],
        answer:           str = "",
        weights:          tuple[float, ...] | None = None,
    ) -> ConfidenceScores:
        """
        Compute all confidence components and return ConfidenceScores.

        Parameters
        ----------
        context          : assembled context from ContextBuilder
        grounding_report : from GroundingVerifier.verify()
        citations        : list of Citation objects from CitationEngine
        answer           : generated answer text (used for citation fraction)
        weights          : (retrieval_w, context_w, grounding_w, citation_w)
        """
        w = weights or self._DEFAULT_WEIGHTS

        # 1. Retrieval confidence
        retrieval_conf = self._retrieval_confidence(context)

        # 2. Context confidence
        ctx_conf = self._context_confidence(context)

        # 3. Grounding confidence (direct from report)
        grounding_conf = grounding_report.grounding_score

        # 4. Citation confidence
        citation_conf = self._citation_confidence(answer, citations)

        # 5. Weighted combination
        overall = (
            w[0] * retrieval_conf +
            w[1] * ctx_conf +
            w[2] * grounding_conf +
            w[3] * citation_conf
        )
        overall = round(min(1.0, max(0.0, overall)), 4)
        tier    = self._tier(overall)

        logger.debug(
            "Confidence: ret=%.3f ctx=%.3f grd=%.3f cit=%.3f → %.3f (%s)",
            retrieval_conf, ctx_conf, grounding_conf, citation_conf, overall, tier,
        )

        return ConfidenceScores(
            retrieval_confidence = round(retrieval_conf, 4),
            context_confidence   = round(ctx_conf, 4),
            grounding_confidence = round(grounding_conf, 4),
            citation_confidence  = round(citation_conf, 4),
            overall_confidence   = overall,
            tier                 = tier,
        )

    # ── Component scorers ─────────────────────────────────────────────────

    @staticmethod
    def _retrieval_confidence(context: Context) -> float:
        """Mean normalised retrieval score of selected chunks."""
        if not context.chunks:
            return 0.0
        scores = [c.score for c in context.chunks]
        # Clamp to [0, 1] (BM25 scores can exceed 1.0)
        clamped = [min(1.0, max(0.0, s)) for s in scores]
        return sum(clamped) / len(clamped)

    @staticmethod
    def _context_confidence(context: Context) -> float:
        """Quality score based on source diversity and low redundancy."""
        if not context.chunks:
            return 0.0
        meta       = context.metadata
        diversity  = meta.diversity_score            # 0 → all same source, 1 → all different
        redundancy = 1.0 - meta.redundancy_score     # flip: low redundancy = high score
        return (diversity + redundancy) / 2.0

    @staticmethod
    def _citation_confidence(answer: str, citations: list[Citation]) -> float:
        """
        Fraction of answer sentences that have at least one citation marker.
        A fully cited answer gets 1.0; uncited gets 0.0.
        """
        if not answer.strip() or not citations:
            return 0.0
        import re
        sentences = [s for s in re.split(r'(?<=[.!?])\s+', answer.strip()) if s]
        if not sentences:
            return 0.0
        cited = sum(
            1 for s in sentences
            if re.search(r'\[\d+\]|\(Source:', s)
        )
        return cited / len(sentences)

    @staticmethod
    def _tier(score: float) -> str:
        if score >= 0.65:
            return "high"
        if score >= 0.40:
            return "medium"
        return "low"
