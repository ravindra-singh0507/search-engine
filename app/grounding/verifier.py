"""
Grounding Verification

=== THEORY ===

Grounding verification answers: "Is the generated answer supported by the
retrieved context, or did the LLM hallucinate?"

This is one of the most critical components of a production RAG system.
Without it, the system silently returns plausible-sounding but fabricated
answers.

=== GROUNDING vs HALLUCINATION ===

  Grounded answer  — every factual claim traces back to the retrieved context.
  Hallucination    — the LLM produced a claim not present in any source.

Sources of hallucination in RAG:
  1. The LLM "fills in" details from training data
  2. The retrieved context is irrelevant or empty
  3. The LLM ignores the "only use context" instruction
  4. The query is out of domain

=== OVERLAP-BASED VERIFICATION ===

We use token n-gram overlap (Jaccard at the bigram level) to measure how
much each answer sentence is supported by the context:

  support(sentence, context) = max_{chunk in context} jaccard_bigram(sentence, chunk)

  grounding_score = mean support across all sentences

This is a lightweight, dependency-free approach.  It gives a reasonable
signal without running an NLI model.

=== NLI-BASED VERIFICATION (Future) ===

A more accurate approach uses a Natural Language Inference model
(e.g. cross-encoder/nli-deberta-v3-small) to predict "entailment" /
"contradiction" / "neutral" for each (sentence, context) pair.
Production systems like Ragas and DeepEval use this approach.

=== DATA STRUCTURES ===

  ClaimSupport — one answer sentence and its support evidence
  GroundingReport — full report with score, risk tier, claim breakdown

=== THRESHOLDS ===

  score ≥ 0.5  → "low" hallucination risk
  score ≥ 0.25 → "medium" risk
  score <  0.25 → "high" risk

These thresholds are configurable via GroundingConfig.

=== COMPLEXITY ===

  Verify(S sentences, C chunks):  O(S × C × avg_len)
  Practical: S ≤ 20, C ≤ 10 → < 1 ms

=== PRODUCTION EQUIVALENTS ===

  Ragas:      faithfulness metric using LLM-based NLI
  DeepEval:   HallucinationMetric — cross-encoder NLI
  TruLens:    Groundedness feedback function
  LlamaIndex: FaithfulnessEvaluator
"""

import logging
import re
from dataclasses import dataclass, field

from app.context_builder.builder import Context
from app.config import GroundingConfig

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ClaimSupport:
    """Grounding evidence for a single sentence in the answer."""
    sentence:      str
    support_score: float             # 0.0 (no support) → 1.0 (fully grounded)
    best_chunk_id: str               # which context chunk best supports it
    best_snippet:  str               # the supporting text excerpt
    is_supported:  bool = False


@dataclass
class GroundingReport:
    """Full grounding verification report."""
    grounding_score:    float            # mean support across all sentences
    support_score:      float            # fraction of sentences above threshold
    hallucination_risk: str              # "low" | "medium" | "high"
    claim_supports:     list[ClaimSupport] = field(default_factory=list)
    unsupported_claims: list[str]          = field(default_factory=list)
    supported_claims:   list[str]          = field(default_factory=list)


# ── Grounding verifier ────────────────────────────────────────────────────────

class GroundingVerifier:
    """
    Verifies whether a generated answer is supported by retrieved context.

    Injection:
        verifier = GroundingVerifier(config.grounding)
        report   = verifier.verify(answer, context)
    """

    def __init__(self, config: GroundingConfig | None = None):
        self.config = config or GroundingConfig()

    def verify(self, answer: str, context: Context) -> GroundingReport:
        """
        Score each sentence in the answer against the context chunks.
        """
        if not answer.strip():
            return GroundingReport(
                grounding_score=0.0, support_score=0.0,
                hallucination_risk="high",
            )

        if context.is_empty():
            return GroundingReport(
                grounding_score=0.0, support_score=0.0,
                hallucination_risk="high",
            )

        sentences    = _split_sentences(answer)
        claim_sup:   list[ClaimSupport] = []
        context_text = context.text

        for sentence in sentences:
            if len(sentence.split()) < self.config.min_support_len:
                # Skip very short sentences (e.g. "Sure." or "Yes.")
                continue

            best_score    = 0.0
            best_chunk_id = ""
            best_snippet  = ""

            for chunk in context.chunks:
                score = _bigram_jaccard(sentence, chunk.text)
                if score > best_score:
                    best_score    = score
                    best_chunk_id = chunk.chunk_id
                    # Find the most overlapping window in the chunk
                    best_snippet  = _extract_support_snippet(sentence, chunk.text)

            is_supported = best_score >= self.config.threshold
            claim_sup.append(ClaimSupport(
                sentence=sentence.strip(),
                support_score=round(best_score, 4),
                best_chunk_id=best_chunk_id,
                best_snippet=best_snippet,
                is_supported=is_supported,
            ))

        if not claim_sup:
            return GroundingReport(
                grounding_score=0.0, support_score=0.0,
                hallucination_risk="high",
            )

        grounding_score = sum(c.support_score for c in claim_sup) / len(claim_sup)
        support_fraction = sum(1 for c in claim_sup if c.is_supported) / len(claim_sup)

        risk = self._risk_tier(grounding_score)

        report = GroundingReport(
            grounding_score    = round(grounding_score, 4),
            support_score      = round(support_fraction, 4),
            hallucination_risk = risk,
            claim_supports     = claim_sup,
            unsupported_claims = [c.sentence for c in claim_sup if not c.is_supported],
            supported_claims   = [c.sentence for c in claim_sup if c.is_supported],
        )

        logger.debug(
            "Grounding: score=%.3f, risk=%s, sentences=%d",
            grounding_score, risk, len(claim_sup),
        )
        return report

    def score_only(self, answer: str, context: Context) -> float:
        """Return just the grounding score (fast path for observability)."""
        return self.verify(answer, context).grounding_score

    # ── Internal ──────────────────────────────────────────────────────────

    def _risk_tier(self, score: float) -> str:
        if score >= 0.50:
            return "low"
        if score >= 0.25:
            return "medium"
        return "high"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _bigrams(text: str) -> set[tuple[str, str]]:
    tokens = re.sub(r"[^\w]", " ", text.lower()).split()
    if len(tokens) < 2:
        return set()
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def _bigram_jaccard(a: str, b: str) -> float:
    ba, bb = _bigrams(a), _bigrams(b)
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)


def _extract_support_snippet(sentence: str, chunk: str, window: int = 80) -> str:
    """Find the substring of chunk most similar to sentence."""
    words = chunk.split()
    if len(words) <= window:
        return chunk[:200]

    s_words = set(re.sub(r"[^\w]", " ", sentence.lower()).split())
    best_start = 0
    best_overlap = 0
    for i in range(max(1, len(words) - window)):
        window_words = set(w.lower() for w in words[i: i + window])
        overlap = len(s_words & window_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_start   = i

    snippet = " ".join(words[best_start: best_start + window])
    return snippet[:200]
