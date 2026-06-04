"""
RAG Evaluation Framework

=== THEORY ===

RAG evaluation answers: "How good is the generated answer?"

Unlike traditional IR evaluation (P@K, NDCG), RAG evaluation must measure
the quality of a generated natural language answer, not just the ranked list.

=== METRICS ===

  Faithfulness         — Is the answer faithful to (entailed by) the retrieved
                         context?  High faithfulness = no hallucination.
                         Measured via token overlap between answer and context.

  Groundedness         — What fraction of answer sentences are supported by
                         at least one source chunk?  (= GroundingReport.support_score)

  Answer Relevance     — Does the answer address the question?
                         Measured via token overlap between answer and query.

  Context Precision    — What fraction of retrieved chunks are relevant to
                         the query?  High precision = retrieval isn't noisy.

  Context Recall       — Do the retrieved chunks contain the information
                         needed to answer?  Requires a ground-truth answer.

  Citation Accuracy    — Do the citation indices in the answer actually
                         correspond to the correct source chunks?

  Response Completeness— Is the answer complete relative to the question?
                         Approximate: length ratio answer / question.

=== WITHOUT GROUND TRUTH ===

When no reference answer is provided, Context Recall is estimated from
the overlap between context and a "self-contained" summary of the answer.

=== RAGAS COMPARISON ===

Our metrics are lightweight proxies for the Ragas framework metrics:
  Ragas Faithfulness        ≈ our faithfulness
  Ragas Answer Relevancy    ≈ our answer_relevance
  Ragas Context Precision   ≈ our context_precision
  Ragas Context Recall      ≈ our context_recall

Ragas uses an LLM as evaluator (expensive).  We use token overlap (fast,
free, no API calls).  For production, add optional LLM-based evaluation.

=== DATA STRUCTURES ===

  RAGEvalCase   — one (query, answer, context, ground_truth?) triple
  RAGEvalResult — metric scores for one case
  RAGEvalReport — aggregate report over many cases

=== COMPLEXITY ===

  Evaluate one case: O(S × C × len) — S sentences, C chunks
  Practical: < 5 ms per case
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.context_builder.builder import Context, ContextChunk
from app.grounding.verifier import GroundingReport
from app.citations.engine import CitedAnswer

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RAGEvalCase:
    """One evaluation example."""
    query_id:        str
    query:           str
    answer:          str
    context:         Context
    citations:       list          = field(default_factory=list)
    grounding:       GroundingReport | None = None
    ground_truth:    str           = ""   # optional reference answer


@dataclass
class RAGEvalResult:
    """Evaluation scores for one case."""
    query_id:             str
    query:                str
    faithfulness:         float
    groundedness:         float
    answer_relevance:     float
    context_precision:    float
    context_recall:       float
    citation_accuracy:    float
    response_completeness: float
    overall_score:        float

    def to_dict(self) -> dict:
        return {
            "query_id":             self.query_id,
            "query":                self.query,
            "faithfulness":         round(self.faithfulness, 4),
            "groundedness":         round(self.groundedness, 4),
            "answer_relevance":     round(self.answer_relevance, 4),
            "context_precision":    round(self.context_precision, 4),
            "context_recall":       round(self.context_recall, 4),
            "citation_accuracy":    round(self.citation_accuracy, 4),
            "response_completeness": round(self.response_completeness, 4),
            "overall_score":        round(self.overall_score, 4),
        }


@dataclass
class RAGEvalReport:
    """Aggregate evaluation report over multiple cases."""
    total_cases:  int
    avg_scores:   dict[str, float]
    per_case:     list[RAGEvalResult]
    generated_at: str = ""

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()


# ── Evaluator ─────────────────────────────────────────────────────────────────

class RAGEvaluator:
    """
    Evaluates RAG pipeline outputs across multiple quality dimensions.

    Usage:
        evaluator = RAGEvaluator()
        result    = evaluator.evaluate(case)
        report    = evaluator.generate_report([case1, case2, ...])
    """

    def evaluate(self, case: RAGEvalCase) -> RAGEvalResult:
        """Evaluate all metrics for a single RAGEvalCase."""
        faithfulness    = self._faithfulness(case)
        groundedness    = self._groundedness(case)
        answer_rel      = self._answer_relevance(case)
        ctx_precision   = self._context_precision(case)
        ctx_recall      = self._context_recall(case)
        citation_acc    = self._citation_accuracy(case)
        completeness    = self._response_completeness(case)

        # Overall = weighted mean (bias toward faithfulness and relevance)
        overall = (
            0.25 * faithfulness +
            0.20 * groundedness +
            0.20 * answer_rel +
            0.10 * ctx_precision +
            0.10 * ctx_recall +
            0.10 * citation_acc +
            0.05 * completeness
        )

        return RAGEvalResult(
            query_id             = case.query_id,
            query                = case.query,
            faithfulness         = faithfulness,
            groundedness         = groundedness,
            answer_relevance     = answer_rel,
            context_precision    = ctx_precision,
            context_recall       = ctx_recall,
            citation_accuracy    = citation_acc,
            response_completeness = completeness,
            overall_score        = round(overall, 4),
        )

    def batch_evaluate(self, cases: list[RAGEvalCase]) -> list[RAGEvalResult]:
        """Evaluate a list of cases."""
        return [self.evaluate(c) for c in cases]

    def generate_report(self, cases: list[RAGEvalCase]) -> RAGEvalReport:
        """Evaluate all cases and compute aggregate statistics."""
        if not cases:
            return RAGEvalReport(total_cases=0, avg_scores={}, per_case=[])

        results = self.batch_evaluate(cases)

        metric_names = [
            "faithfulness", "groundedness", "answer_relevance",
            "context_precision", "context_recall", "citation_accuracy",
            "response_completeness", "overall_score",
        ]
        avg: dict[str, float] = {}
        for m in metric_names:
            vals = [getattr(r, m) for r in results]
            avg[m] = round(sum(vals) / len(vals), 4)

        logger.info(
            "RAG eval complete: %d cases, overall=%.3f",
            len(cases), avg["overall_score"],
        )
        return RAGEvalReport(total_cases=len(cases), avg_scores=avg, per_case=results)

    def save_report(self, report: RAGEvalReport, path: Path) -> None:
        """Persist an evaluation report to JSON."""
        data = {
            "generated_at": report.generated_at,
            "total_cases":  report.total_cases,
            "avg_scores":   report.avg_scores,
            "per_case":     [r.to_dict() for r in report.per_case],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug("Saved RAG eval report to %s", path)

    # ── Metric implementations ────────────────────────────────────────────

    @staticmethod
    def _faithfulness(case: RAGEvalCase) -> float:
        """
        Fraction of answer tokens that appear in the context.
        Perfect faithfulness = every word in the answer came from the context.
        """
        if not case.answer or case.context.is_empty():
            return 0.0
        answer_tokens  = _tokenize(case.answer)
        context_tokens = _tokenize(case.context.text)
        if not answer_tokens:
            return 0.0
        overlap = len(answer_tokens & context_tokens)
        return round(overlap / len(answer_tokens), 4)

    @staticmethod
    def _groundedness(case: RAGEvalCase) -> float:
        """Fraction of answer sentences supported by context chunks."""
        if case.grounding:
            return case.grounding.support_score
        # Fallback: compute from context overlap
        if not case.answer or case.context.is_empty():
            return 0.0
        sentences = _split_sentences(case.answer)
        if not sentences:
            return 0.0
        supported = sum(
            1 for s in sentences
            if any(_jaccard(s, c.text) >= 0.08 for c in case.context.chunks)
        )
        return round(supported / len(sentences), 4)

    @staticmethod
    def _answer_relevance(case: RAGEvalCase) -> float:
        """Token overlap between answer and query."""
        if not case.answer or not case.query:
            return 0.0
        q_tokens = _tokenize(case.query)
        a_tokens = _tokenize(case.answer)
        if not q_tokens or not a_tokens:
            return 0.0
        return round(len(q_tokens & a_tokens) / len(q_tokens), 4)

    @staticmethod
    def _context_precision(case: RAGEvalCase) -> float:
        """Fraction of retrieved chunks that overlap with the answer."""
        if not case.context.chunks:
            return 0.0
        a_tokens = _tokenize(case.answer)
        if not a_tokens:
            return 0.0
        relevant = sum(
            1 for c in case.context.chunks
            if len(_tokenize(c.text) & a_tokens) / max(len(a_tokens), 1) >= 0.05
        )
        return round(relevant / len(case.context.chunks), 4)

    @staticmethod
    def _context_recall(case: RAGEvalCase) -> float:
        """
        If ground_truth is provided: fraction of ground_truth tokens in context.
        Otherwise: fraction of answer tokens in context (proxy).
        """
        reference = case.ground_truth if case.ground_truth else case.answer
        if not reference or case.context.is_empty():
            return 0.0
        ref_tokens = _tokenize(reference)
        ctx_tokens = _tokenize(case.context.text)
        if not ref_tokens:
            return 0.0
        overlap = len(ref_tokens & ctx_tokens)
        return round(overlap / len(ref_tokens), 4)

    @staticmethod
    def _citation_accuracy(case: RAGEvalCase) -> float:
        """
        Fraction of [N] tags in the answer that correspond to an actual source chunk.
        If no citation tags present, returns 0.5 (neutral).
        """
        if not case.citations:
            return 0.5
        tags = re.findall(r'\[(\d+)\]', case.answer)
        if not tags:
            return 0.5
        valid_indices = {c.index for c in case.citations}
        valid = sum(1 for t in tags if int(t) in valid_indices)
        return round(valid / len(tags), 4)

    @staticmethod
    def _response_completeness(case: RAGEvalCase) -> float:
        """
        Rough completeness: ratio of answer tokens to query tokens.
        A very short answer to a complex question is incomplete.
        Capped at 1.0.
        """
        q_len = len(case.query.split())
        a_len = len(case.answer.split())
        if q_len == 0:
            return 1.0
        # Expect answer to be at least 3× the query length
        return round(min(1.0, a_len / (q_len * 3)), 4)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    return set(re.sub(r"[^\w]", " ", text.lower()).split())


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
