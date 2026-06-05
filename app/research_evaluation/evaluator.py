"""
Research Evaluation Framework — Phase 7

=== THEORY ===

Research evaluation extends Phase 6 RAG evaluation to cover the full
agentic pipeline.  While RAG evaluation measures answer quality,
research evaluation measures the PROCESS quality:

  - Did the planner decompose the goal correctly?
  - Did retrieval agents find sufficient evidence?
  - Did the critic identify real issues?
  - Is the final report complete and well-cited?

=== METRICS ===

  Task Completion Rate     — fraction of planned tasks that succeeded
  Research Completeness    — evidence coverage across all sub-topics
  Citation Accuracy        — from CitationValidationAgent output
  Evidence Coverage        — fraction of topics with ≥1 evidence
  Grounding Quality        — from grounding verifier
  Hallucination Rate       — 1 - grounding_score
  Report Quality           — composite of completeness, citations, structure

=== DATA STRUCTURES ===

  ResearchEvalCase    — one workflow run to evaluate
  ResearchEvalResult  — metric scores for one case
  ResearchEvalReport  — aggregate report over multiple cases

=== COMPLEXITY ===

  evaluate():  O(S) where S = steps in the workflow
  report():    O(N × S) where N = number of cases

=== PRODUCTION EQUIVALENTS ===

  LangSmith:     trace-based evaluation with custom metrics
  Ragas:         RAG-specific evaluation (faithfulness, etc.)
  PromptFoo:     automated evaluation harness
  Braintrust:    LLM application evaluation platform
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ResearchEvalCase:
    """One research workflow to evaluate."""
    case_id:         str
    goal:            str
    workflow_name:   str                 = ""
    total_steps:     int                 = 0
    completed_steps: int                 = 0
    failed_steps:    int                 = 0
    evidence_count:  int                 = 0
    topic_count:     int                 = 1
    topics_covered:  int                 = 0
    citation_accuracy: float             = 0.0
    grounding_score: float               = 0.0
    report_text:     str                 = ""
    total_latency_ms: float              = 0.0
    metadata:        dict[str, Any]      = field(default_factory=dict)


@dataclass
class ResearchEvalResult:
    """Evaluation scores for one research case."""
    case_id:               str
    task_completion:        float
    research_completeness: float
    citation_accuracy:     float
    evidence_coverage:     float
    grounding_quality:     float
    hallucination_rate:    float
    report_quality:        float
    overall_score:         float

    def to_dict(self) -> dict:
        return {
            "case_id":               self.case_id,
            "task_completion":       round(self.task_completion, 4),
            "research_completeness": round(self.research_completeness, 4),
            "citation_accuracy":     round(self.citation_accuracy, 4),
            "evidence_coverage":     round(self.evidence_coverage, 4),
            "grounding_quality":     round(self.grounding_quality, 4),
            "hallucination_rate":    round(self.hallucination_rate, 4),
            "report_quality":        round(self.report_quality, 4),
            "overall_score":         round(self.overall_score, 4),
        }


@dataclass
class ResearchEvalReport:
    """Aggregate evaluation over multiple research cases."""
    total_cases:  int
    avg_scores:   dict[str, float]
    per_case:     list[ResearchEvalResult]
    generated_at: str = ""

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()


class ResearchEvaluator:
    """
    Evaluates research workflow quality across multiple dimensions.

    Usage:
        evaluator = ResearchEvaluator()
        result    = evaluator.evaluate(case)
        report    = evaluator.generate_report([case1, case2])
    """

    def evaluate(self, case: ResearchEvalCase) -> ResearchEvalResult:
        task_completion = (
            case.completed_steps / case.total_steps
            if case.total_steps > 0 else 0.0
        )

        research_completeness = self._compute_completeness(case)
        evidence_coverage     = (
            case.topics_covered / case.topic_count
            if case.topic_count > 0 else 0.0
        )
        grounding_quality     = case.grounding_score
        hallucination_rate    = max(0.0, 1.0 - case.grounding_score)
        report_quality        = self._compute_report_quality(case)

        overall = (
            0.20 * task_completion +
            0.20 * research_completeness +
            0.15 * case.citation_accuracy +
            0.15 * evidence_coverage +
            0.15 * grounding_quality +
            0.05 * (1.0 - hallucination_rate) +
            0.10 * report_quality
        )

        return ResearchEvalResult(
            case_id               = case.case_id,
            task_completion       = round(task_completion, 4),
            research_completeness = round(research_completeness, 4),
            citation_accuracy     = round(case.citation_accuracy, 4),
            evidence_coverage     = round(evidence_coverage, 4),
            grounding_quality     = round(grounding_quality, 4),
            hallucination_rate    = round(hallucination_rate, 4),
            report_quality        = round(report_quality, 4),
            overall_score         = round(overall, 4),
        )

    def batch_evaluate(self, cases: list[ResearchEvalCase]) -> list[ResearchEvalResult]:
        return [self.evaluate(c) for c in cases]

    def generate_report(self, cases: list[ResearchEvalCase]) -> ResearchEvalReport:
        if not cases:
            return ResearchEvalReport(total_cases=0, avg_scores={}, per_case=[])

        results = self.batch_evaluate(cases)
        metrics = [
            "task_completion", "research_completeness", "citation_accuracy",
            "evidence_coverage", "grounding_quality", "hallucination_rate",
            "report_quality", "overall_score",
        ]
        avg: dict[str, float] = {}
        for m in metrics:
            vals = [getattr(r, m) for r in results]
            avg[m] = round(sum(vals) / len(vals), 4)

        return ResearchEvalReport(
            total_cases = len(cases),
            avg_scores  = avg,
            per_case    = results,
        )

    def save_report(self, report: ResearchEvalReport, path: Path) -> None:
        data = {
            "generated_at": report.generated_at,
            "total_cases":  report.total_cases,
            "avg_scores":   report.avg_scores,
            "per_case":     [r.to_dict() for r in report.per_case],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _compute_completeness(case: ResearchEvalCase) -> float:
        if case.topic_count == 0:
            return 0.0
        evidence_per_topic = case.evidence_count / max(case.topic_count, 1)
        return min(1.0, evidence_per_topic / 3.0)

    @staticmethod
    def _compute_report_quality(case: ResearchEvalCase) -> float:
        if not case.report_text:
            return 0.0
        word_count = len(case.report_text.split())
        length_score = min(1.0, word_count / 200)
        has_sections = "##" in case.report_text
        has_citations = "[" in case.report_text and "]" in case.report_text
        structure_score = (0.5 * int(has_sections) + 0.5 * int(has_citations))
        return round(0.5 * length_score + 0.5 * structure_score, 4)
