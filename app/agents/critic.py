"""
Critic Agent — Phase 7

=== THEORY ===

The critic implements adversarial evaluation — a pattern where one
agent reviews another's work to catch errors, gaps, and weak evidence
BEFORE it reaches the user.

This is inspired by:
  - Constitutional AI (Bai et al., 2022): self-critique loops
  - Debate (Irving et al., 2018): adversarial agents improve reasoning
  - OpenAI Deep Research: verification step after evidence gathering

The critic checks:
  1. Evidence coverage — are all sub-topics addressed?
  2. Evidence quality  — are scores above threshold?
  3. Contradictions    — do sources disagree?
  4. Missing areas     — what the retrieval didn't find

=== CRITIQUE DIMENSIONS ===

  Coverage:      fraction of sub-topics that have ≥1 evidence
  Quality:       fraction of evidence with score ≥ quality_threshold
  Consistency:   1.0 - contradiction_rate (token overlap disagreement)
  Completeness:  heuristic based on evidence density per topic

=== COMPLEXITY ===

  _execute():   O(E × T) where E = evidence count, T = topics
  Contradiction detection: O(E²) pairwise comparison (bounded by top_k²)

=== PRODUCTION EQUIVALENTS ===

  CrewAI:        manager agent that reviews and sends back for revision
  AutoGen:       critic role in group chat
  LangGraph:     conditional edge back to retrieval if quality < threshold
  Perplexity:    internal quality gate before answer generation
"""

import logging
import re
from typing import Any

from app.agents.base import (
    Agent, AgentContext, AgentMemory, AgentResult, AgentStatus,
    AgentTask, AgentType,
)

logger = logging.getLogger(__name__)


class CriticAgent(Agent):
    """
    Reviews evidence gathered by retrieval agents and produces a critique.

    Reads from task.params["_prior_results"] — the retrieval outputs.

    Returns:
      AgentResult.output = {
        "critique_id":       str,
        "overall_score":     float,
        "coverage_score":    float,
        "quality_score":     float,
        "consistency_score": float,
        "completeness_score": float,
        "issues":            list[dict],  # type, severity, description
        "recommendations":   list[str],
        "strong_evidence":   list[dict],  # evidence items above threshold
        "weak_evidence":     list[dict],  # evidence items below threshold
      }
    """

    QUALITY_THRESHOLD = 0.3

    @property
    def agent_type(self) -> AgentType:
        return AgentType.CRITIC

    def _execute(
        self,
        task:    AgentTask,
        context: AgentContext,
        memory:  AgentMemory,
    ) -> AgentResult:
        prior = task.params.get("_prior_results", {})
        strategy = task.params.get("strategy", "simple")

        # Collect all evidence from retrieval results
        all_evidence: list[dict] = []
        retrieval_outputs: list[dict] = []
        for step_id, result_dict in prior.items():
            if result_dict.get("agent_type") == "retrieval":
                output = result_dict.get("output", {})
                if isinstance(output, dict):
                    retrieval_outputs.append(output)
                    all_evidence.extend(output.get("evidence", []))

        if not all_evidence:
            return AgentResult(
                task_id    = task.task_id,
                agent_type = AgentType.CRITIC,
                status     = AgentStatus.DONE,
                output     = {
                    "critique_id":       task.task_id,
                    "overall_score":     0.0,
                    "coverage_score":    0.0,
                    "quality_score":     0.0,
                    "consistency_score": 1.0,
                    "completeness_score": 0.0,
                    "issues":            [{"type": "no_evidence", "severity": "critical",
                                           "description": "No evidence was gathered"}],
                    "recommendations":   ["Re-run retrieval with broader queries"],
                    "strong_evidence":   [],
                    "weak_evidence":     [],
                },
                confidence = 0.9,
            )

        # Evaluate dimensions
        coverage    = self._evaluate_coverage(retrieval_outputs)
        quality     = self._evaluate_quality(all_evidence)
        consistency = self._evaluate_consistency(all_evidence)
        completeness = self._evaluate_completeness(all_evidence, retrieval_outputs)

        overall = round(
            0.30 * coverage +
            0.30 * quality +
            0.20 * consistency +
            0.20 * completeness,
            4,
        )

        issues          = self._find_issues(coverage, quality, consistency, completeness, all_evidence)
        recommendations = self._generate_recommendations(issues)
        strong = [e for e in all_evidence if e.get("score", 0) >= self.QUALITY_THRESHOLD]
        weak   = [e for e in all_evidence if e.get("score", 0) < self.QUALITY_THRESHOLD]

        critique = {
            "critique_id":        task.task_id,
            "overall_score":      overall,
            "coverage_score":     coverage,
            "quality_score":      quality,
            "consistency_score":  consistency,
            "completeness_score": completeness,
            "issues":             issues,
            "recommendations":    recommendations,
            "strong_evidence":    strong,
            "weak_evidence":      weak,
        }

        memory.remember("critique", critique)

        return AgentResult(
            task_id    = task.task_id,
            agent_type = AgentType.CRITIC,
            status     = AgentStatus.DONE,
            output     = critique,
            confidence = 0.85,
            metadata   = {
                "evidence_reviewed": len(all_evidence),
                "issues_found":     len(issues),
            },
        )

    # ── Dimension evaluators ──────────────────────────────────────────────

    def _evaluate_coverage(self, retrieval_outputs: list[dict]) -> float:
        """Fraction of retrieval queries that returned ≥1 result."""
        if not retrieval_outputs:
            return 0.0
        covered = sum(1 for r in retrieval_outputs if r.get("evidence_count", 0) > 0)
        return round(covered / len(retrieval_outputs), 4)

    def _evaluate_quality(self, evidence: list[dict]) -> float:
        """Fraction of evidence items with score ≥ QUALITY_THRESHOLD."""
        if not evidence:
            return 0.0
        good = sum(1 for e in evidence if e.get("score", 0) >= self.QUALITY_THRESHOLD)
        return round(good / len(evidence), 4)

    def _evaluate_consistency(self, evidence: list[dict]) -> float:
        """
        Detect contradictions via pairwise token overlap on low-similarity pairs.

        Two documents that share < 10% tokens but cover the same topic may
        represent contradicting information.  This is a coarse heuristic.
        """
        if len(evidence) < 2:
            return 1.0

        contradiction_count = 0
        comparisons = 0
        for i in range(len(evidence)):
            for j in range(i + 1, min(i + 5, len(evidence))):
                t_i = set(evidence[i].get("content", "").lower().split())
                t_j = set(evidence[j].get("content", "").lower().split())
                if not t_i or not t_j:
                    continue
                overlap = len(t_i & t_j) / len(t_i | t_j)
                # Very low overlap on same topic = potential contradiction
                if overlap < 0.05 and evidence[i].get("title") == evidence[j].get("title"):
                    contradiction_count += 1
                comparisons += 1

        if comparisons == 0:
            return 1.0
        return round(1.0 - (contradiction_count / comparisons), 4)

    def _evaluate_completeness(
        self, evidence: list[dict], retrieval_outputs: list[dict],
    ) -> float:
        """Heuristic: average evidence density across queries."""
        if not retrieval_outputs:
            return 0.0
        densities = [
            min(1.0, r.get("evidence_count", 0) / 3)
            for r in retrieval_outputs
        ]
        return round(sum(densities) / len(densities), 4)

    # ── Issue detection ───────────────────────────────────────────────────

    def _find_issues(
        self,
        coverage: float, quality: float,
        consistency: float, completeness: float,
        evidence: list[dict],
    ) -> list[dict]:
        issues: list[dict] = []

        if coverage < 0.5:
            issues.append({
                "type": "low_coverage",
                "severity": "high",
                "description": f"Only {coverage:.0%} of topics have evidence",
            })
        if quality < 0.5:
            issues.append({
                "type": "low_quality",
                "severity": "medium",
                "description": f"Only {quality:.0%} of evidence meets quality threshold",
            })
        if consistency < 0.8:
            issues.append({
                "type": "contradictions",
                "severity": "medium",
                "description": f"Consistency score {consistency:.0%} — potential contradictions",
            })
        if completeness < 0.5:
            issues.append({
                "type": "incomplete",
                "severity": "medium",
                "description": f"Completeness {completeness:.0%} — some topics lack depth",
            })
        if len(evidence) < 3:
            issues.append({
                "type": "insufficient_evidence",
                "severity": "high",
                "description": f"Only {len(evidence)} evidence items found",
            })

        return issues

    def _generate_recommendations(self, issues: list[dict]) -> list[str]:
        recs: list[str] = []
        for issue in issues:
            t = issue["type"]
            if t == "low_coverage":
                recs.append("Broaden retrieval queries or add synonyms")
            elif t == "low_quality":
                recs.append("Lower score threshold or index more documents")
            elif t == "contradictions":
                recs.append("Review contradicting sources and flag for user")
            elif t == "incomplete":
                recs.append("Run additional retrieval passes on weak topics")
            elif t == "insufficient_evidence":
                recs.append("Expand search scope or reduce specificity")
        if not recs:
            recs.append("Evidence quality is satisfactory")
        return recs
