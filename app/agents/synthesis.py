"""
Synthesis Agent — Phase 7

=== THEORY ===

The synthesis agent combines evidence from multiple retrieval passes
and critique results into a coherent final report.  This is the
"reduce" step in a MapReduce-style agentic workflow:

  Retrieval (map) → Critique (filter) → Synthesis (reduce)

The synthesis pattern is used in:
  - Multi-document summarisation (Barzilay & McKeown, 2005)
  - Evidence aggregation in systematic reviews
  - OpenAI Deep Research: final report generation from gathered evidence

=== REPORT FORMATS ===

  summary:    2-3 sentence executive summary
  detailed:   structured report with sections per sub-topic
  comparison: side-by-side analysis (for comparison workflows)

=== SYNTHESIS STRATEGY ===

  1. Collect all evidence and critique results from prior steps
  2. Group evidence by topic / sub-query
  3. Filter to strong evidence (score ≥ threshold)
  4. Generate section headers from sub-topics
  5. Produce body text from evidence content
  6. Add citations in [N] format
  7. Append quality metrics from the critique

=== COMPLEXITY ===

  _execute():  O(E × T) where E = evidence items, T = topics
  Report text assembly: O(total evidence text length)

=== PRODUCTION EQUIVALENTS ===

  OpenAI Deep Research: final report generation with inline citations
  Perplexity:           synthesised answer with source attribution
  CrewAI:               final task that aggregates crew outputs
  Glean:                answer generation from retrieved enterprise data
"""

import logging
from typing import Any

from app.agents.base import (
    Agent, AgentContext, AgentMemory, AgentResult, AgentStatus,
    AgentTask, AgentType,
)

logger = logging.getLogger(__name__)


class SynthesisAgent(Agent):
    """
    Combines evidence and critiques into a final research report.

    Reads from task.params["_prior_results"] — all upstream outputs.

    Returns:
      AgentResult.output = {
        "report_id":       str,
        "goal":            str,
        "strategy":        str,
        "summary":         str,
        "sections":        list[dict],  # title, content, citations
        "full_report":     str,         # Markdown report
        "evidence_used":   int,
        "quality_metrics": dict,
      }
    """

    @property
    def agent_type(self) -> AgentType:
        return AgentType.SYNTHESIS

    def _execute(
        self,
        task:    AgentTask,
        context: AgentContext,
        memory:  AgentMemory,
    ) -> AgentResult:
        prior      = task.params.get("_prior_results", {})
        goal       = task.params.get("goal", task.goal)
        strategy   = task.params.get("strategy", "simple")
        sub_topics = task.params.get("sub_topics", [])

        # Collect outputs by agent type
        retrieval_outputs: list[dict] = []
        critique_output:   dict | None = None
        citation_output:   dict | None = None

        for step_id, result_dict in prior.items():
            agent_type = result_dict.get("agent_type", "")
            output     = result_dict.get("output", {})
            if not isinstance(output, dict):
                continue

            if agent_type == "retrieval":
                retrieval_outputs.append(output)
            elif agent_type == "critic":
                critique_output = output
            elif agent_type == "citation_validator":
                citation_output = output

        # Gather all evidence
        all_evidence: list[dict] = []
        for ro in retrieval_outputs:
            all_evidence.extend(ro.get("evidence", []))

        # Build report
        sections = self._build_sections(
            goal, strategy, sub_topics,
            retrieval_outputs, all_evidence,
        )
        summary  = self._build_summary(goal, strategy, sections, all_evidence)
        quality  = self._build_quality_metrics(critique_output, citation_output)
        report   = self._assemble_markdown(goal, summary, sections, quality)

        output = {
            "report_id":       task.task_id,
            "goal":            goal,
            "strategy":        strategy,
            "summary":         summary,
            "sections":        sections,
            "full_report":     report,
            "evidence_used":   len(all_evidence),
            "quality_metrics": quality,
        }

        memory.remember("synthesis", output)

        confidence = min(0.95, 0.5 + 0.05 * len(all_evidence))
        if critique_output:
            confidence = min(0.95, confidence * (0.5 + 0.5 * critique_output.get("overall_score", 0.5)))

        return AgentResult(
            task_id    = task.task_id,
            agent_type = AgentType.SYNTHESIS,
            status     = AgentStatus.DONE,
            output     = output,
            evidence   = all_evidence[:10],
            confidence = round(confidence, 4),
            metadata   = {
                "sections":      len(sections),
                "evidence_used": len(all_evidence),
            },
        )

    # ── Section building ──────────────────────────────────────────────────

    def _build_sections(
        self,
        goal:       str,
        strategy:   str,
        sub_topics: list[str],
        retrieval_outputs: list[dict],
        all_evidence: list[dict],
    ) -> list[dict]:
        sections: list[dict] = []

        if strategy == "comparison" and len(sub_topics) >= 2:
            # One section per entity, plus comparison section
            for i, topic in enumerate(sub_topics):
                evidence_for_topic = (
                    retrieval_outputs[i].get("evidence", [])
                    if i < len(retrieval_outputs) else []
                )
                sections.append({
                    "title":     topic,
                    "content":   self._evidence_to_text(evidence_for_topic, start_idx=sum(
                        len(retrieval_outputs[j].get("evidence", []))
                        for j in range(i)
                    ) + 1),
                    "citations": list(range(1, len(evidence_for_topic) + 1)),
                })
            sections.append({
                "title":     "Comparison",
                "content":   self._comparison_text(sub_topics, retrieval_outputs),
                "citations": [],
            })

        elif strategy == "multi_part":
            for i, topic in enumerate(sub_topics):
                evidence_for_topic = (
                    retrieval_outputs[i].get("evidence", [])
                    if i < len(retrieval_outputs) else []
                )
                sections.append({
                    "title":     topic[:80],
                    "content":   self._evidence_to_text(evidence_for_topic, start_idx=1),
                    "citations": list(range(1, len(evidence_for_topic) + 1)),
                })

        else:
            # Single-section for simple/investigation strategies
            sections.append({
                "title":     goal[:80],
                "content":   self._evidence_to_text(all_evidence, start_idx=1),
                "citations": list(range(1, len(all_evidence) + 1)),
            })

        return sections

    def _evidence_to_text(self, evidence: list[dict], start_idx: int = 1) -> str:
        """Convert evidence list to readable text with citation markers."""
        if not evidence:
            return "No evidence found for this topic."

        parts: list[str] = []
        for i, ev in enumerate(evidence):
            idx     = start_idx + i
            content = ev.get("content", "").strip()
            title   = ev.get("title", "")
            if content:
                snippet = content[:300]
                parts.append(f"{snippet} [{idx}]")

        return " ".join(parts) if parts else "No relevant content found."

    def _comparison_text(
        self, topics: list[str], retrieval_outputs: list[dict],
    ) -> str:
        """Generate a comparison summary across topics."""
        if len(topics) < 2:
            return "Insufficient topics for comparison."

        lines = [f"Comparing {', '.join(topics)}:"]
        for i, topic in enumerate(topics):
            count = (
                retrieval_outputs[i].get("evidence_count", 0)
                if i < len(retrieval_outputs) else 0
            )
            quality = (
                retrieval_outputs[i].get("quality_score", 0.0)
                if i < len(retrieval_outputs) else 0.0
            )
            lines.append(
                f"- {topic}: {count} evidence items, "
                f"quality score {quality:.2f}"
            )

        return "\n".join(lines)

    # ── Summary ───────────────────────────────────────────────────────────

    def _build_summary(
        self, goal: str, strategy: str,
        sections: list[dict], evidence: list[dict],
    ) -> str:
        topic_count = len(sections)
        ev_count    = len(evidence)
        return (
            f"Research completed for: {goal}. "
            f"Analysed {ev_count} evidence items across {topic_count} section(s) "
            f"using a {strategy} strategy."
        )

    # ── Quality metrics ───────────────────────────────────────────────────

    def _build_quality_metrics(
        self,
        critique: dict | None,
        citation_val: dict | None,
    ) -> dict:
        metrics: dict = {}
        if critique:
            metrics["critique_score"]      = critique.get("overall_score", 0.0)
            metrics["coverage"]            = critique.get("coverage_score", 0.0)
            metrics["evidence_quality"]    = critique.get("quality_score", 0.0)
            metrics["consistency"]         = critique.get("consistency_score", 0.0)
            metrics["issue_count"]         = len(critique.get("issues", []))
        if citation_val:
            metrics["citation_accuracy"]   = citation_val.get("citation_accuracy", 0.0)
            metrics["citations_validated"] = citation_val.get("total_citations", 0)
        return metrics

    # ── Report assembly ───────────────────────────────────────────────────

    def _assemble_markdown(
        self,
        goal:     str,
        summary:  str,
        sections: list[dict],
        quality:  dict,
    ) -> str:
        lines: list[str] = []
        lines.append(f"# Research Report: {goal}")
        lines.append("")
        lines.append(f"**Summary:** {summary}")
        lines.append("")

        for section in sections:
            lines.append(f"## {section['title']}")
            lines.append("")
            lines.append(section["content"])
            lines.append("")

        if quality:
            lines.append("## Quality Metrics")
            lines.append("")
            for k, v in quality.items():
                label = k.replace("_", " ").title()
                if isinstance(v, float):
                    lines.append(f"- **{label}:** {v:.2%}")
                else:
                    lines.append(f"- **{label}:** {v}")
            lines.append("")

        return "\n".join(lines)
