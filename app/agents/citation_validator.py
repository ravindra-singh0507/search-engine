"""
Citation Validation Agent — Phase 7

=== THEORY ===

Citation validation ensures every claim in a generated answer is
actually supported by the cited source.  This closes the gap between
"the LLM said [1]" and "source [1] actually supports that claim."

In production systems:
  - Perplexity highlights inline citations with hover-to-verify
  - Google Search Generative Experience fact-checks each claim
  - Academic tools (Semantic Scholar) verify reference accuracy

=== VALIDATION PROCESS ===

  For each (claim, citation) pair:
    1. Extract the sentence containing the citation marker [N]
    2. Look up the source chunk referenced by [N]
    3. Compute token overlap between claim sentence and source
    4. If overlap ≥ threshold → SUPPORTED
    5. If overlap < threshold → UNSUPPORTED (citation drift)

=== METRICS ===

  citation_accuracy:  fraction of citations that are supported
  drift_count:        number of citations that don't match source
  unsupported_claims: list of claim sentences with no valid citation

=== COMPLEXITY ===

  _execute():  O(C × S) where C = citation count, S = sentence count
  Token overlap per pair: O(len(sentence) + len(chunk))

=== PRODUCTION EQUIVALENTS ===

  Perplexity:   inline citation verification
  Bing Chat:    source grounding check per statement
  LangSmith:    evaluation trace with faithfulness scores
"""

import logging
import re
from typing import Any

from app.agents.base import (
    Agent, AgentContext, AgentMemory, AgentResult, AgentStatus,
    AgentTask, AgentType,
)

logger = logging.getLogger(__name__)


class CitationValidationAgent(Agent):
    """
    Validates that citations in generated answers are supported by sources.

    Reads from task.params["_prior_results"] — expects critic output
    which contains strong_evidence and weak_evidence lists.

    Returns:
      AgentResult.output = {
        "validation_id":      str,
        "citation_accuracy":  float,
        "total_citations":    int,
        "supported_count":    int,
        "unsupported_count":  int,
        "drift_count":        int,
        "validations":        list[dict],  # per-citation detail
        "unsupported_claims": list[str],
      }
    """

    SUPPORT_THRESHOLD = 0.08

    @property
    def agent_type(self) -> AgentType:
        return AgentType.CITATION_VALIDATOR

    def _execute(
        self,
        task:    AgentTask,
        context: AgentContext,
        memory:  AgentMemory,
    ) -> AgentResult:
        prior = task.params.get("_prior_results", {})

        # Gather all evidence from prior results
        all_evidence: list[dict] = []
        answers: list[str] = []

        for step_id, result_dict in prior.items():
            agent_type = result_dict.get("agent_type", "")
            output     = result_dict.get("output", {})

            if agent_type == "retrieval" and isinstance(output, dict):
                all_evidence.extend(output.get("evidence", []))
                if output.get("answer"):
                    answers.append(output["answer"])

            if agent_type == "critic" and isinstance(output, dict):
                all_evidence.extend(output.get("strong_evidence", []))

        # Build source lookup by doc_id
        source_map: dict[int, dict] = {}
        for ev in all_evidence:
            doc_id = ev.get("doc_id")
            if doc_id is not None and doc_id not in source_map:
                source_map[doc_id] = ev

        # Validate citations in each answer
        validations: list[dict]  = []
        unsupported_claims: list[str] = []
        total_citations  = 0
        supported_count  = 0
        drift_count      = 0

        for answer in answers:
            sentences = self._split_sentences(answer)
            for sentence in sentences:
                tags = re.findall(r'\[(\d+)\]', sentence)
                if not tags:
                    continue

                for tag in tags:
                    total_citations += 1
                    idx = int(tag)

                    # Find the source this index refers to
                    source = self._find_source(idx, all_evidence)

                    if source is None:
                        drift_count += 1
                        validations.append({
                            "citation_index": idx,
                            "sentence":       sentence[:200],
                            "status":         "missing_source",
                            "overlap":        0.0,
                        })
                        unsupported_claims.append(sentence[:200])
                        continue

                    overlap = self._token_overlap(sentence, source.get("content", ""))
                    is_supported = overlap >= self.SUPPORT_THRESHOLD

                    if is_supported:
                        supported_count += 1
                    else:
                        drift_count += 1
                        unsupported_claims.append(sentence[:200])

                    validations.append({
                        "citation_index": idx,
                        "sentence":       sentence[:200],
                        "source_title":   source.get("title", ""),
                        "status":         "supported" if is_supported else "unsupported",
                        "overlap":        round(overlap, 4),
                    })

        accuracy = (supported_count / total_citations) if total_citations > 0 else 1.0

        result_data = {
            "validation_id":      task.task_id,
            "citation_accuracy":  round(accuracy, 4),
            "total_citations":    total_citations,
            "supported_count":    supported_count,
            "unsupported_count":  total_citations - supported_count,
            "drift_count":        drift_count,
            "validations":        validations,
            "unsupported_claims": unsupported_claims,
        }

        memory.remember("citation_validation", result_data)

        return AgentResult(
            task_id    = task.task_id,
            agent_type = AgentType.CITATION_VALIDATOR,
            status     = AgentStatus.DONE,
            output     = result_data,
            confidence = accuracy,
            metadata   = {
                "citations_checked": total_citations,
                "accuracy":          round(accuracy, 4),
            },
        )

    def _find_source(self, index: int, evidence: list[dict]) -> dict | None:
        """Map a 1-based citation index to an evidence item."""
        if 1 <= index <= len(evidence):
            return evidence[index - 1]
        return None

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    @staticmethod
    def _token_overlap(a: str, b: str) -> float:
        ta = set(re.sub(r"[^\w]", " ", a.lower()).split())
        tb = set(re.sub(r"[^\w]", " ", b.lower()).split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)
