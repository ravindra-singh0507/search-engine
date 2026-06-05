"""
Planner Agent — Phase 7

=== THEORY ===

Task decomposition is the first step in agentic retrieval.  The planner
takes a high-level research goal and produces a dependency-ordered list
of sub-tasks that specialist agents (retrieval, critic, synthesis) can
execute independently.

This mirrors the "plan-and-solve" prompting pattern (Wang et al., 2023):
  1. Devise a plan to solve the problem
  2. Carry out the plan step by step

Our planner is RULE-BASED (no LLM call) for determinism and speed.
It uses keyword heuristics and structural patterns to decompose queries.
Phase 8 can upgrade this to LLM-based planning for open-ended goals.

=== DECOMPOSITION STRATEGIES ===

  Comparison:   "Compare A, B, C"  → one retrieval per entity + comparison step
  Investigation: "Why does X..."   → retrieval + evidence + analysis
  Multi-part:    "Q1? Q2? Q3?"     → one retrieval per question + synthesis
  Simple:        "What is X?"      → single retrieval + answer

=== DEPENDENCY GENERATION ===

  All retrieval tasks are independent (can run in parallel).
  Critic depends on all retrieval tasks.
  Citation validation depends on critic.
  Synthesis depends on everything.

  This forms a DAG:
    R1 ─┬─ Critic ─── CitVal ─── Synthesis
    R2 ─┤
    R3 ─┘

=== COMPLEXITY ===

  plan():  O(len(query)) — regex matching + string splits
  Output:  O(K) WorkflowSteps where K = number of sub-topics

=== PRODUCTION EQUIVALENTS ===

  OpenAI Deep Research: planning phase before retrieval loops
  Perplexity Pro:       query decomposition for multi-step search
  LangGraph:            Planner node in plan-and-execute graph
  CrewAI:               hierarchical process with manager agent
"""

import logging
import re
from typing import Any

from app.agents.base import (
    Agent, AgentContext, AgentMemory, AgentResult, AgentStatus,
    AgentTask, AgentType, TaskPriority,
)
from app.orchestration.engine import WorkflowStep

logger = logging.getLogger(__name__)


class PlannerAgent(Agent):
    """
    Decomposes a research goal into an ordered set of WorkflowSteps.

    Returns:
      AgentResult.output = {
        "plan_id":     str,
        "goal":        str,
        "strategy":    str,   # "comparison" | "investigation" | "multi_part" | "simple"
        "sub_topics":  list[str],
        "steps":       list[dict],   # serialised WorkflowStep data
        "step_count":  int,
        "confidence":  float,
      }
    """

    @property
    def agent_type(self) -> AgentType:
        return AgentType.PLANNER

    def _execute(
        self,
        task:    AgentTask,
        context: AgentContext,
        memory:  AgentMemory,
    ) -> AgentResult:
        goal       = task.goal
        max_topics = task.params.get("max_topics", 6)

        strategy, sub_topics = self._decompose(goal, max_topics)
        steps                = self._build_steps(goal, sub_topics, strategy)

        plan = {
            "plan_id":    task.task_id,
            "goal":       goal,
            "strategy":   strategy,
            "sub_topics": sub_topics,
            "steps":      [self._step_to_dict(s) for s in steps],
            "step_count": len(steps),
        }

        confidence = self._assess_confidence(strategy, sub_topics)

        memory.remember("plan", plan)

        return AgentResult(
            task_id    = task.task_id,
            agent_type = AgentType.PLANNER,
            status     = AgentStatus.DONE,
            output     = plan,
            confidence = confidence,
            metadata   = {"strategy": strategy, "topic_count": len(sub_topics)},
        )

    # ── Decomposition ─────────────────────────────────────────────────────

    def _decompose(self, goal: str, max_topics: int) -> tuple[str, list[str]]:
        """
        Detect the query structure and extract sub-topics.

        Returns (strategy_name, list_of_subtopics).
        """
        goal_stripped = goal.strip()

        # Multi-question: "Q1? Q2? Q3?"
        questions = [q.strip() + "?" for q in goal_stripped.split("?") if q.strip()]
        if len(questions) > 1:
            return "multi_part", questions[:max_topics]

        # Comparison: "Compare A, B, and C" / "A vs B vs C"
        compare_match = re.search(
            r"compare\s+(.+)|(.+?)\s+vs\.?\s+(.+)|"
            r"difference(?:s)?\s+between\s+(.+?)\s+and\s+(.+)|"
            r"(.+?)\s+(?:or|versus)\s+(.+)",
            goal_stripped, re.IGNORECASE,
        )
        if compare_match:
            raw = compare_match.group(0)
            parts = re.split(
                r"\s*,\s*|\s+and\s+|\s+vs\.?\s+|\s+or\s+|\s+versus\s+",
                raw, flags=re.IGNORECASE,
            )
            parts = [
                re.sub(r"^(compare|differences?\s+between)\s+", "", p, flags=re.IGNORECASE).strip()
                for p in parts if p.strip()
            ]
            if len(parts) > 1:
                return "comparison", parts[:max_topics]

        # Investigation: "why" / "how" / "explain" / "analyze" / "investigate"
        if re.match(r"(why|how|explain|analyze|analyse|investigate)\b", goal_stripped, re.IGNORECASE):
            topics = self._extract_key_entities(goal_stripped)
            if topics:
                return "investigation", topics[:max_topics]

        # Simple: single-topic query
        topics = self._extract_key_entities(goal_stripped)
        return "simple", topics[:max_topics] if topics else [goal_stripped]

    def _extract_key_entities(self, text: str) -> list[str]:
        """
        Extract meaningful noun-phrase-like chunks from the query.

        Heuristic: split on common delimiters (commas, "and", semicolons),
        strip stop words, return non-empty fragments.
        """
        _STOP = {
            "what", "is", "the", "a", "an", "of", "for", "in", "on", "to",
            "with", "how", "why", "does", "do", "can", "should", "would",
            "about", "this", "that", "these", "those", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "it", "its", "my",
        }
        parts = re.split(r"\s*[,;]\s*|\s+and\s+", text)
        entities = []
        for part in parts:
            words = [w for w in part.split() if w.lower() not in _STOP and len(w) > 1]
            if words:
                entities.append(" ".join(words))
        return entities

    # ── Step construction ─────────────────────────────────────────────────

    def _build_steps(
        self, goal: str, sub_topics: list[str], strategy: str,
    ) -> list[WorkflowStep]:
        steps: list[WorkflowStep] = []
        retrieval_ids: list[str] = []

        # Retrieval steps — one per sub-topic
        for i, topic in enumerate(sub_topics):
            step_id = f"retrieve_{i}"
            steps.append(WorkflowStep(
                step_id    = step_id,
                agent_type = AgentType.RETRIEVAL,
                goal       = f"Retrieve evidence for: {topic}",
                params     = {"query": topic, "topic": topic, "topic_index": i},
                priority   = TaskPriority.HIGH,
                timeout_sec = 60.0,
            ))
            retrieval_ids.append(step_id)

        # Critic step — depends on all retrieval
        critic_id = "critic_0"
        steps.append(WorkflowStep(
            step_id    = critic_id,
            agent_type = AgentType.CRITIC,
            goal       = f"Critique evidence quality for: {goal}",
            params     = {"strategy": strategy},
            depends_on = list(retrieval_ids),
            priority   = TaskPriority.NORMAL,
            timeout_sec = 60.0,
        ))

        # Citation validation step
        citval_id = "citval_0"
        steps.append(WorkflowStep(
            step_id    = citval_id,
            agent_type = AgentType.CITATION_VALIDATOR,
            goal       = f"Validate citation accuracy for: {goal}",
            params     = {"strategy": strategy},
            depends_on = [critic_id],
            priority   = TaskPriority.NORMAL,
            optional   = True,
            timeout_sec = 60.0,
        ))

        # Synthesis step — depends on everything
        steps.append(WorkflowStep(
            step_id    = "synthesis_0",
            agent_type = AgentType.SYNTHESIS,
            goal       = f"Synthesise final report for: {goal}",
            params     = {"strategy": strategy, "goal": goal, "sub_topics": sub_topics},
            depends_on = [citval_id],
            priority   = TaskPriority.NORMAL,
            timeout_sec = 120.0,
        ))

        return steps

    def _assess_confidence(self, strategy: str, sub_topics: list[str]) -> float:
        if not sub_topics:
            return 0.3
        if strategy == "comparison" and len(sub_topics) >= 2:
            return 0.85
        if strategy == "multi_part":
            return 0.80
        if strategy == "investigation":
            return 0.70
        return 0.60

    @staticmethod
    def _step_to_dict(step: WorkflowStep) -> dict:
        return {
            "step_id":    step.step_id,
            "agent_type": step.agent_type.value,
            "goal":       step.goal,
            "depends_on": step.depends_on,
            "optional":   step.optional,
        }
