"""
Workflow Templates — Phase 7

=== THEORY ===

Workflow templates are parameterised DAGs that encode common research
patterns.  Instead of the planner constructing a workflow from scratch
for every query, it can select a template and fill in the variables.

This mirrors the "prompt template" pattern from Phase 6, elevated to
multi-agent orchestration:
  Phase 6 template: system + context + question → LLM prompt
  Phase 7 template: goal + parameters → agent workflow DAG

=== TEMPLATES ===

  Comparison:     Side-by-side analysis of 2+ entities
  Investigation:  Deep dive into a single topic with evidence + critique
  Documentation:  Generate reference documentation from indexed docs
  Summarization:  Multi-source summary of a broad topic
  TechEval:       Technology evaluation with pros/cons/recommendation
  RootCause:      Diagnostic investigation with hypothesis testing

=== TEMPLATE INTERFACE ===

  WorkflowTemplate.name         — unique identifier
  WorkflowTemplate.description  — human-readable purpose
  WorkflowTemplate.generate()   — produces a list[WorkflowStep]

=== COMPLEXITY ===

  generate():  O(T) where T = number of topics/parameters
  All templates produce linear or tree-shaped DAGs.

=== PRODUCTION EQUIVALENTS ===

  LangGraph:     pre-built graph templates
  CrewAI:        example flows (research, content creation)
  Prefect:       flow templates with parameters
  n8n / Zapier:  workflow templates marketplace
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from app.agents.base import AgentType, TaskPriority
from app.orchestration.engine import WorkflowStep

logger = logging.getLogger(__name__)


class WorkflowTemplate(ABC):
    """Base class for workflow templates."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def generate(self, goal: str, params: dict) -> list[WorkflowStep]: ...


class ComparisonWorkflow(WorkflowTemplate):
    """Compare 2+ entities side by side."""

    @property
    def name(self) -> str:
        return "comparison"

    @property
    def description(self) -> str:
        return "Side-by-side comparison of multiple entities with evidence"

    def generate(self, goal: str, params: dict) -> list[WorkflowStep]:
        entities = params.get("entities", [])
        if len(entities) < 2:
            entities = [goal]

        steps: list[WorkflowStep] = []
        retrieval_ids: list[str] = []

        for i, entity in enumerate(entities):
            sid = f"retrieve_{i}"
            steps.append(WorkflowStep(
                step_id    = sid,
                agent_type = AgentType.RETRIEVAL,
                goal       = f"Retrieve evidence for: {entity}",
                params     = {"query": entity, "topic": entity, "topic_index": i},
                priority   = TaskPriority.HIGH,
            ))
            retrieval_ids.append(sid)

        steps.append(WorkflowStep(
            step_id    = "critic_0",
            agent_type = AgentType.CRITIC,
            goal       = f"Critique evidence quality for comparison: {goal}",
            params     = {"strategy": "comparison"},
            depends_on = retrieval_ids,
        ))
        steps.append(WorkflowStep(
            step_id    = "citval_0",
            agent_type = AgentType.CITATION_VALIDATOR,
            goal       = f"Validate citations for: {goal}",
            depends_on = ["critic_0"],
            optional   = True,
        ))
        steps.append(WorkflowStep(
            step_id    = "synthesis_0",
            agent_type = AgentType.SYNTHESIS,
            goal       = f"Generate comparison report for: {goal}",
            params     = {"strategy": "comparison", "goal": goal, "sub_topics": entities},
            depends_on = ["citval_0"],
        ))
        return steps


class InvestigationWorkflow(WorkflowTemplate):
    """Deep investigation into a single topic."""

    @property
    def name(self) -> str:
        return "investigation"

    @property
    def description(self) -> str:
        return "Deep-dive investigation with evidence gathering and critique"

    def generate(self, goal: str, params: dict) -> list[WorkflowStep]:
        aspects = params.get("aspects", [goal])

        steps: list[WorkflowStep] = []
        retrieval_ids: list[str] = []

        for i, aspect in enumerate(aspects):
            sid = f"retrieve_{i}"
            steps.append(WorkflowStep(
                step_id    = sid,
                agent_type = AgentType.RETRIEVAL,
                goal       = f"Investigate: {aspect}",
                params     = {"query": aspect, "topic": aspect},
                priority   = TaskPriority.HIGH,
            ))
            retrieval_ids.append(sid)

        steps.append(WorkflowStep(
            step_id    = "critic_0",
            agent_type = AgentType.CRITIC,
            goal       = f"Critique investigation evidence for: {goal}",
            params     = {"strategy": "investigation"},
            depends_on = retrieval_ids,
        ))
        steps.append(WorkflowStep(
            step_id    = "synthesis_0",
            agent_type = AgentType.SYNTHESIS,
            goal       = f"Synthesise investigation report for: {goal}",
            params     = {"strategy": "investigation", "goal": goal, "sub_topics": aspects},
            depends_on = ["critic_0"],
        ))
        return steps


class DocumentationWorkflow(WorkflowTemplate):
    """Generate documentation from indexed content."""

    @property
    def name(self) -> str:
        return "documentation"

    @property
    def description(self) -> str:
        return "Generate reference documentation from indexed documents"

    def generate(self, goal: str, params: dict) -> list[WorkflowStep]:
        topics = params.get("topics", [goal])

        steps: list[WorkflowStep] = []
        retrieval_ids: list[str] = []

        for i, topic in enumerate(topics):
            sid = f"retrieve_{i}"
            steps.append(WorkflowStep(
                step_id    = sid,
                agent_type = AgentType.RETRIEVAL,
                goal       = f"Gather documentation for: {topic}",
                params     = {"query": topic, "topic": topic},
            ))
            retrieval_ids.append(sid)

        steps.append(WorkflowStep(
            step_id    = "synthesis_0",
            agent_type = AgentType.SYNTHESIS,
            goal       = f"Generate documentation: {goal}",
            params     = {"strategy": "simple", "goal": goal, "sub_topics": topics},
            depends_on = retrieval_ids,
        ))
        return steps


class SummarizationWorkflow(WorkflowTemplate):
    """Multi-source summarization."""

    @property
    def name(self) -> str:
        return "summarization"

    @property
    def description(self) -> str:
        return "Summarize information from multiple sources"

    def generate(self, goal: str, params: dict) -> list[WorkflowStep]:
        steps: list[WorkflowStep] = [
            WorkflowStep(
                step_id    = "retrieve_0",
                agent_type = AgentType.RETRIEVAL,
                goal       = f"Retrieve content for summary: {goal}",
                params     = {"query": goal, "top_k": params.get("top_k", 10)},
                priority   = TaskPriority.HIGH,
            ),
            WorkflowStep(
                step_id    = "critic_0",
                agent_type = AgentType.CRITIC,
                goal       = f"Evaluate content quality for: {goal}",
                depends_on = ["retrieve_0"],
            ),
            WorkflowStep(
                step_id    = "synthesis_0",
                agent_type = AgentType.SYNTHESIS,
                goal       = f"Generate summary for: {goal}",
                params     = {"strategy": "simple", "goal": goal, "sub_topics": [goal]},
                depends_on = ["critic_0"],
            ),
        ]
        return steps


class TechEvalWorkflow(WorkflowTemplate):
    """Technology evaluation with pros/cons/recommendation."""

    @property
    def name(self) -> str:
        return "tech_evaluation"

    @property
    def description(self) -> str:
        return "Evaluate a technology: pros, cons, use cases, recommendation"

    def generate(self, goal: str, params: dict) -> list[WorkflowStep]:
        tech = params.get("technology", goal)
        aspects = [
            f"{tech} features capabilities",
            f"{tech} limitations drawbacks",
            f"{tech} performance scalability",
            f"{tech} use cases examples",
        ]

        steps: list[WorkflowStep] = []
        retrieval_ids: list[str] = []

        for i, aspect in enumerate(aspects):
            sid = f"retrieve_{i}"
            steps.append(WorkflowStep(
                step_id    = sid,
                agent_type = AgentType.RETRIEVAL,
                goal       = f"Research: {aspect}",
                params     = {"query": aspect, "topic": aspect},
                priority   = TaskPriority.HIGH,
            ))
            retrieval_ids.append(sid)

        steps.append(WorkflowStep(
            step_id    = "critic_0",
            agent_type = AgentType.CRITIC,
            goal       = f"Evaluate evidence quality for {tech} assessment",
            depends_on = retrieval_ids,
        ))
        steps.append(WorkflowStep(
            step_id    = "citval_0",
            agent_type = AgentType.CITATION_VALIDATOR,
            goal       = f"Validate citations for {tech} evaluation",
            depends_on = ["critic_0"],
            optional   = True,
        ))
        steps.append(WorkflowStep(
            step_id    = "synthesis_0",
            agent_type = AgentType.SYNTHESIS,
            goal       = f"Generate technology evaluation report for: {tech}",
            params     = {"strategy": "comparison", "goal": goal, "sub_topics": aspects},
            depends_on = ["citval_0"],
        ))
        return steps


class RootCauseWorkflow(WorkflowTemplate):
    """Root cause analysis / diagnostic investigation."""

    @property
    def name(self) -> str:
        return "root_cause"

    @property
    def description(self) -> str:
        return "Diagnostic investigation: identify root cause with evidence"

    def generate(self, goal: str, params: dict) -> list[WorkflowStep]:
        hypotheses = params.get("hypotheses", [goal])

        steps: list[WorkflowStep] = []
        retrieval_ids: list[str] = []

        for i, hyp in enumerate(hypotheses):
            sid = f"retrieve_{i}"
            steps.append(WorkflowStep(
                step_id    = sid,
                agent_type = AgentType.RETRIEVAL,
                goal       = f"Test hypothesis: {hyp}",
                params     = {"query": hyp, "topic": hyp},
                priority   = TaskPriority.HIGH,
            ))
            retrieval_ids.append(sid)

        steps.append(WorkflowStep(
            step_id    = "critic_0",
            agent_type = AgentType.CRITIC,
            goal       = f"Evaluate hypothesis evidence for: {goal}",
            params     = {"strategy": "investigation"},
            depends_on = retrieval_ids,
        ))
        steps.append(WorkflowStep(
            step_id    = "synthesis_0",
            agent_type = AgentType.SYNTHESIS,
            goal       = f"Root cause analysis report for: {goal}",
            params     = {"strategy": "investigation", "goal": goal, "sub_topics": hypotheses},
            depends_on = ["critic_0"],
        ))
        return steps


# ── Registry ──────────────────────────────────────────────────────────────────

_WORKFLOW_REGISTRY: dict[str, WorkflowTemplate] | None = None

def get_workflow_registry() -> dict[str, WorkflowTemplate]:
    global _WORKFLOW_REGISTRY
    if _WORKFLOW_REGISTRY is None:
        templates = [
            ComparisonWorkflow(),
            InvestigationWorkflow(),
            DocumentationWorkflow(),
            SummarizationWorkflow(),
            TechEvalWorkflow(),
            RootCauseWorkflow(),
        ]
        _WORKFLOW_REGISTRY = {t.name: t for t in templates}
    return _WORKFLOW_REGISTRY
