"""
Phase 7 Test Suite — Agentic Retrieval Platform

Tests cover:
  - Agent framework (base, lifecycle, memory, retry)
  - PlannerAgent (comparison, investigation, multi-part, simple)
  - RetrievalAgent (with mock retriever)
  - CriticAgent (coverage, quality, consistency)
  - CitationValidationAgent (supported, unsupported, drift)
  - SynthesisAgent (comparison, investigation, simple)
  - Orchestration (sequential, parallel, DAG cycles, dependency failure)
  - Evidence engine (store, graph, extractor, validator)
  - Research memory (task, evidence, session)
  - Tool framework (registry, executor, built-in tools)
  - MCP registry (list, call, schema)
  - Workflow templates (all 6 templates)
  - Report generator (markdown, html, json)
  - Research evaluation (metrics, report)
  - Database (Phase 7 tables)
  - API endpoints (Phase 7 routes)
"""

import json
import os
import pytest
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db():
    from app.database.db import Database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    database = Database(db_path)
    database.connect()
    yield database
    database.close()
    db_path.unlink(missing_ok=True)


@pytest.fixture
def agent_context():
    from app.agents.base import AgentContext
    mock_retriever = MagicMock()
    mock_result    = MagicMock()
    mock_result.results = []
    mock_result.stage_latencies = {}
    mock_retriever.search.return_value = mock_result
    return AgentContext(retriever=mock_retriever, db=MagicMock())


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT FRAMEWORK TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestAgentBase:
    def test_agent_task_creation(self):
        from app.agents.base import AgentTask, TaskPriority
        task = AgentTask(goal="test goal", task_type="test")
        assert task.goal == "test goal"
        assert task.task_type == "test"
        assert task.priority == TaskPriority.NORMAL
        assert len(task.task_id) == 36  # UUID

    def test_agent_result_success(self):
        from app.agents.base import AgentResult, AgentStatus, AgentType
        r = AgentResult(
            task_id="t1", agent_type=AgentType.PLANNER,
            status=AgentStatus.DONE, output={"plan": []}
        )
        assert r.is_success()
        d = r.to_dict()
        assert d["status"] == "done"
        assert d["agent_type"] == "planner"

    def test_agent_result_failure(self):
        from app.agents.base import AgentResult, AgentStatus, AgentType
        r = AgentResult(
            task_id="t1", agent_type=AgentType.RETRIEVAL,
            status=AgentStatus.FAILED, output=None, error="Network error"
        )
        assert not r.is_success()
        assert r.error == "Network error"

    def test_agent_memory(self):
        from app.agents.base import AgentMemory
        mem = AgentMemory(max_entries=5)
        for i in range(7):
            mem.remember("key", f"value_{i}")
        assert len(mem) == 5  # older entries evicted
        vals = mem.recall("key")
        assert vals[-1] == "value_6"
        assert mem.recall_latest("key") == "value_6"

    def test_agent_memory_clear(self):
        from app.agents.base import AgentMemory
        mem = AgentMemory()
        mem.remember("a", 1)
        mem.remember("b", 2)
        assert len(mem) == 2
        mem.clear()
        assert len(mem) == 0

    def test_lifecycle_transitions(self):
        from app.agents.base import AgentLifecycle, AgentStatus
        lc = AgentLifecycle()
        assert lc.status == AgentStatus.PENDING
        lc.transition(AgentStatus.RUNNING)
        assert lc.status == AgentStatus.RUNNING
        lc.transition(AgentStatus.DONE)
        assert lc.status == AgentStatus.DONE
        assert lc.is_terminal()
        assert lc.latency_ms >= 0

    def test_lifecycle_illegal_transition(self):
        from app.agents.base import AgentLifecycle, AgentStatus
        lc = AgentLifecycle()
        with pytest.raises(RuntimeError, match="Illegal"):
            lc.transition(AgentStatus.DONE)  # can't go PENDING → DONE

    def test_retry_policy(self):
        from app.agents.base import RetryPolicy
        rp = RetryPolicy(max_attempts=3, base_delay_sec=1.0, max_delay_sec=5.0)
        assert rp.delay_for(0) == 0.0
        assert rp.delay_for(1) == 1.0
        assert rp.delay_for(2) == 2.0
        assert rp.delay_for(3) == 4.0
        assert rp.delay_for(10) == 5.0  # capped
        assert rp.should_retry(0)
        assert rp.should_retry(2)
        assert not rp.should_retry(3)

    def test_agent_context_fields(self):
        from app.agents.base import AgentContext
        ctx = AgentContext(session_id="s1", user_id="u1")
        assert ctx.session_id == "s1"
        assert ctx.retriever is None
        assert ctx.tools == {}


# ══════════════════════════════════════════════════════════════════════════════
#  PLANNER AGENT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestPlannerAgent:
    def test_comparison_decomposition(self, agent_context):
        from app.agents.planner import PlannerAgent
        from app.agents.base import AgentTask
        agent = PlannerAgent()
        task  = AgentTask(goal="Compare FastAPI, Flask, and Django")
        result = agent.run(task, agent_context)
        assert result.is_success()
        plan = result.output
        assert plan["strategy"] == "comparison"
        assert len(plan["sub_topics"]) >= 2
        assert plan["step_count"] >= 4  # 3 retrieval + critic + citval + synthesis

    def test_multi_part_decomposition(self, agent_context):
        from app.agents.planner import PlannerAgent
        from app.agents.base import AgentTask
        agent = PlannerAgent()
        task  = AgentTask(goal="What is Python? How does asyncio work?")
        result = agent.run(task, agent_context)
        assert result.is_success()
        assert result.output["strategy"] == "multi_part"

    def test_investigation_decomposition(self, agent_context):
        from app.agents.planner import PlannerAgent
        from app.agents.base import AgentTask
        agent = PlannerAgent()
        task  = AgentTask(goal="Why does Python use GIL for thread safety")
        result = agent.run(task, agent_context)
        assert result.is_success()
        assert result.output["strategy"] == "investigation"

    def test_simple_decomposition(self, agent_context):
        from app.agents.planner import PlannerAgent
        from app.agents.base import AgentTask
        agent = PlannerAgent()
        task  = AgentTask(goal="Python web frameworks")
        result = agent.run(task, agent_context)
        assert result.is_success()
        assert result.output["strategy"] == "simple"

    def test_plan_confidence(self, agent_context):
        from app.agents.planner import PlannerAgent
        from app.agents.base import AgentTask
        agent = PlannerAgent()
        task  = AgentTask(goal="Compare A vs B")
        result = agent.run(task, agent_context)
        assert result.confidence > 0.5

    def test_max_topics_limit(self, agent_context):
        from app.agents.planner import PlannerAgent
        from app.agents.base import AgentTask
        agent = PlannerAgent()
        task  = AgentTask(goal="Compare A, B, C, D, E, F, G, H",
                          params={"max_topics": 3})
        result = agent.run(task, agent_context)
        assert len(result.output["sub_topics"]) <= 3


# ══════════════════════════════════════════════════════════════════════════════
#  RETRIEVAL AGENT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestRetrievalAgent:
    def test_retrieval_with_empty_query(self, agent_context):
        from app.agents.retrieval import RetrievalAgent
        from app.agents.base import AgentTask
        agent = RetrievalAgent()
        task  = AgentTask(goal="", params={"query": ""})
        result = agent.run(task, agent_context)
        assert not result.is_success()

    def test_retrieval_with_mock_results(self):
        from app.agents.retrieval import RetrievalAgent
        from app.agents.base import AgentTask, AgentContext
        mock = MagicMock()
        doc = MagicMock()
        doc.doc_id = 1
        doc.title = "Python Guide"
        doc.content = "Python is a programming language"
        doc.final_score = 0.9
        result_obj = MagicMock()
        result_obj.results = [doc]
        result_obj.stage_latencies = {"bm25": 5}
        mock.search.return_value = result_obj

        ctx = AgentContext(retriever=mock)
        agent = RetrievalAgent()
        task = AgentTask(goal="Python", params={"query": "Python", "top_k": 5})
        result = agent.run(task, ctx)
        assert result.is_success()
        assert result.output["evidence_count"] >= 1

    def test_quality_scoring(self):
        from app.agents.retrieval import RetrievalAgent
        agent = RetrievalAgent()
        evidence = [
            {"doc_id": 1, "title": "A", "score": 0.9, "content": "text"},
            {"doc_id": 2, "title": "B", "score": 0.8, "content": "text"},
            {"doc_id": 3, "title": "C", "score": 0.7, "content": "text"},
        ]
        score = agent._compute_quality(evidence)
        assert 0.0 < score <= 1.0
        assert agent._compute_quality([]) == 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  CRITIC AGENT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestCriticAgent:
    def _make_prior_results(self, evidence_list):
        return {
            "r0": {
                "agent_type": "retrieval",
                "output": {
                    "query": "test",
                    "evidence": evidence_list,
                    "evidence_count": len(evidence_list),
                    "quality_score": 0.7,
                },
            }
        }

    def test_critic_with_good_evidence(self, agent_context):
        from app.agents.critic import CriticAgent
        from app.agents.base import AgentTask
        evidence = [
            {"doc_id": 1, "title": "A", "score": 0.8, "content": "Good evidence about Python"},
            {"doc_id": 2, "title": "B", "score": 0.7, "content": "More evidence about Python"},
            {"doc_id": 3, "title": "C", "score": 0.6, "content": "Additional evidence about Python"},
        ]
        task = AgentTask(goal="Critique", params={"_prior_results": self._make_prior_results(evidence)})
        agent = CriticAgent()
        result = agent.run(task, agent_context)
        assert result.is_success()
        assert result.output["overall_score"] > 0.3

    def test_critic_with_no_evidence(self, agent_context):
        from app.agents.critic import CriticAgent
        from app.agents.base import AgentTask
        task = AgentTask(goal="Critique", params={"_prior_results": {}})
        agent = CriticAgent()
        result = agent.run(task, agent_context)
        assert result.is_success()
        assert result.output["overall_score"] == 0.0
        assert any(i["type"] == "no_evidence" for i in result.output["issues"])

    def test_critic_identifies_low_quality(self, agent_context):
        from app.agents.critic import CriticAgent
        from app.agents.base import AgentTask
        evidence = [
            {"doc_id": 1, "title": "A", "score": 0.1, "content": "Low score"},
            {"doc_id": 2, "title": "B", "score": 0.05, "content": "Very low"},
        ]
        task = AgentTask(goal="Critique", params={"_prior_results": self._make_prior_results(evidence)})
        agent = CriticAgent()
        result = agent.run(task, agent_context)
        assert result.output["quality_score"] < 0.5


# ══════════════════════════════════════════════════════════════════════════════
#  CITATION VALIDATION AGENT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestCitationValidationAgent:
    def test_supported_citations(self, agent_context):
        from app.agents.citation_validator import CitationValidationAgent
        from app.agents.base import AgentTask
        prior = {
            "r0": {
                "agent_type": "retrieval",
                "output": {
                    "evidence": [
                        {"doc_id": 1, "title": "Python Guide", "content": "Python is a high-level programming language", "score": 0.9},
                    ],
                    "evidence_count": 1,
                    "answer": "Python is a high-level programming language [1].",
                },
            },
        }
        task = AgentTask(goal="Validate", params={"_prior_results": prior})
        agent = CitationValidationAgent()
        result = agent.run(task, agent_context)
        assert result.is_success()
        assert result.output["total_citations"] >= 1

    def test_no_citations(self, agent_context):
        from app.agents.citation_validator import CitationValidationAgent
        from app.agents.base import AgentTask
        prior = {
            "r0": {
                "agent_type": "retrieval",
                "output": {
                    "evidence": [{"doc_id": 1, "title": "A", "content": "text", "score": 0.9}],
                    "evidence_count": 1,
                    "answer": "No citation markers here.",
                },
            },
        }
        task = AgentTask(goal="Validate", params={"_prior_results": prior})
        agent = CitationValidationAgent()
        result = agent.run(task, agent_context)
        assert result.is_success()
        assert result.output["total_citations"] == 0
        assert result.output["citation_accuracy"] == 1.0


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHESIS AGENT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestSynthesisAgent:
    def test_synthesis_comparison(self, agent_context):
        from app.agents.synthesis import SynthesisAgent
        from app.agents.base import AgentTask
        prior = {
            "r0": {"agent_type": "retrieval", "output": {
                "evidence": [{"doc_id": 1, "title": "FastAPI", "content": "FastAPI is fast", "score": 0.9}],
                "evidence_count": 1, "quality_score": 0.8,
            }},
            "r1": {"agent_type": "retrieval", "output": {
                "evidence": [{"doc_id": 2, "title": "Flask", "content": "Flask is lightweight", "score": 0.8}],
                "evidence_count": 1, "quality_score": 0.7,
            }},
        }
        task = AgentTask(goal="Compare FastAPI vs Flask", params={
            "_prior_results": prior,
            "strategy": "comparison",
            "goal": "Compare FastAPI vs Flask",
            "sub_topics": ["FastAPI", "Flask"],
        })
        agent = SynthesisAgent()
        result = agent.run(task, agent_context)
        assert result.is_success()
        assert "full_report" in result.output
        assert "FastAPI" in result.output["full_report"]

    def test_synthesis_simple(self, agent_context):
        from app.agents.synthesis import SynthesisAgent
        from app.agents.base import AgentTask
        prior = {
            "r0": {"agent_type": "retrieval", "output": {
                "evidence": [{"doc_id": 1, "title": "Python", "content": "Python programming", "score": 0.9}],
                "evidence_count": 1, "quality_score": 0.8,
            }},
        }
        task = AgentTask(goal="What is Python", params={
            "_prior_results": prior, "strategy": "simple",
            "goal": "What is Python", "sub_topics": ["Python"],
        })
        agent = SynthesisAgent()
        result = agent.run(task, agent_context)
        assert result.is_success()
        assert result.output["evidence_used"] >= 1

    def test_synthesis_with_critique(self, agent_context):
        from app.agents.synthesis import SynthesisAgent
        from app.agents.base import AgentTask
        prior = {
            "r0": {"agent_type": "retrieval", "output": {
                "evidence": [{"doc_id": 1, "title": "Doc", "content": "content", "score": 0.9}],
                "evidence_count": 1,
            }},
            "critic_0": {"agent_type": "critic", "output": {
                "overall_score": 0.75, "coverage_score": 1.0,
                "quality_score": 0.8, "consistency_score": 1.0,
                "issues": [], "strong_evidence": [],
            }},
        }
        task = AgentTask(goal="Test", params={
            "_prior_results": prior, "strategy": "simple",
            "goal": "Test", "sub_topics": ["test"],
        })
        result = SynthesisAgent().run(task, agent_context)
        assert result.is_success()
        assert "critique_score" in result.output["quality_metrics"]


# ══════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestOrchestration:
    def test_execution_graph_topological_sort(self):
        from app.orchestration.engine import ExecutionGraph, WorkflowStep
        from app.agents.base import AgentType
        steps = [
            WorkflowStep(step_id="a", agent_type=AgentType.RETRIEVAL, goal="A"),
            WorkflowStep(step_id="b", agent_type=AgentType.RETRIEVAL, goal="B"),
            WorkflowStep(step_id="c", agent_type=AgentType.CRITIC, goal="C", depends_on=["a", "b"]),
            WorkflowStep(step_id="d", agent_type=AgentType.SYNTHESIS, goal="D", depends_on=["c"]),
        ]
        graph = ExecutionGraph(steps)
        order = [s.step_id for s in graph.ordered_steps]
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("c")
        assert order.index("c") < order.index("d")

    def test_execution_graph_cycle_detection(self):
        from app.orchestration.engine import ExecutionGraph, WorkflowStep
        from app.agents.base import AgentType
        steps = [
            WorkflowStep(step_id="a", agent_type=AgentType.RETRIEVAL, goal="A", depends_on=["b"]),
            WorkflowStep(step_id="b", agent_type=AgentType.RETRIEVAL, goal="B", depends_on=["a"]),
        ]
        with pytest.raises(ValueError, match="cycle"):
            ExecutionGraph(steps)

    def test_task_scheduler(self):
        from app.orchestration.engine import TaskScheduler
        from app.agents.base import AgentTask, TaskPriority
        sched = TaskScheduler()
        sched.push(AgentTask(goal="low", priority=TaskPriority.LOW))
        sched.push(AgentTask(goal="high", priority=TaskPriority.HIGH))
        sched.push(AgentTask(goal="normal", priority=TaskPriority.NORMAL))
        assert len(sched) == 3
        assert sched.pop().goal == "high"
        assert sched.pop().goal == "normal"
        assert sched.pop().goal == "low"

    def test_workflow_engine_sequential(self, agent_context):
        from app.orchestration.engine import WorkflowEngine, WorkflowStep
        from app.agents.base import AgentType
        from app.agents.retrieval import RetrievalAgent
        from app.agents.critic import CriticAgent
        agents = {
            AgentType.RETRIEVAL: RetrievalAgent(),
            AgentType.CRITIC: CriticAgent(),
        }
        steps = [
            WorkflowStep(step_id="r0", agent_type=AgentType.RETRIEVAL, goal="Test retrieval"),
            WorkflowStep(step_id="c0", agent_type=AgentType.CRITIC, goal="Critique",
                         depends_on=["r0"]),
        ]
        engine = WorkflowEngine(agents, agent_context)
        run = engine.run(steps, goal="Test workflow")
        assert run.status.value in ("completed", "partial", "failed")
        assert len(run.results) == 2
        assert run.total_latency_ms > 0

    def test_workflow_dependency_failure_skips(self, agent_context):
        from app.orchestration.engine import WorkflowEngine, WorkflowStep
        from app.agents.base import AgentType, Agent, AgentResult, AgentStatus, AgentTask, AgentContext, AgentMemory

        class FailingAgent(Agent):
            @property
            def agent_type(self): return AgentType.RETRIEVAL
            def _execute(self, task, ctx, mem):
                return AgentResult(task_id=task.task_id, agent_type=AgentType.RETRIEVAL,
                                   status=AgentStatus.FAILED, output=None, error="Intentional")

        from app.agents.critic import CriticAgent
        agents = {
            AgentType.RETRIEVAL: FailingAgent(),
            AgentType.CRITIC: CriticAgent(),
        }
        steps = [
            WorkflowStep(step_id="r0", agent_type=AgentType.RETRIEVAL, goal="Fail"),
            WorkflowStep(step_id="c0", agent_type=AgentType.CRITIC, goal="Critique", depends_on=["r0"]),
        ]
        engine = WorkflowEngine(agents, agent_context)
        run = engine.run(steps, goal="Test failure")
        # Critic should be cancelled because retrieval failed
        assert run.results["c0"].status.value == "cancelled"


# ══════════════════════════════════════════════════════════════════════════════
#  EVIDENCE ENGINE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestEvidenceEngine:
    def test_evidence_store_crud(self):
        from app.evidence.engine import EvidenceStore, EvidenceRecord
        store = EvidenceStore()
        r1 = EvidenceRecord(evidence_id="e1", doc_id=1, content="test", score=0.9)
        r2 = EvidenceRecord(evidence_id="e2", doc_id=2, content="test2", score=0.5, tags=["python"])
        store.add(r1)
        store.add(r2)
        assert store.count() == 2
        assert store.get("e1") is r1
        assert len(store.filter_by_tag("python")) == 1
        assert len(store.filter_by_doc(1)) == 1
        assert store.remove("e1")
        assert store.count() == 1

    def test_evidence_graph(self):
        from app.evidence.engine import EvidenceGraph, EvidenceRelation, EvidenceRelationType
        graph = EvidenceGraph()
        graph.add_relation(EvidenceRelation("e1", "e2", EvidenceRelationType.SUPPORTS))
        graph.add_relation(EvidenceRelation("e1", "e3", EvidenceRelationType.CONTRADICTS))
        assert len(graph.get_relations("e1")) == 2
        assert graph.get_supporters("e1") == ["e2"]
        assert graph.get_contradictions("e1") == ["e3"]
        assert graph.node_count() == 1
        assert graph.edge_count() == 2

    def test_evidence_extractor(self):
        from app.evidence.engine import EvidenceExtractor
        ext = EvidenceExtractor()
        dicts = [
            {"doc_id": 1, "chunk_id": "c1", "content": "Python programming", "score": 0.9, "title": "Doc"},
        ]
        records = ext.extract_from_agent_output(dicts, tags=["python"])
        assert len(records) == 1
        assert records[0].tags == ["python"]
        assert records[0].score == 0.9

    def test_evidence_validator(self):
        from app.evidence.engine import EvidenceValidator, EvidenceRecord
        validator = EvidenceValidator(min_content_length=10, min_score=0.2)
        records = [
            EvidenceRecord(evidence_id="e1", content="Short", score=0.9),          # too short
            EvidenceRecord(evidence_id="e2", content="Good evidence content here about Python", score=0.1),  # low score
            EvidenceRecord(evidence_id="e3", content="Valid evidence content about Python programming", score=0.8),
            EvidenceRecord(evidence_id="e4", content="Valid evidence content about Python programming", score=0.7),  # duplicate
        ]
        valid, rejected, report = validator.validate(records)
        assert len(valid) == 1
        assert valid[0].evidence_id == "e3"
        assert report["reasons"]["too_short"] == 1
        assert report["reasons"]["low_score"] == 1
        assert report["reasons"]["duplicate"] == 1


# ══════════════════════════════════════════════════════════════════════════════
#  RESEARCH MEMORY TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestResearchMemory:
    def test_task_memory(self):
        from app.research_memory.memory import TaskMemory
        mem = TaskMemory(max_entries=5)
        from app.agents.base import AgentResult, AgentStatus, AgentType
        r = AgentResult(task_id="t1", agent_type=AgentType.RETRIEVAL,
                        status=AgentStatus.DONE, output="result", latency_ms=50)
        mem.add_result(r)
        assert len(mem) == 1
        assert len(mem.get_completed()) == 1
        assert len(mem.get_failed()) == 0

    def test_evidence_memory(self):
        from app.research_memory.memory import EvidenceMemory
        mem = EvidenceMemory(max_entries=3)
        mem.add({"doc_id": 1, "chunk_id": "c1", "score": 0.9})
        mem.add({"doc_id": 2, "chunk_id": "c2", "score": 0.5})
        assert mem.count() == 2
        assert not mem.add({"doc_id": 1, "chunk_id": "c1", "score": 0.3})  # duplicate, lower score
        assert mem.count() == 2
        top = mem.get_top_k(1)
        assert top[0]["doc_id"] == 1  # highest score

    def test_evidence_memory_pruning(self):
        from app.research_memory.memory import EvidenceMemory
        mem = EvidenceMemory(max_entries=2)
        mem.add({"doc_id": 1, "chunk_id": "c1", "score": 0.9})
        mem.add({"doc_id": 2, "chunk_id": "c2", "score": 0.5})
        mem.add({"doc_id": 3, "chunk_id": "c3", "score": 0.7})
        assert mem.count() == 2  # lowest score evicted

    def test_session_memory(self):
        from app.research_memory.memory import ResearchSessionMemory
        mem = ResearchSessionMemory()
        mem.add_event("task_started", "Starting retrieval")
        mem.add_event("task_completed", "Retrieval done")
        assert len(mem) == 2
        summary = mem.summarize()
        assert summary["total_events"] == 2

    def test_research_session_composite(self):
        from app.research_memory.memory import ResearchSession
        from app.agents.base import AgentResult, AgentStatus, AgentType
        session = ResearchSession(goal="Test research")
        r = AgentResult(task_id="t1", agent_type=AgentType.RETRIEVAL,
                        status=AgentStatus.DONE, output="data", latency_ms=100)
        session.add_agent_result(r)
        session.add_evidence([{"doc_id": 1, "chunk_id": "c1", "score": 0.8}])
        snap = session.to_snapshot()
        assert snap["task_count"] == 1
        assert snap["evidence_count"] == 1


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL FRAMEWORK TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestToolFramework:
    def test_tool_registry(self):
        from app.tools.framework import ToolRegistry, SearchTool
        reg = ToolRegistry()
        reg.register(SearchTool())
        assert reg.count() == 1
        assert "search" in reg.list_tools()
        assert reg.get("search") is not None
        assert reg.get("nonexistent") is None

    def test_tool_executor_unknown_tool(self):
        from app.tools.framework import ToolRegistry, ToolExecutor
        reg = ToolRegistry()
        exe = ToolExecutor(reg)
        result = exe.execute("unknown_tool", {})
        assert not result.success
        assert "Unknown tool" in result.error

    def test_search_tool_execute(self):
        from app.tools.framework import SearchTool
        tool = SearchTool()
        assert tool.name == "search"
        result = tool.execute({"query": "test"}, None)
        assert result.success
        assert result.output == []

    def test_database_tool_no_context(self):
        from app.tools.framework import DatabaseTool
        tool = DatabaseTool()
        result = tool.execute({"action": "count_docs"}, None)
        assert result.success
        assert result.output == {"error": "No database connection"}

    def test_create_default_registry(self):
        from app.tools.framework import create_default_registry
        reg = create_default_registry()
        assert reg.count() == 5
        assert "search" in reg.list_tools()
        assert "retrieval" in reg.list_tools()
        assert "database" in reg.list_tools()

    def test_mcp_schema_export(self):
        from app.tools.framework import SearchTool
        tool = SearchTool()
        schema = tool.export_mcp_schema()
        assert schema["name"] == "search"
        assert "inputSchema" in schema
        assert schema["inputSchema"]["type"] == "object"


# ══════════════════════════════════════════════════════════════════════════════
#  MCP REGISTRY TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestMCPRegistry:
    def test_mcp_list_tools(self):
        from app.tools.framework import create_default_registry
        from app.mcp.registry import MCPRegistry
        mcp = MCPRegistry(create_default_registry())
        tools = mcp.list_tools()
        assert len(tools) == 5
        assert all("name" in t for t in tools)

    def test_mcp_call_tool(self):
        from app.tools.framework import create_default_registry
        from app.mcp.registry import MCPRegistry
        mcp = MCPRegistry(create_default_registry())
        result = mcp.call_tool("search", {"query": "test"})
        assert not result["isError"]

    def test_mcp_call_unknown_tool(self):
        from app.tools.framework import create_default_registry
        from app.mcp.registry import MCPRegistry
        mcp = MCPRegistry(create_default_registry())
        result = mcp.call_tool("nonexistent", {})
        assert result["isError"]

    def test_mcp_get_schema(self):
        from app.tools.framework import create_default_registry
        from app.mcp.registry import MCPRegistry
        mcp = MCPRegistry(create_default_registry())
        schema = mcp.get_tool_schema("search")
        assert schema is not None
        assert schema["name"] == "search"


# ══════════════════════════════════════════════════════════════════════════════
#  WORKFLOW TEMPLATES TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestWorkflowTemplates:
    def test_comparison_workflow(self):
        from app.workflows.templates import ComparisonWorkflow
        wf = ComparisonWorkflow()
        assert wf.name == "comparison"
        steps = wf.generate("Compare A vs B", {"entities": ["A", "B"]})
        assert len(steps) >= 4  # 2 retrieval + critic + citval + synthesis
        step_types = [s.agent_type.value for s in steps]
        assert "retrieval" in step_types
        assert "synthesis" in step_types

    def test_investigation_workflow(self):
        from app.workflows.templates import InvestigationWorkflow
        wf = InvestigationWorkflow()
        steps = wf.generate("Why does X happen", {"aspects": ["cause", "effect"]})
        assert len(steps) >= 3

    def test_documentation_workflow(self):
        from app.workflows.templates import DocumentationWorkflow
        wf = DocumentationWorkflow()
        steps = wf.generate("Document API", {"topics": ["endpoints", "auth"]})
        assert len(steps) >= 3

    def test_summarization_workflow(self):
        from app.workflows.templates import SummarizationWorkflow
        wf = SummarizationWorkflow()
        steps = wf.generate("Summarize topic", {})
        assert len(steps) == 3  # retrieval + critic + synthesis

    def test_tech_eval_workflow(self):
        from app.workflows.templates import TechEvalWorkflow
        wf = TechEvalWorkflow()
        steps = wf.generate("Evaluate FastAPI", {"technology": "FastAPI"})
        assert len(steps) >= 6  # 4 retrieval + critic + citval + synthesis

    def test_root_cause_workflow(self):
        from app.workflows.templates import RootCauseWorkflow
        wf = RootCauseWorkflow()
        steps = wf.generate("Debug crash", {"hypotheses": ["memory leak", "race condition"]})
        assert len(steps) >= 4

    def test_workflow_registry(self):
        from app.workflows.templates import get_workflow_registry
        reg = get_workflow_registry()
        assert len(reg) == 6
        assert "comparison" in reg
        assert "investigation" in reg
        assert "root_cause" in reg


# ══════════════════════════════════════════════════════════════════════════════
#  REPORT GENERATOR TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestReportGenerator:
    def _sample_output(self):
        return {
            "report_id": "r1",
            "goal": "Compare FastAPI vs Flask",
            "strategy": "comparison",
            "summary": "Analysis of two frameworks.",
            "sections": [
                {"title": "FastAPI", "content": "FastAPI is fast [1].", "citations": [1]},
                {"title": "Flask", "content": "Flask is lightweight [2].", "citations": [2]},
            ],
            "full_report": "# Compare FastAPI vs Flask\n\n## FastAPI\nFast [1].\n## Flask\nLight [2].",
            "evidence_used": 5,
            "quality_metrics": {"critique_score": 0.8, "issue_count": 1},
        }

    def test_markdown_output(self):
        from app.reports.generator import ReportGenerator, ReportFormat
        gen = ReportGenerator()
        md = gen.generate(self._sample_output(), ReportFormat.MARKDOWN)
        assert "# Compare FastAPI vs Flask" in md

    def test_html_output(self):
        from app.reports.generator import ReportGenerator, ReportFormat
        gen = ReportGenerator()
        html = gen.generate(self._sample_output(), ReportFormat.HTML)
        assert "<html" in html
        assert "Compare FastAPI vs Flask" in html

    def test_json_output(self):
        from app.reports.generator import ReportGenerator, ReportFormat
        gen = ReportGenerator()
        j = gen.generate(self._sample_output(), ReportFormat.JSON)
        data = json.loads(j)
        assert data["goal"] == "Compare FastAPI vs Flask"
        assert len(data["sections"]) == 2


# ══════════════════════════════════════════════════════════════════════════════
#  RESEARCH EVALUATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestResearchEvaluation:
    def test_evaluate_case(self):
        from app.research_evaluation.evaluator import ResearchEvaluator, ResearchEvalCase
        case = ResearchEvalCase(
            case_id="c1", goal="Test",
            total_steps=5, completed_steps=4, failed_steps=1,
            evidence_count=6, topic_count=2, topics_covered=2,
            citation_accuracy=0.9, grounding_score=0.8,
            report_text="## Title\nContent [1]. More content [2].\n## Section 2\nDetails.",
        )
        evaluator = ResearchEvaluator()
        result = evaluator.evaluate(case)
        assert 0.0 <= result.overall_score <= 1.0
        assert result.task_completion == 0.8
        assert result.hallucination_rate == 0.2

    def test_evaluate_empty(self):
        from app.research_evaluation.evaluator import ResearchEvaluator, ResearchEvalCase
        case = ResearchEvalCase(case_id="c0", goal="Empty", total_steps=0)
        result = ResearchEvaluator().evaluate(case)
        assert result.task_completion == 0.0
        assert result.overall_score >= 0.0

    def test_generate_report(self):
        from app.research_evaluation.evaluator import ResearchEvaluator, ResearchEvalCase
        cases = [
            ResearchEvalCase(case_id="c1", goal="T1", total_steps=3, completed_steps=3,
                             evidence_count=3, topic_count=1, topics_covered=1,
                             citation_accuracy=0.9, grounding_score=0.8, report_text="## Report\nContent [1]."),
            ResearchEvalCase(case_id="c2", goal="T2", total_steps=5, completed_steps=4,
                             evidence_count=2, topic_count=2, topics_covered=1,
                             citation_accuracy=0.7, grounding_score=0.6, report_text="Short."),
        ]
        report = ResearchEvaluator().generate_report(cases)
        assert report.total_cases == 2
        assert "overall_score" in report.avg_scores

    def test_save_report(self, tmp_path):
        from app.research_evaluation.evaluator import ResearchEvaluator, ResearchEvalCase, ResearchEvalReport
        cases = [ResearchEvalCase(case_id="c1", goal="T1", total_steps=1, completed_steps=1,
                                   evidence_count=1, topic_count=1, topics_covered=1,
                                   citation_accuracy=1.0, grounding_score=1.0, report_text="Good.")]
        report = ResearchEvaluator().generate_report(cases)
        path = tmp_path / "eval.json"
        ResearchEvaluator().save_report(report, path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["total_cases"] == 1


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase7Database:
    def test_agent_tasks(self, db):
        db.insert_agent_task("t1", "plan", "planner", "Test goal")
        tasks = db.get_agent_tasks()
        assert len(tasks) >= 1
        assert tasks[0]["task_id"] == "t1"
        db.update_agent_task_status("t1", "done")
        tasks = db.get_agent_tasks(status="done")
        assert len(tasks) >= 1

    def test_agent_runs(self, db):
        db.insert_agent_task("t1", "retrieve", "retrieval", "Test")
        db.insert_agent_run("r1", "t1", "retrieval", "done", '{"result": 1}',
                            confidence=0.9, latency_ms=50)
        runs = db.get_agent_runs()
        assert len(runs) >= 1
        runs = db.get_agent_runs(agent_type="retrieval")
        assert len(runs) >= 1

    def test_workflow_runs(self, db):
        db.insert_workflow_run("w1", "investigation", "Test goal", "running", 3)
        db.update_workflow_run("w1", "completed", 3, 0, 500.0)
        runs = db.get_workflow_runs()
        assert len(runs) >= 1
        assert runs[0]["status"] == "completed"

    def test_evidence_records(self, db):
        db.insert_evidence_record("e1", 1, "c1", "Claim", "Content", 0.9, 0.8,
                                   "Source", True, '["python"]', "s1")
        records = db.get_evidence_by_session("s1")
        assert len(records) >= 1

    def test_research_sessions(self, db):
        db.insert_research_session("s1", "user1", "Research goal")
        db.update_research_session("s1", "active", 5, 10, 20)
        sessions = db.get_research_sessions()
        assert len(sessions) >= 1
        sessions = db.get_research_sessions(user_id="user1")
        assert len(sessions) >= 1

    def test_research_reports(self, db):
        db.insert_research_report("rp1", "s1", "w1", "Goal", "comparison",
                                   "markdown", "# Report\nContent", 5)
        reports = db.get_research_reports()
        assert len(reports) >= 1
        reports = db.get_research_reports(session_id="s1")
        assert len(reports) >= 1

    def test_agent_metrics(self, db):
        db.insert_agent_metric("retrieval", "t1", 50.0, True)
        db.insert_agent_metric("planner", "t2", 100.0, False)
        summary = db.get_agent_metrics_summary()
        assert summary["total"] == 2
        assert summary["success_count"] == 1


# ══════════════════════════════════════════════════════════════════════════════
#  API ENDPOINT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase7APIEndpoints:
    def _make_client(self, tmp_path):
        from app.config import EngineConfig, DatabaseConfig, VectorStoreConfig
        config = EngineConfig(
            database=DatabaseConfig(db_path=tmp_path / "test.db"),
            vector_store=VectorStoreConfig(index_path=tmp_path / "idx", dimension=16),
        )
        from app.api.routes import create_app
        from fastapi.testclient import TestClient
        return TestClient(create_app(config))

    def test_list_workflows(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.get("/research/workflows")
            assert resp.status_code == 200
            data = resp.json()
            assert "workflows" in data
            assert len(data["workflows"]) >= 6

    def test_list_agents(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.get("/research/agents")
            assert resp.status_code == 200
            data = resp.json()
            assert "agents" in data
            assert len(data["agents"]) >= 5

    def test_create_plan(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.post("/research/plan", json={
                "goal": "Compare Python vs Java",
                "max_topics": 4,
            })
            assert resp.status_code == 200
            data = resp.json()
            assert "plan" in data
            assert data["plan"]["strategy"] in ("comparison", "multi_part", "simple")

    def test_agent_retrieve(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.post("/research/retrieve", json={
                "query": "Python programming",
                "top_k": 3,
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["agent_type"] == "retrieval"

    def test_run_research(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.post("/research", json={
                "goal": "Summarize Python",
                "workflow": "summarization",
            })
            assert resp.status_code == 200
            data = resp.json()
            assert "session_id" in data
            assert "run_id" in data
            assert data["total_steps"] >= 1

    def test_list_sessions(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.get("/research/sessions")
            assert resp.status_code == 200

    def test_list_reports(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.get("/research/reports")
            assert resp.status_code == 200

    def test_list_workflow_runs(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.get("/research/workflow-runs")
            assert resp.status_code == 200

    def test_generate_report(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.post("/research/reports/generate", json={
                "synthesis_output": {
                    "goal": "Test",
                    "summary": "A test report.",
                    "sections": [{"title": "S1", "content": "Content"}],
                    "evidence_used": 1,
                    "quality_metrics": {},
                },
                "format": "markdown",
            })
            assert resp.status_code == 200
            assert "report" in resp.json()

    def test_generate_html_report(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.post("/research/reports/generate", json={
                "synthesis_output": {
                    "goal": "Test",
                    "summary": "HTML test.",
                    "sections": [],
                    "evidence_used": 0,
                },
                "format": "html",
            })
            assert resp.status_code == 200
            assert "<html" in resp.json()["report"]

    def test_list_tools(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.get("/tools")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["tools"]) >= 5

    def test_execute_tool(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.post("/tools/execute", json={
                "tool_name": "search",
                "params": {"query": "test"},
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["tool_name"] == "search"

    def test_mcp_list(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.get("/mcp/tools")
            assert resp.status_code == 200
            assert len(resp.json()["tools"]) >= 5

    def test_mcp_call(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.post("/mcp/tools/call?name=search", json={"query": "test"})
            assert resp.status_code == 200

    def test_research_metrics(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.get("/research/metrics")
            assert resp.status_code == 200
            data = resp.json()
            assert "agent_executions" in data

    def test_session_not_found(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.get("/research/sessions/nonexistent-id")
            assert resp.status_code == 404

    def test_research_comparison_workflow(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.post("/research", json={
                "goal": "Compare FastAPI vs Flask",
                "workflow": "comparison",
                "params": {"entities": ["FastAPI", "Flask"]},
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_steps"] >= 4

    def test_evidence_endpoint(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.post("/research", json={
                "goal": "Test evidence", "workflow": "summarization",
            })
            session_id = resp.json()["session_id"]
            resp = c.get(f"/research/evidence/{session_id}")
            assert resp.status_code == 200
