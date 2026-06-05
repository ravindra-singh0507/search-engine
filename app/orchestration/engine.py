"""
Agent Orchestration Engine — Phase 7

=== THEORY ===

The orchestration layer coordinates multiple agents across a directed
acyclic graph (DAG) of tasks.  This pattern is used in:

  LangGraph  — StateGraph with edges and conditional routing
  CrewAI     — Process.sequential | Process.hierarchical
  Prefect    — Flow + Task with dependency tracking
  Airflow    — DAG + Operator with upstream/downstream

=== ARCHITECTURE ===

  WorkflowStep  — one node in the execution graph (agent + task template)
  ExecutionGraph — DAG of WorkflowSteps with topological ordering
  TaskScheduler  — priority queue; dispatches ready tasks respecting deps
  AgentOrchestrator — single-step execution (one agent, one task)
  WorkflowEngine    — multi-step orchestration across the full DAG

=== EXECUTION MODEL ===

  Sequential:
    step_1 → step_2 → step_3
    Simple; no parallelism; each step waits for the prior to finish.

  Parallel (where deps allow):
    step_1 ─┬─ step_2 ─┐
            └─ step_3 ─┴─ step_4
    Steps 2 and 3 run concurrently; step 4 waits for both.
    Implemented via threading.Thread in this single-process version.

  DAG invariant:
    Before step_i runs, all steps in step_i.depends_on must be DONE.
    If any dependency FAILED, the dependent step is SKIPPED with an error.

=== DEPENDENCY TRACKING ===

  ExecutionGraph uses Kahn's topological sort algorithm:
    1. Build in-degree map for each node
    2. Enqueue all nodes with in_degree == 0
    3. For each dequeued node, process it, decrement dependents' in-degrees
    4. If any node remains (cycle), raise ValueError

  O(V + E) time, O(V) space.

=== RETRY POLICY ===

  WorkflowEngine applies the agent's retry policy automatically.
  Additional workflow-level retries can be configured via WorkflowConfig.

=== DATA STRUCTURES ===

  WorkflowStep.depends_on  — set[step_id]
  ExecutionGraph._adj      — dict[step_id → list[step_id]] (adjacency list)
  ExecutionGraph._order    — topological ordering (Kahn's)
  TaskScheduler._queue     — sorted list by priority (insertion sort)

=== COMPLEXITY ===

  ExecutionGraph.topological_order: O(V + E)
  TaskScheduler.push/pop:           O(log N) — heapq
  WorkflowEngine.run_sequential:    O(S) steps, each agent's own complexity

=== PRODUCTION EQUIVALENTS ===

  LangGraph:  StateGraph.compile() + invoke()
  CrewAI:     Crew.kickoff()
  Prefect:    flow() decorator + .serve()
  AutoGen:    GroupChatManager
"""

import heapq
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from app.agents.base import (
    Agent, AgentContext, AgentMemory, AgentResult, AgentStatus,
    AgentTask, AgentType, TaskPriority,
)

logger = logging.getLogger(__name__)


# ── Workflow data structures ──────────────────────────────────────────────────

class WorkflowStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    PARTIAL   = "partial"   # some steps succeeded, some failed


@dataclass
class WorkflowStep:
    """
    One node in the execution graph.

    step_id:    unique identifier within the workflow
    agent_type: which agent class handles this step
    goal:       task goal text
    params:     task params dict
    depends_on: step_ids that must complete before this step runs
    optional:   if True, failure of this step does not abort the workflow
    timeout_sec: per-step timeout
    """
    goal:        str
    agent_type:  AgentType
    step_id:     str                    = field(default_factory=lambda: str(uuid.uuid4())[:8])
    params:      dict[str, Any]         = field(default_factory=dict)
    depends_on:  list[str]              = field(default_factory=list)
    optional:    bool                   = False
    timeout_sec: float                  = 120.0
    priority:    TaskPriority           = TaskPriority.NORMAL


@dataclass
class WorkflowRun:
    """
    Execution record for one workflow invocation.

    run_id:        unique identifier
    workflow_name: human-readable workflow label
    goal:          top-level research goal
    steps:         ordered WorkflowStep list
    results:       dict[step_id → AgentResult]
    status:        terminal state
    total_latency_ms
    created_at / finished_at
    """
    run_id:       str                        = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_name: str                       = "unnamed"
    goal:          str                       = ""
    steps:         list[WorkflowStep]        = field(default_factory=list)
    results:       dict[str, AgentResult]    = field(default_factory=dict)
    status:        WorkflowStatus            = WorkflowStatus.PENDING
    total_latency_ms: float                  = 0.0
    created_at:    float                     = field(default_factory=time.time)
    finished_at:   Optional[float]           = None
    metadata:      dict[str, Any]            = field(default_factory=dict)

    def success_count(self) -> int:
        return sum(1 for r in self.results.values() if r.is_success())

    def failure_count(self) -> int:
        return sum(1 for r in self.results.values() if not r.is_success())

    def to_summary(self) -> dict:
        return {
            "run_id":           self.run_id,
            "workflow_name":    self.workflow_name,
            "goal":             self.goal,
            "status":           self.status.value,
            "total_steps":      len(self.steps),
            "success_count":    self.success_count(),
            "failure_count":    self.failure_count(),
            "total_latency_ms": round(self.total_latency_ms, 2),
        }


# ── Execution Graph ────────────────────────────────────────────────────────────

class ExecutionGraph:
    """
    DAG of WorkflowSteps with topological ordering.

    Uses Kahn's algorithm to produce a linear execution order that
    respects all dependency constraints.

    Raises ValueError if the graph contains a cycle (not a valid DAG).
    """

    def __init__(self, steps: list[WorkflowStep]) -> None:
        self._steps   = {s.step_id: s for s in steps}
        self._adj:    dict[str, list[str]] = defaultdict(list)  # step → dependents
        self._in_deg: dict[str, int]       = {s.step_id: 0 for s in steps}
        self._order:  list[str]            = []

        for s in steps:
            for dep in s.depends_on:
                if dep not in self._steps:
                    raise ValueError(
                        f"Step {s.step_id!r} depends on unknown step {dep!r}"
                    )
                self._adj[dep].append(s.step_id)
                self._in_deg[s.step_id] += 1

        self._order = self._topological_sort()

    def _topological_sort(self) -> list[str]:
        in_deg = dict(self._in_deg)
        queue  = deque(sid for sid, deg in in_deg.items() if deg == 0)
        order: list[str] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for dependent in self._adj.get(node, []):
                in_deg[dependent] -= 1
                if in_deg[dependent] == 0:
                    queue.append(dependent)

        if len(order) != len(self._steps):
            raise ValueError(
                "ExecutionGraph contains a cycle — not a valid DAG"
            )
        return order

    @property
    def ordered_steps(self) -> list[WorkflowStep]:
        return [self._steps[sid] for sid in self._order]

    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        return self._steps.get(step_id)

    def dependents_of(self, step_id: str) -> list[str]:
        return self._adj.get(step_id, [])


# ── Task Scheduler ─────────────────────────────────────────────────────────────

class TaskScheduler:
    """
    Priority queue for AgentTask objects.

    Higher priority = dispatched first.
    Uses Python's heapq with a negated priority key so max-priority
    is extracted with heappop (which gives min).

    Push:   O(log N)
    Pop:    O(log N)
    Peek:   O(1)
    """

    def __init__(self) -> None:
        self._heap: list[tuple[int, float, AgentTask]] = []
        self._counter = 0

    def push(self, task: AgentTask) -> None:
        key = (-task.priority.value, task.created_at, self._counter)
        heapq.heappush(self._heap, (key[0], key[1], self._counter, task))
        self._counter += 1

    def pop(self) -> AgentTask:
        if not self._heap:
            raise IndexError("Scheduler queue is empty")
        _, _, _, task = heapq.heappop(self._heap)
        return task

    def peek(self) -> Optional[AgentTask]:
        if not self._heap:
            return None
        return self._heap[0][3]

    def __len__(self) -> int:
        return len(self._heap)


# ── Agent Orchestrator ─────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    Executes a single agent for a single task.

    This is the simplest coordination unit — no DAG, no workflow.
    Useful for one-shot agent invocations from the API layer.

    Responsibilities:
      - look up the correct Agent instance from the registry
      - execute with the provided context
      - record metrics if available
    """

    def __init__(
        self,
        agents:  dict[AgentType, Agent],
        context: AgentContext,
        metrics  = None,
    ) -> None:
        self._agents  = agents
        self._context = context
        self._metrics = metrics

    def run(self, task: AgentTask, agent_type: AgentType) -> AgentResult:
        agent = self._agents.get(agent_type)
        if not agent:
            return AgentResult(
                task_id    = task.task_id,
                agent_type = agent_type,
                status     = AgentStatus.FAILED,
                output     = None,
                error      = f"No agent registered for type {agent_type.value}",
            )

        result = agent.run(task, self._context)

        if self._metrics:
            try:
                self._metrics.record_agent_execution(
                    agent_type  = agent_type.value,
                    latency_ms  = result.latency_ms,
                    success     = result.is_success(),
                )
            except Exception:
                pass

        return result


# ── Workflow Engine ───────────────────────────────────────────────────────────

class WorkflowEngine:
    """
    Multi-step workflow orchestrator with DAG-based execution.

    Modes:
      sequential — execute each step in topological order
      parallel   — where deps allow, run independent steps concurrently
                   (uses threading.Thread; GIL limits CPU parallelism
                    but IO-bound agents still benefit)

    Failure semantics:
      - If a non-optional step fails, all dependent steps are skipped
      - Optional steps' failures do not propagate
      - WorkflowRun.status:
          COMPLETED if all non-optional steps succeeded
          PARTIAL   if some optional steps failed
          FAILED    if any non-optional step failed

    Architecture note:
      Each step gets its own AgentMemory instance so agents don't
      accidentally share mutable state across steps.  The workflow
      assembles a combined results dict post-run for synthesis agents.
    """

    def __init__(
        self,
        agents:   dict[AgentType, Agent],
        context:  AgentContext,
        metrics   = None,
        parallel: bool = False,
    ) -> None:
        self._agents   = agents
        self._context  = context
        self._metrics  = metrics
        self._parallel = parallel

    def run(
        self,
        steps:         list[WorkflowStep],
        goal:          str,
        workflow_name: str = "research",
        metadata:      dict | None = None,
    ) -> WorkflowRun:
        run = WorkflowRun(
            workflow_name = workflow_name,
            goal          = goal,
            steps         = steps,
            status        = WorkflowStatus.RUNNING,
            metadata      = metadata or {},
        )

        t_start = time.perf_counter()

        try:
            graph = ExecutionGraph(steps)
        except ValueError as exc:
            run.status = WorkflowStatus.FAILED
            run.metadata["error"] = str(exc)
            return run

        if self._parallel:
            self._run_parallel(run, graph)
        else:
            self._run_sequential(run, graph)

        run.total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        run.finished_at      = time.time()
        run.status           = self._compute_status(run)

        logger.info(
            "WorkflowRun %s %s: %d/%d steps succeeded in %.0fms",
            run.run_id[:8], run.status.value,
            run.success_count(), len(run.steps), run.total_latency_ms,
        )
        return run

    def _run_sequential(self, run: WorkflowRun, graph: ExecutionGraph) -> None:
        done_set: set[str] = set()
        failed_set: set[str] = set()

        for step in graph.ordered_steps:
            blocked_deps = [d for d in step.depends_on if d in failed_set]
            if blocked_deps:
                run.results[step.step_id] = AgentResult(
                    task_id    = step.step_id,
                    agent_type = step.agent_type,
                    status     = AgentStatus.CANCELLED,
                    output     = None,
                    error      = f"Skipped: dependencies failed: {blocked_deps}",
                )
                failed_set.add(step.step_id)
                continue

            result = self._execute_step(step, run)
            run.results[step.step_id] = result

            if result.is_success():
                done_set.add(step.step_id)
            elif not step.optional:
                failed_set.add(step.step_id)

    def _run_parallel(self, run: WorkflowRun, graph: ExecutionGraph) -> None:
        """
        Execute the graph in parallel where dependency constraints allow.
        Groups steps into waves: wave_i contains all steps whose dependencies
        are satisfied after wave_{i-1} completes.
        """
        done_set:   set[str] = set()
        failed_set: set[str] = set()
        remaining = {s.step_id for s in graph.ordered_steps}

        while remaining:
            ready = [
                graph.get_step(sid)
                for sid in remaining
                if all(d in done_set for d in (graph.get_step(sid).depends_on or []))
                and not any(d in failed_set for d in (graph.get_step(sid).depends_on or []))
            ]
            blocked = [
                sid for sid in remaining
                if any(d in failed_set for d in (graph.get_step(sid).depends_on or []))
            ]

            if not ready and not blocked:
                break

            # Cancel blocked
            for sid in blocked:
                remaining.discard(sid)
                step = graph.get_step(sid)
                run.results[sid] = AgentResult(
                    task_id    = sid,
                    agent_type = step.agent_type,
                    status     = AgentStatus.CANCELLED,
                    output     = None,
                    error      = "Skipped: upstream failure",
                )
                failed_set.add(sid)

            if not ready:
                break

            # Run ready steps concurrently
            threads: list[threading.Thread] = []
            lock    = threading.Lock()
            local_results: dict[str, AgentResult] = {}

            def run_one(s: WorkflowStep) -> None:
                res = self._execute_step(s, run)
                with lock:
                    local_results[s.step_id] = res

            for step in ready:
                t = threading.Thread(target=run_one, args=(step,), daemon=True)
                threads.append(t)
                t.start()
            for t in threads:
                t.join(timeout=300)

            for step in ready:
                sid    = step.step_id
                result = local_results.get(sid)
                if result is None:
                    result = AgentResult(
                        task_id    = sid,
                        agent_type = step.agent_type,
                        status     = AgentStatus.TIMED_OUT,
                        output     = None,
                        error      = "Thread join timed out",
                    )
                run.results[sid] = result
                remaining.discard(sid)
                if result.is_success():
                    done_set.add(sid)
                elif not step.optional:
                    failed_set.add(sid)

    def _execute_step(self, step: WorkflowStep, run: WorkflowRun) -> AgentResult:
        agent = self._agents.get(step.agent_type)
        if not agent:
            return AgentResult(
                task_id    = step.step_id,
                agent_type = step.agent_type,
                status     = AgentStatus.FAILED,
                output     = None,
                error      = f"No agent registered for {step.agent_type.value}",
            )

        # Inject prior results into params for downstream agents
        enriched_params = dict(step.params)
        enriched_params["_prior_results"] = {
            sid: r.to_dict()
            for sid, r in run.results.items()
            if r.is_success()
        }
        enriched_params["_workflow_goal"] = run.goal

        task = AgentTask(
            goal        = step.goal,
            task_type   = step.agent_type.value,
            params      = enriched_params,
            priority    = step.priority,
            timeout_sec = step.timeout_sec,
            task_id     = step.step_id,
        )

        result = agent.run(task, self._context)

        if self._metrics:
            try:
                self._metrics.record_agent_execution(
                    agent_type = step.agent_type.value,
                    latency_ms = result.latency_ms,
                    success    = result.is_success(),
                )
            except Exception:
                pass

        return result

    @staticmethod
    def _compute_status(run: WorkflowRun) -> WorkflowStatus:
        if not run.results:
            return WorkflowStatus.FAILED
        if all(r.is_success() for r in run.results.values()):
            return WorkflowStatus.COMPLETED
        # Check if any non-optional step failed
        failed_required = any(
            not r.is_success()
            for step, r in (
                (next((s for s in run.steps if s.step_id == sid), None), r)
                for sid, r in run.results.items()
            )
            if step and not step.optional
            and r.status not in (AgentStatus.DONE, AgentStatus.CANCELLED)
        )
        return WorkflowStatus.FAILED if failed_required else WorkflowStatus.PARTIAL
