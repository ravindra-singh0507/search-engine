"""
Distributed Workflow Execution Engine — Phase 8 Batch 3

=== THEORY ===

The DistributedWorkflowEngine extends the Phase 7 WorkflowEngine with
production-grade capabilities for running workflows across distributed
infrastructure:

  1. Persistent execution state (survives crashes)
     - After each step, the engine writes a checkpoint containing the
       full execution state.  On restart, recover() loads all pending
       checkpoints and resumes from the last successful step.

  2. Checkpointing after each successful step
     - Eliminates wasted computation: if step 5/10 crashes, only step 5
       is re-executed on recovery (not steps 1-4 again).

  3. Resume from checkpoint on restart
     - The recover() method scans the CheckpointStore for incomplete
       executions and resumes each one from its last checkpoint.

  4. Concurrent workflow execution
     - Multiple workflows run in parallel via threads.
     - Each execution has its own state; no shared mutable state between
       concurrent workflows.

  5. Execution history and tracking
     - Every lifecycle event is recorded in the ExecutionTracker for
       debugging, monitoring, and audit compliance.

  6. Event-driven step completion
     - The engine publishes events (WORKFLOW_STARTED, WORKFLOW_COMPLETED)
       to the EventBus, enabling downstream consumers to react.

=== ARCHITECTURE ===

  DistributedWorkflowEngine
    │
    ├── _executions: dict[execution_id → WorkflowExecution]
    │     Active execution registry (thread-safe via Lock)
    │
    ├── _checkpoint_store: CheckpointStore
    │     Persists execution state for crash recovery
    │
    ├── _tracker: ExecutionTracker
    │     Audit trail of all lifecycle events
    │
    ├── _scheduler: WorkflowScheduler
    │     Periodic/one-shot workflow triggering
    │
    └── Methods:
          start_workflow()   — create + begin executing a workflow
          pause_workflow()   — suspend execution
          resume_workflow()  — resume from pause
          cancel_workflow()  — terminate execution
          recover()          — resume all checkpointed workflows after restart
          cleanup_old()      — garbage-collect finished executions

=== EXECUTION MODEL ===

  start_workflow() spawns a thread that:
    1. Transitions state to RUNNING
    2. Iterates steps in order (topological if using ExecutionGraph)
    3. After each successful step: save checkpoint
    4. After each failed step: check if optional; if not, fail workflow
    5. On completion: transition to COMPLETED, publish event
    6. On failure: transition to FAILED, publish event

  The thread is daemon=True so it doesn't prevent process exit.

=== COMPLEXITY ===

  start_workflow():    O(1) to enqueue (thread spawn)
  pause_workflow():    O(1) state transition
  resume_workflow():   O(1) + spawns resume thread
  cancel_workflow():   O(1) state transition
  get_execution():     O(1) dict lookup
  list_executions():   O(N) filter + sort
  recover():           O(C) where C = checkpointed workflows
  cleanup_old():       O(N) scan all executions

=== PRODUCTION EQUIVALENTS ===

  Temporal:         Worker polls for workflow tasks, maintains state in DB
  Airflow:          Scheduler + Worker with XCom state passing
  Prefect:          Orion server + Agent with state management
  Step Functions:   State machine with automatic checkpointing
"""

import logging
import threading
import time
import uuid
from typing import Any

from app.config import DistributedWorkflowConfig
from app.distributed.workflows.checkpoint import CheckpointStore, InMemoryCheckpointStore
from app.distributed.workflows.scheduler import WorkflowScheduler
from app.distributed.workflows.state import WorkflowExecution, WorkflowState
from app.distributed.workflows.tracker import ExecutionTracker
from app.orchestration.engine import ExecutionGraph, WorkflowStep, WorkflowStatus

logger = logging.getLogger(__name__)


class DistributedWorkflowEngine:
    """
    Distributed workflow execution engine with checkpointing and recovery.

    === THEORY ===

    Extends the Phase 7 WorkflowEngine with:
      - Persistent execution state (survives crashes)
      - Checkpointing after each successful step
      - Resume from checkpoint on restart
      - Concurrent workflow execution (multiple workflows in parallel)
      - Execution history and tracking
      - Event-driven step completion

    The engine maintains a registry of active executions and provides
    methods to query, pause, resume, and cancel them.

    === THREAD SAFETY ===

    All mutable state is protected by self._lock.  Each workflow
    execution runs in its own daemon thread.  The pause/cancel
    mechanisms use per-execution threading.Event objects as
    cooperative signalling.

    === USAGE ===

    engine = DistributedWorkflowEngine(config, agents=agents, context=ctx)
    execution = engine.start_workflow(steps, goal="Research AI safety")

    # Later:
    engine.pause_workflow(execution.execution_id)
    engine.resume_workflow(execution.execution_id)

    # After restart:
    recovered = engine.recover()
    """

    def __init__(
        self,
        config: DistributedWorkflowConfig,
        agents: dict | None = None,
        context: Any = None,
        metrics: Any = None,
        event_bus: Any = None,
        checkpoint_store: CheckpointStore | None = None,
    ) -> None:
        """
        Initialise the distributed workflow engine.

        Args:
            config:           Distributed workflow configuration.
            agents:           dict[AgentType → Agent] for step execution.
            context:          AgentContext shared across agents.
            metrics:          Optional metrics recorder.
            event_bus:        Optional EventBus for publishing lifecycle events.
            checkpoint_store: Optional CheckpointStore for persistence.
                              Defaults to InMemoryCheckpointStore.
        """
        self._config = config
        self._agents = agents or {}
        self._context = context
        self._metrics = metrics
        self._event_bus = event_bus
        self._checkpoint_store: CheckpointStore = (
            checkpoint_store or InMemoryCheckpointStore(max_size=100)
        )

        # Execution registry
        self._executions: dict[str, WorkflowExecution] = {}
        self._lock = threading.Lock()

        # Pause/cancel signalling: execution_id → Event (set = paused/cancelled)
        self._pause_signals: dict[str, threading.Event] = {}
        self._cancel_signals: dict[str, threading.Event] = {}

        # Sub-components
        self._tracker = ExecutionTracker(max_history=1000)
        self._scheduler = WorkflowScheduler(config)

        # Counters for stats
        self._total_started = 0
        self._total_completed = 0
        self._total_failed = 0
        self._total_latency_sum = 0.0

    # ── Public API ──────────────────────────────────────────────────────────

    def start_workflow(
        self,
        steps: list[WorkflowStep],
        goal: str,
        workflow_name: str = "research",
        metadata: dict | None = None,
    ) -> WorkflowExecution:
        """
        Start a new workflow execution.

        Creates a WorkflowExecution, registers it, saves an initial checkpoint,
        and spawns a thread to execute steps sequentially.

        Args:
            steps:         Ordered list of WorkflowStep objects.
            goal:          Human-readable workflow objective.
            workflow_name: Label for this workflow type.
            metadata:      Optional annotations.

        Returns:
            The WorkflowExecution object (state will be RUNNING).

        Raises:
            RuntimeError: If max_concurrent_workflows limit is reached.
        """
        with self._lock:
            active_count = sum(
                1 for ex in self._executions.values()
                if not ex.is_terminal()
            )
            if active_count >= self._config.max_concurrent_workflows:
                raise RuntimeError(
                    f"Max concurrent workflows reached: {self._config.max_concurrent_workflows}"
                )

        execution = WorkflowExecution(
            workflow_name=workflow_name,
            goal=goal,
            steps=steps,
            metadata=metadata or {},
        )

        # Register execution
        with self._lock:
            self._executions[execution.execution_id] = execution
            self._pause_signals[execution.execution_id] = threading.Event()
            self._cancel_signals[execution.execution_id] = threading.Event()
            self._total_started += 1

        # Save initial checkpoint
        self._save_checkpoint(execution)

        # Record start in tracker
        self._tracker.record_start(execution)

        # Publish start event
        self._publish_event("workflow.started", {
            "execution_id":  execution.execution_id,
            "workflow_name": workflow_name,
            "goal":          goal,
            "total_steps":   len(steps),
        })

        # Spawn execution thread
        thread = threading.Thread(
            target=self._execute_workflow,
            args=(execution,),
            daemon=True,
            name=f"workflow-{execution.execution_id[:8]}",
        )
        thread.start()

        logger.info(
            "Started workflow '%s' (execution=%s, steps=%d)",
            workflow_name, execution.execution_id[:8], len(steps),
        )
        return execution

    def pause_workflow(self, execution_id: str) -> bool:
        """
        Pause a running workflow.

        The execution will stop after completing its current step.
        Paused workflows can be resumed with resume_workflow().

        Returns True if the workflow was successfully paused, False if
        it cannot be paused (not found, already terminal, etc.).
        """
        with self._lock:
            execution = self._executions.get(execution_id)
            if execution is None:
                return False
            if execution.state != WorkflowState.RUNNING:
                return False

            try:
                execution.transition(WorkflowState.PAUSED)
            except RuntimeError:
                return False

            # Signal the execution thread to pause
            pause_event = self._pause_signals.get(execution_id)
            if pause_event:
                pause_event.set()

        self._save_checkpoint(execution)
        logger.info("Paused workflow %s", execution_id[:8])
        return True

    def resume_workflow(self, execution_id: str) -> bool:
        """
        Resume a paused or checkpointed workflow.

        Spawns a new execution thread that picks up from the last
        completed step.

        Returns True if successfully resumed, False otherwise.
        """
        with self._lock:
            execution = self._executions.get(execution_id)
            if execution is None:
                return False
            if execution.state not in (WorkflowState.PAUSED, WorkflowState.CHECKPOINTED):
                return False

            try:
                execution.transition(WorkflowState.RUNNING)
            except RuntimeError:
                return False

            # Clear the pause signal
            pause_event = self._pause_signals.get(execution_id)
            if pause_event:
                pause_event.clear()

        # Spawn a new thread to continue execution
        thread = threading.Thread(
            target=self._execute_workflow,
            args=(execution,),
            daemon=True,
            name=f"workflow-resume-{execution_id[:8]}",
        )
        thread.start()

        logger.info("Resumed workflow %s", execution_id[:8])
        return True

    def cancel_workflow(self, execution_id: str) -> bool:
        """
        Cancel a workflow execution.

        The execution will stop after completing its current step.
        Cancelled workflows cannot be resumed.

        Returns True if successfully cancelled, False otherwise.
        """
        with self._lock:
            execution = self._executions.get(execution_id)
            if execution is None:
                return False
            if execution.is_terminal():
                return False

            try:
                execution.transition(WorkflowState.CANCELLED)
            except RuntimeError:
                return False

            # Signal the execution thread to stop
            cancel_event = self._cancel_signals.get(execution_id)
            if cancel_event:
                cancel_event.set()

        # Clean up checkpoint (no recovery needed for cancelled workflows)
        self._checkpoint_store.delete(execution_id)

        logger.info("Cancelled workflow %s", execution_id[:8])
        return True

    def get_execution(self, execution_id: str) -> WorkflowExecution | None:
        """
        Look up an execution by ID.

        Returns None if not found.
        """
        with self._lock:
            return self._executions.get(execution_id)

    def list_executions(
        self,
        state: WorkflowState | None = None,
        limit: int = 50,
    ) -> list[WorkflowExecution]:
        """
        List executions, optionally filtered by state.

        Args:
            state: If provided, only return executions in this state.
            limit: Maximum number of results.

        Returns:
            List of WorkflowExecution objects, newest first.
        """
        with self._lock:
            executions = list(self._executions.values())

        if state is not None:
            executions = [ex for ex in executions if ex.state == state]

        # Sort by created_at descending (newest first)
        executions.sort(key=lambda ex: ex.created_at, reverse=True)
        return executions[:limit]

    def recover(self) -> int:
        """
        Recover workflows from checkpoints after a restart.

        Scans the checkpoint store for all saved checkpoints, rebuilds
        WorkflowExecution objects, and resumes any that were not in a
        terminal state.

        Returns:
            The number of workflows recovered and resumed.
        """
        checkpoint_ids = self._checkpoint_store.list_checkpoints()
        recovered = 0

        for execution_id in checkpoint_ids:
            # Skip if already in our registry (already running)
            with self._lock:
                if execution_id in self._executions:
                    continue

            checkpoint = self._checkpoint_store.load(execution_id)
            if checkpoint is None:
                continue

            # Rebuild execution from checkpoint data
            execution = self._rebuild_from_checkpoint(execution_id, checkpoint)
            if execution is None:
                continue

            # Only recover non-terminal workflows
            if execution.is_terminal():
                # Clean up the stale checkpoint
                self._checkpoint_store.delete(execution_id)
                continue

            # Register and resume
            with self._lock:
                self._executions[execution_id] = execution
                self._pause_signals[execution_id] = threading.Event()
                self._cancel_signals[execution_id] = threading.Event()

            # Set to CHECKPOINTED then resume
            if execution.state == WorkflowState.RUNNING:
                execution.state = WorkflowState.PAUSED
                execution.updated_at = time.time()

            try:
                execution.transition(WorkflowState.CHECKPOINTED)
            except RuntimeError:
                pass

            # Resume from checkpoint
            if self.resume_workflow(execution_id):
                recovered += 1

        if recovered > 0:
            logger.info("Recovered %d workflows from checkpoints", recovered)
        return recovered

    def cleanup_old(self, max_age_hours: int = 24) -> int:
        """
        Remove old completed/failed/cancelled executions from memory.

        Args:
            max_age_hours: Executions older than this are eligible for cleanup.

        Returns:
            Number of executions cleaned up.
        """
        cutoff = time.time() - (max_age_hours * 3600)
        cleaned = 0

        with self._lock:
            to_remove = [
                eid for eid, ex in self._executions.items()
                if ex.is_terminal() and ex.finished_at is not None
                and ex.finished_at < cutoff
            ]
            for eid in to_remove:
                del self._executions[eid]
                self._pause_signals.pop(eid, None)
                self._cancel_signals.pop(eid, None)
                cleaned += 1

        # Clean up checkpoints and tracker history
        for eid in to_remove:
            self._checkpoint_store.delete(eid)
            self._tracker.clear_execution(eid)

        if cleaned > 0:
            logger.info("Cleaned up %d old executions", cleaned)
        return cleaned

    def stats(self) -> dict:
        """
        Return engine statistics.

        Returns:
            dict with keys: active, completed, failed, cancelled,
            total_started, avg_latency_ms, scheduler stats, tracker stats.
        """
        with self._lock:
            states: dict[str, int] = {}
            for ex in self._executions.values():
                states[ex.state.value] = states.get(ex.state.value, 0) + 1

        avg_latency = (
            self._total_latency_sum / self._total_completed
            if self._total_completed > 0
            else 0.0
        )

        return {
            "active":          states.get("running", 0),
            "paused":          states.get("paused", 0),
            "completed":       self._total_completed,
            "failed":          self._total_failed,
            "cancelled":       states.get("cancelled", 0),
            "total_started":   self._total_started,
            "avg_latency_ms":  round(avg_latency, 2),
            "max_concurrent":  self._config.max_concurrent_workflows,
            "checkpoint_enabled": self._config.checkpoint_enabled,
            "states":          states,
            "scheduler":       self._scheduler.stats(),
            "tracker":         self._tracker.stats(),
        }

    @property
    def tracker(self) -> ExecutionTracker:
        """Access the execution tracker for history queries."""
        return self._tracker

    @property
    def scheduler(self) -> WorkflowScheduler:
        """Access the workflow scheduler."""
        return self._scheduler

    # ── Internal execution logic ────────────────────────────────────────────

    def _execute_workflow(self, execution: WorkflowExecution) -> None:
        """
        Execute workflow steps sequentially in a background thread.

        Steps are processed in order.  After each successful step,
        a checkpoint is saved.  The loop checks for pause/cancel signals
        between steps for cooperative interruption.

        This method handles all exceptions internally — it never raises.
        """
        execution_id = execution.execution_id

        # Transition to RUNNING if not already
        if execution.state != WorkflowState.RUNNING:
            try:
                execution.transition(WorkflowState.RUNNING)
            except RuntimeError:
                return

        try:
            # Build execution graph for ordering
            steps = execution.steps
            if not steps:
                execution.transition(WorkflowState.COMPLETED)
                self._on_workflow_complete(execution)
                return

            # Determine which steps still need execution
            completed_set = set(execution.completed_steps)

            for step in steps:
                step_id = step.step_id if hasattr(step, "step_id") else str(step)

                # Skip already-completed steps (resume scenario)
                if step_id in completed_set:
                    continue

                # Check for cancel signal
                cancel_event = self._cancel_signals.get(execution_id)
                if cancel_event and cancel_event.is_set():
                    logger.info("Workflow %s cancelled during execution", execution_id[:8])
                    return

                # Check for pause signal
                pause_event = self._pause_signals.get(execution_id)
                if pause_event and pause_event.is_set():
                    logger.info("Workflow %s paused during execution", execution_id[:8])
                    self._save_checkpoint(execution)
                    return

                # Execute the step
                result = self._execute_step(step, execution)

                if result is not None and self._is_step_success(result):
                    # Step succeeded
                    execution.completed_steps.append(step_id)
                    execution.step_results[step_id] = (
                        result.to_dict() if hasattr(result, "to_dict") else result
                    )
                    execution.updated_at = time.time()

                    self._tracker.record_step_complete(
                        execution_id, step_id,
                        result.to_dict() if hasattr(result, "to_dict") else {"output": str(result)},
                    )

                    # Save checkpoint after each successful step
                    if self._config.checkpoint_enabled:
                        self._save_checkpoint(execution)

                else:
                    # Step failed
                    error_msg = self._extract_error(result)
                    is_optional = getattr(step, "optional", False)

                    execution.failed_steps.append(step_id)
                    execution.step_results[step_id] = {"error": error_msg}
                    execution.updated_at = time.time()

                    self._tracker.record_step_failed(execution_id, step_id, error_msg)

                    if not is_optional:
                        # Non-optional step failure terminates the workflow
                        execution.transition(WorkflowState.FAILED)
                        self._on_workflow_failed(execution, error_msg)
                        return

            # All steps processed successfully (or only optional ones failed)
            execution.transition(WorkflowState.COMPLETED)
            self._on_workflow_complete(execution)

        except Exception as exc:
            # Catch-all for unexpected errors
            logger.exception(
                "Unexpected error in workflow %s: %s", execution_id[:8], exc
            )
            try:
                if not execution.is_terminal():
                    execution.transition(WorkflowState.FAILED)
                    self._on_workflow_failed(execution, str(exc))
            except RuntimeError:
                pass

    def _execute_step(self, step: Any, execution: WorkflowExecution) -> Any:
        """
        Execute a single workflow step using the registered agent.

        Follows the same pattern as Phase 7 WorkflowEngine._execute_step:
        look up the agent, build a task, execute, and return the result.

        Returns the AgentResult (or None on catastrophic failure).
        """
        from app.agents.base import AgentTask, AgentResult, AgentStatus, AgentType

        if not hasattr(step, "agent_type"):
            # Step is not a WorkflowStep — cannot execute
            logger.error("Step has no agent_type: %s", step)
            return None

        agent = self._agents.get(step.agent_type)
        if agent is None:
            return AgentResult(
                task_id=step.step_id,
                agent_type=step.agent_type,
                status=AgentStatus.FAILED,
                output=None,
                error=f"No agent registered for type {step.agent_type.value}",
            )

        # Enrich params with prior results
        enriched_params = dict(step.params) if hasattr(step, "params") else {}
        enriched_params["_prior_results"] = {
            sid: r for sid, r in execution.step_results.items()
        }
        enriched_params["_workflow_goal"] = execution.goal

        task = AgentTask(
            goal=step.goal,
            task_type=step.agent_type.value,
            params=enriched_params,
            priority=step.priority if hasattr(step, "priority") else None,
            timeout_sec=step.timeout_sec if hasattr(step, "timeout_sec") else self._config.step_timeout_sec,
            task_id=step.step_id,
        )

        try:
            result = agent.run(task, self._context)

            if self._metrics:
                try:
                    self._metrics.record_agent_execution(
                        agent_type=step.agent_type.value,
                        latency_ms=result.latency_ms,
                        success=result.is_success(),
                    )
                except Exception:
                    pass

            return result
        except Exception as exc:
            logger.error(
                "Step %s execution error: %s", step.step_id[:8], exc
            )
            return AgentResult(
                task_id=step.step_id,
                agent_type=step.agent_type,
                status=AgentStatus.FAILED,
                output=None,
                error=str(exc),
            )

    def _is_step_success(self, result: Any) -> bool:
        """Check if a step result indicates success."""
        if hasattr(result, "is_success"):
            return result.is_success()
        return result is not None

    def _extract_error(self, result: Any) -> str:
        """Extract error message from a failed result."""
        if result is None:
            return "Step returned None (no agent or catastrophic failure)"
        if hasattr(result, "error") and result.error:
            return result.error
        if hasattr(result, "status"):
            return f"Step failed with status: {result.status}"
        return "Unknown error"

    # ── Lifecycle hooks ─────────────────────────────────────────────────────

    def _on_workflow_complete(self, execution: WorkflowExecution) -> None:
        """Handle workflow completion: update stats, publish event, cleanup."""
        with self._lock:
            self._total_completed += 1
            self._total_latency_sum += execution.total_latency_ms

        self._tracker.record_complete(execution.execution_id)

        # Clean up checkpoint (no longer needed)
        self._checkpoint_store.delete(execution.execution_id)

        # Publish completion event
        self._publish_event("workflow.completed", {
            "execution_id":    execution.execution_id,
            "workflow_name":   execution.workflow_name,
            "total_latency_ms": round(execution.total_latency_ms, 2),
            "completed_steps": len(execution.completed_steps),
            "total_steps":     len(execution.steps),
        })

        logger.info(
            "Workflow %s completed: %d/%d steps in %.0fms",
            execution.execution_id[:8],
            len(execution.completed_steps), len(execution.steps),
            execution.total_latency_ms,
        )

    def _on_workflow_failed(self, execution: WorkflowExecution, error: str) -> None:
        """Handle workflow failure: update stats, publish event."""
        with self._lock:
            self._total_failed += 1
            self._total_latency_sum += execution.total_latency_ms

        self._tracker.record_failed(execution.execution_id, error)

        # Keep checkpoint for potential manual recovery/inspection
        self._save_checkpoint(execution)

        # Publish failure event
        self._publish_event("workflow.failed", {
            "execution_id":    execution.execution_id,
            "workflow_name":   execution.workflow_name,
            "error":           error,
            "completed_steps": len(execution.completed_steps),
            "failed_steps":    len(execution.failed_steps),
            "total_steps":     len(execution.steps),
        })

        logger.warning(
            "Workflow %s failed: %s (completed %d/%d steps)",
            execution.execution_id[:8], error,
            len(execution.completed_steps), len(execution.steps),
        )

    # ── Checkpointing ──────────────────────────────────────────────────────

    def _save_checkpoint(self, execution: WorkflowExecution) -> None:
        """
        Persist the current execution state to the checkpoint store.

        The checkpoint contains the full execution state serialised as a dict.
        On recovery, this is sufficient to rebuild the WorkflowExecution and
        resume from the last successful step.
        """
        if not self._config.checkpoint_enabled:
            return

        try:
            checkpoint_data = execution.to_dict()
            execution.checkpoint_data = {
                "last_checkpoint_time": time.time(),
                "completed_steps_count": len(execution.completed_steps),
            }
            self._checkpoint_store.save(execution.execution_id, checkpoint_data)
        except Exception as exc:
            logger.error(
                "Failed to save checkpoint for %s: %s",
                execution.execution_id[:8], exc,
            )

    def _rebuild_from_checkpoint(
        self, execution_id: str, checkpoint: dict
    ) -> WorkflowExecution | None:
        """
        Rebuild a WorkflowExecution from checkpoint data.

        Returns None if the checkpoint data is malformed.
        """
        try:
            state_str = checkpoint.get("state", "created")
            try:
                state = WorkflowState(state_str)
            except ValueError:
                state = WorkflowState.CREATED

            execution = WorkflowExecution(
                execution_id=execution_id,
                workflow_name=checkpoint.get("workflow_name", "recovered"),
                goal=checkpoint.get("goal", ""),
                state=state,
                steps=checkpoint.get("steps", []),
                completed_steps=checkpoint.get("completed_steps", []),
                failed_steps=checkpoint.get("failed_steps", []),
                step_results=checkpoint.get("step_results", {}),
                checkpoint_data=checkpoint.get("checkpoint_data", {}),
                created_at=checkpoint.get("created_at", time.time()),
                updated_at=checkpoint.get("updated_at", time.time()),
                started_at=checkpoint.get("started_at"),
                finished_at=checkpoint.get("finished_at"),
                total_latency_ms=checkpoint.get("total_latency_ms", 0.0),
                metadata=checkpoint.get("metadata", {}),
            )
            return execution
        except Exception as exc:
            logger.error(
                "Failed to rebuild execution from checkpoint %s: %s",
                execution_id[:8], exc,
            )
            return None

    # ── Event publishing ────────────────────────────────────────────────────

    def _publish_event(self, topic: str, payload: dict) -> None:
        """
        Publish a lifecycle event to the event bus (if available).

        Fails silently if no event bus is configured.
        """
        if self._event_bus is None:
            return

        try:
            from app.events.models import Event, EventMetadata

            event = Event(
                topic=topic,
                payload=payload,
                metadata=EventMetadata(source="distributed_workflow_engine"),
            )
            self._event_bus.publish(event)
        except Exception as exc:
            logger.debug("Failed to publish event '%s': %s", topic, exc)
