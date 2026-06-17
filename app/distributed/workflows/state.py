"""
Distributed Workflow State — Phase 8 Batch 3

=== THEORY ===

Workflow state management is the foundation of distributed workflow execution.
Unlike the Phase 7 WorkflowRun (which lives for a single in-process invocation),
WorkflowExecution tracks the full lifecycle of a workflow that may:

  - Span multiple processes (workers can pick up where another left off)
  - Survive crashes (state is persisted via checkpointing)
  - Be paused and resumed (explicit user control)
  - Be cancelled mid-flight (graceful termination)

The WorkflowState enum extends WorkflowStatus with additional states needed
for distributed coordination:

  CREATED       — registered but not yet dispatched
  RUNNING       — actively executing steps
  PAUSED        — execution suspended by user or system
  COMPLETED     — all steps finished successfully
  FAILED        — terminal failure (non-optional step failed, retries exhausted)
  CANCELLED     — user-initiated termination
  CHECKPOINTED  — paused with a recoverable checkpoint saved

State transitions follow a strict state machine:

  CREATED → RUNNING → COMPLETED
                    → FAILED
                    → PAUSED → RUNNING (resume)
                             → CANCELLED
                    → CANCELLED
  CREATED → CANCELLED
  PAUSED → CHECKPOINTED → RUNNING (recovery)

=== DATA STRUCTURES ===

  WorkflowState     — enum of valid states
  WorkflowExecution — full execution record with checkpointing support

=== COMPLEXITY ===

  WorkflowExecution.progress(): O(1) — division of two lengths
  WorkflowExecution.to_dict():  O(S) where S = number of steps
  WorkflowExecution.is_terminal(): O(1) — set membership check

=== PRODUCTION EQUIVALENTS ===

  Temporal:       WorkflowExecution with RunId + WorkflowId
  Airflow:        DagRun with state tracking
  Prefect:        FlowRun with state transitions
  Step Functions: Execution with status + history
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class WorkflowState(str, Enum):
    """
    Lifecycle states for a distributed workflow execution.

    State machine:
      CREATED → RUNNING → COMPLETED
                        → FAILED
                        → PAUSED → RUNNING (resume)
                                 → CANCELLED
                                 → CHECKPOINTED → RUNNING
                        → CANCELLED
      CREATED → CANCELLED

    Terminal states: COMPLETED, FAILED, CANCELLED
    """
    CREATED       = "created"
    RUNNING       = "running"
    PAUSED        = "paused"
    COMPLETED     = "completed"
    FAILED        = "failed"
    CANCELLED     = "cancelled"
    CHECKPOINTED  = "checkpointed"


# Valid state transitions for enforcement
_VALID_TRANSITIONS: dict[WorkflowState, set[WorkflowState]] = {
    WorkflowState.CREATED:      {WorkflowState.RUNNING, WorkflowState.CANCELLED},
    WorkflowState.RUNNING:      {WorkflowState.COMPLETED, WorkflowState.FAILED,
                                 WorkflowState.PAUSED, WorkflowState.CANCELLED},
    WorkflowState.PAUSED:       {WorkflowState.RUNNING, WorkflowState.CANCELLED,
                                 WorkflowState.CHECKPOINTED},
    WorkflowState.CHECKPOINTED: {WorkflowState.RUNNING, WorkflowState.CANCELLED},
    WorkflowState.COMPLETED:    set(),
    WorkflowState.FAILED:       set(),
    WorkflowState.CANCELLED:    set(),
}

# Terminal states (no further transitions possible)
_TERMINAL_STATES: set[WorkflowState] = {
    WorkflowState.COMPLETED,
    WorkflowState.FAILED,
    WorkflowState.CANCELLED,
}


@dataclass
class WorkflowExecution:
    """
    Tracks the full execution state of a distributed workflow.

    Extends the Phase 7 WorkflowRun concept with:
      - Checkpoint data for crash recovery
      - Explicit pause/resume semantics
      - Step-level result tracking
      - Progress reporting (0.0 to 1.0)
      - Metadata for arbitrary annotations

    === ARCHITECTURE ===

    The execution record is the single source of truth for workflow state.
    The DistributedWorkflowEngine writes to it after each step, and the
    CheckpointStore persists it so recovery is possible.

    Fields:
      execution_id    — unique identifier (UUID)
      workflow_name   — human-readable label
      goal            — top-level workflow objective
      state           — current lifecycle state
      steps           — ordered list of step definitions (dicts or WorkflowStep)
      completed_steps — step_ids that finished successfully
      failed_steps    — step_ids that failed
      step_results    — dict[step_id → result dict]
      checkpoint_data — arbitrary data saved at last checkpoint
      created_at      — Unix epoch when execution was created
      updated_at      — Unix epoch of last state mutation
      started_at      — Unix epoch when execution entered RUNNING
      finished_at     — Unix epoch when execution reached terminal state
      total_latency_ms — wall-clock time from start to finish
      metadata        — arbitrary key-value annotations
    """
    execution_id:    str                    = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_name:   str                    = "unnamed"
    goal:            str                    = ""
    state:           WorkflowState          = WorkflowState.CREATED
    steps:           list[Any]             = field(default_factory=list)
    completed_steps: list[str]             = field(default_factory=list)
    failed_steps:    list[str]             = field(default_factory=list)
    step_results:    dict[str, Any]        = field(default_factory=dict)
    checkpoint_data: dict[str, Any]        = field(default_factory=dict)
    created_at:      float                 = field(default_factory=time.time)
    updated_at:      float                 = field(default_factory=time.time)
    started_at:      float | None          = None
    finished_at:     float | None          = None
    total_latency_ms: float                = 0.0
    metadata:        dict[str, Any]        = field(default_factory=dict)

    def transition(self, new_state: WorkflowState) -> None:
        """
        Transition to a new state with validation.

        Raises RuntimeError if the transition is not valid per the state machine.
        Updates timestamps as appropriate.
        """
        allowed = _VALID_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise RuntimeError(
                f"Invalid workflow state transition: {self.state.value} -> {new_state.value}"
            )
        self.state = new_state
        self.updated_at = time.time()

        if new_state == WorkflowState.RUNNING and self.started_at is None:
            self.started_at = time.time()
        elif new_state in _TERMINAL_STATES:
            self.finished_at = time.time()
            if self.started_at is not None:
                self.total_latency_ms = (self.finished_at - self.started_at) * 1000

    def to_dict(self) -> dict:
        """
        Serialise the execution state to a plain dictionary.

        Used for checkpointing, API responses, and event payloads.
        """
        return {
            "execution_id":    self.execution_id,
            "workflow_name":   self.workflow_name,
            "goal":            self.goal,
            "state":           self.state.value,
            "steps":           [
                s.to_dict() if hasattr(s, "to_dict") else s
                for s in self.steps
            ],
            "completed_steps": list(self.completed_steps),
            "failed_steps":    list(self.failed_steps),
            "step_results":    dict(self.step_results),
            "checkpoint_data": dict(self.checkpoint_data),
            "created_at":      self.created_at,
            "updated_at":      self.updated_at,
            "started_at":      self.started_at,
            "finished_at":     self.finished_at,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "metadata":        dict(self.metadata),
        }

    def progress(self) -> float:
        """
        Return execution progress as a float in [0.0, 1.0].

        Progress = completed_steps / total_steps.
        Returns 0.0 if there are no steps defined.
        """
        total = len(self.steps)
        if total == 0:
            return 0.0
        completed = len(self.completed_steps)
        return min(completed / total, 1.0)

    def is_terminal(self) -> bool:
        """
        Return True if the execution is in a terminal state.

        Terminal states: COMPLETED, FAILED, CANCELLED.
        No further state transitions are possible once terminal.
        """
        return self.state in _TERMINAL_STATES
