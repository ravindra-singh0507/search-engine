"""
Distributed Workflow Execution — Phase 8 Batch 3

=== THEORY ===

This module provides the Distributed Workflow Execution Engine,
extending Phase 7's WorkflowEngine with production-grade capabilities:

  - Persistent state via CheckpointStore (survives crashes)
  - Checkpoint after each successful step (no wasted computation)
  - Recovery from checkpoints on restart
  - Concurrent workflow execution (multiple workflows in parallel)
  - Execution history and audit trail via ExecutionTracker
  - Interval-based scheduling via WorkflowScheduler
  - Cooperative pause/resume/cancel semantics

=== ARCHITECTURE ===

  WorkflowExecution (state.py)
    │  Tracks the full lifecycle of one workflow invocation
    │
  CheckpointStore (checkpoint.py)
    │  Protocol + implementations for persisting execution state
    │
  ExecutionTracker (tracker.py)
    │  Audit trail of all lifecycle events
    │
  WorkflowScheduler (scheduler.py)
    │  Interval-based workflow triggering
    │
  DistributedWorkflowEngine (engine.py)
       Orchestrates everything: execution, checkpointing, recovery,
       scheduling, event publishing

=== PRODUCTION EQUIVALENTS ===

  Temporal:       Durable workflow execution
  Airflow:        DAG-based workflow scheduling + state tracking
  Prefect:        Flow orchestration with state management
  Step Functions: AWS state machine execution + checkpointing
"""

from app.distributed.workflows.state import WorkflowExecution, WorkflowState
from app.distributed.workflows.checkpoint import (
    CheckpointStore,
    InMemoryCheckpointStore,
    RedisCheckpointStore,
)
from app.distributed.workflows.tracker import ExecutionTracker
from app.distributed.workflows.scheduler import WorkflowSchedule, WorkflowScheduler
from app.distributed.workflows.engine import DistributedWorkflowEngine

__all__ = [
    # State
    "WorkflowExecution",
    "WorkflowState",
    # Checkpoint
    "CheckpointStore",
    "InMemoryCheckpointStore",
    "RedisCheckpointStore",
    # Tracker
    "ExecutionTracker",
    # Scheduler
    "WorkflowSchedule",
    "WorkflowScheduler",
    # Engine
    "DistributedWorkflowEngine",
]
