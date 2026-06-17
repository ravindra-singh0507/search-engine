"""
Workflow Execution Tracker — Phase 8 Batch 3

=== THEORY ===

The ExecutionTracker provides an audit trail for distributed workflow execution.
Every significant event (start, step completion, step failure, workflow completion,
workflow failure) is recorded with timestamps for post-hoc debugging and observability.

This implements the Event Sourcing pattern at the workflow level: instead of
only tracking current state, we maintain the full history of state transitions.
This enables:

  - Debugging: "Why did this workflow fail?" — check the step history
  - Monitoring: "How long does each step take on average?"
  - Auditing: "Who triggered this workflow and when?"
  - Replay: reconstruct execution state from history

=== ARCHITECTURE ===

  ExecutionTracker
    │
    ├── _history: dict[execution_id → list[event_dict]]
    │     Each event has: type, timestamp, execution_id, step_id?, details?
    │
    └── _recent: deque[event_dict]
          Bounded FIFO of all events across all executions for dashboard/API

=== DATA STRUCTURES ===

  _history: dict[str, list[dict]]  — per-execution event log
  _recent:  deque[dict]            — bounded global event stream
  Event types: "start", "step_complete", "step_failed", "complete", "failed"

=== COMPLEXITY ===

  record_*():       O(1) amortised (deque/list append)
  get_history():    O(1) (dict lookup + list copy)
  get_recent():     O(min(limit, len(recent)))
  get_step_history(): O(H) where H = events for that execution
  stats():          O(N) where N = total events in _recent

=== SPACE COMPLEXITY ===

  O(max_history) for _recent (bounded deque)
  O(sum of all execution histories) for _history (unbounded per execution,
    but executions are cleaned up by the engine's cleanup_old)

=== PRODUCTION EQUIVALENTS ===

  Temporal:       Workflow event history (immutable append-only log)
  Airflow:        Task instance log + DAG run history
  Prefect:        Flow run timeline with task states
  Datadog APM:    Distributed tracing spans
"""

import logging
import threading
import time
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)


class ExecutionTracker:
    """
    Tracks workflow execution history and step-level details.

    Provides an audit trail for debugging and observability.
    All methods are thread-safe via a shared Lock.

    === USAGE ===

    The DistributedWorkflowEngine calls tracker methods at each
    lifecycle boundary:

      1. record_start()         — when execution begins
      2. record_step_complete() — after each successful step
      3. record_step_failed()   — after each failed step
      4. record_complete()      — when all steps finish successfully
      5. record_failed()        — when the workflow terminates in failure

    External consumers (API, dashboard) read via:
      - get_history()      — full event log for one execution
      - get_recent()       — most recent events across all executions
      - get_step_history() — events for a specific step within an execution
      - stats()            — aggregate statistics
    """

    def __init__(self, max_history: int = 1000) -> None:
        """
        Initialise the tracker.

        Args:
            max_history: Maximum events retained in the global _recent stream.
                         Older events are evicted FIFO. Per-execution history
                         is not bounded here (cleaned up by engine.cleanup_old).
        """
        self._history: dict[str, list[dict[str, Any]]] = {}
        self._recent: deque[dict[str, Any]] = deque(maxlen=max_history)
        self._lock = threading.Lock()
        self._total_recorded = 0

    def record_start(self, execution: Any) -> None:
        """
        Record workflow execution start.

        Args:
            execution: WorkflowExecution instance (uses execution_id,
                       workflow_name, goal attributes).
        """
        event = {
            "type":         "start",
            "execution_id": execution.execution_id,
            "workflow_name": execution.workflow_name,
            "goal":         execution.goal,
            "timestamp":    time.time(),
            "details":      {
                "total_steps": len(execution.steps),
                "state":       execution.state.value if hasattr(execution.state, "value") else str(execution.state),
            },
        }
        self._append_event(execution.execution_id, event)

    def record_step_complete(
        self, execution_id: str, step_id: str, result: dict
    ) -> None:
        """
        Record successful completion of a workflow step.

        Args:
            execution_id: The workflow execution identifier.
            step_id:      The step that completed.
            result:       Result data from the step (arbitrary dict).
        """
        event = {
            "type":         "step_complete",
            "execution_id": execution_id,
            "step_id":      step_id,
            "timestamp":    time.time(),
            "details":      result,
        }
        self._append_event(execution_id, event)

    def record_step_failed(
        self, execution_id: str, step_id: str, error: str
    ) -> None:
        """
        Record failure of a workflow step.

        Args:
            execution_id: The workflow execution identifier.
            step_id:      The step that failed.
            error:        Error message describing the failure.
        """
        event = {
            "type":         "step_failed",
            "execution_id": execution_id,
            "step_id":      step_id,
            "timestamp":    time.time(),
            "details":      {"error": error},
        }
        self._append_event(execution_id, event)

    def record_complete(self, execution_id: str) -> None:
        """
        Record successful completion of the entire workflow.

        Args:
            execution_id: The workflow execution identifier.
        """
        event = {
            "type":         "complete",
            "execution_id": execution_id,
            "timestamp":    time.time(),
            "details":      {},
        }
        self._append_event(execution_id, event)

    def record_failed(self, execution_id: str, error: str) -> None:
        """
        Record terminal failure of the workflow.

        Args:
            execution_id: The workflow execution identifier.
            error:        Error message describing the failure reason.
        """
        event = {
            "type":         "failed",
            "execution_id": execution_id,
            "timestamp":    time.time(),
            "details":      {"error": error},
        }
        self._append_event(execution_id, event)

    def get_history(self, execution_id: str) -> list[dict]:
        """
        Get the full event history for a specific execution.

        Returns an empty list if no events exist for this execution_id.
        """
        with self._lock:
            events = self._history.get(execution_id, [])
            return list(events)

    def get_recent(self, limit: int = 20) -> list[dict]:
        """
        Get the most recent events across all executions.

        Args:
            limit: Maximum number of events to return (most recent first).

        Returns events in reverse chronological order (newest first).
        """
        with self._lock:
            # deque is ordered oldest-first, so we reverse and slice
            all_events = list(self._recent)
            all_events.reverse()
            return all_events[:limit]

    def get_step_history(self, execution_id: str, step_id: str) -> list[dict]:
        """
        Get events for a specific step within an execution.

        Filters the execution's history to only events mentioning this step_id.
        """
        with self._lock:
            events = self._history.get(execution_id, [])
            return [e for e in events if e.get("step_id") == step_id]

    def stats(self) -> dict:
        """
        Return aggregate statistics about tracked executions.

        Returns:
            dict with keys:
              - total_executions: number of distinct executions tracked
              - total_events: total events in the global _recent stream
              - total_recorded: lifetime count of all events recorded
              - event_type_counts: breakdown by event type
        """
        with self._lock:
            type_counts: dict[str, int] = {}
            for event in self._recent:
                etype = event.get("type", "unknown")
                type_counts[etype] = type_counts.get(etype, 0) + 1

            return {
                "total_executions":  len(self._history),
                "total_events":      len(self._recent),
                "total_recorded":    self._total_recorded,
                "event_type_counts": type_counts,
            }

    def _append_event(self, execution_id: str, event: dict) -> None:
        """
        Internal: append event to both per-execution history and global stream.

        Thread-safe via self._lock.
        """
        with self._lock:
            if execution_id not in self._history:
                self._history[execution_id] = []
            self._history[execution_id].append(event)
            self._recent.append(event)
            self._total_recorded += 1

    def clear_execution(self, execution_id: str) -> None:
        """
        Remove all history for a specific execution.

        Used by the engine's cleanup_old to free memory.
        """
        with self._lock:
            self._history.pop(execution_id, None)
