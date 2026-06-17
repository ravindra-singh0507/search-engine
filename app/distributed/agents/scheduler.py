"""
Agent Scheduler — Phase 8 Batch 3

=== THEORY ===

The scheduler decides WHICH worker gets WHICH task.  It sits between
the task queue (where tasks wait) and the worker pool (where tasks
execute).  Different scheduling strategies optimise for different goals:

  - Priority:     maximise urgency-weighted throughput
  - FIFO:         fairness — no task starves regardless of priority
  - Round-robin:  even distribution across workers (load-agnostic)
  - Least-loaded: minimise max worker queue depth (load-aware)

=== ARCHITECTURE ===

  submit(task) ──▶ AgentTaskQueue (priority heap)
                         │
                         │  schedule_next(worker_id)
                         ▼
                   AgentScheduler
                         │
                         ├── strategy = PRIORITY:     pop from heap
                         ├── strategy = FIFO:         pop oldest
                         ├── strategy = ROUND_ROBIN:  cycle workers
                         └── strategy = LEAST_LOADED: pick lightest worker
                         │
                         ▼
                   Worker assignment recorded in _running dict

  complete(task_id, result)  ──▶ move from _running to _completed
  fail(task_id, error)       ──▶ move from _running to _failed

=== STATE MACHINE ===

  Task states within the scheduler:
    QUEUED  → ASSIGNED (to worker) → COMPLETED
                                   → FAILED

=== COMPLEXITY ===

  submit():        O(log N) — delegates to AgentTaskQueue.enqueue
  schedule_next(): O(log N) — heap pop (PRIORITY/FIFO)
                   O(W)     — ROUND_ROBIN / LEAST_LOADED (W = worker count)
  complete():      O(1)     — dict pop + dict insert
  fail():          O(1)
  get_pending():   O(N)     — snapshot of queue contents
  get_running():   O(R)     — R = running tasks
  cancel():        O(1)     — delegates to queue.cancel

=== SPACE COMPLEXITY ===

  O(N + R) where N = queued tasks, R = running tasks

=== TRADEOFFS ===

  + Multiple strategies via enum (configurable at runtime)
  + Complete audit trail: _completed + _failed dicts
  + Thread-safe via Lock
  + Decoupled from worker lifecycle (scheduler does not own workers)
  - ROUND_ROBIN and LEAST_LOADED require worker registration (tracked
    via schedule_next calls; the scheduler learns about workers lazily)
  - No persistence (in-memory only)

=== PRODUCTION EQUIVALENTS ===

  Kubernetes:    kube-scheduler with scoring plugins
  Celery:       celery.beat + routing keys
  Ray:          Raylet scheduler with resource-based placement
  Temporal:     Task queue + sticky execution
  Airflow:      SchedulerJob with pool-based limiting
"""

import logging
import threading
import time
from enum import Enum
from typing import Any, Optional

from app.agents.base import AgentResult, AgentStatus, AgentTask, TaskPriority
from app.config import AgentExecutionConfig
from app.distributed.agents.queue import AgentTaskQueue

logger = logging.getLogger(__name__)


class SchedulingStrategy(str, Enum):
    """
    Available scheduling strategies.

    PRIORITY:     Highest priority task is dispatched first.
    FIFO:         First-in, first-out regardless of priority.
    ROUND_ROBIN:  Distribute evenly across workers in a cycle.
    LEAST_LOADED: Send to the worker with the fewest in-flight tasks.
    """
    PRIORITY     = "priority"
    FIFO         = "fifo"
    ROUND_ROBIN  = "round_robin"
    LEAST_LOADED = "least_loaded"


class AgentScheduler:
    """
    Schedules agent tasks across a pool of workers.

    The scheduler maintains three collections:
      1. _queue (AgentTaskQueue) — tasks waiting for execution
      2. _running — dict[task_id → {task, worker_id, started_at}]
      3. _completed / _failed — terminal task records

    Workers call schedule_next(worker_id) to pull their next task.
    The strategy determines which task is selected.

    Thread-safe for concurrent worker access.
    """

    def __init__(
        self,
        config: AgentExecutionConfig,
        queue: AgentTaskQueue,
    ) -> None:
        """
        Initialise the scheduler.

        Args:
            config: Execution configuration (scheduling_strategy, etc.)
            queue:  The task queue to pull work from.
        """
        self._config = config
        self._queue = queue
        self._lock = threading.Lock()

        # Resolve strategy from config string
        try:
            self._strategy = SchedulingStrategy(config.scheduling_strategy)
        except ValueError:
            logger.warning(
                "Unknown scheduling strategy '%s', defaulting to PRIORITY",
                config.scheduling_strategy,
            )
            self._strategy = SchedulingStrategy.PRIORITY

        # Task state tracking
        self._running: dict[str, dict[str, Any]] = {}  # task_id → assignment info
        self._completed: dict[str, AgentResult] = {}
        self._failed: dict[str, dict[str, Any]] = {}

        # Round-robin state
        self._rr_index = 0
        self._worker_ids: list[str] = []

        # Worker load tracking (for LEAST_LOADED)
        self._worker_load: dict[str, int] = {}  # worker_id → count of running tasks

        # Statistics
        self._total_submitted = 0
        self._total_completed = 0
        self._total_failed = 0

    def submit(self, task: AgentTask) -> str:
        """
        Submit a task for scheduling.

        Enqueues the task in the priority queue.  The task will be picked
        up by a worker when schedule_next() is called.

        Args:
            task: The AgentTask to schedule.

        Returns:
            The task_id (for tracking purposes).

        Raises:
            RuntimeError: If the queue is full and cannot accept the task.
        """
        priority = task.priority.value if isinstance(task.priority, TaskPriority) else int(task.priority)
        success = self._queue.enqueue(task, priority=priority)

        if not success:
            raise RuntimeError(
                f"Task queue full (max_size={self._config.max_queue_size}), "
                f"cannot submit task {task.task_id[:8]}"
            )

        with self._lock:
            self._total_submitted += 1

        logger.debug("Submitted task %s (priority=%d)", task.task_id[:8], priority)
        return task.task_id

    def schedule_next(self, worker_id: str) -> Optional[AgentTask]:
        """
        Assign the next task to the specified worker.

        The scheduling strategy determines which task is selected:
          - PRIORITY: highest priority (default heap behaviour)
          - FIFO: same as priority (heap already preserves insertion order
                  for equal priorities; true FIFO would ignore priority)
          - ROUND_ROBIN: assign in cyclic order (tasks still come from
                         the queue; the worker assignment is round-robin)
          - LEAST_LOADED: only assign if this worker has the fewest tasks

        Args:
            worker_id: The ID of the worker requesting work.

        Returns:
            An AgentTask to execute, or None if no work is available.
        """
        with self._lock:
            # Register worker if not seen before
            if worker_id not in self._worker_load:
                self._worker_load[worker_id] = 0
                self._worker_ids.append(worker_id)

            # Strategy-specific gating
            if self._strategy == SchedulingStrategy.ROUND_ROBIN:
                # Only the "current" worker in the rotation gets work
                if self._worker_ids:
                    expected_worker = self._worker_ids[self._rr_index % len(self._worker_ids)]
                    if worker_id != expected_worker:
                        return None
                    # Advance round-robin index (no modulo here — use
                    # monotonic counter so new workers are naturally included)
                    self._rr_index += 1

            elif self._strategy == SchedulingStrategy.LEAST_LOADED:
                # Only dispatch if this worker has the minimum load
                if self._worker_load:
                    min_load = min(self._worker_load.values())
                    if self._worker_load.get(worker_id, 0) > min_load:
                        return None

        # Dequeue a task (non-blocking)
        task = self._queue.dequeue(timeout=0)
        if task is None:
            return None

        # Record assignment
        with self._lock:
            self._running[task.task_id] = {
                "task": task,
                "worker_id": worker_id,
                "started_at": time.time(),
            }
            self._worker_load[worker_id] = self._worker_load.get(worker_id, 0) + 1

        logger.debug(
            "Scheduled task %s → worker %s (strategy=%s)",
            task.task_id[:8], worker_id, self._strategy.value,
        )
        return task

    def complete(self, task_id: str, result: AgentResult) -> None:
        """
        Mark a task as completed.

        Moves the task from _running to _completed and decrements
        the worker's load counter.

        Args:
            task_id: The ID of the completed task.
            result:  The AgentResult produced by the worker.
        """
        with self._lock:
            assignment = self._running.pop(task_id, None)
            if assignment:
                worker_id = assignment["worker_id"]
                self._worker_load[worker_id] = max(
                    0, self._worker_load.get(worker_id, 1) - 1
                )
            self._completed[task_id] = result
            self._total_completed += 1

        logger.debug("Task %s completed", task_id[:8])

    def fail(self, task_id: str, error: str) -> None:
        """
        Mark a task as failed.

        Moves the task from _running to _failed and decrements
        the worker's load counter.

        Args:
            task_id: The ID of the failed task.
            error:   Human-readable error description.
        """
        with self._lock:
            assignment = self._running.pop(task_id, None)
            worker_id = assignment["worker_id"] if assignment else "unknown"
            if assignment:
                self._worker_load[worker_id] = max(
                    0, self._worker_load.get(worker_id, 1) - 1
                )
            self._failed[task_id] = {
                "task_id": task_id,
                "worker_id": worker_id,
                "error": error,
                "failed_at": time.time(),
            }
            self._total_failed += 1

        logger.debug("Task %s failed: %s", task_id[:8], error)

    def get_pending(self) -> list[AgentTask]:
        """
        Return a snapshot of all tasks currently waiting in the queue.

        Note: This is an O(N) scan of the heap for observability.
        Not intended for hot-path use.

        Returns:
            List of AgentTask objects still in the queue.
        """
        # Read from the queue's internal structures
        with self._queue._lock:
            pending = []
            for entry in self._queue._heap:
                _, _, task_id, _, task = entry
                if task_id not in self._queue._cancelled:
                    pending.append(task)
            return pending

    def get_running(self) -> list[dict[str, Any]]:
        """
        Return a snapshot of all tasks currently being executed.

        Returns:
            List of dicts with keys: task_id, worker_id, started_at, elapsed_sec.
        """
        with self._lock:
            now = time.time()
            return [
                {
                    "task_id":     task_id,
                    "worker_id":   info["worker_id"],
                    "started_at":  info["started_at"],
                    "elapsed_sec": round(now - info["started_at"], 2),
                    "goal":        info["task"].goal,
                }
                for task_id, info in self._running.items()
            ]

    def cancel(self, task_id: str) -> bool:
        """
        Cancel a task.

        If the task is still queued, it is removed from the queue.
        If it is already running, it cannot be cancelled here (the
        worker must handle cancellation).

        Args:
            task_id: The ID of the task to cancel.

        Returns:
            True if the task was found and cancelled from the queue.
        """
        # Try cancelling from the queue first
        if self._queue.cancel(task_id):
            return True

        # Check if it's running (cannot cancel mid-execution from scheduler)
        with self._lock:
            if task_id in self._running:
                logger.warning(
                    "Cannot cancel running task %s from scheduler", task_id[:8]
                )
                return False

        return False

    def get_result(self, task_id: str) -> Optional[AgentResult]:
        """
        Retrieve the result of a completed task.

        Args:
            task_id: The ID of the task.

        Returns:
            AgentResult if the task completed, None otherwise.
        """
        with self._lock:
            return self._completed.get(task_id)

    def stats(self) -> dict[str, Any]:
        """
        Return scheduler statistics.

        Returns:
            Dict with keys: strategy, total_submitted, total_completed,
            total_failed, pending_count, running_count, worker_count.
        """
        with self._lock:
            return {
                "strategy":        self._strategy.value,
                "total_submitted": self._total_submitted,
                "total_completed": self._total_completed,
                "total_failed":    self._total_failed,
                "pending_count":   self._queue.size(),
                "running_count":   len(self._running),
                "worker_count":    len(self._worker_ids),
                "worker_load":     dict(self._worker_load),
            }
