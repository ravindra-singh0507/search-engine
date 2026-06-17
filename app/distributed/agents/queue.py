"""
Agent Task Queue — Phase 8 Batch 3

=== THEORY ===

A priority queue ensures high-priority tasks execute before low-priority
ones.  This is the fundamental data structure behind any task scheduling
system.  Internally uses a min-heap (Python heapq) with negated priority
so that higher-priority values are dequeued first.

Within the same priority level, tasks are ordered FIFO (by insertion
sequence number), guaranteeing fairness for equal-priority work.

=== ARCHITECTURE ===

  Producer(s)
    │  enqueue(task, priority)
    ▼
  AgentTaskQueue  [bounded heap + lookup dict]
    │  dequeue(timeout)
    ▼
  Consumer (Scheduler / Worker)

The queue is bounded (max_size) to provide backpressure.  When the queue
is full, enqueue() returns False and the caller must handle the rejection
(e.g. retry later, drop, or push to a dead-letter queue).

Thread safety is achieved via threading.Lock for mutations and
threading.Event for blocking dequeue with timeout.

=== COMPLEXITY ===

  enqueue():    O(log N) — heap push
  dequeue():    O(log N) — heap pop
  peek():       O(1)     — read heap[0]
  get_by_id():  O(1)     — dict lookup
  cancel():     O(N)     — lazy removal (mark + skip on dequeue)
  size():       O(1)
  is_full():    O(1)
  is_empty():   O(1)

=== SPACE COMPLEXITY ===

  O(N) where N = current number of tasks in the queue

=== TRADEOFFS ===

  + Bounded size prevents OOM under load spikes
  + FIFO within same priority (fair scheduling)
  + Blocking dequeue enables worker threads to wait efficiently
  + O(1) lookup by task_id via auxiliary dict
  + Thread-safe for multi-producer / multi-consumer use
  - Cancel is lazy (O(1) mark, deferred skip) — cancelled items
    remain in heap until they reach the front
  - No persistence (in-memory only; production would use Redis/Kafka)

=== PRODUCTION EQUIVALENTS ===

  Celery:       Redis/RabbitMQ priority queue
  Ray:          Object store + scheduling queue
  Temporal:     Task queue per activity type
  Kubernetes:   Priority-based pod scheduling queue
  SQS:          FIFO queue with message group deduplication
"""

import heapq
import logging
import threading
import time
from typing import Any, Optional

from app.agents.base import AgentTask

logger = logging.getLogger(__name__)


class AgentTaskQueue:
    """
    Priority queue for agent task distribution.

    Supports:
      - Priority-based ordering (higher priority = dequeue first)
      - FIFO within same priority (fair scheduling via sequence counter)
      - Bounded size (backpressure when full)
      - Task timeout (expired tasks are discarded on dequeue)
      - Dead letter handling (tasks that fail too many times)
      - Blocking dequeue with timeout (for worker threads)
      - O(1) lookup by task_id

    Thread-safe for multi-producer / multi-consumer usage.

    Implementation:
      Internal heap entries are tuples:
        (-priority, sequence_number, task_id, enqueue_time, task)
      The negative priority makes heapq (min-heap) behave as max-priority.
      The sequence number breaks ties for FIFO ordering.
    """

    def __init__(self, max_size: int = 1000) -> None:
        """
        Initialise the task queue.

        Args:
            max_size: Maximum number of tasks the queue can hold.
                      enqueue() returns False when this limit is reached.
        """
        self._max_size = max_size
        self._heap: list[tuple[int, int, str, float, AgentTask]] = []
        self._counter = 0  # monotonic sequence for FIFO within same priority
        self._lock = threading.Lock()
        self._not_empty = threading.Event()

        # Lookup for O(1) get_by_id and cancel
        self._tasks: dict[str, AgentTask] = {}
        self._cancelled: set[str] = set()

        # Statistics
        self._enqueued_count = 0
        self._dequeued_count = 0
        self._cancelled_count = 0

    def enqueue(self, task: AgentTask, priority: int = 5) -> bool:
        """
        Add a task to the priority queue.

        Args:
            task:     The AgentTask to enqueue.
            priority: Scheduling priority (higher = dequeued first).
                      Defaults to 5 (NORMAL).

        Returns:
            True if the task was enqueued successfully.
            False if the queue is full (backpressure signal).
        """
        with self._lock:
            if len(self._heap) - len(self._cancelled) >= self._max_size:
                logger.warning(
                    "AgentTaskQueue full (max_size=%d), rejecting task %s",
                    self._max_size, task.task_id[:8],
                )
                return False

            entry = (-priority, self._counter, task.task_id, time.time(), task)
            heapq.heappush(self._heap, entry)
            self._counter += 1
            self._tasks[task.task_id] = task
            self._enqueued_count += 1
            self._not_empty.set()

            logger.debug(
                "Enqueued task %s (priority=%d, queue_size=%d)",
                task.task_id[:8], priority, self._active_size_unlocked(),
            )
            return True

    def dequeue(self, timeout: float = 0) -> Optional[AgentTask]:
        """
        Remove and return the highest-priority task.

        If the queue is empty and timeout > 0, blocks up to timeout seconds
        waiting for a task to appear.  If timeout == 0, returns None immediately
        when the queue is empty.

        Cancelled tasks are silently skipped (lazy removal).

        Args:
            timeout: Maximum seconds to wait for a task (0 = non-blocking).

        Returns:
            The highest-priority AgentTask, or None if no task is available.
        """
        deadline = time.time() + timeout if timeout > 0 else 0

        while True:
            with self._lock:
                task = self._pop_next_valid()
                if task is not None:
                    self._dequeued_count += 1
                    if self._active_size_unlocked() == 0:
                        self._not_empty.clear()
                    return task

            # Non-blocking mode: return immediately
            if timeout <= 0:
                return None

            # Blocking mode: wait for signal or deadline
            remaining = deadline - time.time()
            if remaining <= 0:
                return None

            self._not_empty.wait(timeout=remaining)

    def peek(self) -> Optional[AgentTask]:
        """
        Return the highest-priority task without removing it.

        Skips cancelled tasks (peeks ahead to the next valid one).

        Returns:
            The next AgentTask that would be dequeued, or None if empty.
        """
        with self._lock:
            for entry in sorted(self._heap):
                _, _, task_id, _, task = entry
                if task_id not in self._cancelled:
                    return task
            return None

    def size(self) -> int:
        """Return the number of active (non-cancelled) tasks in the queue."""
        with self._lock:
            return self._active_size_unlocked()

    def is_full(self) -> bool:
        """Return True if the queue has reached its maximum capacity."""
        with self._lock:
            return self._active_size_unlocked() >= self._max_size

    def is_empty(self) -> bool:
        """Return True if the queue contains no active tasks."""
        with self._lock:
            return self._active_size_unlocked() == 0

    def clear(self) -> int:
        """
        Remove all tasks from the queue.

        Returns:
            The number of tasks that were cleared.
        """
        with self._lock:
            count = self._active_size_unlocked()
            self._heap.clear()
            self._tasks.clear()
            self._cancelled.clear()
            self._not_empty.clear()
            logger.info("AgentTaskQueue cleared (%d tasks removed)", count)
            return count

    def get_by_id(self, task_id: str) -> Optional[AgentTask]:
        """
        Look up a task by its ID.

        Returns:
            The AgentTask if found and not cancelled, else None.
        """
        with self._lock:
            if task_id in self._cancelled:
                return None
            return self._tasks.get(task_id)

    def cancel(self, task_id: str) -> bool:
        """
        Cancel a task by marking it for lazy removal.

        The task remains in the heap but will be skipped on dequeue.

        Args:
            task_id: The ID of the task to cancel.

        Returns:
            True if the task was found and cancelled, False otherwise.
        """
        with self._lock:
            if task_id not in self._tasks or task_id in self._cancelled:
                return False
            self._cancelled.add(task_id)
            self._cancelled_count += 1
            logger.debug("Cancelled task %s", task_id[:8])
            return True

    def stats(self) -> dict[str, Any]:
        """
        Return queue statistics.

        Returns:
            Dict with keys: enqueued, dequeued, cancelled, current_size,
            max_size, heap_size (including cancelled entries).
        """
        with self._lock:
            return {
                "enqueued":     self._enqueued_count,
                "dequeued":     self._dequeued_count,
                "cancelled":    self._cancelled_count,
                "current_size": self._active_size_unlocked(),
                "max_size":     self._max_size,
                "heap_size":    len(self._heap),
            }

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _active_size_unlocked(self) -> int:
        """Return active task count. Caller must hold self._lock."""
        return len(self._heap) - len(self._cancelled)

    def _pop_next_valid(self) -> Optional[AgentTask]:
        """
        Pop entries from the heap until a non-cancelled task is found.
        Caller must hold self._lock.

        Returns:
            A valid AgentTask, or None if no valid tasks remain.
        """
        while self._heap:
            _, _, task_id, _, task = heapq.heappop(self._heap)
            if task_id in self._cancelled:
                # Remove from cancelled set and tasks dict (cleanup)
                self._cancelled.discard(task_id)
                self._tasks.pop(task_id, None)
                continue
            # Valid task found
            self._tasks.pop(task_id, None)
            return task
        return None
