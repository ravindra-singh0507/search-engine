"""
Distributed Agent Executor — Phase 8 Batch 3

=== THEORY ===

The DistributedAgentExecutor is the high-level facade that combines
queue + scheduler + worker pool into a single, easy-to-use interface.

It replaces/wraps the Phase 7 AgentOrchestrator for distributed use:
  - Phase 7 AgentOrchestrator: single-process, single-task execution
  - Phase 8 DistributedAgentExecutor: multi-worker, queue-based execution

The executor provides:
  1. Simple submit/wait API (submit_task → wait_for_result)
  2. Non-blocking result retrieval (get_result polling)
  3. Task cancellation
  4. Pool lifecycle management (setup → start → stop)
  5. Observability (stats, health check)

=== ARCHITECTURE ===

  Client
    │  submit_task(AgentTask)
    ▼
  DistributedAgentExecutor
    │
    ├── AgentTaskQueue        — bounded priority heap
    ├── AgentScheduler        — strategy-based dispatch
    └── AgentWorkerPool       — thread pool executing tasks
         ├── AgentWorker[0]
         ├── AgentWorker[1]
         └── ...

  Results flow back:
    Worker → Scheduler.complete() → Pool._results → Executor.get_result()

=== COMPLEXITY ===

  submit_task():      O(log N) — enqueue
  get_result():       O(1)     — dict lookup
  wait_for_result():  O(1) + blocking wait
  cancel_task():      O(1)     — mark in queue
  stats():            O(W)     — aggregate from W workers

=== SPACE COMPLEXITY ===

  O(N + R + C) where N = queued, R = running, C = completed results

=== TRADEOFFS ===

  + Single entry point — simplifies API layer integration
  + Abstracts queue/scheduler/pool complexity
  + Event bus integration for cross-service communication
  + Optional Redis for distributed state (future extension)
  - Additional indirection layer (slightly more memory)
  - Results accumulate in memory (bounded by completed task count)

=== PRODUCTION EQUIVALENTS ===

  Celery app:        Celery() instance combining broker + backend + workers
  Ray:               ray.init() + ray.remote() + ray.get()
  Temporal Client:   WorkflowClient submitting to TaskQueue
  Dramatiq:         dramatiq.broker + dramatiq.actor + worker process
"""

import logging
import threading
import time
from typing import Any, Optional

from app.agents.base import (
    Agent, AgentContext, AgentResult, AgentStatus, AgentTask, AgentType,
)
from app.config import AgentExecutionConfig
from app.distributed.agents.queue import AgentTaskQueue
from app.distributed.agents.scheduler import AgentScheduler, SchedulingStrategy
from app.distributed.agents.worker_pool import AgentWorkerPool

logger = logging.getLogger(__name__)


class DistributedAgentExecutor:
    """
    High-level API for distributed agent execution.

    Combines queue + scheduler + worker pool into a single interface.
    This replaces/wraps the Phase 7 AgentOrchestrator for distributed use.

    Usage:
        executor = DistributedAgentExecutor(
            config=config,
            agents={AgentType.RETRIEVAL: retrieval_agent, ...},
            context=agent_context,
            event_bus=bus,
        )
        executor.setup()
        executor.start()

        task_id = executor.submit_task(task)
        result = executor.wait_for_result(task_id, timeout=30.0)

        executor.stop()
    """

    def __init__(
        self,
        config: AgentExecutionConfig,
        agents: dict[AgentType, Agent],
        context: AgentContext,
        event_bus=None,
        redis_client=None,
    ) -> None:
        """
        Initialise the distributed executor.

        Args:
            config:       Execution configuration (workers, queue size, strategy).
            agents:       Registry of available agents (AgentType → Agent).
            context:      Shared agent context (retriever, DB, tools, etc.).
            event_bus:    Optional EventBus for publishing lifecycle events.
            redis_client: Optional Redis client for distributed state (future).
        """
        self._config = config
        self._agents = agents
        self._context = context
        self._event_bus = event_bus
        self._redis_client = redis_client

        # Components (created in setup())
        self._pool: Optional[AgentWorkerPool] = None
        self._running = False
        self._setup_done = False
        self._lock = threading.Lock()

    def setup(self) -> None:
        """
        Create the internal queue, scheduler, and worker pool.

        Must be called before start().  Idempotent — calling setup()
        multiple times has no effect after the first call.
        """
        if self._setup_done:
            return

        self._pool = AgentWorkerPool(
            config=self._config,
            agents=self._agents,
            context=self._context,
            event_bus=self._event_bus,
        )

        self._setup_done = True
        logger.info(
            "DistributedAgentExecutor setup complete "
            "(max_workers=%d, max_queue=%d, strategy=%s)",
            self._config.max_workers,
            self._config.max_queue_size,
            self._config.scheduling_strategy,
        )

    def start(self) -> None:
        """
        Start the worker pool and begin processing tasks.

        Calls setup() automatically if not already done.

        Idempotent — calling start() on a running executor is a no-op.
        """
        if self._running:
            return

        if not self._setup_done:
            self.setup()

        self._pool.start()
        self._running = True
        logger.info("DistributedAgentExecutor started")

    def stop(self) -> None:
        """
        Stop the executor gracefully.

        Waits for in-flight tasks to complete, then shuts down the
        worker pool.

        Idempotent — calling stop() on a stopped executor is a no-op.
        """
        if not self._running:
            return

        self._running = False
        if self._pool:
            self._pool.stop(graceful=True)

        logger.info("DistributedAgentExecutor stopped")

    def submit_task(self, task: AgentTask) -> str:
        """
        Submit a task for distributed execution.

        The task is enqueued and will be picked up by the next available
        worker.

        Args:
            task: The AgentTask to execute.

        Returns:
            The task_id for tracking.

        Raises:
            RuntimeError: If the executor is not running.
        """
        if not self._running:
            raise RuntimeError("DistributedAgentExecutor is not running")

        return self._pool.submit(task)

    def get_result(self, task_id: str) -> Optional[AgentResult]:
        """
        Retrieve the result of a completed task (non-blocking).

        Args:
            task_id: The ID of the task.

        Returns:
            AgentResult if the task has completed, None if still running.
        """
        if not self._pool:
            return None
        return self._pool.get_result(task_id)

    def wait_for_result(self, task_id: str, timeout: float = 60.0) -> Optional[AgentResult]:
        """
        Block until a task completes or timeout expires.

        This is the synchronous convenience method for callers that need
        to wait for a result inline (e.g. API request handlers).

        Args:
            task_id: The ID of the task to wait for.
            timeout: Maximum seconds to wait.

        Returns:
            AgentResult if the task completed within timeout, else None.
        """
        if not self._pool:
            return None
        return self._pool.wait_for_result(task_id, timeout=timeout)

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a queued task.

        If the task is still waiting in the queue, it is marked for
        removal.  If it is already being executed by a worker, this
        method returns False (the worker must finish or timeout).

        Args:
            task_id: The ID of the task to cancel.

        Returns:
            True if the task was cancelled, False otherwise.
        """
        if not self._pool:
            return False
        return self._pool.scheduler.cancel(task_id)

    def stats(self) -> dict[str, Any]:
        """
        Return executor-level statistics.

        Aggregates metrics from the queue, scheduler, and worker pool.

        Returns:
            Dict with comprehensive execution statistics.
        """
        if not self._pool:
            return {
                "running": False,
                "setup_done": self._setup_done,
            }

        pool_stats = self._pool.stats()
        scheduler_stats = self._pool.scheduler.stats()
        queue_stats = self._pool.queue.stats()

        return {
            "running":         self._running,
            "config": {
                "max_workers":        self._config.max_workers,
                "max_queue_size":     self._config.max_queue_size,
                "scheduling_strategy": self._config.scheduling_strategy,
                "worker_timeout_sec": self._config.worker_timeout_sec,
            },
            "pool":      pool_stats,
            "scheduler": scheduler_stats,
            "queue":     queue_stats,
        }

    def is_running(self) -> bool:
        """Return whether the executor is currently active."""
        return self._running
