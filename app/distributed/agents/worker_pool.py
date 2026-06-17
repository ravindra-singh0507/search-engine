"""
Agent Worker Pool — Phase 8 Batch 3

=== THEORY ===

A worker pool maintains a fixed or elastic set of workers that consume
tasks from a scheduler and execute them concurrently.  This pattern
decouples task submission from execution, enabling:

  1. Bounded concurrency — limit resource consumption
  2. Elastic scaling — grow/shrink pool based on demand
  3. Fault isolation — one worker failure does not crash the system
  4. Graceful shutdown — drain in-flight work before stopping

Each worker runs in its own thread and loops:
  poll scheduler for next task → execute → report result → repeat

=== ARCHITECTURE ===

  AgentWorkerPool
    ├── AgentWorker[0]  ─── thread_0 ─── loop(schedule → execute → complete)
    ├── AgentWorker[1]  ─── thread_1 ─── loop(...)
    ├── ...
    └── AgentWorker[N-1] ── thread_N ─── loop(...)

  Each worker:
    1. Calls scheduler.schedule_next(worker_id) to get a task
    2. Looks up the agent for the task's task_type
    3. Calls agent.run(task, context) to produce a result
    4. Reports result back to the scheduler via complete() or fail()
    5. Publishes events (AGENT_TASK_COMPLETED / AGENT_TASK_FAILED)

=== COMPLEXITY ===

  AgentWorker.execute():    O(agent.run) — depends on agent type
  AgentWorkerPool.start():  O(W) — spawn W threads
  AgentWorkerPool.stop():   O(W) — join W threads
  AgentWorkerPool.submit(): O(log N) — delegates to scheduler.submit
  AgentWorkerPool.scale_to(): O(|delta|) — spawn or stop workers

=== SPACE COMPLEXITY ===

  O(W * S) where W = workers, S = per-worker state

=== TRADEOFFS ===

  + Fixed pool: predictable resource usage, no runaway thread creation
  + Elastic scale_to(): adapt to load without restart
  + Graceful drain: finish in-flight tasks before shutdown
  + Event bus integration: other services react to task completion
  + Per-worker metrics: identify slow or failing workers
  - Thread-based (GIL limits CPU parallelism; fine for IO-bound agents)
  - No work-stealing (idle worker cannot take work from busy worker's local queue)
  - No checkpointing (task restarts from scratch on worker failure)

=== PRODUCTION EQUIVALENTS ===

  Celery:          Worker pool with prefork/gevent/eventlet execution pools
  Ray:             Actor pool with auto-scaling and placement groups
  ThreadPoolExecutor: stdlib concurrent.futures (simpler, no task routing)
  Temporal:        Activity worker with poller and task token
  Kubernetes:      ReplicaSet of pod workers consuming from a queue
"""

import logging
import threading
import time
import uuid
from enum import Enum
from typing import Any, Optional

from app.agents.base import (
    Agent, AgentContext, AgentResult, AgentStatus, AgentTask, AgentType,
)
from app.config import AgentExecutionConfig
from app.distributed.agents.queue import AgentTaskQueue
from app.distributed.agents.scheduler import AgentScheduler
from app.events.models import Event, EventMetadata
from app.events import topics as event_topics

logger = logging.getLogger(__name__)


class AgentWorkerState(str, Enum):
    """
    State of an individual agent worker.

    IDLE:      Worker is waiting for a task.
    BUSY:      Worker is currently executing a task.
    DRAINING:  Worker will not accept new tasks; finishing current one.
    STOPPED:   Worker has terminated.
    """
    IDLE     = "idle"
    BUSY     = "busy"
    DRAINING = "draining"
    STOPPED  = "stopped"


class AgentWorker:
    """
    Single worker that processes agent tasks.

    The worker runs in a dedicated thread and continuously polls the
    scheduler for work.  When a task is received, the worker looks up
    the correct Agent instance, executes the task, and reports the result.

    Lifecycle:
      IDLE → BUSY → IDLE → BUSY → ... → DRAINING → STOPPED
    """

    def __init__(
        self,
        worker_id: str,
        agents: dict[AgentType, Agent],
        context: AgentContext,
    ) -> None:
        """
        Initialise a worker.

        Args:
            worker_id: Unique identifier for this worker.
            agents:    Registry of available agents (type → instance).
            context:   Shared agent context (retriever, DB, tools, etc.).
        """
        self.worker_id = worker_id
        self._agents = agents
        self._context = context
        self._state = AgentWorkerState.IDLE
        self._current_task: Optional[AgentTask] = None
        self._lock = threading.Lock()

        # Metrics
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._total_latency_ms = 0.0

    def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute a single agent task.

        Looks up the agent by task_type, transitions state to BUSY,
        runs the agent, and returns the result.

        Args:
            task: The AgentTask to execute.

        Returns:
            AgentResult with the execution outcome.
        """
        with self._lock:
            self._state = AgentWorkerState.BUSY
            self._current_task = task

        t_start = time.perf_counter()

        try:
            # Resolve agent type from task_type string
            agent = self._resolve_agent(task.task_type)

            if agent is None:
                result = AgentResult(
                    task_id=task.task_id,
                    agent_type=AgentType.RETRIEVAL,  # fallback type
                    status=AgentStatus.FAILED,
                    output=None,
                    error=f"No agent registered for task_type '{task.task_type}'",
                )
            else:
                result = agent.run(task, self._context)

        except Exception as exc:
            logger.exception(
                "Worker %s unhandled exception executing task %s: %s",
                self.worker_id, task.task_id[:8], exc,
            )
            result = AgentResult(
                task_id=task.task_id,
                agent_type=AgentType.RETRIEVAL,
                status=AgentStatus.FAILED,
                output=None,
                error=f"Worker exception: {exc}",
            )

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        # Update metrics
        with self._lock:
            self._current_task = None
            if result.is_success():
                self._tasks_completed += 1
            else:
                self._tasks_failed += 1
            self._total_latency_ms += elapsed_ms

            if self._state == AgentWorkerState.BUSY:
                self._state = AgentWorkerState.IDLE

        return result

    def _resolve_agent(self, task_type: str) -> Optional[Agent]:
        """
        Resolve an Agent instance from a task_type string.

        Tries to match the task_type to an AgentType enum value,
        then looks up the corresponding agent in the registry.
        """
        # Direct match by AgentType value
        for agent_type, agent in self._agents.items():
            if agent_type.value == task_type:
                return agent

        # Fallback: try first available agent
        if self._agents:
            return next(iter(self._agents.values()))

        return None

    @property
    def state(self) -> AgentWorkerState:
        """Return the current worker state."""
        with self._lock:
            return self._state

    @state.setter
    def state(self, new_state: AgentWorkerState) -> None:
        """Set the worker state (used by the pool for drain/stop)."""
        with self._lock:
            self._state = new_state

    @property
    def current_task(self) -> Optional[AgentTask]:
        """Return the task currently being executed, if any."""
        with self._lock:
            return self._current_task

    def stats(self) -> dict[str, Any]:
        """
        Return worker performance metrics.

        Returns:
            Dict with keys: worker_id, state, tasks_completed, tasks_failed,
            avg_latency_ms, current_task_id.
        """
        with self._lock:
            total_tasks = self._tasks_completed + self._tasks_failed
            avg_latency = (
                self._total_latency_ms / total_tasks if total_tasks > 0 else 0.0
            )
            return {
                "worker_id":       self.worker_id,
                "state":           self._state.value,
                "tasks_completed": self._tasks_completed,
                "tasks_failed":    self._tasks_failed,
                "avg_latency_ms":  round(avg_latency, 2),
                "current_task_id": self._current_task.task_id if self._current_task else None,
            }


class AgentWorkerPool:
    """
    Pool of agent workers for concurrent task execution.

    Manages a set of AgentWorker instances, each running in its own
    thread.  The pool coordinates task dispatch via the AgentScheduler
    and provides lifecycle management (start, stop, drain, scale).

    Usage:
        pool = AgentWorkerPool(config, agents, context, event_bus=bus)
        pool.start()
        task_id = pool.submit(task)
        result = pool.get_result(task_id)
        pool.stop()
    """

    def __init__(
        self,
        config: AgentExecutionConfig,
        agents: dict[AgentType, Agent],
        context: AgentContext,
        event_bus=None,
    ) -> None:
        """
        Initialise the worker pool.

        Args:
            config:    Execution configuration (max_workers, queue size, etc.).
            agents:    Registry of available agents.
            context:   Shared agent context.
            event_bus: Optional EventBus for publishing task lifecycle events.
        """
        self._config = config
        self._agents = agents
        self._context = context
        self._event_bus = event_bus

        # Build internal queue and scheduler
        self._queue = AgentTaskQueue(max_size=config.max_queue_size)
        self._scheduler = AgentScheduler(config=config, queue=self._queue)

        # Workers and threads
        self._workers: list[AgentWorker] = []
        self._threads: list[threading.Thread] = []
        self._running = False
        self._draining = False
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Results storage
        self._results: dict[str, AgentResult] = {}
        self._result_events: dict[str, threading.Event] = {}  # for wait_for_result

    def start(self) -> None:
        """
        Start all workers in the pool.

        Creates worker instances and spawns a thread for each one.
        Each thread runs a loop that polls the scheduler for tasks.

        Idempotent — calling start() on a running pool is a no-op.
        """
        if self._running:
            logger.debug("AgentWorkerPool already running")
            return

        self._running = True
        self._draining = False
        self._stop_event.clear()

        num_workers = self._config.max_workers
        for i in range(num_workers):
            worker_id = f"agent-worker-{i}"
            worker = AgentWorker(
                worker_id=worker_id,
                agents=self._agents,
                context=self._context,
            )
            self._workers.append(worker)

            thread = threading.Thread(
                target=self._worker_loop,
                args=(worker,),
                name=f"AgentWorkerThread-{i}",
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()

        logger.info(
            "AgentWorkerPool started with %d workers", num_workers
        )

    def stop(self, graceful: bool = True) -> None:
        """
        Stop the worker pool.

        Args:
            graceful: If True, wait for in-flight tasks to complete before
                      stopping.  If False, stop immediately.
        """
        if not self._running:
            return

        if graceful:
            self._draining = True
            # Wait for running tasks to complete
            timeout = self._config.worker_timeout_sec
            deadline = time.time() + timeout
            while time.time() < deadline:
                with self._lock:
                    busy_count = sum(
                        1 for w in self._workers
                        if w.state == AgentWorkerState.BUSY
                    )
                if busy_count == 0:
                    break
                time.sleep(0.05)

        self._running = False
        self._stop_event.set()

        # Join all threads with a timeout
        for thread in self._threads:
            thread.join(timeout=5.0)

        # Mark all workers as stopped
        for worker in self._workers:
            worker.state = AgentWorkerState.STOPPED

        logger.info(
            "AgentWorkerPool stopped (workers=%d)", len(self._workers)
        )

    def submit(self, task: AgentTask) -> str:
        """
        Submit a task for execution by the worker pool.

        The task is enqueued via the scheduler and will be picked up
        by the next available worker.

        Args:
            task: The AgentTask to execute.

        Returns:
            The task_id for tracking.

        Raises:
            RuntimeError: If the pool is not running or the queue is full.
        """
        if not self._running:
            raise RuntimeError("AgentWorkerPool is not running")
        if self._draining:
            raise RuntimeError("AgentWorkerPool is draining; not accepting new tasks")

        # Create a result event for wait_for_result
        with self._lock:
            self._result_events[task.task_id] = threading.Event()

        task_id = self._scheduler.submit(task)

        # Publish task created event
        self._publish_event(
            event_topics.AGENT_TASK_CREATED,
            {"task_id": task_id, "goal": task.goal, "priority": task.priority.value},
        )

        return task_id

    def get_result(self, task_id: str) -> Optional[AgentResult]:
        """
        Retrieve the result of a completed task.

        Args:
            task_id: The ID of the task.

        Returns:
            AgentResult if available, None if still running or unknown.
        """
        with self._lock:
            return self._results.get(task_id)

    def wait_for_result(self, task_id: str, timeout: float = 60.0) -> Optional[AgentResult]:
        """
        Block until a task completes or timeout expires.

        Args:
            task_id: The ID of the task to wait for.
            timeout: Maximum seconds to wait.

        Returns:
            AgentResult if the task completed within timeout, else None.
        """
        with self._lock:
            event = self._result_events.get(task_id)
            # If result already available, return immediately
            if task_id in self._results:
                return self._results[task_id]

        if event is None:
            return None

        event.wait(timeout=timeout)

        with self._lock:
            return self._results.get(task_id)

    def scale_to(self, num_workers: int) -> None:
        """
        Adjust the pool size to the specified number of workers.

        If num_workers > current: spawn new workers.
        If num_workers < current: drain and stop excess workers.

        Args:
            num_workers: Target number of workers (must be >= 1).
        """
        if num_workers < 1:
            num_workers = 1

        current = len(self._workers)

        if num_workers > current:
            # Scale up
            for i in range(current, num_workers):
                worker_id = f"agent-worker-{i}"
                worker = AgentWorker(
                    worker_id=worker_id,
                    agents=self._agents,
                    context=self._context,
                )
                self._workers.append(worker)

                thread = threading.Thread(
                    target=self._worker_loop,
                    args=(worker,),
                    name=f"AgentWorkerThread-{i}",
                    daemon=True,
                )
                self._threads.append(thread)
                if self._running:
                    thread.start()

            logger.info("Scaled up to %d workers (+%d)", num_workers, num_workers - current)

        elif num_workers < current:
            # Scale down: mark excess workers as draining
            excess = self._workers[num_workers:]
            for worker in excess:
                worker.state = AgentWorkerState.DRAINING

            # Wait briefly for them to finish current tasks
            time.sleep(0.1)

            # Stop excess workers
            for worker in excess:
                worker.state = AgentWorkerState.STOPPED

            self._workers = self._workers[:num_workers]
            self._threads = self._threads[:num_workers]
            logger.info("Scaled down to %d workers (-%d)", num_workers, current - num_workers)

    def drain(self) -> None:
        """
        Stop accepting new tasks and finish current work.

        After drain(), submit() will raise RuntimeError.
        Workers continue processing their current tasks until complete.
        """
        self._draining = True
        for worker in self._workers:
            if worker.state == AgentWorkerState.IDLE:
                worker.state = AgentWorkerState.DRAINING

        logger.info("AgentWorkerPool draining (no new tasks accepted)")

    def get_workers(self) -> list[dict[str, Any]]:
        """
        Return status of all workers.

        Returns:
            List of dicts with keys: worker_id, state, current_task_id.
        """
        return [
            {
                "worker_id":       w.worker_id,
                "state":           w.state.value,
                "current_task_id": w.current_task.task_id if w.current_task else None,
            }
            for w in self._workers
        ]

    def stats(self) -> dict[str, Any]:
        """
        Return pool-level statistics.

        Returns:
            Dict with keys: total_workers, idle, busy, draining, stopped,
            tasks_completed, tasks_failed, queue_size, running_tasks.
        """
        states = [w.state for w in self._workers]
        worker_stats = [w.stats() for w in self._workers]

        total_completed = sum(ws["tasks_completed"] for ws in worker_stats)
        total_failed = sum(ws["tasks_failed"] for ws in worker_stats)

        return {
            "total_workers":   len(self._workers),
            "idle":            states.count(AgentWorkerState.IDLE),
            "busy":            states.count(AgentWorkerState.BUSY),
            "draining":        states.count(AgentWorkerState.DRAINING),
            "stopped":         states.count(AgentWorkerState.STOPPED),
            "tasks_completed": total_completed,
            "tasks_failed":    total_failed,
            "queue_size":      self._queue.size(),
            "running_tasks":   len(self._scheduler.get_running()),
            "is_running":      self._running,
            "is_draining":     self._draining,
        }

    @property
    def scheduler(self) -> AgentScheduler:
        """Expose the scheduler for direct access (e.g. by DistributedAgentExecutor)."""
        return self._scheduler

    @property
    def queue(self) -> AgentTaskQueue:
        """Expose the queue for direct access."""
        return self._queue

    # ── Internal ─────────────────────────────────────────────────────────────

    def _worker_loop(self, worker: AgentWorker) -> None:
        """
        Main loop for a worker thread.

        Continuously polls the scheduler for tasks, executes them,
        and reports results.  Exits when the pool is stopped or the
        worker is marked STOPPED/DRAINING with no task.
        """
        logger.debug("Worker %s loop started", worker.worker_id)

        while not self._stop_event.is_set():
            # Check if this worker should stop
            if worker.state in (AgentWorkerState.STOPPED, AgentWorkerState.DRAINING):
                break

            # Poll for a task
            task = self._scheduler.schedule_next(worker.worker_id)

            if task is None:
                # No work available; brief sleep to avoid busy-waiting
                self._stop_event.wait(timeout=0.05)
                continue

            # Execute the task
            result = worker.execute(task)

            # Report to scheduler
            if result.is_success():
                self._scheduler.complete(task.task_id, result)
                self._publish_event(
                    event_topics.AGENT_TASK_COMPLETED,
                    {
                        "task_id": task.task_id,
                        "agent_type": result.agent_type.value,
                        "latency_ms": result.latency_ms,
                    },
                )
            else:
                self._scheduler.fail(task.task_id, result.error or "Unknown error")
                self._publish_event(
                    event_topics.AGENT_TASK_FAILED,
                    {
                        "task_id": task.task_id,
                        "error": result.error,
                    },
                )

            # Store result and signal waiters
            with self._lock:
                self._results[task.task_id] = result
                event = self._result_events.get(task.task_id)
                if event:
                    event.set()

        logger.debug("Worker %s loop exited", worker.worker_id)

    def _publish_event(self, topic: str, payload: dict) -> None:
        """Publish an event to the event bus if available."""
        if self._event_bus is None:
            return
        try:
            event = Event(
                topic=topic,
                payload=payload,
                metadata=EventMetadata(source="agent_worker_pool"),
            )
            self._event_bus.publish(event)
        except Exception as exc:
            logger.debug("Failed to publish event %s: %s", topic, exc)
