"""
Workflow Scheduler — Phase 8 Batch 3

=== THEORY ===

Workflow scheduling enables periodic or event-triggered execution of
workflows without manual intervention.  The scheduler maintains a
registry of schedule definitions and determines which workflows are
"due" for execution based on elapsed time since last run.

This is a simplified interval-based scheduler (not full cron).
Schedules fire when:
  now >= last_run + interval_seconds

Compared to production schedulers like Airflow:
  - No DAG parsing step
  - No complex dependency resolution between schedules
  - No backfill semantics (missed intervals are not retroactively fired)
  - Interval-based only (no cron expression parsing in this implementation)

The scheduler is passive — it does not run a background thread.
The engine polls get_due_schedules() at a chosen frequency.

=== ARCHITECTURE ===

  WorkflowScheduler
    │
    ├── _schedules: dict[schedule_id → WorkflowSchedule]
    │     Thread-safe via Lock
    │
    └── get_due_schedules() → list[WorkflowSchedule]
          Returns schedules where now >= next_run and enabled=True

=== DATA STRUCTURES ===

  WorkflowSchedule — dataclass defining when/how to run a workflow
  _schedules:      dict[str, WorkflowSchedule] — O(1) lookup by id

=== COMPLEXITY ===

  add_schedule():      O(1)
  remove_schedule():   O(1)
  get_due_schedules(): O(N) where N = total schedules
  mark_executed():     O(1)
  list_schedules():    O(N)
  stats():             O(N)

=== PRODUCTION EQUIVALENTS ===

  Airflow:        Scheduler daemon with DagBag parsing
  Temporal:       Schedule + CronSchedule
  Prefect:        Deployments with schedules (IntervalSchedule, CronSchedule)
  Celery Beat:    Periodic task scheduler
  Kubernetes:     CronJob resources
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.config import DistributedWorkflowConfig

logger = logging.getLogger(__name__)


@dataclass
class WorkflowSchedule:
    """
    Defines when a workflow should run.

    Fields:
      schedule_id:      unique identifier for this schedule (UUID)
      workflow_name:    which workflow template to execute
      cron_expression:  optional cron string (reserved for future use)
      interval_seconds: interval between executions (0 = one-shot/disabled)
      enabled:          whether this schedule is active
      last_run:         Unix epoch of last execution (None if never run)
      next_run:         Unix epoch when the next execution is due
      params:           parameters to pass to the workflow when triggered

    === SCHEDULING SEMANTICS ===

    A schedule fires when:
      - enabled is True
      - current_time >= next_run

    After firing:
      - last_run is updated to current_time
      - next_run is updated to current_time + interval_seconds
      - If interval_seconds == 0, the schedule is disabled (one-shot)
    """
    schedule_id:      str              = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_name:    str              = ""
    cron_expression:  str | None       = None
    interval_seconds: int              = 0
    enabled:          bool             = True
    last_run:         float | None     = None
    next_run:         float            = field(default_factory=time.time)
    params:           dict[str, Any]   = field(default_factory=dict)
    created_at:       float            = field(default_factory=time.time)


class WorkflowScheduler:
    """
    Schedules workflows for execution at specified times/intervals.

    === THEORY ===

    Workflow scheduling enables periodic or event-triggered execution.
    The scheduler maintains a list of registered schedules and checks
    which ones are due for execution.

    Simpler than cron — just interval-based or one-shot scheduling.
    The scheduler is a passive component: it does not spawn threads
    or timers.  The owning engine calls get_due_schedules() to poll.

    === THREAD SAFETY ===

    All mutations are protected by a threading.Lock.  The scheduler
    may be called from multiple threads (e.g., API thread registers
    a schedule while the engine thread polls for due schedules).

    === USAGE ===

    scheduler = WorkflowScheduler(config)
    sid = scheduler.add_schedule("research", interval_seconds=3600)

    # In engine loop:
    for schedule in scheduler.get_due_schedules():
        engine.start_workflow(...)
        scheduler.mark_executed(schedule.schedule_id)
    """

    def __init__(self, config: DistributedWorkflowConfig) -> None:
        """
        Initialise the scheduler.

        Args:
            config: Distributed workflow configuration containing
                    schedule_enabled and related settings.
        """
        self._config = config
        self._schedules: dict[str, WorkflowSchedule] = {}
        self._lock = threading.Lock()
        self._total_executions = 0

    def add_schedule(
        self,
        workflow_name: str,
        interval_seconds: int = 0,
        params: dict | None = None,
    ) -> str:
        """
        Register a new workflow schedule.

        Args:
            workflow_name:    Name of the workflow to execute.
            interval_seconds: Seconds between executions.
                              0 means one-shot (fires once, then disables).
            params:           Parameters passed to the workflow on trigger.

        Returns:
            schedule_id: unique identifier for the new schedule.
        """
        schedule = WorkflowSchedule(
            workflow_name=workflow_name,
            interval_seconds=interval_seconds,
            params=params or {},
            next_run=time.time(),  # Due immediately on first creation
        )

        with self._lock:
            self._schedules[schedule.schedule_id] = schedule

        logger.info(
            "Added schedule %s for workflow '%s' (interval=%ds)",
            schedule.schedule_id[:8], workflow_name, interval_seconds,
        )
        return schedule.schedule_id

    def remove_schedule(self, schedule_id: str) -> bool:
        """
        Remove a schedule by its ID.

        Returns True if the schedule was found and removed, False otherwise.
        """
        with self._lock:
            if schedule_id in self._schedules:
                del self._schedules[schedule_id]
                logger.info("Removed schedule %s", schedule_id[:8])
                return True
            return False

    def get_due_schedules(self) -> list[WorkflowSchedule]:
        """
        Return all schedules that are due for execution.

        A schedule is due when:
          - enabled is True
          - current_time >= next_run

        Returns a list of WorkflowSchedule objects (copies to prevent
        mutation while iterating).
        """
        now = time.time()
        due: list[WorkflowSchedule] = []

        with self._lock:
            for schedule in self._schedules.values():
                if schedule.enabled and now >= schedule.next_run:
                    due.append(schedule)

        return due

    def mark_executed(self, schedule_id: str) -> None:
        """
        Mark a schedule as having been executed.

        Updates last_run to now, computes next_run, and disables
        one-shot schedules (interval_seconds == 0).
        """
        now = time.time()

        with self._lock:
            schedule = self._schedules.get(schedule_id)
            if schedule is None:
                logger.warning(
                    "mark_executed called for unknown schedule: %s", schedule_id[:8]
                )
                return

            schedule.last_run = now
            self._total_executions += 1

            if schedule.interval_seconds > 0:
                schedule.next_run = now + schedule.interval_seconds
            else:
                # One-shot schedule — disable after execution
                schedule.enabled = False
                schedule.next_run = float("inf")

        logger.debug("Schedule %s marked executed", schedule_id[:8])

    def list_schedules(self) -> list[WorkflowSchedule]:
        """
        Return all registered schedules (both enabled and disabled).

        Returns copies to prevent external mutation.
        """
        with self._lock:
            return list(self._schedules.values())

    def get_schedule(self, schedule_id: str) -> WorkflowSchedule | None:
        """
        Look up a specific schedule by ID.

        Returns None if not found.
        """
        with self._lock:
            return self._schedules.get(schedule_id)

    def enable_schedule(self, schedule_id: str) -> bool:
        """
        Enable a disabled schedule.

        Returns True if the schedule was found, False otherwise.
        """
        with self._lock:
            schedule = self._schedules.get(schedule_id)
            if schedule is None:
                return False
            schedule.enabled = True
            # Reset next_run if it was set to infinity
            if schedule.next_run == float("inf"):
                schedule.next_run = time.time()
            return True

    def disable_schedule(self, schedule_id: str) -> bool:
        """
        Disable an active schedule (stops it from firing).

        Returns True if the schedule was found, False otherwise.
        """
        with self._lock:
            schedule = self._schedules.get(schedule_id)
            if schedule is None:
                return False
            schedule.enabled = False
            return True

    def stats(self) -> dict:
        """
        Return statistics about the scheduler state.

        Returns:
            dict with keys:
              - total_schedules: number of registered schedules
              - enabled_schedules: number of active (enabled) schedules
              - disabled_schedules: number of inactive schedules
              - total_executions: lifetime count of mark_executed calls
              - schedule_enabled: whether scheduling is enabled in config
        """
        with self._lock:
            total = len(self._schedules)
            enabled = sum(1 for s in self._schedules.values() if s.enabled)

            return {
                "total_schedules":    total,
                "enabled_schedules":  enabled,
                "disabled_schedules": total - enabled,
                "total_executions":   self._total_executions,
                "schedule_enabled":   self._config.schedule_enabled,
            }
