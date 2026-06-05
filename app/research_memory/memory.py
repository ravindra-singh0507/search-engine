"""
Research Memory — Phase 7

=== THEORY ===

Research memory extends Phase 6 conversation memory into a multi-layer
state that persists across agent runs within a research session.

Layers:
  1. Task Memory       — tracks completed/pending/failed agent tasks
  2. Evidence Memory   — accumulated evidence records across retrievals
  3. Session Memory    — research session metadata, goals, and progress

This mirrors how a human researcher works:
  - Keeps a to-do list of what to investigate (task memory)
  - Maintains a folder of gathered sources (evidence memory)
  - Writes a research log of progress and decisions (session memory)

=== PERSISTENCE ===

In-memory by default; SQLite persistence through the DB layer.
The research_sessions table (Phase 7 schema addition) stores snapshots.

=== MEMORY PRUNING ===

When evidence memory exceeds max_entries, the lowest-scored records
are evicted first (priority queue by score, ascending).

=== COMPLEXITY ===

  TaskMemory.add_result:       O(1)
  EvidenceMemory.add:          O(log N) — heapq for eviction
  SessionMemory.add_event:     O(1)
  SessionMemory.summarize:     O(N) — linear scan of events

=== SPACE ===

  Per session: O(T + E + S) where T=tasks, E=evidence, S=session events
  Bounded by max_entries on each layer.

=== PRODUCTION EQUIVALENTS ===

  LangGraph:    checkpointer (SQLite, Postgres, Redis)
  CrewAI:       short_term_memory / long_term_memory
  AutoGen:      teachable agent memory
  Mem0:         persistent memory layer for AI agents
"""

import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Task Memory ───────────────────────────────────────────────────────────────

@dataclass
class TaskRecord:
    """One completed or failed agent task."""
    task_id:    str
    agent_type: str
    goal:       str
    status:     str
    output_summary: str   = ""
    latency_ms: float     = 0.0
    timestamp:  float     = field(default_factory=time.time)


class TaskMemory:
    """
    Tracks all agent task executions within a research session.

    Supports queries:
      - completed tasks by agent type
      - failed tasks for retry decisions
      - full execution history
    """

    def __init__(self, max_entries: int = 200) -> None:
        self._records: list[TaskRecord] = []
        self._max = max_entries

    def add_result(self, result: Any) -> None:
        """Add from an AgentResult object."""
        record = TaskRecord(
            task_id        = result.task_id,
            agent_type     = result.agent_type.value if hasattr(result.agent_type, "value") else str(result.agent_type),
            goal           = "",
            status         = result.status.value if hasattr(result.status, "value") else str(result.status),
            output_summary = str(result.output)[:200] if result.output else "",
            latency_ms     = result.latency_ms,
        )
        self._records.append(record)
        if len(self._records) > self._max:
            self._records.pop(0)

    def get_by_type(self, agent_type: str) -> list[TaskRecord]:
        return [r for r in self._records if r.agent_type == agent_type]

    def get_completed(self) -> list[TaskRecord]:
        return [r for r in self._records if r.status == "done"]

    def get_failed(self) -> list[TaskRecord]:
        return [r for r in self._records if r.status == "failed"]

    def all_records(self) -> list[TaskRecord]:
        return list(self._records)

    def clear(self) -> None:
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)


# ── Evidence Memory ───────────────────────────────────────────────────────────

class EvidenceMemory:
    """
    Accumulated evidence across multiple retrieval passes.

    Deduplicates by doc_id + chunk_id.
    Prunes lowest-scored records when exceeding max_entries.
    """

    def __init__(self, max_entries: int = 500) -> None:
        self._evidence: dict[str, dict] = {}   # key = doc_id:chunk_id
        self._max = max_entries

    def add(self, evidence: dict) -> bool:
        """Add an evidence dict. Returns False if duplicate."""
        key = f"{evidence.get('doc_id', 0)}:{evidence.get('chunk_id', '')}"
        if key in self._evidence:
            # Keep the higher-scored version
            if evidence.get("score", 0) > self._evidence[key].get("score", 0):
                self._evidence[key] = evidence
            return False
        self._evidence[key] = evidence
        self._prune()
        return True

    def add_many(self, evidence_list: list[dict]) -> int:
        """Add multiple evidence dicts. Returns count of new additions."""
        return sum(1 for e in evidence_list if self.add(e))

    def get_all(self) -> list[dict]:
        return list(self._evidence.values())

    def get_by_doc(self, doc_id: int) -> list[dict]:
        return [e for e in self._evidence.values() if e.get("doc_id") == doc_id]

    def get_top_k(self, k: int = 10) -> list[dict]:
        """Return top-K evidence by score."""
        return sorted(
            self._evidence.values(),
            key=lambda e: e.get("score", 0),
            reverse=True,
        )[:k]

    def count(self) -> int:
        return len(self._evidence)

    def clear(self) -> None:
        self._evidence.clear()

    def _prune(self) -> None:
        if len(self._evidence) <= self._max:
            return
        # Evict lowest-scored records
        sorted_keys = sorted(
            self._evidence.keys(),
            key=lambda k: self._evidence[k].get("score", 0),
        )
        to_remove = len(self._evidence) - self._max
        for key in sorted_keys[:to_remove]:
            del self._evidence[key]


# ── Research Session Memory ──────────────────────────────────────────────────

@dataclass
class SessionEvent:
    """One event in the research session log."""
    event_type: str          # "goal_set", "task_started", "task_completed", etc.
    description: str
    metadata:   dict[str, Any] = field(default_factory=dict)
    timestamp:  float          = field(default_factory=time.time)


class ResearchSessionMemory:
    """
    Tracks the full lifecycle of a research session:
      - Research goal
      - Events (task starts, completions, errors, replanning)
      - Progress metrics
      - Summary generation
    """

    def __init__(self, max_events: int = 500) -> None:
        self._events: list[SessionEvent] = []
        self._max = max_events

    def add_event(self, event_type: str, description: str, metadata: dict | None = None) -> None:
        self._events.append(SessionEvent(
            event_type  = event_type,
            description = description,
            metadata    = metadata or {},
        ))
        if len(self._events) > self._max:
            self._events.pop(0)

    def get_events(self, event_type: str | None = None) -> list[SessionEvent]:
        if event_type:
            return [e for e in self._events if e.event_type == event_type]
        return list(self._events)

    def get_latest(self, n: int = 10) -> list[SessionEvent]:
        return self._events[-n:]

    def summarize(self) -> dict:
        """Generate a summary of the session."""
        type_counts: dict[str, int] = defaultdict(int)
        for e in self._events:
            type_counts[e.event_type] += 1
        return {
            "total_events":     len(self._events),
            "event_types":      dict(type_counts),
            "first_event_time": self._events[0].timestamp if self._events else None,
            "last_event_time":  self._events[-1].timestamp if self._events else None,
        }

    def clear(self) -> None:
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)


# ── Research Session (composite) ──────────────────────────────────────────────

class ResearchSession:
    """
    Composite memory container for one research session.

    Bundles all three memory layers into a single object that the
    orchestration layer manages per-session.
    """

    def __init__(
        self,
        session_id:  str | None = None,
        goal:        str        = "",
        user_id:     str        = "",
        max_tasks:   int        = 200,
        max_evidence: int       = 500,
        max_events:  int        = 500,
    ) -> None:
        self.session_id   = session_id or str(uuid.uuid4())
        self.goal         = goal
        self.user_id      = user_id
        self.created_at   = time.time()
        self.tasks        = TaskMemory(max_tasks)
        self.evidence     = EvidenceMemory(max_evidence)
        self.session_log  = ResearchSessionMemory(max_events)

        self.session_log.add_event("session_created", f"Research session created: {goal}")

    def add_agent_result(self, result: Any) -> None:
        """Record a completed agent result into task memory."""
        self.tasks.add_result(result)
        status = result.status.value if hasattr(result.status, "value") else str(result.status)
        self.session_log.add_event(
            "task_completed",
            f"{result.agent_type.value if hasattr(result.agent_type, 'value') else result.agent_type} "
            f"task {result.task_id[:8]} → {status}",
        )

    def add_evidence(self, evidence_list: list[dict]) -> int:
        """Add evidence and return count of new items."""
        count = self.evidence.add_many(evidence_list)
        if count:
            self.session_log.add_event(
                "evidence_added", f"Added {count} new evidence records",
            )
        return count

    def to_snapshot(self) -> dict:
        """Serialise session state for persistence."""
        return {
            "session_id":     self.session_id,
            "goal":           self.goal,
            "user_id":        self.user_id,
            "created_at":     self.created_at,
            "task_count":     len(self.tasks),
            "evidence_count": self.evidence.count(),
            "event_count":    len(self.session_log),
            "session_summary": self.session_log.summarize(),
        }
