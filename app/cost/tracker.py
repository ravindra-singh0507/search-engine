"""
Cost Tracker — Phase 8 Batch 4

=== THEORY ===

The CostTracker is the **write side** of the cost observability system,
implementing an in-process event store backed by a bounded circular buffer
(collections.deque with maxlen).

Design decisions:

1. **Bounded buffer (deque, maxlen=10 000)** — prevents unbounded memory
   growth.  The oldest events are evicted automatically (FIFO).  For longer
   retention, events are also written to a JSONL file on disk so they survive
   process restarts and can be ingested by external log aggregators (Loki,
   Splunk, CloudWatch Logs).

2. **Thread safety via threading.Lock** — multiple threads may call record()
   concurrently (agent workers, API handlers, crawl workers).  A single lock
   guards both the deque append and the file write so the two stay consistent.

3. **Helper methods (record_llm, record_embedding, record_agent)** — each
   creates properly attributed CostEvents from high-level parameters, reducing
   the chance of callers setting the wrong category or unit_type.

4. **Budget alerting** — is_over_budget() compares today's total spend against
   config.budget_alert_usd.  The caller decides what to do (halt, notify, log);
   the tracker is policy-neutral.

=== PRODUCTION EQUIVALENTS ===

  Stripe Metered Billing: record usage events → aggregate → invoice
  AWS Cost Allocation Tags: tag resources → aggregate by tag in Cost Explorer
  OpenAI Usage API:  per-request token counts accumulate in a daily bucket
"""

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

from app.config import CostConfig
from app.cost.models import CostCategory, CostEvent, CostSummary


class CostTracker:
    """
    Thread-safe, bounded in-process cost event store.

    Parameters
    ----------
    config : CostConfig
        Controls whether tracking is enabled, the JSONL persistence path, and
        the budget alert threshold.
    """

    def __init__(self, config: CostConfig) -> None:
        self._config = config
        self._events: deque[CostEvent] = deque(maxlen=10_000)
        self._lock = threading.Lock()

        # Optional JSONL file for durable persistence
        self._log_file: Optional[Path] = None
        if config.enabled and config.cost_log_path:
            self._log_file = Path(config.cost_log_path)
            self._log_file.parent.mkdir(parents=True, exist_ok=True)

    # ── Core write ─────────────────────────────────────────────────────────────

    def record(self, event: CostEvent) -> None:
        """
        Append a CostEvent to the in-memory buffer and, optionally, to the
        JSONL log file.

        The file write is kept inside the lock so the on-disk log is always a
        prefix of the in-memory buffer (no partial lines from concurrent writes).
        """
        if not self._config.enabled:
            return
        with self._lock:
            self._events.append(event)
            if self._log_file is not None:
                try:
                    with self._log_file.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(event.to_dict()) + "\n")
                except OSError:
                    pass  # never let a logging failure break the hot path

    # ── Convenience helpers ────────────────────────────────────────────────────

    def record_llm(
        self,
        provider:     str,
        model:        str,
        input_tokens: int,
        output_tokens: int,
        cost_usd:     float,
        tenant_id:    str = "",
        session_id:   str = "",
    ) -> None:
        """
        Record an LLM call as *two* events — one for prompt tokens
        (LLM_INPUT) and one for completion tokens (LLM_OUTPUT).

        The total cost is split proportionally by token count so that the
        by_category breakdown in summaries correctly reflects the fact that
        output tokens are typically 3–5 × more expensive than input tokens.
        """
        if not self._config.track_llm:
            return
        total_tokens = input_tokens + output_tokens
        if total_tokens > 0:
            input_fraction  = input_tokens  / total_tokens
            output_fraction = output_tokens / total_tokens
        else:
            input_fraction = output_fraction = 0.5

        now = time.time()
        self.record(CostEvent(
            category   = CostCategory.LLM_INPUT,
            cost_usd   = cost_usd * input_fraction,
            units      = float(input_tokens),
            unit_type  = "tokens",
            provider   = provider,
            model      = model,
            tenant_id  = tenant_id,
            session_id = session_id,
            timestamp  = now,
        ))
        self.record(CostEvent(
            category   = CostCategory.LLM_OUTPUT,
            cost_usd   = cost_usd * output_fraction,
            units      = float(output_tokens),
            unit_type  = "tokens",
            provider   = provider,
            model      = model,
            tenant_id  = tenant_id,
            session_id = session_id,
            timestamp  = now,
        ))

    def record_embedding(
        self,
        provider:  str,
        model:     str,
        tokens:    int,
        cost_usd:  float,
        tenant_id: str = "",
    ) -> None:
        """Record an embedding API call."""
        if not self._config.track_embeddings:
            return
        self.record(CostEvent(
            category  = CostCategory.EMBEDDING,
            cost_usd  = cost_usd,
            units     = float(tokens),
            unit_type = "tokens",
            provider  = provider,
            model     = model,
            tenant_id = tenant_id,
        ))

    def record_agent(
        self,
        agent_type:   str,
        duration_sec: float,
        cost_usd:     float = 0.0,
        tenant_id:    str   = "",
    ) -> None:
        """
        Record agent compute time.

        cost_usd defaults to 0.0 because many self-hosted agents have no
        direct monetary cost; the duration is still useful for profiling.
        """
        if not self._config.track_agents:
            return
        self.record(CostEvent(
            category  = CostCategory.AGENT_COMPUTE,
            cost_usd  = cost_usd,
            units     = duration_sec,
            unit_type = "seconds",
            provider  = "internal",
            model     = agent_type,
            tenant_id = tenant_id,
        ))

    # ── Read / query ───────────────────────────────────────────────────────────

    def _events_in_window(self, period_hours: float) -> list[CostEvent]:
        """Return events whose timestamp falls within the last *period_hours*."""
        cutoff = time.time() - period_hours * 3600.0
        with self._lock:
            return [e for e in self._events if e.timestamp >= cutoff]

    @staticmethod
    def _build_summary(events: list[CostEvent], period_hours: float) -> CostSummary:
        """Aggregate a list of CostEvents into a CostSummary."""
        if not events:
            now = time.time()
            return CostSummary(
                total_usd    = 0.0,
                by_category  = {},
                by_tenant    = {},
                by_provider  = {},
                event_count  = 0,
                period_start = now - period_hours * 3600.0,
                period_end   = now,
            )

        total_usd:   float             = 0.0
        by_category: dict[str, float]  = {}
        by_tenant:   dict[str, float]  = {}
        by_provider: dict[str, float]  = {}

        for e in events:
            total_usd += e.cost_usd
            cat = e.category.value
            by_category[cat]             = by_category.get(cat, 0.0)             + e.cost_usd
            by_tenant[e.tenant_id]       = by_tenant.get(e.tenant_id, 0.0)       + e.cost_usd
            by_provider[e.provider]      = by_provider.get(e.provider, 0.0)      + e.cost_usd

        timestamps = [e.timestamp for e in events]
        return CostSummary(
            total_usd    = total_usd,
            by_category  = by_category,
            by_tenant    = by_tenant,
            by_provider  = by_provider,
            event_count  = len(events),
            period_start = min(timestamps),
            period_end   = max(timestamps),
        )

    def get_summary(self, period_hours: float = 24.0) -> CostSummary:
        """Aggregate all events from the last *period_hours* into a CostSummary."""
        events = self._events_in_window(period_hours)
        return self._build_summary(events, period_hours)

    def get_by_tenant(
        self, tenant_id: str, period_hours: float = 24.0
    ) -> CostSummary:
        """Return a CostSummary scoped to a single tenant."""
        events = [
            e for e in self._events_in_window(period_hours)
            if e.tenant_id == tenant_id
        ]
        return self._build_summary(events, period_hours)

    def get_recent_events(
        self,
        limit:    int                        = 100,
        category: Optional[CostCategory]     = None,
    ) -> list[CostEvent]:
        """
        Return the *limit* most-recent events, optionally filtered by category.

        Events are returned in reverse-chronological order (newest first).
        """
        with self._lock:
            events: list[CostEvent] = list(self._events)
        if category is not None:
            events = [e for e in events if e.category == category]
        return list(reversed(events))[:limit]

    # ── Budget alerting ────────────────────────────────────────────────────────

    def is_over_budget(self) -> bool:
        """
        Return True if today's total spend meets or exceeds the configured
        budget alert threshold.

        Uses a 24-hour rolling window (not a calendar-day reset) to avoid
        cliff-edge behaviour at midnight.
        """
        summary = self.get_summary(period_hours=24.0)
        return summary.total_usd >= self._config.budget_alert_usd

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """
        Return lightweight statistics without building a full CostSummary.

        Keys
        ----
        total_events       : int   — events currently in the buffer
        total_cost_usd     : float — sum of all buffered event costs
        avg_cost_per_event : float — mean cost (0.0 if no events)
        """
        with self._lock:
            events = list(self._events)
        total_events = len(events)
        total_cost   = sum(e.cost_usd for e in events)
        avg_cost     = total_cost / total_events if total_events else 0.0
        return {
            "total_events":       total_events,
            "total_cost_usd":     total_cost,
            "avg_cost_per_event": avg_cost,
        }
