"""
Cost Dashboard — Phase 8 Batch 4

=== THEORY ===

The CostDashboard is the **read side** of the cost observability system.  It
queries the CostTracker (the write side) to produce actionable reports:

  - Daily / weekly rolling summaries (trend detection)
  - Per-tenant reports (multi-tenancy billing accountability)
  - Budget status (real-time spend-vs-budget gauge)
  - CSV export (feed to external BI tools: Metabase, Grafana, Tableau)

This separation of concerns — tracker owns writes, dashboard owns reads —
follows the **CQRS** (Command–Query Responsibility Segregation) pattern:
commands change state; queries only read it.  CQRS lets the read side be
scaled, cached, or moved to a read replica independently of the write side.

Thread safety is achieved by delegating all data access to CostTracker, which
is already thread-safe.  The dashboard itself is stateless: every method
builds its result fresh from the tracker.

=== PRODUCTION EQUIVALENTS ===

  Grafana dashboards polling Prometheus metrics
  OpenAI usage dashboard (daily token / cost graphs)
  AWS Cost Explorer (daily/weekly breakdowns with filters)
  Stripe Revenue Recognition (tenant-level revenue and cost reports)
"""

import csv
import io
import time
import threading

from app.config import CostConfig
from app.cost.tracker import CostTracker


class CostDashboard:
    """
    Read-side cost reporting layer built on top of CostTracker.

    Parameters
    ----------
    tracker : CostTracker
        The source of truth for cost events.
    config  : CostConfig
        Platform cost configuration (budget thresholds, feature flags).
    """

    def __init__(self, tracker: CostTracker, config: CostConfig) -> None:
        self._tracker = tracker
        self._config  = config
        self._lock    = threading.Lock()  # guards any future mutable dashboard state

    # ── Period reports ─────────────────────────────────────────────────────────

    def daily_report(self) -> dict:
        """
        Return a cost summary for the last 24 hours.

        The report includes the full CostSummary projection plus metadata
        fields (report_type, generated_at) for dashboard display.
        """
        summary = self._tracker.get_summary(period_hours=24.0)
        return {
            "report_type":  "daily",
            "generated_at": time.time(),
            **summary.to_dict(),
        }

    def weekly_report(self) -> dict:
        """
        Return a 7-day rolling cost summary.

        Uses a 168-hour window (7 × 24) rather than calendar-week boundaries
        so the report is always current and requires no timezone handling.
        """
        summary = self._tracker.get_summary(period_hours=168.0)
        return {
            "report_type":  "weekly",
            "generated_at": time.time(),
            **summary.to_dict(),
        }

    def tenant_report(self, tenant_id: str) -> dict:
        """
        Return a 24-hour cost summary scoped to a single tenant.

        Parameters
        ----------
        tenant_id : str
            The tenant identifier to filter by.
        """
        summary = self._tracker.get_by_tenant(tenant_id, period_hours=24.0)
        return {
            "report_type":  "tenant",
            "tenant_id":    tenant_id,
            "generated_at": time.time(),
            **summary.to_dict(),
        }

    # ── Budget status ──────────────────────────────────────────────────────────

    def budget_status(self) -> dict:
        """
        Return the current spend-vs-budget gauge for the last 24 hours.

        Keys
        ----
        current_spend : float — USD spent in the last 24 h
        budget        : float — configured alert threshold (config.budget_alert_usd)
        percentage    : float — (current_spend / budget) * 100, or 0.0 if budget == 0
        over_budget   : bool  — True when current_spend >= budget
        """
        summary = self._tracker.get_summary(period_hours=24.0)
        budget  = self._config.budget_alert_usd
        pct     = (summary.total_usd / budget * 100.0) if budget > 0 else 0.0
        return {
            "current_spend": summary.total_usd,
            "budget":        budget,
            "percentage":    round(pct, 2),
            "over_budget":   summary.total_usd >= budget,
        }

    # ── CSV export ─────────────────────────────────────────────────────────────

    def export_csv(self, period_hours: float = 24.0) -> str:
        """
        Export cost events for the requested window as a CSV string.

        Columns: timestamp, category, provider, model, cost_usd, tenant_id

        The returned string can be written to a file, sent as an HTTP
        response (Content-Type: text/csv), or piped to a spreadsheet tool.

        Parameters
        ----------
        period_hours : float
            Rolling window in hours to export.  Defaults to 24.
        """
        events = self._tracker.get_recent_events(limit=10_000)
        cutoff = time.time() - period_hours * 3600.0
        events = [e for e in events if e.timestamp >= cutoff]

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["timestamp", "category", "provider", "model",
                         "cost_usd", "tenant_id"])
        for e in reversed(events):   # chronological order in the file
            writer.writerow([
                e.timestamp,
                e.category.value,
                e.provider,
                e.model,
                e.cost_usd,
                e.tenant_id,
            ])
        return buf.getvalue()

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """
        Return lightweight dashboard metadata.

        Keys
        ----
        tracker_stats   : dict — passthrough from CostTracker.stats()
        budget_status   : dict — current spend-vs-budget gauge
        config_enabled  : bool — whether cost tracking is active
        """
        return {
            "tracker_stats":  self._tracker.stats(),
            "budget_status":  self.budget_status(),
            "config_enabled": self._config.enabled,
        }
