"""
Cost Observability Models — Phase 8 Batch 4

=== THEORY ===

Cost attribution is the practice of tracking *who* caused *what* spend so that
the platform can:
  - Alert when a budget threshold is exceeded
  - Produce per-tenant invoices (cost-based multi-tenancy billing)
  - Identify the most expensive operations for optimisation
  - Feed dashboards and anomaly detection

The two core modelling choices made here:

1. CostEvent as an **immutable record** (dataclass with a UUID + timestamp
   assigned at creation).  This mirrors the event-sourcing pattern: every
   cost incurred produces an append-only log entry.  Summaries are then
   derived by aggregating those events, not by mutating a running total
   (which would be hard to audit and impossible to replay).

2. CostSummary as a **projection** — a read-model computed on demand from
   the event log for a given time window.  Projections can be re-generated
   at any point without losing accuracy; storing only the summary would lose
   the ability to re-bucket by a new dimension.

=== PRODUCTION EQUIVALENTS ===

  OpenAI:  /v1/usage endpoint returns token-level cost records per model
  AWS:     Cost Explorer line items + resource tags
  GCP:     Cloud Billing export to BigQuery (event-level rows)
  Stripe:  Meter events for metered billing (exact same pattern)
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


# ── Category taxonomy ──────────────────────────────────────────────────────────

class CostCategory(str, Enum):
    """
    Enumeration of cost categories for the AI search platform.

    Using ``str`` as a mixin makes the enum JSON-serialisable without a custom
    encoder — ``json.dumps(CostCategory.LLM_INPUT)`` yields ``"LLM_INPUT"``.
    """
    LLM_INPUT       = "LLM_INPUT"        # LLM prompt tokens billed
    LLM_OUTPUT      = "LLM_OUTPUT"       # LLM completion tokens billed
    EMBEDDING       = "EMBEDDING"        # Embedding model token costs
    VECTOR_STORAGE  = "VECTOR_STORAGE"   # Per-GB/month vector index storage
    AGENT_COMPUTE   = "AGENT_COMPUTE"    # CPU/wall-clock time for agent tasks
    SEARCH_QUERY    = "SEARCH_QUERY"     # Per-query retrieval API costs
    CRAWL_COMPUTE   = "CRAWL_COMPUTE"    # Bandwidth + compute for crawling


# ── CostEvent ─────────────────────────────────────────────────────────────────

@dataclass
class CostEvent:
    """
    Immutable record of a single cost-incurring operation.

    Attributes
    ----------
    category:   Broad type of cost (see CostCategory).
    cost_usd:   Monetary value in US dollars.
    units:      Number of billable units consumed (tokens, seconds, GB, etc.).
    unit_type:  Human-readable name for the unit  (e.g. "tokens", "seconds").
    provider:   Upstream provider  ("openai", "anthropic", "internal", …).
    model:      Model or resource identifier  ("gpt-4o", "faiss-hnsw", …).
    tenant_id:  Tenant that incurred the cost (empty = platform-level cost).
    session_id: Session or request ID for fine-grained attribution.
    metadata:   Arbitrary key-value pairs for additional context.
    timestamp:  Unix epoch seconds at event creation.
    event_id:   Globally unique identifier (UUID4 string).
    """
    category:   CostCategory
    cost_usd:   float
    units:      float
    unit_type:  str
    provider:   str
    model:      str
    tenant_id:  str              = ""
    session_id: str              = ""
    metadata:   dict[str, Any]  = field(default_factory=dict)
    timestamp:  float            = field(default_factory=time.time)
    event_id:   str              = field(default_factory=lambda: str(uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary (JSON-safe)."""
        return {
            "event_id":   self.event_id,
            "category":   self.category.value,
            "cost_usd":   self.cost_usd,
            "units":      self.units,
            "unit_type":  self.unit_type,
            "provider":   self.provider,
            "model":      self.model,
            "tenant_id":  self.tenant_id,
            "session_id": self.session_id,
            "metadata":   self.metadata,
            "timestamp":  self.timestamp,
        }


# ── CostSummary ───────────────────────────────────────────────────────────────

@dataclass
class CostSummary:
    """
    Aggregated cost projection for a time window.

    Computed on demand from raw CostEvents; never stored persistently so it
    can always be re-derived with a different bucketing strategy.

    Attributes
    ----------
    total_usd:    Sum of all event costs in the window.
    by_category:  Cost broken down by CostCategory value.
    by_tenant:    Cost broken down by tenant_id.
    by_provider:  Cost broken down by provider.
    event_count:  Number of events included in this summary.
    period_start: Unix timestamp of the earliest event in the window.
    period_end:   Unix timestamp of the latest event (or window end).
    """
    total_usd:    float
    by_category:  dict[str, float]
    by_tenant:    dict[str, float]
    by_provider:  dict[str, float]
    event_count:  int
    period_start: float
    period_end:   float

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary (JSON-safe)."""
        return {
            "total_usd":    self.total_usd,
            "by_category":  self.by_category,
            "by_tenant":    self.by_tenant,
            "by_provider":  self.by_provider,
            "event_count":  self.event_count,
            "period_start": self.period_start,
            "period_end":   self.period_end,
        }
