"""
Tenant Data Models — Multi-Tenancy

=== THEORY ===

These dataclasses define the core entities for multi-tenant operation:

  Tenant       — represents one customer/organization
  TenantQuotas — resource limits for a tenant
  TenantUsage  — current resource consumption for a tenant

The separation of quotas from usage follows the separation of concerns
principle: quotas are configuration (set by admin), while usage is
runtime state (updated by the system).

=== PRODUCTION EQUIVALENTS ===

  Salesforce: Organization object with edition-based limits
  AWS:        Account with service quotas (soft/hard limits)
  OpenAI:     Organization with rate limits and usage caps
  Stripe:     Account with subscription-based feature flags
"""

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tenant:
    """
    Represents one tenant (customer/organization) in the system.

    Fields:
      tenant_id   — unique identifier (typically org slug or UUID)
      name        — human-readable display name
      status      — lifecycle state: active, suspended, deleted
      config      — tenant-specific configuration overrides
      quotas      — resource limits (max documents, sessions, etc.)
      created_at  — unix timestamp of creation
      metadata    — arbitrary key-value metadata (plan, region, etc.)
    """
    tenant_id:  str
    name:       str
    status:     str = "active"
    config:     dict[str, Any] = field(default_factory=dict)
    quotas:     dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    metadata:   dict[str, Any] = field(default_factory=dict)

    def is_active(self) -> bool:
        """Check if tenant can serve requests."""
        return self.status == "active"

    def to_dict(self) -> dict:
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "status": self.status,
            "config": self.config,
            "quotas": self.quotas,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class TenantQuotas:
    """
    Resource quotas for a tenant.

    These represent the maximum allowed resource usage.  When a tenant
    exceeds a quota, the system should reject new operations for that
    resource type (returning HTTP 429 or equivalent).

    Fields:
      max_documents         — maximum documents in the index
      max_sessions          — maximum concurrent conversation sessions
      max_agents            — maximum agent instances
      max_queries_per_minute — rate limit on search queries
      max_storage_mb        — total storage cap in megabytes
    """
    max_documents:         int = 100000
    max_sessions:          int = 1000
    max_agents:            int = 50
    max_queries_per_minute: int = 120
    max_storage_mb:        int = 10000

    def to_dict(self) -> dict:
        return {
            "max_documents": self.max_documents,
            "max_sessions": self.max_sessions,
            "max_agents": self.max_agents,
            "max_queries_per_minute": self.max_queries_per_minute,
            "max_storage_mb": self.max_storage_mb,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TenantQuotas":
        return cls(
            max_documents=data.get("max_documents", 100000),
            max_sessions=data.get("max_sessions", 1000),
            max_agents=data.get("max_agents", 50),
            max_queries_per_minute=data.get("max_queries_per_minute", 120),
            max_storage_mb=data.get("max_storage_mb", 10000),
        )


@dataclass
class TenantUsage:
    """
    Current resource consumption for a tenant.

    Updated incrementally as the tenant uses resources.  Compared
    against TenantQuotas to enforce limits.

    Fields:
      tenant_id      — which tenant this usage belongs to
      document_count — number of indexed documents
      session_count  — current active sessions
      agent_count    — current running agents
      queries_today  — queries executed today (resets daily)
      storage_mb     — total storage consumed in megabytes
    """
    tenant_id:      str
    document_count: int   = 0
    session_count:  int   = 0
    agent_count:    int   = 0
    queries_today:  int   = 0
    storage_mb:     float = 0.0

    def to_dict(self) -> dict:
        return {
            "tenant_id": self.tenant_id,
            "document_count": self.document_count,
            "session_count": self.session_count,
            "agent_count": self.agent_count,
            "queries_today": self.queries_today,
            "storage_mb": self.storage_mb,
        }
