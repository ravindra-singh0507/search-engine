"""
Multi-Tenancy — Phase 8 Batch 3

=== THEORY ===

Multi-tenancy is an architecture where a single instance of software serves
multiple customers (tenants).  Each tenant's data is isolated and invisible
to other tenants, even though they share the same underlying infrastructure.

Isolation strategies (from least to most isolated):
  1. Logical isolation (this implementation):
     - Shared database, shared schema
     - tenant_id column/prefix on every record
     - Pros: simple, cost-effective, easy to manage
     - Cons: requires careful access control, noisy neighbor risk

  2. Schema-per-tenant:
     - Shared database, separate schema per tenant
     - Pros: better isolation, easier backup/restore per tenant
     - Cons: schema migration complexity, connection pool pressure

  3. Database-per-tenant:
     - Completely separate database per tenant
     - Pros: strongest isolation, independent scaling
     - Cons: operational overhead, higher cost

  4. Cluster-per-tenant:
     - Separate compute + storage
     - Pros: physical isolation, regulatory compliance
     - Cons: highest cost, complex orchestration

Our implementation uses logical isolation (strategy 1) with:
  - TenantContext: thread-local tenant state for request scoping
  - TenantManager: tenant lifecycle (CRUD, quotas, usage tracking)
  - TenantIsolation: key prefixing and access validation
  - TenantMiddleware: request-level tenant extraction and validation

=== PRODUCTION EQUIVALENTS ===

  Salesforce:     org_id partitioning in shared Oracle DB
  Slack:          workspace-level isolation with data locality
  AWS:            account-level isolation with IAM boundaries
  Elastic Cloud:  index-per-tenant or cluster-per-tenant (configurable)
  OpenAI:         organization-level API key scoping and rate limits
"""

from app.tenancy.models import Tenant, TenantQuotas, TenantUsage
from app.tenancy.context import TenantContext, tenant_scope
from app.tenancy.manager import TenantManager
from app.tenancy.isolation import TenantIsolation
from app.tenancy.middleware import TenantMiddleware

__all__ = [
    "Tenant",
    "TenantQuotas",
    "TenantUsage",
    "TenantContext",
    "tenant_scope",
    "TenantManager",
    "TenantIsolation",
    "TenantMiddleware",
]
