"""
Tenant Manager — Multi-Tenancy Lifecycle Management

=== THEORY ===

The TenantManager is the central coordinator for all tenant operations.
It handles the complete tenant lifecycle:

  create  -> active -> suspended -> deleted
                    -> deleted (direct)

Lifecycle states:
  active    — tenant can perform all operations
  suspended — tenant exists but all operations are blocked
              (used for billing issues, abuse, maintenance)
  deleted   — tenant is logically deleted (soft delete)
              data may be retained for N days before physical deletion

Resource management:
  - Quotas define the maximum resource usage per tenant
  - Usage tracking monitors actual consumption
  - check_quota() compares usage against quotas before allowing operations

=== DATA STORAGE ===

In-memory dict for development/testing.  In production, this would be
backed by PostgreSQL or a dedicated tenant database with:
  - Tenant table (id, name, status, config, quotas, created_at)
  - TenantUsage table (tenant_id, resource, count, updated_at)

=== COMPLEXITY ===

  create_tenant():  O(1) — dict insertion
  get_tenant():     O(1) — dict lookup
  list_tenants():   O(N) — N = total tenants (filtered)
  check_quota():    O(1) — compare two values
  increment_usage(): O(1) — counter increment

=== PRODUCTION EQUIVALENTS ===

  Salesforce: Organization provisioning service
  AWS:        Organizations API + Service Control Policies
  Stripe:     Account management with subscription tiers
  Auth0:      Tenant provisioning with feature flags
"""

import logging
import threading
import time
from typing import Any, Optional

from app.config import TenancyConfig
from app.tenancy.models import Tenant, TenantQuotas, TenantUsage

logger = logging.getLogger(__name__)


class TenantManager:
    """
    Manages tenant lifecycle and provides tenant-scoped access.

    === THEORY ===

    Multi-tenancy with logical isolation uses a tenant_id column/prefix
    to partition data within shared infrastructure.  This is simpler than
    physical isolation but requires careful access control.

    All data access methods accept a tenant context and scope queries
    to that tenant's data only.

    Usage:
        manager = TenantManager(config)
        tenant = manager.create_tenant("acme", "Acme Corp")
        manager.increment_usage("acme", "documents", 10)
        if manager.check_quota("acme", "documents"):
            # proceed with indexing
            ...
    """

    def __init__(self, config: TenancyConfig, redis_client: Any = None):
        """
        Args:
            config: TenancyConfig with limits and settings.
            redis_client: Optional RedisClient for distributed state.
        """
        self._config = config
        self._redis = redis_client
        self._lock = threading.Lock()
        # tenant_id -> Tenant
        self._tenants: dict[str, Tenant] = {}
        # tenant_id -> TenantUsage
        self._usage: dict[str, TenantUsage] = {}
        logger.info(
            "TenantManager initialized: max_tenants=%d, isolation=%s",
            config.max_tenants, config.isolation_level,
        )

    def create_tenant(
        self,
        tenant_id: str,
        name: str,
        quotas: dict[str, Any] | None = None,
    ) -> Tenant:
        """
        Create a new tenant.

        Args:
            tenant_id: Unique identifier for the tenant.
            name: Human-readable display name.
            quotas: Optional quota overrides (dict form of TenantQuotas).

        Returns:
            The created Tenant object.

        Raises:
            ValueError: If tenant_id already exists or max_tenants reached.
        """
        if quotas is None:
            quotas = {}

        with self._lock:
            if tenant_id in self._tenants:
                raise ValueError(f"Tenant '{tenant_id}' already exists")

            if len(self._tenants) >= self._config.max_tenants:
                raise ValueError(
                    f"Maximum tenant limit reached ({self._config.max_tenants})"
                )

            # Build default quotas from config if not provided
            default_quotas = {
                "max_documents": self._config.max_docs_per_tenant,
                "max_sessions": self._config.max_sessions_per_tenant,
                "max_agents": self._config.max_agents_per_tenant,
            }
            default_quotas.update(quotas)

            tenant = Tenant(
                tenant_id=tenant_id,
                name=name,
                status="active",
                quotas=default_quotas,
                created_at=time.time(),
            )

            self._tenants[tenant_id] = tenant
            self._usage[tenant_id] = TenantUsage(tenant_id=tenant_id)

        logger.info("Created tenant: %s (%s)", tenant_id, name)
        return tenant

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """
        Retrieve a tenant by ID.

        Returns:
            Tenant object if found, None otherwise.
        """
        with self._lock:
            return self._tenants.get(tenant_id)

    def update_tenant(self, tenant_id: str, **kwargs) -> Optional[Tenant]:
        """
        Update tenant attributes.

        Supported kwargs: name, status, config, quotas, metadata.

        Returns:
            Updated Tenant object, or None if tenant not found.
        """
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if tenant is None:
                return None

            for key, value in kwargs.items():
                if hasattr(tenant, key) and key != "tenant_id":
                    setattr(tenant, key, value)

        logger.info("Updated tenant %s: %s", tenant_id, list(kwargs.keys()))
        return tenant

    def delete_tenant(self, tenant_id: str) -> bool:
        """
        Soft-delete a tenant (mark as deleted).

        The tenant record is preserved for audit purposes but the
        tenant can no longer perform operations.

        Returns:
            True if tenant was found and deleted, False otherwise.
        """
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if tenant is None:
                return False

            tenant.status = "deleted"

        logger.info("Deleted tenant: %s", tenant_id)
        return True

    def suspend_tenant(self, tenant_id: str) -> bool:
        """
        Suspend a tenant (block all operations).

        Used for billing issues, abuse, or maintenance.
        The tenant can be reactivated later via update_tenant(status="active").

        Returns:
            True if tenant was found and suspended, False otherwise.
        """
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if tenant is None:
                return False

            tenant.status = "suspended"

        logger.info("Suspended tenant: %s", tenant_id)
        return True

    def list_tenants(self, status: Optional[str] = None) -> list[Tenant]:
        """
        List all tenants, optionally filtered by status.

        Args:
            status: If provided, only return tenants with this status.

        Returns:
            List of Tenant objects.
        """
        with self._lock:
            tenants = list(self._tenants.values())

        if status is not None:
            tenants = [t for t in tenants if t.status == status]

        return tenants

    def get_usage(self, tenant_id: str) -> TenantUsage:
        """
        Get current resource usage for a tenant.

        Returns:
            TenantUsage object. Returns zero-usage if tenant not found.
        """
        with self._lock:
            usage = self._usage.get(tenant_id)
            if usage is None:
                return TenantUsage(tenant_id=tenant_id)
            return usage

    def check_quota(self, tenant_id: str, resource: str) -> bool:
        """
        Check if a tenant is within quota for a given resource.

        Args:
            tenant_id: The tenant to check.
            resource: Resource type: "documents", "sessions", "agents".

        Returns:
            True if within quota (operation allowed), False if at limit.
        """
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if tenant is None:
                return False

            usage = self._usage.get(tenant_id)
            if usage is None:
                return True

            # Map resource names to quota keys and usage fields
            quota_map = {
                "documents": ("max_documents", usage.document_count),
                "sessions": ("max_sessions", usage.session_count),
                "agents": ("max_agents", usage.agent_count),
            }

            if resource not in quota_map:
                logger.warning("Unknown resource type for quota check: %s", resource)
                return True

            quota_key, current_usage = quota_map[resource]
            max_allowed = tenant.quotas.get(quota_key, float("inf"))

            return current_usage < max_allowed

    def increment_usage(
        self, tenant_id: str, resource: str, amount: int = 1
    ) -> None:
        """
        Increment resource usage for a tenant.

        Args:
            tenant_id: The tenant whose usage to update.
            resource: Resource type: "documents", "sessions", "agents", "queries".
            amount: Amount to increment by (default 1).
        """
        with self._lock:
            usage = self._usage.get(tenant_id)
            if usage is None:
                usage = TenantUsage(tenant_id=tenant_id)
                self._usage[tenant_id] = usage

            field_map = {
                "documents": "document_count",
                "sessions": "session_count",
                "agents": "agent_count",
                "queries": "queries_today",
            }

            field_name = field_map.get(resource)
            if field_name:
                current = getattr(usage, field_name)
                setattr(usage, field_name, current + amount)
            else:
                logger.warning("Unknown resource type for usage increment: %s", resource)

    def decrement_usage(
        self, tenant_id: str, resource: str, amount: int = 1
    ) -> None:
        """
        Decrement resource usage for a tenant.

        Args:
            tenant_id: The tenant whose usage to update.
            resource: Resource type.
            amount: Amount to decrement by (default 1).
        """
        with self._lock:
            usage = self._usage.get(tenant_id)
            if usage is None:
                return

            field_map = {
                "documents": "document_count",
                "sessions": "session_count",
                "agents": "agent_count",
                "queries": "queries_today",
            }

            field_name = field_map.get(resource)
            if field_name:
                current = getattr(usage, field_name)
                setattr(usage, field_name, max(0, current - amount))

    def stats(self) -> dict:
        """
        Return tenant management statistics.

        Returns:
            Dict with total_tenants, active/suspended/deleted counts,
            and aggregate usage.
        """
        with self._lock:
            tenants = list(self._tenants.values())
            active = sum(1 for t in tenants if t.status == "active")
            suspended = sum(1 for t in tenants if t.status == "suspended")
            deleted = sum(1 for t in tenants if t.status == "deleted")

            total_docs = sum(u.document_count for u in self._usage.values())
            total_sessions = sum(u.session_count for u in self._usage.values())

        return {
            "total_tenants": len(tenants),
            "active": active,
            "suspended": suspended,
            "deleted": deleted,
            "total_documents": total_docs,
            "total_sessions": total_sessions,
        }
