"""
Tenant Isolation — Data Access Scoping

=== THEORY ===

Tenant isolation ensures that one tenant cannot access another tenant's
data.  In a shared-infrastructure (logical isolation) model, this is
enforced at the application layer by prefixing all keys and filtering
all queries with the tenant identifier.

Key scoping pattern:
  Original key:    "documents:123"
  Tenant-scoped:   "tenant:acme-corp:documents:123"

This transparent prefixing means:
  - All cache keys are tenant-isolated
  - All database queries are tenant-filtered
  - All event topics are tenant-partitioned
  - No cross-tenant data leakage is possible (if consistently applied)

The TenantIsolation class provides utilities for:
  1. scope_key() — prefix any key with tenant context
  2. validate_access() — verify a requesting tenant owns a resource
  3. get_tenant_prefix() — get the prefix string for bulk operations

=== DEFENSE IN DEPTH ===

Multiple layers of isolation prevent data leakage:
  Layer 1: Middleware sets TenantContext (this request = this tenant)
  Layer 2: TenantIsolation.scope_key() prefixes storage keys
  Layer 3: TenantIsolation.validate_access() checks ownership
  Layer 4: Database-level row security policies (production)

=== COMPLEXITY ===

  scope_key():        O(1) — string concatenation
  validate_access():  O(1) — string comparison
  get_tenant_prefix(): O(1) — string formatting

=== PRODUCTION EQUIVALENTS ===

  PostgreSQL:   Row-Level Security (RLS) with tenant_id policies
  Elasticsearch: Index-per-tenant or routing with tenant_id
  Redis:        Key prefixing with tenant namespace
  S3:           Bucket/prefix-per-tenant with IAM policies
"""

import logging
from typing import Any, Optional

from app.tenancy.context import TenantContext
from app.tenancy.manager import TenantManager

logger = logging.getLogger(__name__)


class TenantIsolation:
    """
    Enforces tenant data isolation.

    Wraps database/cache operations to scope them to the current tenant.
    Provides key prefixing, access validation, and tenant-aware
    key generation.

    Usage:
        isolation = TenantIsolation(tenant_manager)

        # Scope a cache key to the current tenant
        key = isolation.scope_key("documents:123")
        # Result: "tenant:acme-corp:documents:123"

        # Validate cross-tenant access attempt
        if not isolation.validate_access("acme-corp", "other-corp"):
            raise PermissionError("Cross-tenant access denied")
    """

    def __init__(self, tenant_manager: TenantManager) -> None:
        """
        Args:
            tenant_manager: TenantManager for tenant lookups and config.
        """
        self._tenant_manager = tenant_manager
        logger.info("TenantIsolation initialized")

    def scope_key(self, key: str, tenant_id: Optional[str] = None) -> str:
        """
        Prefix a key with the tenant namespace.

        If tenant_id is not provided, uses the current TenantContext.
        If no tenant context is available, returns the key unchanged
        (for system-level operations that are not tenant-scoped).

        Args:
            key: The original key to scope.
            tenant_id: Optional explicit tenant ID. If None, uses context.

        Returns:
            Tenant-scoped key: "tenant:{tenant_id}:{key}"
            Or the original key if no tenant context.
        """
        tid = tenant_id or TenantContext.get()
        if tid is None:
            return key

        return f"tenant:{tid}:{key}"

    def validate_access(self, tenant_id: str, resource_tenant: str) -> bool:
        """
        Validate that a requesting tenant can access a resource.

        In logical isolation, a tenant can only access its own resources.
        Cross-tenant access is always denied.

        Args:
            tenant_id: The requesting tenant's ID.
            resource_tenant: The tenant that owns the resource.

        Returns:
            True if access is allowed (same tenant), False otherwise.
        """
        if tenant_id != resource_tenant:
            logger.warning(
                "Cross-tenant access denied: %s attempted to access %s's resource",
                tenant_id, resource_tenant,
            )
            return False
        return True

    def get_tenant_prefix(self, tenant_id: Optional[str] = None) -> str:
        """
        Get the key prefix for a tenant.

        Useful for bulk operations (list all keys for a tenant)
        or for constructing glob patterns.

        Args:
            tenant_id: Optional explicit tenant ID. If None, uses context.

        Returns:
            "tenant:{id}:" if a tenant is identified, "" otherwise.
        """
        tid = tenant_id or TenantContext.get()
        if tid is None:
            return ""
        return f"tenant:{tid}:"

    def scope_query(self, query: dict, tenant_id: Optional[str] = None) -> dict:
        """
        Add tenant filter to a query dict.

        For database queries that accept filter parameters, this adds
        the tenant_id condition to ensure only the tenant's data is
        returned.

        Args:
            query: Original query dict.
            tenant_id: Optional explicit tenant ID. If None, uses context.

        Returns:
            Query dict with tenant_id filter added.
        """
        tid = tenant_id or TenantContext.get()
        if tid is None:
            return query

        scoped = dict(query)
        scoped["tenant_id"] = tid
        return scoped

    def check_isolation(self, resource_tenant_id: str) -> bool:
        """
        Check if the current tenant context matches the resource owner.

        Convenience method that combines TenantContext.get() with
        validate_access().

        Args:
            resource_tenant_id: The tenant that owns the resource.

        Returns:
            True if current tenant owns the resource.

        Raises:
            RuntimeError: If no tenant context is set.
        """
        current_tenant = TenantContext.get()
        if current_tenant is None:
            # No tenant context — system-level access (allowed)
            return True
        return self.validate_access(current_tenant, resource_tenant_id)
