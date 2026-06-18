"""
Security Enforcement Layer — Phase 8 Final

Provides endpoint-level security enforcement without modifying
each endpoint individually. Uses a permission matrix that maps
URL patterns to required permissions/roles.

=== THEORY ===

Rather than adding @require_permission decorators to 111 endpoints
(invasive, error-prone), we define a centralized permission matrix
and enforce it in middleware. This is how API gateways work:
  - Kong: route-level plugin configuration
  - Envoy: RBAC filter chain
  - AWS API Gateway: authorizer + resource policies

The matrix approach gives us:
  1. Single source of truth for access control
  2. Easy auditing — "who can access what?" is one dict lookup
  3. No endpoint modification — enforcement is orthogonal to business logic
  4. Pattern matching — prefix-based rules cover entire URL subtrees

Thread-safety: SecurityContext is thread-local, and the matrix is
immutable after module load. No locking is needed.

=== PRODUCTION EQUIVALENTS ===

  Kong:            route-level ACL plugin
  Envoy:           RBAC filter with principal/permission rules
  AWS API Gateway: Lambda authorizer + resource policies
  Istio:           AuthorizationPolicy CRD
  Spring Security: HttpSecurity antMatchers + authorities
"""

import logging
import threading
from enum import Enum
from typing import Optional

from app.config import SecurityConfig
from app.security.middleware import SecurityContext
from app.security.rbac import Permission, RBACEnforcer

logger = logging.getLogger(__name__)


class EndpointAccess(str, Enum):
    """
    Classification of endpoint access requirements.

    Each level builds on the previous:
      PUBLIC         — no authentication required
      AUTHENTICATED — any valid token/API key
      TENANT_SCOPED — authenticated + tenant ID present in context
      ADMIN_ONLY    — requires admin role
      SERVICE_ONLY  — requires service role (machine-to-machine)
    """
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    TENANT_SCOPED = "tenant_scoped"
    ADMIN_ONLY = "admin_only"
    SERVICE_ONLY = "service_only"


# ── Endpoint Permission Matrix ────────────────────────────────────────────────
#
# Maps URL path patterns to (access_level, required_permission).
# Patterns ending with "/" are treated as prefixes (match any sub-path).
# Exact matches take priority over prefix matches.
#
# This is the single source of truth for endpoint-level authorization.
# Adding a new endpoint? Add its entry here — no decorator needed.

ENDPOINT_MATRIX: dict[str, tuple[EndpointAccess, Optional[Permission]]] = {
    # ── Public endpoints (no auth) ────────────────────────────────────────────
    "/health": (EndpointAccess.PUBLIC, None),
    "/docs": (EndpointAccess.PUBLIC, None),
    "/openapi.json": (EndpointAccess.PUBLIC, None),
    "/redoc": (EndpointAccess.PUBLIC, None),
    "/metrics": (EndpointAccess.PUBLIC, None),

    # ── Authenticated: search ─────────────────────────────────────────────────
    "/search": (EndpointAccess.AUTHENTICATED, Permission.READ_SEARCH),
    "/autocomplete": (EndpointAccess.AUTHENTICATED, Permission.READ_SEARCH),
    "/spellcheck": (EndpointAccess.AUTHENTICATED, Permission.READ_SEARCH),
    "/semantic-search": (EndpointAccess.AUTHENTICATED, Permission.READ_SEARCH),
    "/hybrid-search": (EndpointAccess.AUTHENTICATED, Permission.READ_SEARCH),
    "/rerank-search": (EndpointAccess.AUTHENTICATED, Permission.READ_SEARCH),
    "/gateway/search": (EndpointAccess.AUTHENTICATED, Permission.READ_SEARCH),

    # ── Authenticated: retrieval ──────────────────────────────────────────────
    "/explain": (EndpointAccess.AUTHENTICATED, Permission.READ_SEARCH),
    "/evaluation": (EndpointAccess.AUTHENTICATED, Permission.READ_ANALYTICS),
    "/fusion/compare": (EndpointAccess.AUTHENTICATED, Permission.READ_SEARCH),

    # ── Write: indexing (tenant-scoped) ───────────────────────────────────────
    "/index": (EndpointAccess.TENANT_SCOPED, Permission.WRITE_INDEX),
    "/document/": (EndpointAccess.TENANT_SCOPED, Permission.WRITE_INDEX),
    "/embeddings/reindex": (EndpointAccess.TENANT_SCOPED, Permission.WRITE_INDEX),

    # ── Write: crawling (tenant-scoped) ───────────────────────────────────────
    "/crawl": (EndpointAccess.TENANT_SCOPED, Permission.WRITE_CRAWL),

    # ── RAG (authenticated) ───────────────────────────────────────────────────
    "/chat": (EndpointAccess.AUTHENTICATED, Permission.READ_SEARCH),
    "/rag/": (EndpointAccess.AUTHENTICATED, Permission.READ_SEARCH),
    "/research": (EndpointAccess.AUTHENTICATED, Permission.WRITE_AGENTS),

    # ── Agents (authenticated) ────────────────────────────────────────────────
    "/research/": (EndpointAccess.AUTHENTICATED, Permission.READ_AGENTS),

    # ── Analytics (authenticated) ─────────────────────────────────────────────
    "/analytics/": (EndpointAccess.AUTHENTICATED, Permission.READ_ANALYTICS),

    # ── Admin only ────────────────────────────────────────────────────────────
    "/tenants": (EndpointAccess.ADMIN_ONLY, Permission.MANAGE_TENANTS),
    "/security/": (EndpointAccess.ADMIN_ONLY, Permission.ADMIN),
    "/services/register": (EndpointAccess.ADMIN_ONLY, Permission.ADMIN),
    "/infrastructure/": (EndpointAccess.ADMIN_ONLY, Permission.ADMIN),

    # ── Memory/sessions ───────────────────────────────────────────────────────
    "/memory": (EndpointAccess.AUTHENTICATED, Permission.READ_SEARCH),

    # ── Events ────────────────────────────────────────────────────────────────
    "/events": (EndpointAccess.AUTHENTICATED, Permission.READ_ANALYTICS),

    # ── Cost/Performance (admin) ──────────────────────────────────────────────
    "/cost/": (EndpointAccess.ADMIN_ONLY, Permission.ADMIN),
    "/performance/": (EndpointAccess.ADMIN_ONLY, Permission.ADMIN),
    "/observability/": (EndpointAccess.ADMIN_ONLY, Permission.ADMIN),
    "/resilience/": (EndpointAccess.ADMIN_ONLY, Permission.ADMIN),
}


class SecurityEnforcer:
    """
    Centralized security enforcement using the endpoint permission matrix.

    Called from SecurityMiddleware to determine if a request should be
    allowed or denied based on the caller's identity and the endpoint's
    access requirements.

    The enforcement flow:
      1. Classify the endpoint (exact match → prefix match → default)
      2. If PUBLIC, allow immediately
      3. Check authentication (actor must not be anonymous)
      4. Check access level (ADMIN_ONLY, SERVICE_ONLY, TENANT_SCOPED)
      5. Check specific permission if one is required

    Thread-safety: reads from immutable ENDPOINT_MATRIX and thread-local
    SecurityContext. The enforcer itself holds no mutable shared state
    beyond the enable flag (set once at construction).
    """

    def __init__(self, config: SecurityConfig, rbac: RBACEnforcer = None):
        self._config = config
        self._rbac = rbac or RBACEnforcer()
        self._enabled = config.enabled
        self._lock = threading.Lock()

    def classify_endpoint(self, path: str) -> tuple[EndpointAccess, Optional[Permission]]:
        """
        Determine the access requirements for a given URL path.

        Resolution order:
          1. Exact match in ENDPOINT_MATRIX
          2. Longest prefix match (patterns ending with "/")
          3. Default: AUTHENTICATED with no specific permission

        Returns:
            Tuple of (EndpointAccess level, required Permission or None)
        """
        # Check exact matches first (O(1) dict lookup)
        if path in ENDPOINT_MATRIX:
            return ENDPOINT_MATRIX[path]

        # Check prefix matches — find the longest matching prefix
        best_match: Optional[str] = None
        best_length = 0
        for pattern, access_info in ENDPOINT_MATRIX.items():
            if pattern.endswith("/") and path.startswith(pattern):
                if len(pattern) > best_length:
                    best_match = pattern
                    best_length = len(pattern)

        if best_match is not None:
            return ENDPOINT_MATRIX[best_match]

        # Default: require authentication but no specific permission
        return (EndpointAccess.AUTHENTICATED, None)

    def check_access(self, path: str, method: str = "GET") -> tuple[bool, str]:
        """
        Check if the current SecurityContext has access to the endpoint.

        Reads the calling identity from thread-local SecurityContext
        (populated by SecurityMiddleware earlier in the request pipeline).

        Args:
            path:   The URL path being accessed.
            method: HTTP method (reserved for future method-level rules).

        Returns:
            Tuple of (allowed: bool, reason: str).
            Reason is machine-readable for logging/metrics:
              - "security_disabled"
              - "public_endpoint"
              - "authentication_required"
              - "admin_required"
              - "service_role_required"
              - "tenant_required"
              - "permission_denied:<perm_value>"
              - "allowed"
        """
        if not self._enabled:
            return True, "security_disabled"

        access_level, required_perm = self.classify_endpoint(path)

        # Public endpoints are always accessible
        if access_level == EndpointAccess.PUBLIC:
            return True, "public_endpoint"

        # All other levels require authentication
        actor = SecurityContext.get_actor()
        if not actor or actor == "anonymous":
            return False, "authentication_required"

        roles = SecurityContext.get_roles()

        # Admin-only endpoints
        if access_level == EndpointAccess.ADMIN_ONLY:
            if not self._rbac.has_permission(roles, Permission.ADMIN):
                return False, "admin_required"

        # Service-only endpoints (machine-to-machine)
        if access_level == EndpointAccess.SERVICE_ONLY:
            if "service" not in roles:
                return False, "service_role_required"

        # Tenant-scoped endpoints require a tenant ID in context
        if access_level == EndpointAccess.TENANT_SCOPED:
            tenant = SecurityContext.get_tenant()
            if not tenant:
                return False, "tenant_required"

        # Check specific permission if required
        if required_perm and not self._rbac.has_permission(roles, required_perm):
            return False, f"permission_denied:{required_perm.value}"

        return True, "allowed"

    def stats(self) -> dict:
        """Return enforcement statistics for monitoring dashboards."""
        return {
            "enabled": self._enabled,
            "endpoint_rules": len(ENDPOINT_MATRIX),
            "public_endpoints": sum(
                1 for _, (a, _) in ENDPOINT_MATRIX.items()
                if a == EndpointAccess.PUBLIC
            ),
            "authenticated_endpoints": sum(
                1 for _, (a, _) in ENDPOINT_MATRIX.items()
                if a == EndpointAccess.AUTHENTICATED
            ),
            "tenant_scoped_endpoints": sum(
                1 for _, (a, _) in ENDPOINT_MATRIX.items()
                if a == EndpointAccess.TENANT_SCOPED
            ),
            "admin_endpoints": sum(
                1 for _, (a, _) in ENDPOINT_MATRIX.items()
                if a == EndpointAccess.ADMIN_ONLY
            ),
            "service_endpoints": sum(
                1 for _, (a, _) in ENDPOINT_MATRIX.items()
                if a == EndpointAccess.SERVICE_ONLY
            ),
        }
