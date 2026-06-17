"""
Tenant Middleware — Request-Level Tenant Extraction

=== THEORY ===

In a multi-tenant web application, every incoming request must be
associated with a tenant.  The middleware pattern intercepts requests
at the framework level, before they reach business logic, to:

  1. Extract the tenant identifier (from header, API key, JWT, subdomain)
  2. Validate the tenant exists and is active
  3. Set the TenantContext for the duration of the request
  4. Clear the TenantContext when the request completes

This ensures that:
  - All downstream code has access to the tenant context
  - Invalid/missing tenant IDs are rejected early (before business logic)
  - Context is always cleaned up (even on exceptions)
  - Suspended/deleted tenants cannot access the system

=== EXTRACTION STRATEGIES ===

  Header-based:   X-Tenant-ID header (simple, explicit, used in service mesh)
  API key-based:  Lookup tenant from API key (common in public APIs)
  JWT claim:      Extract org_id from JWT token (OAuth2 / OIDC)
  Subdomain:      Parse tenant from subdomain (e.g., acme.app.com)
  Path prefix:    Extract from URL path (e.g., /api/v1/tenants/{id}/...)

This implementation uses header-based extraction as the primary strategy.

=== PRODUCTION EQUIVALENTS ===

  Django:       Custom middleware class with process_request/process_response
  Express.js:   app.use(tenantMiddleware) early in the chain
  Spring:       HandlerInterceptor with ThreadLocal
  FastAPI:      Middleware or Depends() for dependency injection
  Envoy:        Header-based routing rules in the sidecar proxy
"""

import logging
from typing import Any, Callable, Optional

from app.tenancy.context import TenantContext
from app.tenancy.manager import TenantManager

logger = logging.getLogger(__name__)


class TenantMiddleware:
    """
    FastAPI middleware that extracts tenant_id from request headers
    or API key and sets TenantContext for the duration of the request.

    Usage (FastAPI):
        from fastapi import FastAPI
        app = FastAPI()

        tenant_manager = TenantManager(config)
        middleware = TenantMiddleware(tenant_manager)
        app.middleware("http")(middleware)

    Usage (standalone — for testing/non-HTTP):
        middleware = TenantMiddleware(tenant_manager)
        # Simulate request processing
        middleware.extract_and_set("acme-corp")
    """

    def __init__(
        self,
        tenant_manager: TenantManager,
        header_name: str = "X-Tenant-ID",
    ) -> None:
        """
        Args:
            tenant_manager: TenantManager for tenant validation.
            header_name: HTTP header name containing the tenant ID.
        """
        self._tenant_manager = tenant_manager
        self._header_name = header_name
        logger.info("TenantMiddleware initialized: header=%s", header_name)

    async def __call__(self, request: Any, call_next: Callable) -> Any:
        """
        ASGI middleware handler.

        Extracts tenant_id from the request header, validates the tenant,
        sets TenantContext, processes the request, and clears context.

        If no tenant header is provided or the tenant is invalid, returns
        an appropriate error response.
        """
        # Extract tenant_id from header
        tenant_id = self._get_tenant_from_request(request)

        if tenant_id is None:
            # No tenant header — check if tenancy is required
            # Allow requests without tenant for system/health endpoints
            path = getattr(request, "url", None)
            path_str = str(path) if path else ""
            if self._is_exempt_path(path_str):
                return await call_next(request)

            # Return 400 for missing tenant
            return self._error_response(
                status_code=400,
                detail=f"Missing required header: {self._header_name}",
            )

        # Validate tenant exists and is active
        tenant = self._tenant_manager.get_tenant(tenant_id)
        if tenant is None:
            return self._error_response(
                status_code=404,
                detail=f"Tenant '{tenant_id}' not found",
            )

        if not tenant.is_active():
            return self._error_response(
                status_code=403,
                detail=f"Tenant '{tenant_id}' is {tenant.status}",
            )

        # Set context and process request
        TenantContext.set(tenant_id)
        try:
            response = await call_next(request)
            return response
        finally:
            TenantContext.clear()

    def _get_tenant_from_request(self, request: Any) -> Optional[str]:
        """
        Extract tenant_id from request headers.

        Handles both FastAPI Request objects and plain dicts (for testing).
        """
        # FastAPI Request object
        if hasattr(request, "headers"):
            headers = request.headers
            if hasattr(headers, "get"):
                return headers.get(self._header_name.lower()) or headers.get(self._header_name)

        # Dict-based request (testing)
        if isinstance(request, dict):
            headers = request.get("headers", {})
            return headers.get(self._header_name) or headers.get(self._header_name.lower())

        return None

    def _is_exempt_path(self, path: str) -> bool:
        """
        Check if a request path is exempt from tenant requirement.

        Health checks, metrics, and docs endpoints don't need tenant context.
        """
        exempt_prefixes = [
            "/health",
            "/metrics",
            "/docs",
            "/openapi",
            "/redoc",
        ]
        for prefix in exempt_prefixes:
            if prefix in path:
                return True
        return False

    def _error_response(self, status_code: int, detail: str) -> Any:
        """
        Create an error response.

        Attempts to use FastAPI's JSONResponse if available, falls back
        to a simple dict (for testing without FastAPI).
        """
        try:
            from starlette.responses import JSONResponse
            return JSONResponse(
                status_code=status_code,
                content={"detail": detail},
            )
        except ImportError:
            # Fallback for testing without starlette
            return {"status_code": status_code, "detail": detail}

    def extract_and_set(self, tenant_id: str) -> bool:
        """
        Programmatic tenant extraction and validation (non-HTTP use).

        Validates the tenant and sets TenantContext if valid.

        Args:
            tenant_id: The tenant ID to validate and set.

        Returns:
            True if tenant is valid and context was set, False otherwise.
        """
        tenant = self._tenant_manager.get_tenant(tenant_id)
        if tenant is None:
            logger.warning("Tenant not found: %s", tenant_id)
            return False

        if not tenant.is_active():
            logger.warning("Tenant not active: %s (status=%s)", tenant_id, tenant.status)
            return False

        TenantContext.set(tenant_id)
        return True
