"""
Tenant Context — Thread-Local Tenant Scoping

=== THEORY ===

In a multi-tenant system, every request must be associated with a specific
tenant.  The tenant context pattern uses thread-local storage to hold the
current tenant_id for the duration of a request, making it available to
all layers (service, repository, cache) without passing it explicitly
through every function signature.

Flow:
  1. Middleware extracts tenant_id from request header/token
  2. Middleware calls TenantContext.set(tenant_id)
  3. Service/repository layers call TenantContext.get() or .require()
  4. Middleware calls TenantContext.clear() in finally block

The context manager `tenant_scope` wraps this pattern for non-HTTP
use cases (background jobs, tests, CLI commands).

=== THREAD SAFETY ===

threading.local() provides per-thread isolation automatically.  Each
thread (handling one HTTP request in a thread-pool-based server) has
its own tenant_id value.  No locks are needed.

For async frameworks (asyncio), contextvars.ContextVar would be used
instead.  We use threading.local for compatibility with the existing
synchronous architecture.

=== PRODUCTION EQUIVALENTS ===

  Spring:     SecurityContextHolder (thread-local per request)
  Django:     request.user / custom middleware with thread-local
  Flask:      flask.g (request-scoped context)
  gRPC:       Metadata propagation via context
  OpenTelemetry: Context propagation for tracing spans
"""

import logging
import threading
from typing import ClassVar, Optional

logger = logging.getLogger(__name__)


class TenantContext:
    """
    Thread-local tenant context for request-scoped tenant isolation.

    Set at the beginning of each request (middleware), cleared at the end.
    All data access layers check this context to scope queries.

    Usage:
        # In middleware
        TenantContext.set("acme-corp")
        try:
            # ... handle request ...
            tenant_id = TenantContext.require()  # "acme-corp"
        finally:
            TenantContext.clear()

        # Or use the context manager
        with tenant_scope("acme-corp") as tid:
            # tid == "acme-corp"
            # TenantContext.get() == "acme-corp"
    """

    _current: ClassVar[threading.local] = threading.local()

    @classmethod
    def set(cls, tenant_id: str) -> None:
        """
        Set the current tenant for this thread.

        Args:
            tenant_id: The tenant identifier to associate with this thread.
        """
        cls._current.tenant_id = tenant_id
        logger.debug("TenantContext set: %s", tenant_id)

    @classmethod
    def get(cls) -> Optional[str]:
        """
        Get the current tenant for this thread.

        Returns:
            The current tenant_id, or None if no tenant is set.
        """
        return getattr(cls._current, "tenant_id", None)

    @classmethod
    def clear(cls) -> None:
        """
        Clear the current tenant context.

        Must be called at the end of every request to prevent tenant
        context leaking between requests on the same thread.
        """
        cls._current.tenant_id = None
        logger.debug("TenantContext cleared")

    @classmethod
    def require(cls) -> str:
        """
        Get the current tenant, raising if none is set.

        Use this in code paths that absolutely require a tenant context
        (e.g., data access layers).  Fails loudly if the middleware
        forgot to set the context.

        Returns:
            The current tenant_id.

        Raises:
            RuntimeError: If no tenant context is set.
        """
        tenant_id = cls.get()
        if tenant_id is None:
            raise RuntimeError(
                "No tenant context set. Ensure TenantMiddleware is active "
                "or wrap the operation in tenant_scope()."
            )
        return tenant_id


class tenant_scope:
    """
    Context manager for tenant-scoped operations.

    Sets the tenant context on entry, clears it on exit.
    Useful for background jobs, tests, and CLI commands where
    there is no HTTP middleware to set the context.

    Usage:
        with tenant_scope("acme-corp") as tenant_id:
            # All operations in this block are scoped to acme-corp
            docs = repository.list_documents()  # only acme-corp docs
    """

    def __init__(self, tenant_id: str) -> None:
        self._tenant_id = tenant_id
        self._previous: Optional[str] = None

    def __enter__(self) -> str:
        """Set tenant context and return the tenant_id."""
        # Save previous context (for nested scopes)
        self._previous = TenantContext.get()
        TenantContext.set(self._tenant_id)
        return self._tenant_id

    def __exit__(self, *args) -> None:
        """Restore previous tenant context."""
        if self._previous is not None:
            TenantContext.set(self._previous)
        else:
            TenantContext.clear()
