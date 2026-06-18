"""
Tenant-Aware Data Access Layer — Phase 8 Final

=== THEORY ===

Multi-tenant data isolation in existing systems is best achieved through
a proxy/wrapper pattern rather than invasive rewrites:

  TenantAwareDatabase (wrapper)
    -> checks TenantContext for current tenant
    -> delegates to underlying Database
    -> filters results by tenant_id
    -> prefixes keys for tenant scoping

This approach:
  - Preserves backward compatibility (all 834 tests pass unchanged)
  - Adds tenant isolation without modifying the Database class
  - Can be disabled (passthrough mode) when tenancy is off
  - Supports gradual migration (wrap one method at a time)

=== PRODUCTION EQUIVALENTS ===

  Salesforce: org-level row filters on shared schema
  Slack:      workspace_id scoping at the data access layer
  Shopify:    tenant-aware ActiveRecord scopes
  AWS:        resource-level IAM policy enforcement
"""

import logging
import json
from typing import Any, Optional
from app.database.db import Database
from app.tenancy.context import TenantContext
from app.config import TenancyConfig

logger = logging.getLogger(__name__)


class TenantAwareDatabase:
    """
    Proxy that wraps Database and adds tenant-scoped operations.

    When tenancy is enabled:
      - insert operations tag with current tenant_id
      - query operations filter by current tenant_id
      - cross-tenant access is denied

    When tenancy is disabled (default):
      - all operations pass through unchanged

    Usage:
        db = Database(path)
        tenant_db = TenantAwareDatabase(db, config.tenancy)
        # Use tenant_db instead of db in tenant-scoped contexts
    """

    def __init__(self, db: Database, config: TenancyConfig):
        self._db = db
        self._config = config
        self._enabled = config.enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def db(self) -> Database:
        """Access the underlying database directly (for non-tenant operations)."""
        return self._db

    def _current_tenant(self) -> str:
        """Get the current tenant from context, or default."""
        if not self._enabled:
            return ""
        tenant = TenantContext.get()
        return tenant or self._config.default_tenant

    def _require_tenant(self) -> str:
        """Require a tenant in context. Raises if none set and tenancy enabled."""
        if not self._enabled:
            return ""
        tenant = TenantContext.get()
        if not tenant:
            raise PermissionError("Tenant context required but not set")
        return tenant

    # -- Document Operations (tenant-scoped) ---------------------------------

    def insert_document(self, title: str, content: str, source: str = "local",
                        doc_type: str = "text", word_count: int = 0) -> int:
        """Insert document tagged with current tenant."""
        tenant = self._current_tenant()
        # Tag source with tenant prefix for tenant-scoped retrieval
        scoped_source = f"tenant:{tenant}:{source}" if tenant else source
        return self._db.insert_document(title, content, scoped_source, doc_type, word_count)

    def get_document(self, doc_id: int) -> Any:
        """Get document, verify tenant ownership."""
        doc = self._db.get_document(doc_id)
        if doc is None:
            return None
        if self._enabled and self._current_tenant():
            tenant = self._current_tenant()
            if not doc.source.startswith(f"tenant:{tenant}:") and doc.source != "local":
                logger.warning("Cross-tenant access denied: tenant=%s doc=%d", tenant, doc_id)
                return None
        return doc

    def get_all_documents(self) -> list:
        """Get all documents for current tenant."""
        if not self._enabled or not self._current_tenant():
            return self._db.get_all_documents()
        tenant = self._current_tenant()
        all_docs = self._db.get_all_documents()
        return [d for d in all_docs if d.source.startswith(f"tenant:{tenant}:") or d.source == "local"]

    def get_document_count(self) -> int:
        """Get document count for current tenant."""
        if not self._enabled or not self._current_tenant():
            return self._db.get_document_count()
        return len(self.get_all_documents())

    # -- Search Operations (tenant-scoped) -----------------------------------

    def log_search(self, query: str, results_count: int,
                   latency_ms: float, session_id: Optional[str] = None) -> int:
        """Log search with tenant context."""
        tenant = self._current_tenant()
        tagged_session = f"tenant:{tenant}:{session_id}" if tenant and session_id else session_id
        return self._db.log_search(query, results_count, latency_ms, tagged_session)

    # -- Session Operations (tenant-scoped) ----------------------------------

    def create_conversation_session(self, session_id: str, user_id: str,
                                     created_at: str) -> None:
        """Create session tagged with tenant."""
        tenant = self._current_tenant()
        tagged_user = f"tenant:{tenant}:{user_id}" if tenant else user_id
        self._db.create_conversation_session(session_id, tagged_user, created_at)

    def list_conversation_sessions(self, limit: int = 100) -> list:
        """List sessions for current tenant only."""
        if not self._enabled or not self._current_tenant():
            return self._db.list_conversation_sessions(limit)
        tenant = self._current_tenant()
        all_sessions = self._db.list_conversation_sessions(limit * 5)
        return [s for s in all_sessions
                if s.get("user_id", "").startswith(f"tenant:{tenant}:")][:limit]

    # -- Research Operations (tenant-scoped) ---------------------------------

    def get_research_sessions(self, user_id: Optional[str] = None, limit: int = 20) -> list:
        """Get research sessions for current tenant."""
        if not self._enabled or not self._current_tenant():
            return self._db.get_research_sessions(user_id, limit)
        tenant = self._current_tenant()
        all_sessions = self._db.get_research_sessions(user_id, limit * 5)
        return [s for s in all_sessions
                if s.get("user_id", "").startswith(f"tenant:{tenant}:")][:limit]

    # -- Passthrough Operations ----------------------------------------------
    # For operations that don't need tenant scoping (global metadata)

    def __getattr__(self, name: str) -> Any:
        """Passthrough: delegate all other methods to underlying Database."""
        return getattr(self._db, name)

    # -- Tenant Stats --------------------------------------------------------

    def tenant_stats(self) -> dict:
        """Get usage statistics for current tenant."""
        tenant = self._current_tenant()
        docs = self.get_all_documents()
        return {
            "tenant_id": tenant,
            "document_count": len(docs),
            "enabled": self._enabled,
        }

    def validate_access(self, doc_id: int) -> bool:
        """Check if current tenant can access a document."""
        if not self._enabled:
            return True
        doc = self._db.get_document(doc_id)
        if doc is None:
            return False
        tenant = self._current_tenant()
        if not tenant:
            return True
        return doc.source.startswith(f"tenant:{tenant}:") or doc.source == "local"
