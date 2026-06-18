"""
Tenant-Aware Vector Namespaces — Phase 8 Final

Provides tenant-scoped vector operations by prefixing chunk_ids
with the tenant identifier. This ensures vectors from different
tenants don't leak across boundaries in search results.
"""

import logging
from typing import Any, Optional
from app.tenancy.context import TenantContext
from app.config import TenancyConfig

logger = logging.getLogger(__name__)


class TenantVectorNamespace:
    """
    Wraps a VectorStore to provide tenant-scoped namespaces.

    Strategy: prefix chunk_ids with "t:{tenant_id}:" so that
    search results can be filtered by tenant after retrieval.
    """

    def __init__(self, vector_store, config: TenancyConfig):
        self._store = vector_store
        self._config = config
        self._enabled = config.enabled

    @property
    def store(self):
        return self._store

    def _current_tenant(self) -> str:
        if not self._enabled:
            return ""
        return TenantContext.get() or self._config.default_tenant

    def _scope_id(self, chunk_id: str) -> str:
        tenant = self._current_tenant()
        if tenant:
            return f"t:{tenant}:{chunk_id}"
        return chunk_id

    def _unscope_id(self, scoped_id: str) -> str:
        parts = scoped_id.split(":", 2)
        if len(parts) == 3 and parts[0] == "t":
            return parts[2]
        return scoped_id

    def _is_owned(self, scoped_id: str) -> bool:
        if not self._enabled:
            return True
        tenant = self._current_tenant()
        if not tenant:
            return True
        return scoped_id.startswith(f"t:{tenant}:")

    def add(self, chunk_ids: list[str], vectors: list[list[float]]) -> None:
        scoped_ids = [self._scope_id(cid) for cid in chunk_ids]
        self._store.add(scoped_ids, vectors)

    def search(self, query_vector: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        # Retrieve more candidates to account for filtering
        oversample = top_k * 3 if self._enabled else top_k
        results = self._store.search(query_vector, oversample)
        # Filter to current tenant's vectors
        filtered = [(self._unscope_id(cid), score)
                    for cid, score in results if self._is_owned(cid)]
        return filtered[:top_k]

    def delete(self, chunk_ids: list[str]) -> None:
        scoped_ids = [self._scope_id(cid) for cid in chunk_ids]
        self._store.delete(scoped_ids)

    def save(self, path) -> None:
        self._store.save(path)

    def load(self, path) -> None:
        self._store.load(path)

    @property
    def total_vectors(self) -> int:
        return self._store.total_vectors

    def stats(self) -> dict:
        base_stats = self._store.stats() if hasattr(self._store, 'stats') else {}
        return {
            **base_stats,
            "tenant_namespacing": self._enabled,
            "current_tenant": self._current_tenant(),
        }
