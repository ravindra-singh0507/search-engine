"""
Tenant-Scoped Cache Keys — Phase 8 Final

Prefixes all Redis/cache keys with the tenant identifier to prevent
cross-tenant cache pollution.
"""

import logging
from typing import Any, Optional
from app.tenancy.context import TenantContext
from app.config import TenancyConfig

logger = logging.getLogger(__name__)


class TenantCacheScope:
    """
    Wraps a RedisClient or cache to add tenant-scoped key prefixing.

    All keys become: "t:{tenant_id}:{original_key}"
    When tenancy is disabled, keys pass through unchanged.
    """

    def __init__(self, client, config: TenancyConfig):
        self._client = client
        self._config = config
        self._enabled = config.enabled

    def _scope_key(self, key: str) -> str:
        if not self._enabled:
            return key
        tenant = TenantContext.get() or self._config.default_tenant
        return f"t:{tenant}:{key}"

    def get(self, key: str) -> Optional[str]:
        return self._client.get(self._scope_key(key))

    def set(self, key: str, value: str, ex: Optional[int] = None) -> None:
        self._client.set(self._scope_key(key), value, ex=ex)

    def delete(self, key: str) -> bool:
        return self._client.delete(self._scope_key(key))

    def exists(self, key: str) -> bool:
        return self._client.exists(self._scope_key(key))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def stats(self) -> dict:
        return {
            "tenant_scoping": self._enabled,
            "current_tenant": TenantContext.get() or "",
        }
