"""FastAPI security middleware."""
import logging, threading
from typing import ClassVar, Optional
from app.config import SecurityConfig

logger = logging.getLogger(__name__)

class SecurityContext:
    """Thread-local per-request security context."""
    _local: ClassVar[threading.local] = threading.local()

    @classmethod
    def set(cls, actor: str, roles: list, tenant_id: str = "") -> None:
        cls._local.actor = actor
        cls._local.roles = roles
        cls._local.tenant_id = tenant_id

    @classmethod
    def get_actor(cls) -> Optional[str]:
        return getattr(cls._local, "actor", None)

    @classmethod
    def get_roles(cls) -> list:
        return getattr(cls._local, "roles", [])

    @classmethod
    def get_tenant(cls) -> Optional[str]:
        return getattr(cls._local, "tenant_id", None)

    @classmethod
    def clear(cls) -> None:
        cls._local.actor = None
        cls._local.roles = []
        cls._local.tenant_id = None

class SecurityMiddleware:
    """FastAPI middleware for JWT/API-key validation."""
    PUBLIC_PATHS = {"/", "/health", "/docs", "/openapi.json", "/metrics",
                    "/redoc", "/favicon.ico"}

    def __init__(self, jwt_auth=None, api_key_mgr=None,
                 audit_logger=None, enabled: bool = False):
        self._jwt = jwt_auth
        self._keys = api_key_mgr
        self._audit = audit_logger
        self._enabled = enabled

    async def __call__(self, request, call_next):
        SecurityContext.clear()
        if not self._enabled or request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)
        auth = request.headers.get("Authorization", "")
        api_key_raw = request.headers.get("X-API-Key", "")
        actor, roles, tenant = "anonymous", [], ""
        if auth.startswith("Bearer ") and self._jwt:
            try:
                claims = self._jwt.verify_token(auth[7:])
                actor, roles, tenant = claims.sub, claims.roles, claims.tenant_id
            except Exception as e:
                logger.warning("JWT validation failed: %s", e)
        elif api_key_raw and self._keys:
            key = self._keys.verify_key(api_key_raw)
            if key:
                actor, roles, tenant = key.key_id, key.roles, key.tenant_id
        SecurityContext.set(actor, roles, tenant)
        response = await call_next(request)
        SecurityContext.clear()
        return response
