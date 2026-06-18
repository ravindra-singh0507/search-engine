"""Role-Based Access Control."""
import logging
from enum import Enum
from app.config import SecurityConfig

logger = logging.getLogger(__name__)

class Permission(str, Enum):
    READ_SEARCH      = "search:read"
    WRITE_INDEX      = "index:write"
    DELETE_INDEX     = "index:delete"
    READ_ANALYTICS   = "analytics:read"
    WRITE_CRAWL      = "crawl:write"
    READ_AGENTS      = "agents:read"
    WRITE_AGENTS     = "agents:write"
    MANAGE_TENANTS   = "tenants:manage"
    ADMIN            = "admin:*"

ROLE_PERMISSIONS: dict = {
    "reader":     {Permission.READ_SEARCH, Permission.READ_ANALYTICS},
    "indexer":    {Permission.READ_SEARCH, Permission.WRITE_INDEX},
    "operator":   {Permission.READ_SEARCH, Permission.WRITE_INDEX,
                   Permission.DELETE_INDEX, Permission.WRITE_CRAWL,
                   Permission.READ_ANALYTICS},
    "agent_user": {Permission.READ_SEARCH, Permission.READ_AGENTS,
                   Permission.WRITE_AGENTS},
    "admin":      set(Permission),
}

class RBACEnforcer:
    """Role-Based Access Control enforcer."""
    def __init__(self, role_map: dict = None):
        self._roles = dict(role_map or ROLE_PERMISSIONS)

    def has_permission(self, roles: list, required: Permission) -> bool:
        for role in roles:
            perms = self._roles.get(role, set())
            if Permission.ADMIN in perms or required in perms:
                return True
        return False

    def check_permission(self, roles: list, required: Permission) -> None:
        if not self.has_permission(roles, required):
            raise PermissionError(f"Missing permission: {required.value}")

    def get_permissions(self, roles: list) -> set:
        perms: set = set()
        for role in roles:
            perms |= self._roles.get(role, set())
        return perms

    def add_role(self, role: str, permissions: set) -> None:
        self._roles[role] = permissions
