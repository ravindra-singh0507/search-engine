"""API key management with hashed storage."""
import hashlib, logging, secrets, time, uuid
from dataclasses import dataclass, field
from typing import Optional
import threading
from app.config import SecurityConfig

logger = logging.getLogger(__name__)

@dataclass
class APIKey:
    key_id: str
    name: str
    prefix: str
    key_hash: str
    tenant_id: str = ""
    roles: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_used: Optional[float] = None
    expires_at: Optional[float] = None
    enabled: bool = True

    def is_expired(self) -> bool:
        return self.expires_at is not None and time.time() > self.expires_at

class APIKeyManager:
    """Manages API key lifecycle: create, verify, rotate, revoke."""
    def __init__(self, config: SecurityConfig, db=None):
        self._config = config
        self._db = db
        self._keys: dict[str, APIKey] = {}  # key_id -> APIKey
        self._hash_index: dict[str, str] = {}  # sha256(raw_key) -> key_id
        self._lock = threading.Lock()

    @staticmethod
    def _hash_key(raw: str) -> str:
        return hashlib.sha256(raw.encode()).hexdigest()

    def create_key(self, name: str, tenant_id: str = "", roles: list = [],
                   expires_in_days: Optional[int] = None) -> tuple:
        raw = f"sk-{secrets.token_urlsafe(32)}"
        key_id = str(uuid.uuid4())[:8]
        key_hash = self._hash_key(raw)
        expires_at = time.time() + expires_in_days * 86400 if expires_in_days else None
        key = APIKey(key_id=key_id, name=name, prefix=raw[:12],
                     key_hash=key_hash, tenant_id=tenant_id, roles=roles,
                     expires_at=expires_at)
        with self._lock:
            self._keys[key_id] = key
            self._hash_index[key_hash] = key_id
        logger.info("Created API key %s (%s)", key_id, name)
        return raw, key

    def verify_key(self, raw_key: str) -> Optional[APIKey]:
        h = self._hash_key(raw_key)
        with self._lock:
            key_id = self._hash_index.get(h)
            if not key_id:
                return None
            key = self._keys.get(key_id)
        if key and key.enabled and not key.is_expired():
            key.last_used = time.time()
            return key
        return None

    def revoke_key(self, key_id: str) -> bool:
        with self._lock:
            if key_id not in self._keys:
                return False
            self._keys[key_id].enabled = False
        return True

    def list_keys(self, tenant_id: str = "") -> list:
        with self._lock:
            keys = list(self._keys.values())
        if tenant_id:
            keys = [k for k in keys if k.tenant_id == tenant_id]
        return keys

    def rotate_key(self, key_id: str) -> tuple:
        with self._lock:
            old = self._keys.get(key_id)
            if not old:
                raise KeyError(f"Key {key_id} not found")
            old.enabled = False
            if old.key_hash in self._hash_index:
                del self._hash_index[old.key_hash]
        return self.create_key(old.name, old.tenant_id, old.roles)

    def stats(self) -> dict:
        with self._lock:
            total = len(self._keys)
            enabled = sum(1 for k in self._keys.values() if k.enabled)
        return {"total": total, "enabled": enabled, "disabled": total - enabled}
