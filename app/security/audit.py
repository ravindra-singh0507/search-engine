"""Immutable append-only audit log."""
import json, logging, threading, time, uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
from app.config import SecurityConfig

logger = logging.getLogger(__name__)

class AuditEventType(str, Enum):
    AUTH_SUCCESS      = "auth.success"
    AUTH_FAILURE      = "auth.failure"
    API_KEY_CREATED   = "api_key.created"
    API_KEY_REVOKED   = "api_key.revoked"
    DOCUMENT_INDEXED  = "document.indexed"
    DOCUMENT_DELETED  = "document.deleted"
    SEARCH_EXECUTED   = "search.executed"
    ADMIN_ACTION      = "admin.action"
    RATE_LIMIT_HIT    = "rate_limit.hit"
    PERMISSION_DENIED = "permission.denied"

@dataclass
class AuditEvent:
    event_type: AuditEventType
    actor: str
    tenant_id: str = ""
    resource: str = ""
    action: str = ""
    result: str = "success"
    ip_address: str = ""
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return {**self.__dict__, "event_type": self.event_type.value}

class AuditLogger:
    """Append-only audit logger writing JSONL to disk."""
    def __init__(self, config: SecurityConfig, db=None):
        self._config = config
        self._lock = threading.Lock()
        self._buffer: deque = deque(maxlen=1000)
        self._count = 0
        if config.audit_log_enabled:
            Path(config.audit_log_path).parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: AuditEvent) -> None:
        with self._lock:
            self._buffer.append(event)
            self._count += 1
        if self._config.audit_log_enabled:
            try:
                with open(self._config.audit_log_path, "a") as f:
                    f.write(json.dumps(event.to_dict()) + "\n")
            except Exception as e:
                logger.error("Audit log write failed: %s", e)

    def log_auth(self, actor: str, success: bool, ip: str = "", tenant_id: str = "") -> None:
        self.log(AuditEvent(
            event_type=AuditEventType.AUTH_SUCCESS if success else AuditEventType.AUTH_FAILURE,
            actor=actor, tenant_id=tenant_id, ip_address=ip,
            result="success" if success else "failure",
        ))

    def log_action(self, actor: str, event_type: AuditEventType, resource: str,
                   tenant_id: str = "", metadata: dict = {}) -> None:
        self.log(AuditEvent(event_type=event_type, actor=actor,
                            tenant_id=tenant_id, resource=resource, metadata=metadata))

    def get_recent(self, limit: int = 100, tenant_id: Optional[str] = None) -> list:
        with self._lock:
            events = list(self._buffer)
        if tenant_id:
            events = [e for e in events if e.tenant_id == tenant_id]
        return events[-limit:]

    def stats(self) -> dict:
        return {"total_events": self._count, "buffer_size": len(self._buffer),
                "audit_log_enabled": self._config.audit_log_enabled}
