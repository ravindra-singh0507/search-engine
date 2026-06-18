"""JWT Authentication — stateless bearer token auth using stdlib."""
import base64, hashlib, hmac, json, logging, os, time, uuid
from dataclasses import dataclass, field
from typing import Optional
from app.config import SecurityConfig

logger = logging.getLogger(__name__)

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def _b64url_decode(s: str) -> bytes:
    pad = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * (pad % 4))

@dataclass
class JWTClaims:
    sub: str
    tenant_id: str = ""
    roles: list = field(default_factory=list)
    exp: float = 0.0
    iat: float = 0.0

class JWTAuth:
    """JWT token issuer/verifier using HMAC-SHA256 (stdlib only)."""
    def __init__(self, config: SecurityConfig):
        self._config = config
        self._secret = os.environ.get(config.jwt_secret_env, "dev-secret-change-in-prod")

    def create_token(self, sub: str, tenant_id: str = "", roles: list = []) -> str:
        now = time.time()
        header = _b64url_encode(json.dumps({"alg":"HS256","typ":"JWT"}).encode())
        payload = _b64url_encode(json.dumps({
            "sub": sub, "tenant_id": tenant_id, "roles": roles,
            "iat": now, "exp": now + self._config.jwt_expiry_hours * 3600,
            "jti": str(uuid.uuid4()),
        }).encode())
        msg = f"{header}.{payload}".encode()
        sig = _b64url_encode(hmac.new(self._secret.encode(), msg, hashlib.sha256).digest())
        return f"{header}.{payload}.{sig}"

    def verify_token(self, token: str) -> JWTClaims:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid token format")
        header, payload, sig = parts
        msg = f"{header}.{payload}".encode()
        expected = _b64url_encode(hmac.new(self._secret.encode(), msg, hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            raise ValueError("Invalid token signature")
        data = json.loads(_b64url_decode(payload))
        if time.time() > data.get("exp", 0):
            raise ValueError("Token expired")
        return JWTClaims(
            sub=data.get("sub",""), tenant_id=data.get("tenant_id",""),
            roles=data.get("roles",[]), exp=data.get("exp",0), iat=data.get("iat",0),
        )

    def refresh_token(self, token: str) -> str:
        claims = self.verify_token(token)
        return self.create_token(claims.sub, claims.tenant_id, claims.roles)
