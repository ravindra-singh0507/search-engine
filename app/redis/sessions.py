"""
Redis Session Store

=== THEORY ===

Server-side sessions store user state on the backend rather than in cookies
or client-side storage.  The client holds only a session ID (opaque token);
all session data lives in Redis.

Advantages over client-side sessions:
  - No size limit (cookies are capped at ~4 KB)
  - Server controls data — client cannot tamper
  - Can store sensitive data without encryption concerns
  - Easy invalidation (delete the key)

Redis hashes are ideal for sessions because:
  - Each field is independently readable/writable (HGET/HSET)
  - No serialization needed for flat key-value data
  - Atomic field updates (HSET is O(1))
  - TTL on the top-level key auto-expires the entire session

=== SESSION LIFECYCLE ===

  1. create(session_id, data)  — HSET all fields + EXPIRE
  2. get(session_id)           — HGETALL (returns {} if expired/missing)
  3. update(session_id, data)  — HSET individual fields + refresh TTL
  4. extend_ttl(session_id)    — EXPIRE to push deadline forward
  5. delete(session_id)        — DEL key
  6. list_sessions()           — KEYS pattern (admin only; O(N))

=== SECURITY CONSIDERATIONS ===

  - Session IDs should be cryptographically random (uuid4 or secrets.token_hex)
  - TTL prevents abandoned sessions from persisting forever
  - Rotate session IDs after authentication to prevent session fixation
  - In production, encrypt sensitive session fields at rest

=== COMPLEXITY ===

  create:      O(M) where M = number of fields
  get:         O(M)
  update:      O(K) where K = number of updated fields
  delete:      O(1)
  exists:      O(1)
  extend_ttl:  O(1)
  list:        O(N) where N = total keys (avoid in production; use scan)

=== AT PRODUCTION SCALE ===

  Google:    Spanner-backed session store for global consistency
  Netflix:   Zuul + EVCache for API gateway sessions
  Uber:      Redis cluster with consistent hashing for session affinity
  Shopify:   Redis for session storage with 30-minute sliding TTL
"""

import logging

from app.redis.client import RedisClient

logger = logging.getLogger(__name__)


class RedisSessionStore:
    """
    Server-side session management using Redis hashes.

    Each session is stored as a Redis hash with the key:
        {prefix}{session_id}

    Fields within the hash are flat string key-value pairs.  Complex
    values should be JSON-serialized by the caller before storing.

    TTL is set on create and refreshed on update/extend to implement
    sliding expiration.
    """

    def __init__(self, client: RedisClient, prefix: str = "session:", ttl: int = 3600):
        self._client = client
        self._prefix = prefix
        self._ttl = ttl

    def _full_key(self, session_id: str) -> str:
        """Build the Redis key for a session."""
        return f"{self._prefix}{session_id}"

    def create(self, session_id: str, data: dict) -> None:
        """
        Create a new session with the given data.

        All values in data are converted to strings before storage.
        The session TTL is set immediately.
        """
        key = self._full_key(session_id)
        for field, value in data.items():
            self._client.hset(key, str(field), str(value))
        self._client.expire(key, self._ttl)
        logger.debug("Session created: %s (%d fields, TTL=%ds)",
                      session_id, len(data), self._ttl)

    def get(self, session_id: str) -> dict | None:
        """
        Retrieve all session data.

        Returns None if the session does not exist or has expired.
        Returns a dict[str, str] of all session fields otherwise.
        """
        key = self._full_key(session_id)
        if not self._client.exists(key):
            return None
        data = self._client.hgetall(key)
        if not data:
            return None
        return data

    def update(self, session_id: str, data: dict) -> None:
        """
        Update specific fields in an existing session.

        New fields are added; existing fields are overwritten.
        The TTL is refreshed (sliding expiration).
        """
        key = self._full_key(session_id)
        if not self._client.exists(key):
            logger.warning("Session update for non-existent session: %s", session_id)
            return
        for field, value in data.items():
            self._client.hset(key, str(field), str(value))
        self._client.expire(key, self._ttl)

    def delete(self, session_id: str) -> bool:
        """Delete a session.  Returns True if the session existed."""
        key = self._full_key(session_id)
        return self._client.delete(key)

    def exists(self, session_id: str) -> bool:
        """Check whether a session exists and has not expired."""
        return self._client.exists(self._full_key(session_id))

    def extend_ttl(self, session_id: str) -> None:
        """
        Push the session's expiration deadline forward by the configured TTL.

        This implements sliding expiration: active sessions stay alive,
        idle sessions eventually expire.
        """
        key = self._full_key(session_id)
        if self._client.exists(key):
            self._client.expire(key, self._ttl)

    def list_sessions(self, limit: int = 100) -> list[str]:
        """
        List active session IDs.

        Uses KEYS pattern matching (O(N) — use sparingly).
        Returns at most `limit` session IDs with the prefix stripped.
        """
        pattern = f"{self._prefix}*"
        all_keys = self._client.keys(pattern)
        prefix_len = len(self._prefix)
        session_ids = [k[prefix_len:] for k in all_keys[:limit]]
        return session_ids
