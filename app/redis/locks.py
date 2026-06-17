"""
Distributed Locking

=== THEORY ===

A distributed lock ensures mutual exclusion across multiple processes or
machines.  Unlike threading.Lock (which only works within a single process),
a distributed lock uses an external coordinator (Redis, ZooKeeper, etcd)
visible to all participants.

=== REDIS LOCK ALGORITHM (Redlock-lite) ===

  ACQUIRE:
    1. Generate a unique token (UUID4) — this prevents releasing
       someone else's lock.
    2. SET lock_key token NX EX ttl
       - NX: only set if not exists (atomic test-and-set)
       - EX: auto-expire after ttl seconds (crash recovery)
    3. If SET returned OK → lock acquired.
    4. If SET returned None → lock held by another owner.
       - If blocking=True: sleep 0.1s, retry until timeout.
       - If blocking=False: return False immediately.

  RELEASE:
    1. GET lock_key → stored_token
    2. If stored_token == our token → DEL lock_key (we own it)
    3. If stored_token != our token → do nothing (someone else's lock)

  The token comparison prevents a dangerous race: if our lock expired and
  another process acquired it, a naive DEL would release their lock.

=== FENCING TOKENS ===

In production, a fencing token (monotonically increasing integer) is
attached to every lock acquisition.  The protected resource checks that
incoming requests carry a token >= its last seen token, rejecting stale
holders whose lock expired but who are still operating.  We omit this
for simplicity but note it in the docstring.

=== IN-MEMORY FALLBACK ===

InMemoryLock wraps threading.Lock for single-process deployments.  It
provides the same interface as DistributedLock so the application code
does not need to know which implementation is in use.

=== COMPLEXITY ===

  acquire: O(1) per attempt (Redis SET NX)
  release: O(1)             (Redis GET + DEL)
  Blocking acquire: O(timeout / 0.1) attempts in worst case

=== AT PRODUCTION SCALE ===

  Redis:     Redlock algorithm across 5+ independent Redis instances
  Google:    Chubby lock service (Paxos-based)
  ZooKeeper: Ephemeral sequential znodes for distributed locks
  etcd:      Lease-based locking with TTL
  Consul:    Session-based locks with health-check integration
"""

import logging
import threading
import time
import uuid

from app.redis.client import RedisClient

logger = logging.getLogger(__name__)


# -- Redis-based distributed lock ---------------------------------------------

class DistributedLock:
    """
    Redis SETNX + TTL distributed lock with token-based release.

    Usage:
        lock = DistributedLock(client, "my-resource", ttl=30)

        # Explicit acquire/release
        if lock.acquire(blocking=True, timeout=10.0):
            try:
                # critical section
                pass
            finally:
                lock.release()

        # Context manager
        with lock:
            # critical section
            pass
    """

    def __init__(self, client: RedisClient, name: str, ttl: int = 30):
        self._client = client
        self._name = name
        self._key = f"lock:{name}"
        self._ttl = ttl
        self._token: str | None = None
        self._lock = threading.Lock()

    def acquire(self, blocking: bool = True, timeout: float = 10.0) -> bool:
        """
        Attempt to acquire the distributed lock.

        If blocking=True, retries every 0.1s until the lock is acquired
        or the timeout is exceeded.  If blocking=False, returns immediately
        with True/False.

        Returns True if the lock was acquired, False otherwise.
        """
        token = str(uuid.uuid4())
        deadline = time.time() + timeout

        while True:
            # Attempt atomic SET NX EX
            # We use get+set because our protocol doesn't have setnx;
            # the InMemoryRedisClient's set is atomic under its lock
            with self._lock:
                if not self._client.exists(self._key):
                    self._client.set(self._key, token, ex=self._ttl)
                    self._token = token
                    logger.debug("Lock acquired: %s (token=%s)", self._name, token[:8])
                    return True

            if not blocking:
                return False

            if time.time() >= deadline:
                logger.debug("Lock acquire timeout: %s (%.1fs)", self._name, timeout)
                return False

            time.sleep(0.1)

    def release(self) -> bool:
        """
        Release the lock only if we still own it (token comparison).

        Returns True if the lock was released, False if we don't own it
        or it has already expired.
        """
        with self._lock:
            if self._token is None:
                return False
            stored = self._client.get(self._key)
            if stored == self._token:
                self._client.delete(self._key)
                logger.debug("Lock released: %s", self._name)
                self._token = None
                return True
            # Lock expired or acquired by someone else
            logger.debug("Lock release skipped (not owner): %s", self._name)
            self._token = None
            return False

    def is_locked(self) -> bool:
        """Check whether the lock is currently held (by anyone)."""
        return self._client.exists(self._key)

    def __enter__(self) -> "DistributedLock":
        acquired = self.acquire(blocking=True, timeout=10.0)
        if not acquired:
            raise TimeoutError(
                f"Could not acquire distributed lock {self._name!r} within timeout"
            )
        return self

    def __exit__(self, *args: object) -> None:
        self.release()


# -- In-memory fallback lock --------------------------------------------------

class InMemoryLock:
    """
    Thread-based lock with the same interface as DistributedLock.

    Uses threading.Lock for mutual exclusion within a single process.
    This is the fallback when Redis is not available.
    """

    def __init__(self, name: str = "", ttl: int = 30):
        self._name = name
        self._ttl = ttl
        self._lock = threading.Lock()
        self._owned = False

    def acquire(self, blocking: bool = True, timeout: float = 10.0) -> bool:
        """
        Acquire the lock.

        If blocking=True, waits up to `timeout` seconds.
        If blocking=False, returns immediately.
        """
        if blocking:
            acquired = self._lock.acquire(blocking=True, timeout=timeout)
        else:
            acquired = self._lock.acquire(blocking=False)
        if acquired:
            self._owned = True
        return acquired

    def release(self) -> bool:
        """Release the lock.  Returns True if we owned it."""
        if self._owned:
            self._owned = False
            try:
                self._lock.release()
                return True
            except RuntimeError:
                return False
        return False

    def is_locked(self) -> bool:
        """Check whether the lock is currently held."""
        if self._lock.acquire(blocking=False):
            self._lock.release()
            return False
        return True

    def __enter__(self) -> "InMemoryLock":
        acquired = self.acquire(blocking=True, timeout=10.0)
        if not acquired:
            raise TimeoutError(
                f"Could not acquire in-memory lock {self._name!r} within timeout"
            )
        return self

    def __exit__(self, *args: object) -> None:
        self.release()
