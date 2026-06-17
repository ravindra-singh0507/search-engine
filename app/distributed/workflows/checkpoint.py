"""
Workflow Checkpoint Store — Phase 8 Batch 3

=== THEORY ===

Checkpointing enables fault-tolerant workflow execution by persisting
intermediate state to durable storage after each successful step.
If a worker crashes, another worker can load the last checkpoint and
resume from where execution left off — no wasted computation.

This implements the Checkpoint/Restart pattern (also known as
"Sagas with compensation" in distributed transactions literature).

Key properties of a good checkpoint:
  1. Atomic   — either the full checkpoint is written or nothing
  2. Durable  — survives process/machine restarts
  3. Bounded  — old checkpoints are garbage-collected (TTL or count limit)
  4. Fast     — saving a checkpoint should not dominate step latency

=== ARCHITECTURE ===

  CheckpointStore (Protocol)
    │
    ├── InMemoryCheckpointStore  — dict-backed, for dev/testing
    │     bounded by max_size, FIFO eviction
    │
    └── RedisCheckpointStore     — Redis HSET-backed, for production
          TTL-based expiry via EXPIRE

=== DATA STRUCTURES ===

  InMemoryCheckpointStore._store:  OrderedDict[execution_id → checkpoint_dict]
  RedisCheckpointStore:            Redis Hash per execution_id

=== COMPLEXITY ===

  InMemoryCheckpointStore:
    save():   O(1) amortised (dict set + possible eviction)
    load():   O(1) (dict lookup)
    delete(): O(1)
    list():   O(N) where N = stored checkpoints

  RedisCheckpointStore:
    save():   O(F) where F = fields in checkpoint dict (HSET)
    load():   O(F) (HGETALL)
    delete(): O(1) (DEL)
    list():   O(N) (SCAN)

=== PRODUCTION EQUIVALENTS ===

  Temporal:       Workflow history events persisted to Cassandra/MySQL
  Airflow:        XCom + metadata database
  Prefect:        State snapshots in Orion DB
  Step Functions: Execution history in AWS-managed storage
"""

import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ── Protocol ────────────────────────────────────────────────────────────────

@runtime_checkable
class CheckpointStore(Protocol):
    """
    Protocol for workflow checkpoint persistence.

    Any implementation must provide save/load/delete/list_checkpoints.
    The @runtime_checkable decorator enables isinstance() checks.
    """

    def save(self, execution_id: str, data: dict) -> None:
        """
        Persist a checkpoint for the given execution.

        If a checkpoint already exists for this execution_id, it is
        overwritten (only the latest checkpoint matters for recovery).
        """
        ...

    def load(self, execution_id: str) -> dict | None:
        """
        Load the most recent checkpoint for an execution.

        Returns None if no checkpoint exists.
        """
        ...

    def delete(self, execution_id: str) -> None:
        """
        Delete the checkpoint for an execution.

        Called after successful completion or explicit cleanup.
        No-op if the checkpoint does not exist.
        """
        ...

    def list_checkpoints(self) -> list[str]:
        """
        Return all execution_ids that have stored checkpoints.

        Used during recovery to find workflows that need resumption.
        """
        ...


# ── In-Memory Implementation ────────────────────────────────────────────────

class InMemoryCheckpointStore:
    """
    In-memory checkpoint store for development and testing.

    Uses an OrderedDict to maintain insertion order, enabling FIFO
    eviction when the store exceeds max_size.  Thread-safe via Lock.

    === TRADEOFFS ===

      + Zero external dependencies
      + Fast (all in-process)
      - Not durable (lost on process restart)
      - Single-process only (not shared across workers)
      - Bounded by max_size to prevent memory leaks

    === EVICTION POLICY ===

    When max_size is exceeded, the oldest checkpoint (first inserted
    that hasn't been updated) is evicted.  This is acceptable because:
      - Active workflows update their checkpoint frequently
      - Stale checkpoints belong to workflows that likely completed or failed
    """

    def __init__(self, max_size: int = 100) -> None:
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

    def save(self, execution_id: str, data: dict) -> None:
        """
        Save checkpoint data for an execution.

        If the execution already has a checkpoint, it is moved to the end
        (most recent) and overwritten.  If the store exceeds max_size,
        the oldest entry is evicted.
        """
        with self._lock:
            # Move to end if already present (refresh LRU order)
            if execution_id in self._store:
                self._store.move_to_end(execution_id)

            self._store[execution_id] = {
                **data,
                "_checkpoint_time": time.time(),
            }

            # Evict oldest if over capacity
            while len(self._store) > self._max_size:
                evicted_id, _ = self._store.popitem(last=False)
                logger.debug(
                    "Checkpoint store evicted oldest entry: %s", evicted_id[:8]
                )

    def load(self, execution_id: str) -> dict | None:
        """
        Load checkpoint data for an execution.

        Returns None if no checkpoint exists for this execution_id.
        """
        with self._lock:
            data = self._store.get(execution_id)
            if data is None:
                return None
            # Return a copy to prevent mutation of stored state
            return dict(data)

    def delete(self, execution_id: str) -> None:
        """
        Delete the checkpoint for an execution.

        No-op if the execution_id is not found.
        """
        with self._lock:
            self._store.pop(execution_id, None)

    def list_checkpoints(self) -> list[str]:
        """
        Return all execution_ids with stored checkpoints.

        Returns a snapshot of current keys (thread-safe copy).
        """
        with self._lock:
            return list(self._store.keys())

    def __len__(self) -> int:
        """Return the number of stored checkpoints."""
        with self._lock:
            return len(self._store)


# ── Redis Implementation ────────────────────────────────────────────────────

class RedisCheckpointStore:
    """
    Redis-backed checkpoint store for production deployments.

    Uses Redis Hash (HSET) to store checkpoint fields, with EXPIRE for
    automatic TTL-based cleanup of old checkpoints.

    === ARCHITECTURE ===

    Redis key format: {prefix}{execution_id}
    Each key is a Redis Hash with fields from the checkpoint dict.
    Complex values (nested dicts/lists) are JSON-serialised.

    === TRADEOFFS ===

      + Durable across process restarts
      + Shared across workers (distributed recovery)
      + TTL handles cleanup automatically
      + Sub-millisecond reads/writes
      - Requires Redis infrastructure
      - Serialisation overhead for complex objects
      - Bounded by Redis memory

    === PRODUCTION EQUIVALENTS ===

    Netflix EVCache, DoorDash Redis, Uber Redis cluster
    """

    def __init__(
        self,
        redis_client: Any,
        prefix: str = "wf:checkpoint:",
        ttl: int = 86400,
    ) -> None:
        """
        Initialise the Redis checkpoint store.

        Args:
            redis_client: A Redis client instance (redis.Redis or compatible).
            prefix:       Key prefix for all checkpoint keys in Redis.
            ttl:          Time-to-live in seconds for checkpoint keys (default 24h).
        """
        self._redis = redis_client
        self._prefix = prefix
        self._ttl = ttl
        self._lock = threading.Lock()

    def _key(self, execution_id: str) -> str:
        """Build the Redis key for an execution checkpoint."""
        return f"{self._prefix}{execution_id}"

    def save(self, execution_id: str, data: dict) -> None:
        """
        Persist a checkpoint to Redis using HSET.

        All values are JSON-encoded for safe serialisation.
        The key's TTL is refreshed on every save.
        """
        key = self._key(execution_id)
        # Serialise all values to JSON strings for Redis Hash compatibility
        serialised: dict[str, str] = {}
        for field_name, value in data.items():
            serialised[field_name] = json.dumps(value)
        serialised["_checkpoint_time"] = json.dumps(time.time())

        try:
            with self._lock:
                self._redis.hset(key, mapping=serialised)
                self._redis.expire(key, self._ttl)
            logger.debug("Checkpoint saved to Redis: %s", execution_id[:8])
        except Exception as exc:
            logger.error("Failed to save checkpoint to Redis: %s", exc)

    def load(self, execution_id: str) -> dict | None:
        """
        Load a checkpoint from Redis using HGETALL.

        Returns None if the key does not exist.
        Values are JSON-decoded back to Python objects.
        """
        key = self._key(execution_id)
        try:
            with self._lock:
                raw = self._redis.hgetall(key)
            if not raw:
                return None
            # Deserialise JSON values
            result: dict[str, Any] = {}
            for field_name, value in raw.items():
                # Redis may return bytes; decode if needed
                fname = field_name.decode() if isinstance(field_name, bytes) else field_name
                val = value.decode() if isinstance(value, bytes) else value
                try:
                    result[fname] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    result[fname] = val
            return result
        except Exception as exc:
            logger.error("Failed to load checkpoint from Redis: %s", exc)
            return None

    def delete(self, execution_id: str) -> None:
        """
        Delete a checkpoint from Redis using DEL.

        No-op if the key does not exist.
        """
        key = self._key(execution_id)
        try:
            with self._lock:
                self._redis.delete(key)
            logger.debug("Checkpoint deleted from Redis: %s", execution_id[:8])
        except Exception as exc:
            logger.error("Failed to delete checkpoint from Redis: %s", exc)

    def list_checkpoints(self) -> list[str]:
        """
        List all execution_ids with checkpoints in Redis.

        Uses SCAN with the prefix pattern to find all matching keys,
        then strips the prefix to return bare execution_ids.
        """
        pattern = f"{self._prefix}*"
        prefix_len = len(self._prefix)
        execution_ids: list[str] = []

        try:
            with self._lock:
                cursor = 0
                while True:
                    cursor, keys = self._redis.scan(cursor=cursor, match=pattern, count=100)
                    for key in keys:
                        k = key.decode() if isinstance(key, bytes) else key
                        execution_ids.append(k[prefix_len:])
                    if cursor == 0:
                        break
        except Exception as exc:
            logger.error("Failed to list checkpoints from Redis: %s", exc)

        return execution_ids
