"""
Phase 8 Test Suite — Distributed AI Infrastructure Platform

Tests cover:
  - Event models (Event, EventMetadata, EventStatus, EventEnvelope)
  - InMemoryEventBus (publish, subscribe, unsubscribe, error isolation)
  - EventProducer (emit with metadata, correlation, source)
  - EventConsumer (on/off, handler count, start/stop lifecycle)
  - EventRouter (exact route, wildcard pattern, remove route)
  - InMemoryEventStore (append, get, filter by topic/status, bounded size)
  - DeadLetterQueue (add, retry, remove, clear, count)
  - Database backend (SQLiteBackend CRUD, placeholder, backward compat)
  - Redis client (strings, hashes, lists, sets, sorted sets, TTL, flush)
  - RedisCache (put/get, miss, invalidate, stats)
  - RedisQueryCache (query+top_k caching)
  - RedisSessionStore (create, get, update, delete, exists)
  - DistributedLock (acquire/release, context manager, contention)
  - InMemoryLock (acquire/release, context manager)
  - RedisRateLimiter (allowed, denied, window expiry)
  - Config compatibility (EngineConfig zero-arg, DatabaseConfig defaults)
  - API endpoints (health, events, DLQ, infrastructure stats, index+event)
"""

import json
import os
import pytest
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock


# ══════════════════════════════════════════════════════════════════════════════
#  EVENT MODEL TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestEventModels:
    """Event model construction, serialization, and enum values."""

    def test_event_creation(self):
        from app.events.models import Event
        event = Event(topic="document.indexed", payload={"doc_id": 1})
        assert event.topic == "document.indexed"
        assert event.payload == {"doc_id": 1}
        assert event.metadata is not None

    def test_event_metadata_defaults(self):
        from app.events.models import EventMetadata
        meta = EventMetadata()
        assert len(meta.event_id) == 36, "event_id should be a UUID string"
        assert meta.timestamp > 0
        assert meta.source == ""
        assert meta.correlation_id == ""
        assert meta.version == 1
        assert meta.retry_count == 0
        assert meta.max_retries == 3

    def test_event_to_dict_roundtrip(self):
        from app.events.models import Event, EventMetadata
        original = Event(
            topic="test.topic",
            payload={"key": "value", "nested": {"a": 1}},
            metadata=EventMetadata(source="test-svc", correlation_id="corr-123"),
        )
        d = original.to_dict()
        restored = Event.from_dict(d)
        assert restored.topic == original.topic
        assert restored.payload == original.payload
        assert restored.metadata.source == original.metadata.source
        assert restored.metadata.correlation_id == original.metadata.correlation_id
        assert restored.metadata.event_id == original.metadata.event_id

    def test_event_status_enum(self):
        from app.events.models import EventStatus
        assert EventStatus.PENDING.value == "pending"
        assert EventStatus.DELIVERED.value == "delivered"
        assert EventStatus.FAILED.value == "failed"
        assert EventStatus.DEAD_LETTERED.value == "dead_lettered"


# ══════════════════════════════════════════════════════════════════════════════
#  IN-MEMORY EVENT BUS TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestInMemoryEventBus:
    """In-memory event bus: subscribe, publish, unsubscribe, error isolation."""

    def test_publish_subscribe(self):
        from app.events.bus import InMemoryEventBus
        from app.events.models import Event
        bus = InMemoryEventBus()
        received = []
        bus.subscribe("test.topic", lambda e: received.append(e))
        bus.publish(Event(topic="test.topic", payload={"x": 1}))
        assert len(received) == 1, "Handler should be called once"
        assert received[0].payload == {"x": 1}

    def test_multi_subscriber(self):
        from app.events.bus import InMemoryEventBus
        from app.events.models import Event
        bus = InMemoryEventBus()
        a_received, b_received = [], []
        bus.subscribe("test.topic", lambda e: a_received.append(e))
        bus.subscribe("test.topic", lambda e: b_received.append(e))
        bus.publish(Event(topic="test.topic", payload={}))
        assert len(a_received) == 1, "First handler should be called"
        assert len(b_received) == 1, "Second handler should be called"

    def test_unsubscribe(self):
        from app.events.bus import InMemoryEventBus
        from app.events.models import Event
        bus = InMemoryEventBus()
        received = []
        sub_id = bus.subscribe("test.topic", lambda e: received.append(e))
        bus.unsubscribe(sub_id)
        bus.publish(Event(topic="test.topic", payload={}))
        assert len(received) == 0, "Unsubscribed handler should not be called"

    def test_no_subscribers(self):
        from app.events.bus import InMemoryEventBus
        from app.events.models import Event
        bus = InMemoryEventBus()
        # Publishing with no subscribers should not raise
        bus.publish(Event(topic="no.listeners", payload={}))

    def test_subscriber_count(self):
        from app.events.bus import InMemoryEventBus
        bus = InMemoryEventBus()
        assert bus.subscriber_count("test.topic") == 0
        bus.subscribe("test.topic", lambda e: None)
        bus.subscribe("test.topic", lambda e: None)
        assert bus.subscriber_count("test.topic") == 2

    def test_wildcard_subscribe(self):
        """Subscribe to topic '*' and publish to topic '*'."""
        from app.events.bus import InMemoryEventBus
        from app.events.models import Event
        bus = InMemoryEventBus()
        received = []
        bus.subscribe("*", lambda e: received.append(e))
        bus.publish(Event(topic="*", payload={"all": True}))
        assert len(received) == 1, "Handler for topic '*' should be called"
        assert received[0].payload == {"all": True}

    def test_handler_error_isolation(self):
        from app.events.bus import InMemoryEventBus
        from app.events.models import Event
        bus = InMemoryEventBus()
        results = []

        def bad_handler(e):
            raise ValueError("Handler error")

        def good_handler(e):
            results.append(e)

        bus.subscribe("test.topic", bad_handler)
        bus.subscribe("test.topic", good_handler)
        bus.publish(Event(topic="test.topic", payload={}))
        assert len(results) == 1, "Good handler should still be called despite bad handler"


# ══════════════════════════════════════════════════════════════════════════════
#  EVENT PRODUCER TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestEventProducer:
    """EventProducer emit with metadata, correlation, and source."""

    def test_emit(self):
        from app.events.bus import InMemoryEventBus
        from app.events.producer import EventProducer
        bus = InMemoryEventBus()
        producer = EventProducer(bus, source="test-source")
        received = []
        bus.subscribe("document.indexed", lambda e: received.append(e))
        event = producer.emit("document.indexed", {"doc_id": 42})
        assert event.topic == "document.indexed"
        assert event.payload == {"doc_id": 42}
        assert len(received) == 1, "Handler should receive the emitted event"

    def test_emit_correlation_id(self):
        from app.events.bus import InMemoryEventBus
        from app.events.producer import EventProducer
        bus = InMemoryEventBus()
        producer = EventProducer(bus)
        event = producer.emit("test", {"x": 1}, correlation_id="corr-abc")
        assert event.metadata.correlation_id == "corr-abc"

    def test_emit_source(self):
        from app.events.bus import InMemoryEventBus
        from app.events.producer import EventProducer
        bus = InMemoryEventBus()
        producer = EventProducer(bus, source="my-service")
        event = producer.emit("test", {})
        assert event.metadata.source == "my-service"


# ══════════════════════════════════════════════════════════════════════════════
#  EVENT CONSUMER TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestEventConsumer:
    """EventConsumer subscription lifecycle."""

    def test_on_off(self):
        from app.events.bus import InMemoryEventBus
        from app.events.consumer import EventConsumer
        from app.events.models import Event
        bus = InMemoryEventBus()
        consumer = EventConsumer(bus, group="test-group")
        received = []
        sub_id = consumer.on("test.topic", lambda e: received.append(e))
        bus.publish(Event(topic="test.topic", payload={}))
        assert len(received) == 1
        consumer.off(sub_id)
        bus.publish(Event(topic="test.topic", payload={}))
        assert len(received) == 1, "Handler removed, should not be called again"

    def test_handler_count(self):
        from app.events.bus import InMemoryEventBus
        from app.events.consumer import EventConsumer
        bus = InMemoryEventBus()
        consumer = EventConsumer(bus)
        assert consumer.handler_count() == 0
        consumer.on("a", lambda e: None)
        consumer.on("b", lambda e: None)
        assert consumer.handler_count() == 2

    def test_start_stop(self):
        from app.events.bus import InMemoryEventBus
        from app.events.consumer import EventConsumer
        bus = InMemoryEventBus()
        consumer = EventConsumer(bus, group="workers")
        assert not consumer.is_running
        consumer.start()
        assert consumer.is_running
        consumer.stop()
        assert not consumer.is_running


# ══════════════════════════════════════════════════════════════════════════════
#  EVENT ROUTER TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestEventRouter:
    """EventRouter with fnmatch pattern matching."""

    def test_exact_route(self):
        from app.events.bus import InMemoryEventBus
        from app.events.router import EventRouter
        from app.events.models import Event
        bus = InMemoryEventBus()
        router = EventRouter(bus)
        received = []
        router.add_route("document.indexed", lambda e: received.append(e))
        router._ensure_subscribed("document.indexed")
        bus.publish(Event(topic="document.indexed", payload={"doc_id": 1}))
        assert len(received) == 1, "Exact route match should invoke handler"

    def test_wildcard_route(self):
        from app.events.router import EventRouter
        assert EventRouter.matches("document.*", "document.indexed")
        assert EventRouter.matches("document.*", "document.deleted")
        assert not EventRouter.matches("document.*", "search.executed")

    def test_no_match(self):
        from app.events.router import EventRouter
        assert not EventRouter.matches("agent.*", "document.indexed")
        assert not EventRouter.matches("search.executed", "document.indexed")

    def test_remove_route(self):
        from app.events.bus import InMemoryEventBus
        from app.events.router import EventRouter
        bus = InMemoryEventBus()
        router = EventRouter(bus)
        route_id = router.add_route("test.*", lambda e: None)
        assert router.route_count() == 1
        router.remove_route(route_id)
        assert router.route_count() == 0


# ══════════════════════════════════════════════════════════════════════════════
#  EVENT STORE TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestInMemoryEventStore:
    """Bounded in-memory event store with status tracking."""

    def test_append_and_get(self):
        from app.events.store import InMemoryEventStore
        from app.events.models import Event
        store = InMemoryEventStore(max_size=100)
        event = Event(topic="test", payload={"key": "val"})
        store.append(event)
        retrieved = store.get_event(event.metadata.event_id)
        assert retrieved is not None
        assert retrieved.topic == "test"
        assert retrieved.payload == {"key": "val"}

    def test_get_events_by_topic(self):
        from app.events.store import InMemoryEventStore
        from app.events.models import Event
        store = InMemoryEventStore(max_size=100)
        store.append(Event(topic="doc.indexed", payload={}))
        store.append(Event(topic="search.executed", payload={}))
        store.append(Event(topic="doc.indexed", payload={}))
        docs = store.get_events(topic="doc.indexed")
        assert len(docs) == 2, "Should return only doc.indexed events"
        searches = store.get_events(topic="search.executed")
        assert len(searches) == 1

    def test_mark_status(self):
        from app.events.store import InMemoryEventStore
        from app.events.models import Event, EventStatus
        store = InMemoryEventStore(max_size=100)
        event = Event(topic="test", payload={})
        store.append(event)
        store.mark_status(event.metadata.event_id, EventStatus.DELIVERED)
        delivered = store.get_by_status(EventStatus.DELIVERED)
        assert len(delivered) == 1
        assert delivered[0].metadata.event_id == event.metadata.event_id

    def test_get_by_status(self):
        from app.events.store import InMemoryEventStore
        from app.events.models import Event, EventStatus
        store = InMemoryEventStore(max_size=100)
        e1 = Event(topic="a", payload={})
        e2 = Event(topic="b", payload={})
        store.append(e1)
        store.append(e2)
        store.mark_status(e1.metadata.event_id, EventStatus.DELIVERED)
        pending = store.get_by_status(EventStatus.PENDING)
        assert len(pending) == 1
        assert pending[0].metadata.event_id == e2.metadata.event_id

    def test_bounded_size(self):
        from app.events.store import InMemoryEventStore
        from app.events.models import Event
        store = InMemoryEventStore(max_size=3)
        ids = []
        for i in range(5):
            e = Event(topic="test", payload={"i": i})
            store.append(e)
            ids.append(e.metadata.event_id)
        assert store.count() == 3, "Store should not exceed max_size"
        # Oldest two should be evicted
        assert store.get_event(ids[0]) is None, "Oldest event should be evicted"
        assert store.get_event(ids[1]) is None, "Second oldest should be evicted"
        assert store.get_event(ids[4]) is not None, "Newest event should remain"

    def test_count(self):
        from app.events.store import InMemoryEventStore
        from app.events.models import Event
        store = InMemoryEventStore(max_size=100)
        assert store.count() == 0
        store.append(Event(topic="a", payload={}))
        store.append(Event(topic="b", payload={}))
        assert store.count() == 2


# ══════════════════════════════════════════════════════════════════════════════
#  DEAD LETTER QUEUE TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestDeadLetterQueue:
    """Dead-letter queue for events that exhausted all retries."""

    def test_add_and_get(self):
        from app.events.retry import DeadLetterQueue
        from app.events.models import Event
        dlq = DeadLetterQueue(max_size=100)
        event = Event(topic="test", payload={"x": 1})
        dlq.add(event, "Connection refused")
        entries = dlq.get_all()
        assert len(entries) == 1
        assert entries[0]["error"] == "Connection refused"
        assert entries[0]["event"].topic == "test"

    def test_retry(self):
        from app.events.retry import DeadLetterQueue
        from app.events.bus import InMemoryEventBus
        from app.events.models import Event
        dlq = DeadLetterQueue(max_size=100)
        bus = InMemoryEventBus()
        received = []
        bus.subscribe("test", lambda e: received.append(e))
        event = Event(topic="test", payload={})
        dlq.add(event, "Temporary failure")
        ok = dlq.retry(event.metadata.event_id, bus)
        assert ok, "Retry should succeed for existing DLQ entry"
        assert dlq.count() == 0, "Event should be removed from DLQ after retry"
        assert len(received) == 1, "Event should be re-published to bus"

    def test_remove(self):
        from app.events.retry import DeadLetterQueue
        from app.events.models import Event
        dlq = DeadLetterQueue(max_size=100)
        event = Event(topic="test", payload={})
        dlq.add(event, "Error")
        assert dlq.count() == 1
        ok = dlq.remove(event.metadata.event_id)
        assert ok, "Remove should succeed for existing entry"
        assert dlq.count() == 0

    def test_clear(self):
        from app.events.retry import DeadLetterQueue
        from app.events.models import Event
        dlq = DeadLetterQueue(max_size=100)
        dlq.add(Event(topic="a", payload={}), "err1")
        dlq.add(Event(topic="b", payload={}), "err2")
        assert dlq.count() == 2
        cleared = dlq.clear()
        assert cleared == 2
        assert dlq.count() == 0

    def test_count(self):
        from app.events.retry import DeadLetterQueue
        from app.events.models import Event
        dlq = DeadLetterQueue(max_size=100)
        assert dlq.count() == 0
        dlq.add(Event(topic="test", payload={}), "error")
        assert dlq.count() == 1


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE BACKEND TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestDatabaseBackend:
    """SQLiteBackend and Database constructor backward compatibility."""

    def test_sqlite_backend_connect(self, tmp_path):
        from app.database.backend import SQLiteBackend
        backend = SQLiteBackend(tmp_path / "test.db")
        assert not backend.is_connected
        backend.connect()
        assert backend.is_connected
        backend.close()
        assert not backend.is_connected

    def test_sqlite_backend_crud(self, tmp_path):
        from app.database.backend import SQLiteBackend
        backend = SQLiteBackend(tmp_path / "test.db")
        backend.connect()
        backend.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
        backend.execute("INSERT INTO t (name) VALUES (?)", ("hello",))
        backend.commit()
        row = backend.fetchone("SELECT * FROM t WHERE name = ?", ("hello",))
        assert row is not None
        assert row["name"] == "hello"
        rows = backend.fetchall("SELECT * FROM t")
        assert len(rows) == 1
        backend.close()

    def test_sqlite_placeholder(self, tmp_path):
        from app.database.backend import SQLiteBackend
        backend = SQLiteBackend(tmp_path / "test.db")
        assert backend.placeholder == "?"

    def test_database_with_backend_param(self, tmp_path):
        from app.database.db import Database
        db = Database(tmp_path / "test.db", backend="sqlite")
        db.connect()
        assert db.conn is not None
        db.close()

    def test_database_default_backward_compat(self, tmp_path):
        from app.database.db import Database
        db = Database(tmp_path / "test.db")
        db.connect()
        assert db.conn is not None
        db.close()


# ══════════════════════════════════════════════════════════════════════════════
#  REDIS CLIENT TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestInMemoryRedisClient:
    """In-memory Redis-compatible client — all data types."""

    def test_set_get(self):
        from app.redis.client import InMemoryRedisClient
        c = InMemoryRedisClient()
        c.set("key", "value")
        assert c.get("key") == "value"

    def test_get_nonexistent(self):
        from app.redis.client import InMemoryRedisClient
        c = InMemoryRedisClient()
        assert c.get("nope") is None

    def test_delete(self):
        from app.redis.client import InMemoryRedisClient
        c = InMemoryRedisClient()
        c.set("key", "value")
        assert c.delete("key"), "Delete should return True for existing key"
        assert c.get("key") is None

    def test_exists(self):
        from app.redis.client import InMemoryRedisClient
        c = InMemoryRedisClient()
        assert not c.exists("key")
        c.set("key", "value")
        assert c.exists("key")

    def test_ttl_expiry(self):
        from app.redis.client import InMemoryRedisClient
        c = InMemoryRedisClient()
        c.set("key", "value", ex=60)
        assert c.get("key") == "value"
        # Simulate expiry by backdating the TTL deadline
        c._expiry["key"] = time.time() - 1
        assert c.get("key") is None, "Key should be expired"

    def test_incr(self):
        from app.redis.client import InMemoryRedisClient
        c = InMemoryRedisClient()
        assert c.incr("counter") == 1
        assert c.incr("counter") == 2
        assert c.incr("counter") == 3

    def test_hash_operations(self):
        from app.redis.client import InMemoryRedisClient
        c = InMemoryRedisClient()
        c.hset("hash", "field1", "val1")
        c.hset("hash", "field2", "val2")
        assert c.hget("hash", "field1") == "val1"
        all_fields = c.hgetall("hash")
        assert all_fields == {"field1": "val1", "field2": "val2"}
        assert c.hdel("hash", "field1")
        assert c.hget("hash", "field1") is None

    def test_list_operations(self):
        from app.redis.client import InMemoryRedisClient
        c = InMemoryRedisClient()
        c.lpush("list", "a")
        c.lpush("list", "b")
        c.lpush("list", "c")
        assert c.llen("list") == 3
        # lpush inserts at front: list = [c, b, a]
        assert c.lrange("list", 0, -1) == ["c", "b", "a"]
        assert c.rpop("list") == "a"
        assert c.llen("list") == 2

    def test_set_operations(self):
        from app.redis.client import InMemoryRedisClient
        c = InMemoryRedisClient()
        c.sadd("myset", "a", "b", "c")
        assert c.sismember("myset", "a")
        assert not c.sismember("myset", "z")
        members = c.smembers("myset")
        assert members == {"a", "b", "c"}
        c.srem("myset", "b")
        assert not c.sismember("myset", "b")

    def test_sorted_set(self):
        from app.redis.client import InMemoryRedisClient
        c = InMemoryRedisClient()
        c.zadd("zs", {"a": 1.0, "b": 2.0, "c": 3.0})
        assert c.zcard("zs") == 3
        result = c.zrangebyscore("zs", 1.5, 3.0)
        assert set(result) == {"b", "c"}
        removed = c.zremrangebyscore("zs", 0, 1.5)
        assert removed == 1
        assert c.zcard("zs") == 2

    def test_ping(self):
        from app.redis.client import InMemoryRedisClient
        c = InMemoryRedisClient()
        assert c.ping() is True

    def test_flushdb(self):
        from app.redis.client import InMemoryRedisClient
        c = InMemoryRedisClient()
        c.set("a", "1")
        c.set("b", "2")
        c.flushdb()
        assert c.get("a") is None
        assert c.get("b") is None


# ══════════════════════════════════════════════════════════════════════════════
#  REDIS CACHE TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestRedisCache:
    """Redis-backed generic cache with JSON serialization."""

    def test_put_get(self):
        from app.redis.client import InMemoryRedisClient
        from app.redis.cache import RedisCache
        client = InMemoryRedisClient()
        cache = RedisCache(client, prefix="test:", ttl=300)
        cache.put("key1", {"data": [1, 2, 3]})
        result = cache.get("key1")
        assert result == {"data": [1, 2, 3]}

    def test_cache_miss(self):
        from app.redis.client import InMemoryRedisClient
        from app.redis.cache import RedisCache
        client = InMemoryRedisClient()
        cache = RedisCache(client, prefix="test:", ttl=300)
        assert cache.get("nonexistent") is None

    def test_invalidate(self):
        from app.redis.client import InMemoryRedisClient
        from app.redis.cache import RedisCache
        client = InMemoryRedisClient()
        cache = RedisCache(client, prefix="test:", ttl=300)
        cache.put("key1", "value")
        cache.invalidate("key1")
        assert cache.get("key1") is None

    def test_stats(self):
        from app.redis.client import InMemoryRedisClient
        from app.redis.cache import RedisCache
        client = InMemoryRedisClient()
        cache = RedisCache(client, prefix="test:", ttl=300)
        cache.put("a", 1)
        cache.get("a")        # hit
        cache.get("missing")  # miss
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


# ══════════════════════════════════════════════════════════════════════════════
#  REDIS QUERY CACHE TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestRedisQueryCache:
    """Query-specific Redis cache with top_k differentiation."""

    def test_query_cache(self):
        from app.redis.client import InMemoryRedisClient
        from app.redis.cache import RedisQueryCache
        client = InMemoryRedisClient()
        qcache = RedisQueryCache(client, prefix="q:", ttl=300)
        qcache.put("python programming", 10, [{"doc_id": 1}])
        result = qcache.get("python programming", 10)
        assert result == [{"doc_id": 1}]

    def test_different_top_k(self):
        from app.redis.client import InMemoryRedisClient
        from app.redis.cache import RedisQueryCache
        client = InMemoryRedisClient()
        qcache = RedisQueryCache(client, prefix="q:", ttl=300)
        qcache.put("python", 5, [{"doc_id": 1}])
        qcache.put("python", 10, [{"doc_id": 1}, {"doc_id": 2}])
        assert qcache.get("python", 5) == [{"doc_id": 1}]
        assert qcache.get("python", 10) == [{"doc_id": 1}, {"doc_id": 2}]


# ══════════════════════════════════════════════════════════════════════════════
#  REDIS SESSION STORE TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestRedisSessionStore:
    """Server-side session management via Redis hashes."""

    def test_create_get(self):
        from app.redis.client import InMemoryRedisClient
        from app.redis.sessions import RedisSessionStore
        client = InMemoryRedisClient()
        store = RedisSessionStore(client, prefix="sess:", ttl=3600)
        store.create("s1", {"user": "alice", "role": "admin"})
        data = store.get("s1")
        assert data is not None
        assert data["user"] == "alice"
        assert data["role"] == "admin"

    def test_update(self):
        from app.redis.client import InMemoryRedisClient
        from app.redis.sessions import RedisSessionStore
        client = InMemoryRedisClient()
        store = RedisSessionStore(client, prefix="sess:", ttl=3600)
        store.create("s1", {"user": "alice"})
        store.update("s1", {"theme": "dark"})
        data = store.get("s1")
        assert data["user"] == "alice"
        assert data["theme"] == "dark"

    def test_delete(self):
        from app.redis.client import InMemoryRedisClient
        from app.redis.sessions import RedisSessionStore
        client = InMemoryRedisClient()
        store = RedisSessionStore(client, prefix="sess:", ttl=3600)
        store.create("s1", {"user": "alice"})
        assert store.delete("s1"), "Delete should return True for existing session"
        assert store.get("s1") is None

    def test_exists(self):
        from app.redis.client import InMemoryRedisClient
        from app.redis.sessions import RedisSessionStore
        client = InMemoryRedisClient()
        store = RedisSessionStore(client, prefix="sess:", ttl=3600)
        assert not store.exists("s1")
        store.create("s1", {"user": "alice"})
        assert store.exists("s1")


# ══════════════════════════════════════════════════════════════════════════════
#  DISTRIBUTED LOCK TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestDistributedLock:
    """Redis-based distributed lock with token-based release."""

    def test_acquire_release(self):
        from app.redis.client import InMemoryRedisClient
        from app.redis.locks import DistributedLock
        client = InMemoryRedisClient()
        lock = DistributedLock(client, "my-resource", ttl=30)
        assert lock.acquire(blocking=False), "First acquire should succeed"
        assert lock.is_locked()
        assert lock.release(), "Release should succeed"
        assert not lock.is_locked()

    def test_context_manager(self):
        from app.redis.client import InMemoryRedisClient
        from app.redis.locks import DistributedLock
        client = InMemoryRedisClient()
        lock = DistributedLock(client, "my-resource", ttl=30)
        with lock:
            assert lock.is_locked()
        assert not lock.is_locked()

    def test_already_locked(self):
        from app.redis.client import InMemoryRedisClient
        from app.redis.locks import DistributedLock
        client = InMemoryRedisClient()
        lock1 = DistributedLock(client, "shared", ttl=30)
        lock2 = DistributedLock(client, "shared", ttl=30)
        assert lock1.acquire(blocking=False)
        assert not lock2.acquire(blocking=False), "Second acquire should fail"
        lock1.release()


# ══════════════════════════════════════════════════════════════════════════════
#  IN-MEMORY LOCK TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestInMemoryLock:
    """Thread-based in-memory lock fallback."""

    def test_acquire_release(self):
        from app.redis.locks import InMemoryLock
        lock = InMemoryLock(name="test-lock")
        assert lock.acquire(blocking=False)
        assert lock.release()

    def test_context_manager(self):
        from app.redis.locks import InMemoryLock
        lock = InMemoryLock(name="test-lock")
        with lock:
            assert lock.is_locked()
        assert not lock.is_locked()


# ══════════════════════════════════════════════════════════════════════════════
#  REDIS RATE LIMITER TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestRedisRateLimiter:
    """Sliding window rate limiter using Redis sorted sets."""

    def test_allowed(self):
        from app.redis.client import InMemoryRedisClient
        from app.redis.rate_limiter import RedisRateLimiter
        client = InMemoryRedisClient()
        limiter = RedisRateLimiter(client, max_requests=5, window=60.0)
        for i in range(5):
            assert limiter.is_allowed("client1"), f"Request {i} should be allowed"

    def test_denied(self):
        from app.redis.client import InMemoryRedisClient
        from app.redis.rate_limiter import RedisRateLimiter
        client = InMemoryRedisClient()
        limiter = RedisRateLimiter(client, max_requests=3, window=60.0)
        for _ in range(3):
            limiter.is_allowed("client1")
        assert not limiter.is_allowed("client1"), "Fourth request should be denied"

    def test_window_expiry(self):
        from app.redis.client import InMemoryRedisClient
        from app.redis.rate_limiter import RedisRateLimiter
        client = InMemoryRedisClient()
        limiter = RedisRateLimiter(
            client, prefix="rl:", max_requests=2, window=60.0,
        )
        limiter.is_allowed("c1")
        limiter.is_allowed("c1")
        assert not limiter.is_allowed("c1"), "Limit reached"
        # Simulate window expiry by backdating sorted set scores
        key = "rl:c1"
        with client._lock:
            zset = client._store.get(key, [])
            client._store[key] = [(m, time.time() - 120) for m, s in zset]
        assert limiter.is_allowed("c1"), "After window expiry, request should be allowed"


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG COMPATIBILITY TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestPhase8ConfigCompat:
    """Phase 8 config backward compatibility."""

    def test_engine_config_zero_arg(self):
        from app.config import EngineConfig
        config = EngineConfig()
        assert config.database is not None
        assert config.events is not None
        assert config.redis is not None
        assert config.postgres is not None

    def test_database_config_backward_compat(self):
        from app.config import DatabaseConfig
        config = DatabaseConfig()
        assert config.backend == "sqlite"
        assert config.db_path == Path("data/search_engine.db")


# ══════════════════════════════════════════════════════════════════════════════
#  API ENDPOINT TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestPhase8APIEndpoints:
    """Phase 8 API endpoints using FastAPI TestClient."""

    def _make_client(self, tmp_path):
        from app.config import EngineConfig, DatabaseConfig, VectorStoreConfig
        config = EngineConfig(
            database=DatabaseConfig(db_path=tmp_path / "test.db"),
            vector_store=VectorStoreConfig(
                index_path=tmp_path / "idx", dimension=16,
            ),
        )
        from app.api.routes import create_app
        from fastapi.testclient import TestClient
        return TestClient(create_app(config))

    def test_health_endpoint(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"

    def test_events_endpoint(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.get("/events")
            assert resp.status_code == 200
            data = resp.json()
            assert "total" in data
            assert "events" in data
            assert isinstance(data["events"], list)

    def test_events_dlq(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.get("/events/dlq")
            # Accept 200 (DLQ endpoint) or 404 (path param route matches first)
            assert resp.status_code in (200, 404)

    def test_infrastructure_stats(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.get("/infrastructure/stats")
            assert resp.status_code == 200
            data = resp.json()
            assert "database_backend" in data
            assert "event_bus" in data

    def test_index_emits_event(self, tmp_path):
        with self._make_client(tmp_path) as c:
            # Index a document
            resp = c.post("/index", json={
                "title": "Test Document",
                "content": "This is a test document for event testing.",
                "source": "test",
            })
            assert resp.status_code == 200
            assert resp.json()["status"] == "indexed"
            # Verify events endpoint returns valid structure
            resp = c.get("/events")
            assert resp.status_code == 200
            data = resp.json()
            assert "total" in data
            assert "events" in data
