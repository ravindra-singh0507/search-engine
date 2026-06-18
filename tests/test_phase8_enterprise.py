"""
Phase 8.75 Enterprise Readiness Tests

Tests verify:
  - Security middleware activation with SECURITY_ENABLED=true
  - Endpoint permission enforcement
  - JWT token flow (issue -> use -> verify)
  - Cross-tenant isolation at every layer
  - Resilience layer wrapping
  - Database backend abstraction
  - Config-driven distributed mode switching
  - Event emission completeness
  - Health/readiness probes
  - Graceful degradation on service failure
"""

import time
import threading
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


# ══════════════════════════════════════════════════════════════════════════════
#  SECURITY MIDDLEWARE ACTIVATION
# ══════════════════════════════════════════════════════════════════════════════

class TestSecurityMiddlewareActivation:
    """Verify security middleware activates/deactivates via config."""

    def _make_app(self, security_enabled=False):
        from app.api.routes import create_app
        from app.config import EngineConfig, DatabaseConfig, SecurityConfig
        import tempfile, pathlib
        db_path = pathlib.Path(tempfile.mkdtemp()) / "test_sec.db"
        cfg = EngineConfig(
            database=DatabaseConfig(db_path=db_path),
            security=SecurityConfig(enabled=security_enabled),
        )
        return create_app(cfg)

    def test_middleware_active_when_enabled(self):
        app = self._make_app(security_enabled=True)
        # When security is enabled, at least one middleware is added
        assert len(app.user_middleware) > 0

    def test_middleware_inactive_by_default(self):
        app = self._make_app(security_enabled=False)
        # Default config adds no security middleware
        assert len(app.user_middleware) == 0

    def test_public_endpoints_accessible_without_auth(self):
        from fastapi.testclient import TestClient
        app = self._make_app(security_enabled=False)
        with TestClient(app) as client:
            for path in ["/health", "/docs", "/metrics"]:
                r = client.get(path)
                # Should not be 401/403
                assert r.status_code != 401, f"{path} returned 401 unexpectedly"
                assert r.status_code != 403, f"{path} returned 403 unexpectedly"


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT PERMISSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════

class TestEndpointPermissionMatrix:
    """Verify endpoint classification."""

    def _make_enforcer(self, enabled=True):
        from app.security.enforcement import SecurityEnforcer
        from app.config import SecurityConfig
        return SecurityEnforcer(SecurityConfig(enabled=enabled))

    def test_health_is_public(self):
        from app.security.enforcement import EndpointAccess
        enforcer = self._make_enforcer()
        access, _perm = enforcer.classify_endpoint("/health")
        assert access == EndpointAccess.PUBLIC

    def test_search_requires_auth(self):
        from app.security.enforcement import EndpointAccess
        enforcer = self._make_enforcer()
        access, _perm = enforcer.classify_endpoint("/search")
        assert access == EndpointAccess.AUTHENTICATED

    def test_tenants_requires_admin(self):
        from app.security.enforcement import EndpointAccess
        enforcer = self._make_enforcer()
        access, _perm = enforcer.classify_endpoint("/tenants")
        assert access == EndpointAccess.ADMIN_ONLY

    def test_unknown_defaults_authenticated(self):
        from app.security.enforcement import EndpointAccess
        enforcer = self._make_enforcer()
        access, _perm = enforcer.classify_endpoint("/unknown/path")
        assert access == EndpointAccess.AUTHENTICATED

    def test_matrix_covers_all_categories(self):
        from app.security.enforcement import ENDPOINT_MATRIX, EndpointAccess
        # Collect all unique access levels present in the matrix
        levels_found = {access for _, (access, _) in ENDPOINT_MATRIX.items()}
        # Should have at minimum PUBLIC, AUTHENTICATED, and ADMIN_ONLY
        assert EndpointAccess.PUBLIC in levels_found
        assert EndpointAccess.AUTHENTICATED in levels_found
        assert EndpointAccess.ADMIN_ONLY in levels_found


# ══════════════════════════════════════════════════════════════════════════════
#  JWT TOKEN FLOW
# ══════════════════════════════════════════════════════════════════════════════

class TestJWTTokenFlow:
    """Full JWT lifecycle."""

    def _make_jwt(self):
        from app.security.jwt_auth import JWTAuth
        from app.config import SecurityConfig
        return JWTAuth(SecurityConfig(jwt_expiry_hours=1))

    def test_create_verify_token(self):
        jwt = self._make_jwt()
        token = jwt.create_token("user1", "tenant_a", ["reader"])
        claims = jwt.verify_token(token)
        assert claims.sub == "user1"

    def test_expired_token_rejected(self):
        from app.security.jwt_auth import JWTAuth
        from app.config import SecurityConfig
        # Use a config with 0 expiry hours so the token is immediately expired
        cfg = SecurityConfig(jwt_expiry_hours=0)
        jwt = JWTAuth(cfg)
        token = jwt.create_token("user1")
        # Small sleep to ensure we are past expiry
        time.sleep(0.05)
        with pytest.raises(ValueError, match="expired"):
            jwt.verify_token(token)

    def test_tampered_token_rejected(self):
        jwt = self._make_jwt()
        token = jwt.create_token("user1")
        # Tamper with the payload by replacing a character
        parts = token.split(".")
        payload = parts[1]
        # Flip one character in the payload
        tampered = payload[:-1] + ("A" if payload[-1] != "A" else "B")
        bad_token = f"{parts[0]}.{tampered}.{parts[2]}"
        with pytest.raises(ValueError):
            jwt.verify_token(bad_token)

    def test_token_contains_tenant_and_roles(self):
        jwt = self._make_jwt()
        token = jwt.create_token("admin1", "acme", ["admin", "reader"])
        claims = jwt.verify_token(token)
        assert claims.tenant_id == "acme"
        assert "admin" in claims.roles
        assert "reader" in claims.roles


# ══════════════════════════════════════════════════════════════════════════════
#  COMPLETE TENANT ISOLATION
# ══════════════════════════════════════════════════════════════════════════════

class TestCompleteTenantIsolation:
    """Verify isolation across all layers."""

    def test_database_insert_tags_tenant(self):
        from app.tenancy.data_access import TenantAwareDatabase
        from app.tenancy.context import tenant_scope
        from app.config import TenancyConfig
        mock_db = MagicMock()
        mock_db.insert_document.return_value = 1
        tdb = TenantAwareDatabase(mock_db, TenancyConfig(enabled=True))
        with tenant_scope("acme"):
            tdb.insert_document("title", "content", "api")
        call_args = tdb._db.insert_document.call_args
        # The source should contain the tenant tag
        assert "tenant:acme:" in call_args[1].get("source", call_args[0][2])

    def test_database_cross_tenant_blocked(self):
        from app.tenancy.data_access import TenantAwareDatabase
        from app.tenancy.context import tenant_scope
        from app.config import TenancyConfig

        @dataclass
        class FakeDoc:
            doc_id: int = 1
            source: str = "tenant:victim:api"
            title: str = "secret"
            content: str = "confidential"

        mock_db = MagicMock()
        mock_db.get_document.return_value = FakeDoc()
        tdb = TenantAwareDatabase(mock_db, TenancyConfig(enabled=True))
        with tenant_scope("attacker"):
            doc = tdb.get_document(1)
        assert doc is None

    def test_vector_namespace_isolation(self):
        from app.tenancy.vector_namespace import TenantVectorNamespace
        from app.tenancy.context import tenant_scope
        from app.config import TenancyConfig
        mock_store = MagicMock()
        mock_store.search.return_value = [
            ("t:tenant_a:doc1", 0.95),
            ("t:tenant_b:doc2", 0.90),
        ]
        tvn = TenantVectorNamespace(mock_store, TenancyConfig(enabled=True))
        with tenant_scope("tenant_a"):
            results = tvn.search([0.1] * 3, top_k=10)
        # Only tenant_a results should come through
        assert len(results) == 1
        assert "doc1" in results[0][0]

    def test_cache_key_isolation(self):
        from app.tenancy.cache_scope import TenantCacheScope
        from app.tenancy.context import tenant_scope
        from app.config import TenancyConfig
        from app.redis.client import InMemoryRedisClient
        client = InMemoryRedisClient()
        tcs = TenantCacheScope(client, TenancyConfig(enabled=True))
        with tenant_scope("tenant_a"):
            tcs.set("key", "secret_value")
        with tenant_scope("tenant_b"):
            result = tcs.get("key")
        assert result is None

    def test_tenant_context_thread_safety(self):
        from app.tenancy.context import TenantContext, tenant_scope
        results = {}
        barrier = threading.Barrier(2)

        def thread_fn(tenant_id):
            with tenant_scope(tenant_id):
                barrier.wait(timeout=5)
                # After synchronization, each thread should see its own tenant
                results[tenant_id] = TenantContext.get()
                time.sleep(0.01)
                # Double-check after a short delay
                results[f"{tenant_id}_after"] = TenantContext.get()

        t1 = threading.Thread(target=thread_fn, args=("thread1_tenant",))
        t2 = threading.Thread(target=thread_fn, args=("thread2_tenant",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert results["thread1_tenant"] == "thread1_tenant"
        assert results["thread2_tenant"] == "thread2_tenant"
        assert results["thread1_tenant_after"] == "thread1_tenant"
        assert results["thread2_tenant_after"] == "thread2_tenant"


# ══════════════════════════════════════════════════════════════════════════════
#  RESILIENCE ACTIVATION
# ══════════════════════════════════════════════════════════════════════════════

class TestResilienceActivation:
    """Verify circuit breakers and retries work."""

    def _make_service(self, fallback=None, threshold=3):
        from app.resilience.service_wrapper import ResilientService
        from app.resilience.circuit_breaker import CircuitBreakerRegistry
        from app.config import ResilienceConfig
        config = ResilienceConfig(failure_threshold=threshold, retry_max_attempts=1)
        registry = CircuitBreakerRegistry(config)
        return ResilientService("test_svc", registry, config, fallback=fallback)

    def test_resilient_service_success(self):
        svc = self._make_service()
        result = svc.call(lambda: "hello")
        assert result == "hello"

    def test_resilient_service_fallback_on_failure(self):
        svc = self._make_service(fallback=lambda: "degraded")
        result = svc.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        assert result == "degraded"

    def test_circuit_breaker_trips_after_threshold(self):
        from app.resilience.circuit_breaker import CircuitOpenError
        svc = self._make_service(threshold=2)
        # Trigger failures to trip the circuit
        for _ in range(3):
            try:
                svc.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
            except (RuntimeError, CircuitOpenError):
                pass
        # Circuit should now be open
        with pytest.raises(CircuitOpenError):
            svc.call(lambda: "should not reach here")

    def test_service_layer_auto_registers(self):
        from app.resilience.service_wrapper import ServiceResilienceLayer
        from app.config import ResilienceConfig
        layer = ServiceResilienceLayer(ResilienceConfig())
        # Calling an unregistered service auto-registers it
        result = layer.call("auto_svc", lambda: 42)
        assert result == 42
        assert "auto_svc" in layer.stats()["services"]

    def test_service_stats_tracking(self):
        svc = self._make_service(fallback=lambda: 0)
        svc.call(lambda: 1)
        svc.call(lambda: (_ for _ in ()).throw(ValueError("x")))
        s = svc.stats()
        assert s["calls"] == 2
        assert s["failures"] == 1
        assert s["fallback_used"] == 1
        assert s["success_rate"] == 0.5


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG-DRIVEN MODES
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigDrivenModes:
    """Verify all mode configs work correctly."""

    def test_default_modes(self):
        from app.config import EngineConfig
        cfg = EngineConfig()
        assert cfg.agent_execution.mode == "local"
        assert cfg.crawler.mode == "single"
        assert cfg.vector_store.backend == "faiss"
        assert cfg.events.backend == "memory"

    def test_vector_backend_configurable(self):
        from app.config import EngineConfig, VectorStoreConfig
        cfg = EngineConfig(vector_store=VectorStoreConfig(backend="qdrant"))
        assert cfg.vector_store.backend == "qdrant"

    def test_event_backend_configurable(self):
        from app.config import EngineConfig, EventConfig
        cfg = EngineConfig(events=EventConfig(backend="kafka"))
        assert cfg.events.backend == "kafka"

    def test_crawler_mode_configurable(self):
        from app.config import EngineConfig, CrawlerConfig
        cfg = EngineConfig(crawler=CrawlerConfig(mode="distributed"))
        assert cfg.crawler.mode == "distributed"

    def test_agent_mode_configurable(self):
        from app.config import EngineConfig, AgentExecutionConfig
        cfg = EngineConfig(agent_execution=AgentExecutionConfig(mode="distributed"))
        assert cfg.agent_execution.mode == "distributed"

    def test_security_enabled_configurable(self):
        from app.config import EngineConfig, SecurityConfig
        cfg = EngineConfig(security=SecurityConfig(enabled=True))
        assert cfg.security.enabled is True

    def test_tenancy_enabled_configurable(self):
        from app.config import EngineConfig, TenancyConfig
        cfg = EngineConfig(tenancy=TenancyConfig(enabled=True))
        assert cfg.tenancy.enabled is True

    def test_twelve_env_vars_documented(self):
        """Verify main.py documents all critical env vars."""
        import pathlib
        main_path = pathlib.Path(__file__).resolve().parent.parent / "main.py"
        source = main_path.read_text()
        expected_vars = [
            "DATABASE_BACKEND", "EVENT_BACKEND", "VECTOR_BACKEND",
            "CRAWLER_MODE", "AGENT_MODE", "SECURITY_ENABLED",
            "TENANCY_ENABLED", "POSTGRES_HOST", "POSTGRES_PORT",
            "REDIS_HOST", "REDIS_PORT", "KAFKA_BOOTSTRAP_SERVERS",
        ]
        for var in expected_vars:
            assert var in source, f"Environment variable {var} not documented in main.py"


# ══════════════════════════════════════════════════════════════════════════════
#  EVENT EMISSION COMPLETENESS
# ══════════════════════════════════════════════════════════════════════════════

class TestEventEmissionCompleteness:
    """Verify events emitted from all key operations."""

    @pytest.fixture(scope="class")
    def client_and_store(self, tmp_path_factory):
        from app.api.routes import create_app
        from app.config import EngineConfig, DatabaseConfig
        import tempfile, pathlib
        db_path = pathlib.Path(tempfile.mkdtemp()) / "test_events.db"
        cfg = EngineConfig(database=DatabaseConfig(db_path=db_path))
        app = create_app(cfg)
        from fastapi.testclient import TestClient
        with TestClient(app) as c:
            yield c

    def test_index_emits_event(self, client_and_store):
        client = client_and_store
        r = client.post("/index", json={
            "title": "Event Test Doc",
            "content": "Testing event emission on document indexing.",
            "source": "test",
        })
        assert r.status_code == 200
        # Check that events were stored
        events_r = client.get("/events?limit=10")
        assert events_r.status_code == 200
        events = events_r.json()["events"]
        topics = [e["topic"] for e in events]
        assert "document.indexed" in topics

    def test_search_emits_event(self, client_and_store):
        client = client_and_store
        # First index a document to search for
        client.post("/index", json={
            "title": "Searchable Doc",
            "content": "This document is searchable for event testing.",
        })
        r = client.get("/search?q=searchable")
        assert r.status_code == 200
        events_r = client.get("/events?limit=50")
        topics = [e["topic"] for e in events_r.json()["events"]]
        assert "search.executed" in topics

    def test_delete_emits_event(self, client_and_store):
        client = client_and_store
        # Index then delete
        r = client.post("/index", json={
            "title": "To Delete",
            "content": "Will be deleted to test event emission.",
        })
        doc_id = r.json()["doc_id"]
        client.delete(f"/document/{doc_id}")
        events_r = client.get("/events?limit=50")
        topics = [e["topic"] for e in events_r.json()["events"]]
        assert "document.deleted" in topics

    def test_event_topics_catalogue_is_comprehensive(self):
        """Count total event types in topics.py and verify they cover key operations."""
        from app.events import topics
        # Collect all string constants from topics module
        all_topics = [
            v for k, v in vars(topics).items()
            if isinstance(v, str) and not k.startswith("_") and "." in v
        ]
        # Should have at least 15 event types defined
        assert len(all_topics) >= 15, f"Only {len(all_topics)} topics defined, expected >= 15"
        # Core topics must exist
        core = [
            "document.indexed", "document.deleted",
            "search.executed", "crawl.started",
            "workflow.completed", "rag.query.completed",
        ]
        for topic in core:
            assert topic in all_topics, f"Missing core event topic: {topic}"


# ══════════════════════════════════════════════════════════════════════════════
#  HEALTH AND READINESS
# ══════════════════════════════════════════════════════════════════════════════

class TestHealthAndReadiness:
    """Verify platform health infrastructure."""

    @pytest.fixture(scope="class")
    def client(self, tmp_path_factory):
        from app.api.routes import create_app
        from app.config import EngineConfig, DatabaseConfig
        import tempfile, pathlib
        db_path = pathlib.Path(tempfile.mkdtemp()) / "test_health.db"
        app = create_app(EngineConfig(database=DatabaseConfig(db_path=db_path)))
        from fastapi.testclient import TestClient
        with TestClient(app) as c:
            yield c

    def test_health_endpoint_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"

    def test_health_checks_all_components(self, client):
        r = client.get("/health")
        data = r.json()
        # Should report on database, events, and redis
        assert "database" in data
        assert "events" in data
        assert "redis" in data

    def test_infrastructure_stats_reports_backends(self, client):
        r = client.get("/infrastructure/stats")
        assert r.status_code == 200
        data = r.json()
        assert "database_backend" in data
        assert "event_bus" in data
        assert "redis_type" in data
        assert "event_store_size" in data

    def test_resilience_probes_endpoint(self, client):
        r = client.get("/resilience/health-probes")
        assert r.status_code == 200
        data = r.json()
        assert "overall_healthy" in data
        assert "probes" in data
        assert isinstance(data["probes"], list)


# ══════════════════════════════════════════════════════════════════════════════
#  GRACEFUL DEGRADATION
# ══════════════════════════════════════════════════════════════════════════════

class TestGracefulDegradation:
    """Verify platform handles missing services gracefully."""

    def _make_app_with_config(self, **overrides):
        from app.api.routes import create_app
        from app.config import (
            EngineConfig, DatabaseConfig, RedisConfig,
            EventConfig, VectorStoreConfig,
        )
        import tempfile, pathlib
        db_path = pathlib.Path(tempfile.mkdtemp()) / "test_degrade.db"
        cfg = EngineConfig(
            database=DatabaseConfig(db_path=db_path),
            redis=RedisConfig(
                host=overrides.get("redis_host", "localhost"),
                port=overrides.get("redis_port", 6379),
            ),
        )
        return create_app(cfg)

    def test_app_starts_without_redis(self):
        """App should start even when Redis is unreachable (fallback to in-memory)."""
        from app.api.routes import create_app
        from app.config import EngineConfig, DatabaseConfig, RedisConfig
        import tempfile, pathlib
        db_path = pathlib.Path(tempfile.mkdtemp()) / "test_no_redis.db"
        cfg = EngineConfig(
            database=DatabaseConfig(db_path=db_path),
            redis=RedisConfig(host="127.0.0.1", port=59999),  # unreachable port
        )
        app = create_app(cfg)
        assert app is not None
        from fastapi.testclient import TestClient
        with TestClient(app) as client:
            r = client.get("/health")
            assert r.status_code == 200

    def test_app_starts_without_postgres(self):
        """App should start with SQLite when Postgres is not configured."""
        from app.api.routes import create_app
        from app.config import EngineConfig, DatabaseConfig
        import tempfile, pathlib
        db_path = pathlib.Path(tempfile.mkdtemp()) / "test_no_pg.db"
        cfg = EngineConfig(
            database=DatabaseConfig(db_path=db_path, backend="sqlite"),
        )
        app = create_app(cfg)
        assert app is not None
        from fastapi.testclient import TestClient
        with TestClient(app) as client:
            r = client.get("/health")
            assert r.status_code == 200

    def test_app_starts_without_kafka(self):
        """App should start with in-memory event bus when Kafka is unavailable."""
        from app.api.routes import create_app
        from app.config import EngineConfig, DatabaseConfig, EventConfig
        import tempfile, pathlib
        db_path = pathlib.Path(tempfile.mkdtemp()) / "test_no_kafka.db"
        cfg = EngineConfig(
            database=DatabaseConfig(db_path=db_path),
            events=EventConfig(backend="memory"),
        )
        app = create_app(cfg)
        assert app is not None
        from fastapi.testclient import TestClient
        with TestClient(app) as client:
            r = client.get("/health")
            assert r.status_code == 200

    def test_app_starts_without_qdrant(self):
        """App should start with FAISS when Qdrant is unavailable."""
        from app.api.routes import create_app
        from app.config import EngineConfig, DatabaseConfig, VectorStoreConfig
        import tempfile, pathlib
        db_path = pathlib.Path(tempfile.mkdtemp()) / "test_no_qdrant.db"
        cfg = EngineConfig(
            database=DatabaseConfig(db_path=db_path),
            vector_store=VectorStoreConfig(backend="faiss"),
        )
        app = create_app(cfg)
        assert app is not None
        from fastapi.testclient import TestClient
        with TestClient(app) as client:
            r = client.get("/health")
            assert r.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE BACKEND ABSTRACTION
# ══════════════════════════════════════════════════════════════════════════════

class TestDatabaseBackendAbstraction:
    """Verify backend abstraction layer."""

    def test_sqlite_backend_exists(self):
        from app.database.backend import SQLiteBackend
        assert SQLiteBackend is not None

    def test_postgres_backend_exists(self):
        from app.database.backend import PostgreSQLBackend
        assert PostgreSQLBackend is not None

    def test_database_accepts_backend_param(self):
        from app.config import DatabaseConfig
        cfg = DatabaseConfig(backend="postgres")
        assert cfg.backend == "postgres"

    def test_database_defaults_to_sqlite(self):
        from app.config import DatabaseConfig
        cfg = DatabaseConfig()
        assert cfg.backend == "sqlite"

    def test_backend_protocol_methods_exist(self):
        from app.database.backend import DatabaseBackend
        # Verify the protocol defines the expected methods
        expected_methods = [
            "connect", "close", "execute", "executemany",
            "fetchone", "fetchall", "commit", "begin",
        ]
        for method in expected_methods:
            assert hasattr(DatabaseBackend, method), (
                f"DatabaseBackend protocol missing method: {method}"
            )

    def test_sqlite_backend_implements_protocol(self):
        from app.database.backend import SQLiteBackend, DatabaseBackend
        import tempfile, pathlib
        db_path = pathlib.Path(tempfile.mkdtemp()) / "protocol_test.db"
        backend = SQLiteBackend(db_path)
        assert isinstance(backend, DatabaseBackend)

    def test_sqlite_backend_placeholder(self):
        from app.database.backend import SQLiteBackend
        import tempfile, pathlib
        db_path = pathlib.Path(tempfile.mkdtemp()) / "placeholder.db"
        backend = SQLiteBackend(db_path)
        assert backend.placeholder == "?"

    def test_postgres_backend_placeholder(self):
        from app.database.backend import PostgreSQLBackend
        from app.config import PostgresConfig
        backend = PostgreSQLBackend(PostgresConfig())
        assert backend.placeholder == "%s"


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 8.75 API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase875APIEndpoints:
    """API smoke tests for enterprise features."""

    @pytest.fixture(scope="class")
    def client(self, tmp_path_factory):
        from app.api.routes import create_app
        from app.config import EngineConfig, DatabaseConfig
        import tempfile, pathlib
        db_path = pathlib.Path(tempfile.mkdtemp()) / "test_api_enterprise.db"
        app = create_app(EngineConfig(database=DatabaseConfig(db_path=db_path)))
        from fastapi.testclient import TestClient
        with TestClient(app) as c:
            yield c

    def test_resilience_services_endpoint(self, client):
        r = client.get("/resilience/services")
        assert r.status_code == 200
        data = r.json()
        assert "services" in data
        assert "total_services" in data
        assert data["total_services"] > 0

    def test_security_enforcement_endpoint(self, client):
        r = client.get("/security/enforcement")
        assert r.status_code == 200
        data = r.json()
        assert "endpoint_rules" in data
        assert data["endpoint_rules"] > 0
        assert "public_endpoints" in data
        assert "admin_endpoints" in data

    def test_all_endpoints_return_valid_json(self, client):
        """Hit every observability/infrastructure endpoint and verify valid JSON."""
        endpoints = [
            "/health",
            "/infrastructure/stats",
            "/resilience/services",
            "/resilience/circuit-breakers",
            "/resilience/health-probes",
            "/security/enforcement",
            "/security/health",
            "/security/rbac",
            "/cost/summary",
            "/cost/budget",
            "/cost/stats",
            "/observability/traces",
            "/observability/logs",
            "/services",
            "/services/health",
            "/metrics/snapshot",
            "/events",
        ]
        for path in endpoints:
            r = client.get(path)
            assert r.status_code == 200, f"{path} returned {r.status_code}"
            # Metrics endpoint returns text, all others return JSON
            if path != "/metrics":
                data = r.json()
                assert isinstance(data, (dict, list)), f"{path} did not return valid JSON"
