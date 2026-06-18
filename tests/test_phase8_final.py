"""
Phase 8 Final Tests — Tenant Isolation, Security Enforcement, Resilience Wrapping

Tests cover:
  - TenantAwareDatabase (insert with scoping, access control, passthrough)
  - TenantVectorNamespace (scoped add/search, cross-tenant filtering)
  - TenantCacheScope (scoped keys, isolation)
  - SecurityEnforcer (endpoint classification, access checks)
  - ServiceResilienceLayer (circuit breaker wrapping, fallback, retry)
  - ResilientService (call with breaker, failure + fallback)
  - Cross-tenant leakage prevention
  - Config backward compatibility
  - API endpoint smoke tests
"""

import time
import threading
import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass


# ══════════════════════════════════════════════════════════════════════════════
#  TENANT-AWARE DATABASE
# ══════════════════════════════════════════════════════════════════════════════

class TestTenantAwareDatabase:
    """Tenant-scoped data access layer."""

    def _make_tdb(self, enabled=True):
        from app.tenancy.data_access import TenantAwareDatabase
        from app.config import TenancyConfig
        mock_db = MagicMock()
        mock_db.insert_document.return_value = 1
        mock_db.get_all_documents.return_value = []
        mock_db.get_document_count.return_value = 0
        return TenantAwareDatabase(mock_db, TenancyConfig(enabled=enabled))

    def test_insert_tags_with_tenant(self):
        from app.tenancy.context import tenant_scope
        tdb = self._make_tdb()
        with tenant_scope("tenant_a"):
            tdb.insert_document("title", "content", "api")
        tdb._db.insert_document.assert_called_once()
        call_args = tdb._db.insert_document.call_args
        assert "tenant:tenant_a:" in call_args[1].get("source", call_args[0][2])

    def test_get_document_denies_cross_tenant(self):
        from app.tenancy.context import tenant_scope
        tdb = self._make_tdb()

        @dataclass
        class FakeDoc:
            doc_id: int = 1
            source: str = "tenant:tenant_b:api"
            title: str = "test"
            content: str = "test"

        tdb._db.get_document.return_value = FakeDoc()
        with tenant_scope("tenant_a"):
            result = tdb.get_document(1)
        assert result is None

    def test_get_document_allows_own_tenant(self):
        from app.tenancy.context import tenant_scope
        tdb = self._make_tdb()

        @dataclass
        class FakeDoc:
            doc_id: int = 1
            source: str = "tenant:tenant_a:api"
            title: str = "test"
            content: str = "test"

        tdb._db.get_document.return_value = FakeDoc()
        with tenant_scope("tenant_a"):
            result = tdb.get_document(1)
        assert result is not None

    def test_disabled_passes_through(self):
        tdb = self._make_tdb(enabled=False)
        tdb.insert_document("t", "c", "src")
        call_args = tdb._db.insert_document.call_args
        assert "tenant:" not in str(call_args)

    def test_passthrough_delegates(self):
        tdb = self._make_tdb()
        tdb.get_term_count()
        tdb._db.get_term_count.assert_called_once()

    def test_tenant_stats(self):
        from app.tenancy.context import tenant_scope
        tdb = self._make_tdb()
        tdb._db.get_all_documents.return_value = []
        with tenant_scope("t1"):
            stats = tdb.tenant_stats()
        assert stats["tenant_id"] == "t1"
        assert stats["enabled"] is True

    def test_validate_access(self):
        from app.tenancy.context import tenant_scope
        tdb = self._make_tdb()

        @dataclass
        class FakeDoc:
            doc_id: int = 1
            source: str = "tenant:t1:api"

        tdb._db.get_document.return_value = FakeDoc()
        with tenant_scope("t1"):
            assert tdb.validate_access(1) is True
        with tenant_scope("t2"):
            assert tdb.validate_access(1) is False


# ══════════════════════════════════════════════════════════════════════════════
#  TENANT VECTOR NAMESPACE
# ══════════════════════════════════════════════════════════════════════════════

class TestTenantVectorNamespace:
    """Tenant-scoped vector operations."""

    def _make_tvn(self, enabled=True):
        from app.tenancy.vector_namespace import TenantVectorNamespace
        from app.config import TenancyConfig
        mock_store = MagicMock()
        mock_store.total_vectors = 0
        return TenantVectorNamespace(mock_store, TenancyConfig(enabled=enabled))

    def test_add_scopes_chunk_ids(self):
        from app.tenancy.context import tenant_scope
        tvn = self._make_tvn()
        with tenant_scope("t1"):
            tvn.add(["c1", "c2"], [[0.1]*3, [0.2]*3])
        call_args = tvn._store.add.call_args[0]
        assert all(cid.startswith("t:t1:") for cid in call_args[0])

    def test_search_filters_by_tenant(self):
        from app.tenancy.context import tenant_scope
        tvn = self._make_tvn()
        tvn._store.search.return_value = [
            ("t:t1:c1", 0.9),
            ("t:t2:c2", 0.8),
            ("t:t1:c3", 0.7),
        ]
        with tenant_scope("t1"):
            results = tvn.search([0.1]*3, top_k=10)
        assert len(results) == 2
        assert all(not cid.startswith("t:") for cid, _ in results)

    def test_disabled_passes_through(self):
        tvn = self._make_tvn(enabled=False)
        tvn.add(["c1"], [[0.1]*3])
        call_args = tvn._store.add.call_args[0]
        assert call_args[0] == ["c1"]

    def test_cross_tenant_leak_prevented(self):
        from app.tenancy.context import tenant_scope
        tvn = self._make_tvn()
        tvn._store.search.return_value = [
            ("t:other_tenant:secret_doc", 0.95),
        ]
        with tenant_scope("my_tenant"):
            results = tvn.search([0.1]*3, top_k=10)
        assert len(results) == 0


# ══════════════════════════════════════════════════════════════════════════════
#  TENANT CACHE SCOPE
# ══════════════════════════════════════════════════════════════════════════════

class TestTenantCacheScope:
    """Tenant-scoped cache key prefixing."""

    def _make_tcs(self, enabled=True):
        from app.tenancy.cache_scope import TenantCacheScope
        from app.config import TenancyConfig
        from app.redis.client import InMemoryRedisClient
        client = InMemoryRedisClient()
        return TenantCacheScope(client, TenancyConfig(enabled=enabled))

    def test_scoped_set_get(self):
        from app.tenancy.context import tenant_scope
        tcs = self._make_tcs()
        with tenant_scope("t1"):
            tcs.set("key1", "value1")
            assert tcs.get("key1") == "value1"

    def test_cross_tenant_isolation(self):
        from app.tenancy.context import tenant_scope
        tcs = self._make_tcs()
        with tenant_scope("t1"):
            tcs.set("shared_key", "t1_value")
        with tenant_scope("t2"):
            result = tcs.get("shared_key")
        assert result is None

    def test_disabled_no_scoping(self):
        tcs = self._make_tcs(enabled=False)
        tcs.set("k", "v")
        assert tcs.get("k") == "v"

    def test_delete_scoped(self):
        from app.tenancy.context import tenant_scope
        tcs = self._make_tcs()
        with tenant_scope("t1"):
            tcs.set("k", "v")
            tcs.delete("k")
            assert tcs.get("k") is None


# ══════════════════════════════════════════════════════════════════════════════
#  SECURITY ENFORCER
# ══════════════════════════════════════════════════════════════════════════════

class TestSecurityEnforcer:
    """Endpoint-level security enforcement."""

    def _make_enforcer(self, enabled=True):
        from app.security.enforcement import SecurityEnforcer
        from app.config import SecurityConfig
        return SecurityEnforcer(SecurityConfig(enabled=enabled))

    def test_public_endpoint_allowed(self):
        from app.security.middleware import SecurityContext
        enforcer = self._make_enforcer()
        allowed, reason = enforcer.check_access("/health")
        assert allowed
        assert reason == "public_endpoint"

    def test_unauthenticated_denied(self):
        from app.security.middleware import SecurityContext
        SecurityContext.clear()
        enforcer = self._make_enforcer()
        allowed, reason = enforcer.check_access("/search")
        assert not allowed
        assert reason == "authentication_required"

    def test_authenticated_allowed(self):
        from app.security.middleware import SecurityContext
        SecurityContext.set("user1", ["reader"])
        enforcer = self._make_enforcer()
        allowed, reason = enforcer.check_access("/search")
        assert allowed
        SecurityContext.clear()

    def test_admin_endpoint_denied_for_reader(self):
        from app.security.middleware import SecurityContext
        SecurityContext.set("user1", ["reader"])
        enforcer = self._make_enforcer()
        allowed, reason = enforcer.check_access("/tenants")
        assert not allowed
        assert "admin" in reason
        SecurityContext.clear()

    def test_admin_endpoint_allowed_for_admin(self):
        from app.security.middleware import SecurityContext
        SecurityContext.set("admin1", ["admin"])
        enforcer = self._make_enforcer()
        allowed, reason = enforcer.check_access("/tenants")
        assert allowed
        SecurityContext.clear()

    def test_disabled_allows_everything(self):
        enforcer = self._make_enforcer(enabled=False)
        allowed, reason = enforcer.check_access("/tenants")
        assert allowed
        assert reason == "security_disabled"

    def test_classify_unknown_defaults_authenticated(self):
        from app.security.enforcement import EndpointAccess
        enforcer = self._make_enforcer()
        access, perm = enforcer.classify_endpoint("/unknown/path")
        assert access == EndpointAccess.AUTHENTICATED

    def test_stats(self):
        enforcer = self._make_enforcer()
        s = enforcer.stats()
        assert "endpoint_rules" in s
        assert s["endpoint_rules"] > 0


# ══════════════════════════════════════════════════════════════════════════════
#  RESILIENT SERVICE
# ══════════════════════════════════════════════════════════════════════════════

class TestResilientService:
    """Service wrapping with circuit breaker + retry + fallback."""

    def _make_service(self, fallback=None):
        from app.resilience.service_wrapper import ResilientService
        from app.resilience.circuit_breaker import CircuitBreakerRegistry
        from app.config import ResilienceConfig
        config = ResilienceConfig(failure_threshold=3, retry_max_attempts=1)
        registry = CircuitBreakerRegistry(config)
        return ResilientService("test", registry, config, fallback=fallback)

    def test_successful_call(self):
        svc = self._make_service()
        result = svc.call(lambda: 42)
        assert result == 42

    def test_failure_with_fallback(self):
        svc = self._make_service(fallback=lambda: "fallback")
        result = svc.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        assert result == "fallback"

    def test_failure_without_fallback_raises(self):
        svc = self._make_service()
        with pytest.raises(RuntimeError):
            svc.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

    def test_stats_tracking(self):
        svc = self._make_service(fallback=lambda: 0)
        svc.call(lambda: 1)
        svc.call(lambda: (_ for _ in ()).throw(ValueError("x")))
        s = svc.stats()
        assert s["calls"] == 2
        assert s["failures"] == 1
        assert s["fallback_used"] == 1


# ══════════════════════════════════════════════════════════════════════════════
#  SERVICE RESILIENCE LAYER
# ══════════════════════════════════════════════════════════════════════════════

class TestServiceResilienceLayer:
    """Centralized resilience management."""

    def _make_layer(self):
        from app.resilience.service_wrapper import ServiceResilienceLayer
        from app.config import ResilienceConfig
        return ServiceResilienceLayer(ResilienceConfig())

    def test_register_and_call(self):
        layer = self._make_layer()
        layer.register("svc1")
        result = layer.call("svc1", lambda: "ok")
        assert result == "ok"

    def test_auto_register_on_call(self):
        layer = self._make_layer()
        result = layer.call("new_svc", lambda: 99)
        assert result == 99

    def test_stats(self):
        layer = self._make_layer()
        layer.call("a", lambda: 1)
        layer.call("b", lambda: 2)
        s = layer.stats()
        assert s["total_services"] == 2
        assert "a" in s["services"]
        assert "b" in s["services"]


# ══════════════════════════════════════════════════════════════════════════════
#  CROSS-TENANT LEAKAGE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestCrossTenantLeakage:
    """Verify zero data leakage across tenants."""

    def test_vector_search_no_cross_tenant_results(self):
        from app.tenancy.vector_namespace import TenantVectorNamespace
        from app.tenancy.context import tenant_scope
        from app.config import TenancyConfig
        mock_store = MagicMock()
        mock_store.search.return_value = [
            ("t:attacker:stolen_data", 0.99),
            ("t:victim:private_doc", 0.95),
            ("t:legit:my_doc", 0.90),
        ]
        tvn = TenantVectorNamespace(mock_store, TenancyConfig(enabled=True))
        with tenant_scope("legit"):
            results = tvn.search([0.1]*3, top_k=10)
        assert len(results) == 1
        assert results[0][0] == "my_doc"

    def test_cache_no_cross_tenant_reads(self):
        from app.tenancy.cache_scope import TenantCacheScope
        from app.tenancy.context import tenant_scope
        from app.config import TenancyConfig
        from app.redis.client import InMemoryRedisClient
        client = InMemoryRedisClient()
        tcs = TenantCacheScope(client, TenancyConfig(enabled=True))
        with tenant_scope("victim"):
            tcs.set("secret", "sensitive_data")
        with tenant_scope("attacker"):
            leaked = tcs.get("secret")
        assert leaked is None

    def test_database_no_cross_tenant_documents(self):
        from app.tenancy.data_access import TenantAwareDatabase
        from app.tenancy.context import tenant_scope
        from app.config import TenancyConfig

        @dataclass
        class FakeDoc:
            doc_id: int
            source: str
            title: str = "t"
            content: str = "c"

        mock_db = MagicMock()
        mock_db.get_document.return_value = FakeDoc(1, "tenant:victim:api")
        tdb = TenantAwareDatabase(mock_db, TenancyConfig(enabled=True))
        with tenant_scope("attacker"):
            doc = tdb.get_document(1)
        assert doc is None


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG COMPATIBILITY
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase8FinalConfigCompat:
    """Config backward compatibility — final check."""

    def test_engine_config_zero_arg(self):
        from app.config import EngineConfig
        cfg = EngineConfig()
        assert cfg.vector_store.backend == "faiss"
        assert cfg.crawler.mode == "single"
        assert cfg.agent_execution.mode == "local"
        assert cfg.security.enabled is False
        assert cfg.tenancy.enabled is False

    def test_all_mode_fields_have_defaults(self):
        from app.config import EngineConfig
        cfg = EngineConfig()
        assert hasattr(cfg.vector_store, 'backend')
        assert hasattr(cfg.crawler, 'mode')
        assert hasattr(cfg.agent_execution, 'mode')


# ══════════════════════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase8FinalAPIEndpoints:
    """Phase 8 Final API smoke tests."""

    @pytest.fixture(scope="class")
    def client(self, tmp_path_factory):
        from app.api.routes import create_app
        from app.config import EngineConfig, DatabaseConfig
        import tempfile, pathlib
        db_path = pathlib.Path(tempfile.mkdtemp()) / "test.db"
        app = create_app(EngineConfig(database=DatabaseConfig(db_path=db_path)))
        from fastapi.testclient import TestClient
        with TestClient(app) as c:
            yield c

    def test_resilience_services(self, client):
        r = client.get("/resilience/services")
        assert r.status_code == 200
        assert "services" in r.json()

    def test_security_enforcement(self, client):
        r = client.get("/security/enforcement")
        assert r.status_code == 200
        assert "endpoint_rules" in r.json()
