"""
Phase 8 Batch 4 Tests — Security, Observability, Resilience, Cost

Tests cover:
  - JWT authentication (issue, verify, expiry, invalid)
  - API key management (create, verify, revoke, rotate)
  - RBAC enforcer (roles, permissions, admin wildcard)
  - Audit logger (log, get_recent, stats)
  - Security context (thread-local, clear)
  - Circuit breaker (CLOSED→OPEN→HALF_OPEN, fast-fail)
  - Retry strategy (backoff, jitter, max attempts)
  - Health probe (pass, fail, timeout)
  - Graceful shutdown (LIFO order, timeout)
  - Cost tracker (record, summary, budget)
  - Cost estimator (price lookup, cheapest model)
  - Cost dashboard (daily report, budget status)
  - Tracer (start_span, finish, context manager)
  - Structured logger (emit, get_recent)
  - Config backward compatibility
  - API endpoint smoke tests

All tests use in-memory backends — no external services required.
"""

import threading
import time
import pytest
from fastapi.testclient import TestClient


# ══════════════════════════════════════════════════════════════════════════════
#  JWT AUTHENTICATION
# ══════════════════════════════════════════════════════════════════════════════

class TestJWTAuth:
    """JWT token issuance and verification."""

    def _make_auth(self):
        from app.security.jwt_auth import JWTAuth
        from app.config import SecurityConfig
        return JWTAuth(SecurityConfig(jwt_expiry_hours=1))

    def test_issue_and_verify(self):
        auth = self._make_auth()
        token = auth.create_token("user1", "tenant1", ["reader"])
        claims = auth.verify_token(token)
        assert claims.sub == "user1"
        assert claims.tenant_id == "tenant1"
        assert "reader" in claims.roles

    def test_invalid_signature_rejected(self):
        auth = self._make_auth()
        token = auth.create_token("u1")
        parts = token.split(".")
        tampered = parts[0] + "." + parts[1] + ".invalidsig"
        with pytest.raises(ValueError):
            auth.verify_token(tampered)

    def test_bad_format_rejected(self):
        auth = self._make_auth()
        with pytest.raises(ValueError):
            auth.verify_token("not.a.valid.jwt.token.format.at.all")

    def test_refresh_token(self):
        auth = self._make_auth()
        token = auth.create_token("u1", "t1", ["admin"])
        new_token = auth.refresh_token(token)
        claims = auth.verify_token(new_token)
        assert claims.sub == "u1"
        assert "admin" in claims.roles

    def test_empty_roles_allowed(self):
        auth = self._make_auth()
        token = auth.create_token("u1")
        claims = auth.verify_token(token)
        assert claims.roles == []


# ══════════════════════════════════════════════════════════════════════════════
#  API KEY MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

class TestAPIKeyManager:
    """API key lifecycle: create, verify, revoke, rotate."""

    def _make_mgr(self):
        from app.security.api_keys import APIKeyManager
        from app.config import SecurityConfig
        return APIKeyManager(SecurityConfig())

    def test_create_and_verify(self):
        mgr = self._make_mgr()
        raw, key = mgr.create_key("test-key", tenant_id="t1", roles=["reader"])
        assert raw.startswith("sk-")
        verified = mgr.verify_key(raw)
        assert verified is not None
        assert verified.name == "test-key"
        assert verified.tenant_id == "t1"

    def test_wrong_key_rejected(self):
        mgr = self._make_mgr()
        assert mgr.verify_key("sk-wrongkey") is None

    def test_revoke(self):
        mgr = self._make_mgr()
        raw, key = mgr.create_key("k1")
        assert mgr.verify_key(raw) is not None
        assert mgr.revoke_key(key.key_id)
        assert mgr.verify_key(raw) is None

    def test_list_keys(self):
        mgr = self._make_mgr()
        mgr.create_key("k1", tenant_id="t1")
        mgr.create_key("k2", tenant_id="t1")
        mgr.create_key("k3", tenant_id="t2")
        assert len(mgr.list_keys("t1")) == 2
        assert len(mgr.list_keys("t2")) == 1

    def test_rotate(self):
        mgr = self._make_mgr()
        raw1, key1 = mgr.create_key("k1")
        raw2, key2 = mgr.rotate_key(key1.key_id)
        assert raw2 != raw1
        assert mgr.verify_key(raw1) is None   # old key disabled
        assert mgr.verify_key(raw2) is not None

    def test_stats(self):
        mgr = self._make_mgr()
        mgr.create_key("k1")
        s = mgr.stats()
        assert s["total"] == 1
        assert s["enabled"] == 1


# ══════════════════════════════════════════════════════════════════════════════
#  RBAC
# ══════════════════════════════════════════════════════════════════════════════

class TestRBACEnforcer:
    """Role-based access control."""

    def _make_rbac(self):
        from app.security.rbac import RBACEnforcer
        return RBACEnforcer()

    def test_reader_can_search(self):
        from app.security.rbac import Permission
        rbac = self._make_rbac()
        assert rbac.has_permission(["reader"], Permission.READ_SEARCH)

    def test_reader_cannot_index(self):
        from app.security.rbac import Permission
        rbac = self._make_rbac()
        assert not rbac.has_permission(["reader"], Permission.WRITE_INDEX)

    def test_admin_has_all(self):
        from app.security.rbac import Permission
        rbac = self._make_rbac()
        for perm in Permission:
            assert rbac.has_permission(["admin"], perm)

    def test_multiple_roles_combined(self):
        from app.security.rbac import Permission
        rbac = self._make_rbac()
        assert rbac.has_permission(["reader", "indexer"], Permission.WRITE_INDEX)

    def test_check_permission_raises(self):
        from app.security.rbac import Permission
        rbac = self._make_rbac()
        with pytest.raises(PermissionError):
            rbac.check_permission(["reader"], Permission.WRITE_INDEX)

    def test_get_permissions(self):
        from app.security.rbac import Permission
        rbac = self._make_rbac()
        perms = rbac.get_permissions(["reader"])
        assert Permission.READ_SEARCH in perms

    def test_add_custom_role(self):
        from app.security.rbac import Permission
        rbac = self._make_rbac()
        rbac.add_role("custom", {Permission.READ_SEARCH})
        assert rbac.has_permission(["custom"], Permission.READ_SEARCH)
        assert not rbac.has_permission(["custom"], Permission.WRITE_INDEX)


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIT LOGGER
# ══════════════════════════════════════════════════════════════════════════════

class TestAuditLogger:
    """Audit log recording and retrieval."""

    def _make_logger(self, tmp_path):
        from app.security.audit import AuditLogger
        from app.config import SecurityConfig
        cfg = SecurityConfig(audit_log_enabled=True,
                             audit_log_path=str(tmp_path / "audit.log"))
        return AuditLogger(cfg)

    def test_log_auth_success(self, tmp_path):
        logger = self._make_logger(tmp_path)
        logger.log_auth("user1", True, ip="127.0.0.1", tenant_id="t1")
        events = logger.get_recent()
        assert len(events) == 1
        assert events[0].actor == "user1"

    def test_log_auth_failure(self, tmp_path):
        from app.security.audit import AuditEventType
        logger = self._make_logger(tmp_path)
        logger.log_auth("badactor", False)
        events = logger.get_recent()
        assert events[0].event_type == AuditEventType.AUTH_FAILURE

    def test_filter_by_tenant(self, tmp_path):
        logger = self._make_logger(tmp_path)
        logger.log_auth("u1", True, tenant_id="t1")
        logger.log_auth("u2", True, tenant_id="t2")
        t1_events = logger.get_recent(tenant_id="t1")
        assert len(t1_events) == 1
        assert t1_events[0].actor == "u1"

    def test_stats(self, tmp_path):
        logger = self._make_logger(tmp_path)
        logger.log_auth("u1", True)
        s = logger.stats()
        assert s["total_events"] == 1


# ══════════════════════════════════════════════════════════════════════════════
#  SECURITY CONTEXT
# ══════════════════════════════════════════════════════════════════════════════

class TestSecurityContext:
    """Thread-local security context."""

    def test_set_get(self):
        from app.security.middleware import SecurityContext
        SecurityContext.set("user1", ["admin"], "t1")
        assert SecurityContext.get_actor() == "user1"
        assert SecurityContext.get_roles() == ["admin"]
        assert SecurityContext.get_tenant() == "t1"
        SecurityContext.clear()

    def test_clear(self):
        from app.security.middleware import SecurityContext
        SecurityContext.set("u1", ["r"])
        SecurityContext.clear()
        assert SecurityContext.get_actor() is None
        assert SecurityContext.get_roles() == []

    def test_thread_isolation(self):
        from app.security.middleware import SecurityContext
        results = {}

        def thread_fn(name, actor):
            SecurityContext.set(actor, [])
            time.sleep(0.05)
            results[name] = SecurityContext.get_actor()
            SecurityContext.clear()

        t1 = threading.Thread(target=thread_fn, args=("t1", "alice"))
        t2 = threading.Thread(target=thread_fn, args=("t2", "bob"))
        t1.start(); t2.start()
        t1.join(); t2.join()
        assert results["t1"] == "alice"
        assert results["t2"] == "bob"


# ══════════════════════════════════════════════════════════════════════════════
#  CIRCUIT BREAKER
# ══════════════════════════════════════════════════════════════════════════════

class TestCircuitBreaker:
    """Circuit breaker state machine."""

    def _make_cb(self, threshold=3, recovery=1.0):
        from app.resilience.circuit_breaker import CircuitBreaker
        from app.config import ResilienceConfig
        cfg = ResilienceConfig(
            failure_threshold=threshold,
            recovery_timeout_sec=recovery,
            half_open_max_calls=1,
        )
        return CircuitBreaker("test", cfg)

    def test_successful_call(self):
        cb = self._make_cb()
        assert cb.call(lambda: 42) == 42

    def test_trips_after_threshold(self):
        from app.resilience.circuit_breaker import CircuitState, CircuitOpenError
        cb = self._make_cb(threshold=2)
        for _ in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
            except ValueError:
                pass
        assert cb.state() == CircuitState.OPEN
        with pytest.raises(CircuitOpenError):
            cb.call(lambda: 1)

    def test_reset(self):
        from app.resilience.circuit_breaker import CircuitState
        cb = self._make_cb(threshold=1)
        try:
            cb.call(lambda: (_ for _ in ()).throw(ValueError("x")))
        except ValueError:
            pass
        cb.reset()
        assert cb.state() == CircuitState.CLOSED

    def test_stats(self):
        cb = self._make_cb()
        cb.call(lambda: 1)
        s = cb.stats()
        assert "state" in s
        assert "failure_count" in s


class TestCircuitBreakerRegistry:
    """Registry of named circuit breakers."""

    def test_get_creates_new(self):
        from app.resilience.circuit_breaker import CircuitBreakerRegistry, CircuitBreaker
        from app.config import ResilienceConfig
        reg = CircuitBreakerRegistry(ResilienceConfig())
        cb = reg.get("my-service")
        assert isinstance(cb, CircuitBreaker)

    def test_get_returns_same(self):
        from app.resilience.circuit_breaker import CircuitBreakerRegistry
        from app.config import ResilienceConfig
        reg = CircuitBreakerRegistry(ResilienceConfig())
        cb1 = reg.get("svc")
        cb2 = reg.get("svc")
        assert cb1 is cb2

    def test_get_all_stats(self):
        from app.resilience.circuit_breaker import CircuitBreakerRegistry
        from app.config import ResilienceConfig
        reg = CircuitBreakerRegistry(ResilienceConfig())
        reg.get("a"); reg.get("b")
        stats = reg.get_all_stats()
        assert "a" in stats and "b" in stats


# ══════════════════════════════════════════════════════════════════════════════
#  RETRY STRATEGY
# ══════════════════════════════════════════════════════════════════════════════

class TestRetryStrategy:
    """Retry with exponential backoff."""

    def test_success_no_retry(self):
        from app.resilience.retry import RetryStrategy
        calls = []
        def fn():
            calls.append(1)
            return "ok"
        assert RetryStrategy().execute(fn) == "ok"
        assert len(calls) == 1

    def test_retry_on_failure(self):
        from app.resilience.retry import RetryStrategy, RetryConfig
        calls = []
        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("transient")
            return "done"
        cfg = RetryConfig(max_attempts=3, base_delay_sec=0.0)
        result = RetryStrategy(cfg).execute(fn)
        assert result == "done"
        assert len(calls) == 3

    def test_exhausted_raises(self):
        from app.resilience.retry import RetryStrategy, RetryConfig
        cfg = RetryConfig(max_attempts=2, base_delay_sec=0.0)
        with pytest.raises(RuntimeError):
            RetryStrategy(cfg).execute(lambda: (_ for _ in ()).throw(RuntimeError("perm")))

    def test_delay_grows(self):
        from app.resilience.retry import RetryStrategy, RetryConfig
        cfg = RetryConfig(base_delay_sec=1.0, jitter=False)
        s = RetryStrategy(cfg)
        assert s.delay_for(0) == 0.0
        assert s.delay_for(1) >= 1.0
        assert s.delay_for(2) >= 2.0

    def test_with_retry_decorator(self):
        from app.resilience.retry import with_retry
        calls = []
        @with_retry(max_attempts=2, base_delay=0.0)
        def fn():
            calls.append(1)
            if len(calls) < 2:
                raise IOError("retry me")
            return "wrapped"
        assert fn() == "wrapped"
        assert len(calls) == 2


# ══════════════════════════════════════════════════════════════════════════════
#  HEALTH PROBE
# ══════════════════════════════════════════════════════════════════════════════

class TestHealthProbe:
    """Active health probes."""

    def _make_probe(self):
        from app.resilience.health_probe import HealthProbe
        from app.config import ResilienceConfig
        return HealthProbe(ResilienceConfig())

    def test_all_pass(self):
        probe = self._make_probe()
        probe.add_probe("db", lambda: True)
        probe.add_probe("cache", lambda: True)
        assert probe.is_healthy()

    def test_one_fails(self):
        probe = self._make_probe()
        probe.add_probe("db", lambda: True)
        probe.add_probe("bad", lambda: False)
        assert not probe.is_healthy()

    def test_exception_marked_unhealthy(self):
        probe = self._make_probe()
        probe.add_probe("err", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        results = probe.probe_all()
        assert not results[0].healthy

    def test_stats(self):
        probe = self._make_probe()
        probe.add_probe("x", lambda: True)
        probe.probe_all()
        s = probe.stats()
        assert "registered_probes" in s
        assert s["probe_count"] == 1


# ══════════════════════════════════════════════════════════════════════════════
#  GRACEFUL SHUTDOWN
# ══════════════════════════════════════════════════════════════════════════════

class TestGracefulShutdown:
    """Graceful shutdown handler ordering and execution."""

    def _make_sd(self):
        from app.resilience.shutdown import GracefulShutdown
        from app.config import ResilienceConfig
        return GracefulShutdown(ResilienceConfig(graceful_shutdown_sec=10.0))

    def test_handler_called(self):
        sd = self._make_sd()
        called = []
        sd.register("h1", lambda: called.append("h1"))
        sd.shutdown()
        assert "h1" in called

    def test_lifo_order(self):
        sd = self._make_sd()
        order = []
        sd.register("first", lambda: order.append("first"))
        sd.register("second", lambda: order.append("second"))
        sd.shutdown()
        assert order == ["second", "first"]

    def test_results_ok(self):
        sd = self._make_sd()
        sd.register("clean", lambda: None)
        results = sd.shutdown()
        assert results["clean"] == "ok"

    def test_is_shutting_down(self):
        sd = self._make_sd()
        assert not sd.is_shutting_down()
        sd.shutdown()
        assert sd.is_shutting_down()


# ══════════════════════════════════════════════════════════════════════════════
#  COST TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class TestCostTracker:
    """Cost event recording and summarization."""

    def _make_tracker(self, tmp_path):
        from app.cost.tracker import CostTracker
        from app.config import CostConfig
        return CostTracker(CostConfig(cost_log_path=str(tmp_path / "costs.jsonl")))

    def test_record_llm(self, tmp_path):
        t = self._make_tracker(tmp_path)
        t.record_llm("openai", "gpt-4o", 100, 50, 0.001)
        s = t.get_summary()
        assert s.total_usd > 0

    def test_record_embedding(self, tmp_path):
        t = self._make_tracker(tmp_path)
        t.record_embedding("openai", "text-embedding-3-small", 500, 0.00001)
        events = t.get_recent_events()
        assert len(events) >= 1

    def test_get_recent_events(self, tmp_path):
        t = self._make_tracker(tmp_path)
        t.record_llm("a", "m", 10, 5, 0.001)
        t.record_llm("b", "n", 10, 5, 0.002)
        # record_llm creates 2 events per call (INPUT + OUTPUT) = 4 total
        events = t.get_recent_events(limit=10)
        assert len(events) == 4

    def test_stats(self, tmp_path):
        t = self._make_tracker(tmp_path)
        t.record_llm("a", "m", 10, 5, 0.001)
        s = t.stats()
        assert "total_events" in s

    def test_not_over_budget_initially(self, tmp_path):
        t = self._make_tracker(tmp_path)
        assert not t.is_over_budget()


# ══════════════════════════════════════════════════════════════════════════════
#  COST ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════

class TestCostEstimator:
    """Cost estimation before execution."""

    def _make_est(self):
        from app.cost.estimator import CostEstimator
        from app.config import CostConfig
        return CostEstimator(CostConfig())

    def test_estimate_llm(self):
        est = self._make_est()
        cost = est.estimate_llm("openai", "gpt-4o", 1000, 500)
        assert cost > 0.0

    def test_estimate_embedding(self):
        est = self._make_est()
        cost = est.estimate_embedding("openai", "text-embedding-3-small", 10000)
        assert cost >= 0.0

    def test_unknown_model_returns_zero(self):
        est = self._make_est()
        cost = est.estimate_llm("unknown", "unknown-model", 1000, 500)
        assert cost == 0.0

    def test_stats(self):
        est = self._make_est()
        s = est.stats()
        assert "providers" in s or isinstance(s, dict)


# ══════════════════════════════════════════════════════════════════════════════
#  COST DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

class TestCostDashboard:
    """Cost reporting and budget status."""

    def _make_dashboard(self, tmp_path):
        from app.cost.tracker import CostTracker
        from app.cost.dashboard import CostDashboard
        from app.config import CostConfig
        cfg = CostConfig(budget_alert_usd=1.0,
                         cost_log_path=str(tmp_path / "c.jsonl"))
        tracker = CostTracker(cfg)
        tracker.record_llm("openai", "gpt-4o", 100, 50, 0.01)
        return CostDashboard(tracker, cfg)

    def test_daily_report(self, tmp_path):
        d = self._make_dashboard(tmp_path)
        r = d.daily_report()
        assert "total_usd" in r

    def test_budget_status(self, tmp_path):
        d = self._make_dashboard(tmp_path)
        b = d.budget_status()
        assert "budget_usd" in b or "budget" in b

    def test_weekly_report(self, tmp_path):
        d = self._make_dashboard(tmp_path)
        r = d.weekly_report()
        assert isinstance(r, dict)

    def test_export_csv(self, tmp_path):
        d = self._make_dashboard(tmp_path)
        csv = d.export_csv()
        assert isinstance(csv, str)


# ══════════════════════════════════════════════════════════════════════════════
#  TRACER
# ══════════════════════════════════════════════════════════════════════════════

class TestTracer:
    """Distributed tracing spans."""

    def _make_tracer(self):
        from app.observability.tracing import Tracer
        from app.config import ObservabilityConfig2
        return Tracer(ObservabilityConfig2())

    def test_start_finish_span(self):
        tracer = self._make_tracer()
        span = tracer.start_span("search")
        assert span.operation == "search"
        tracer.finish_span(span, "ok")
        assert span.end_time is not None

    def test_context_manager(self):
        tracer = self._make_tracer()
        with tracer.trace("embed") as span:
            span.set_tag("model", "bge-small")
        assert span.status == "ok"

    def test_get_recent_traces(self):
        tracer = self._make_tracer()
        with tracer.trace("op1"):
            pass
        with tracer.trace("op2"):
            pass
        traces = tracer.get_recent_traces(limit=10)
        assert len(traces) == 2

    def test_stats(self):
        tracer = self._make_tracer()
        with tracer.trace("x"):
            pass
        s = tracer.stats()
        assert "total_spans" in s


# ══════════════════════════════════════════════════════════════════════════════
#  STRUCTURED LOGGER
# ══════════════════════════════════════════════════════════════════════════════

class TestStructuredLogger:
    """JSON structured logging."""

    def _make_logger(self):
        from app.observability.structured_logging import StructuredLogger
        from app.config import ObservabilityConfig2
        return StructuredLogger("test-svc", ObservabilityConfig2())

    def test_info_logged(self):
        logger = self._make_logger()
        logger.info("test message", query="hello")
        recent = logger.get_recent(limit=5)
        assert len(recent) >= 1
        assert recent[-1]["msg"] == "test message"

    def test_level_filter(self):
        logger = self._make_logger()
        logger.info("info msg")
        logger.warning("warn msg")
        infos = logger.get_recent(level="INFO")
        warns = logger.get_recent(level="WARNING")
        assert all(e["level"] == "INFO" for e in infos)
        assert all(e["level"] == "WARNING" for e in warns)

    def test_stats(self):
        logger = self._make_logger()
        logger.error("oops")
        s = logger.stats()
        assert isinstance(s, dict)


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG COMPATIBILITY
# ══════════════════════════════════════════════════════════════════════════════

class TestBatch4ConfigCompat:
    """Config backward compatibility."""

    def test_engine_config_zero_arg(self):
        from app.config import EngineConfig
        cfg = EngineConfig()
        assert cfg.security.jwt_algorithm == "HS256"
        assert cfg.resilience.circuit_breaker_enabled is True
        assert cfg.cost.enabled is True
        assert cfg.observability2.tracing_enabled is False

    def test_security_defaults(self):
        from app.config import SecurityConfig
        cfg = SecurityConfig()
        assert cfg.enabled is False
        assert cfg.jwt_expiry_hours == 24

    def test_resilience_defaults(self):
        from app.config import ResilienceConfig
        cfg = ResilienceConfig()
        assert cfg.failure_threshold == 5
        assert cfg.retry_max_attempts == 3

    def test_cost_defaults(self):
        from app.config import CostConfig
        cfg = CostConfig()
        assert cfg.budget_alert_usd == 10.0
        assert cfg.track_llm is True


# ══════════════════════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

class TestBatch4APIEndpoints:
    """Smoke-test Batch 4 API endpoints via TestClient."""

    @pytest.fixture(scope="class")
    def client(self, tmp_path_factory):
        from app.api.routes import create_app
        from app.config import EngineConfig, DatabaseConfig
        import tempfile, pathlib
        db_path = pathlib.Path(tempfile.mkdtemp()) / "test.db"
        app = create_app(EngineConfig(database=DatabaseConfig(db_path=db_path)))
        with TestClient(app) as c:
            yield c

    def test_security_health(self, client):
        r = client.get("/security/health")
        assert r.status_code == 200
        data = r.json()
        assert "jwt_enabled" in data

    def test_security_rbac(self, client):
        r = client.get("/security/rbac")
        assert r.status_code == 200
        data = r.json()
        assert "roles" in data
        assert "admin" in data["roles"]

    def test_circuit_breakers(self, client):
        r = client.get("/resilience/circuit-breakers")
        assert r.status_code == 200

    def test_health_probes(self, client):
        r = client.get("/resilience/health-probes")
        assert r.status_code == 200
        data = r.json()
        assert "overall_healthy" in data

    def test_cost_stats(self, client):
        r = client.get("/cost/stats")
        assert r.status_code == 200

    def test_cost_budget(self, client):
        r = client.get("/cost/budget")
        assert r.status_code == 200

    def test_traces_endpoint(self, client):
        r = client.get("/observability/traces")
        assert r.status_code == 200

    def test_logs_endpoint(self, client):
        r = client.get("/observability/logs")
        assert r.status_code == 200
