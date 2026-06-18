"""
Phase 8 Batch 5 Tests — Performance, Load Testing, CI/CD, K8s

Tests cover:
  - DistributedCacheLayer (L1 hit, L2 hit, miss, invalidate, stats)
  - BatchProcessor (batch flush, max_size trigger, manual flush, stats)
  - PerformanceOptimizer (profile context manager, profile_fn, dedup, summary)
  - BenchmarkResult (throughput, percentiles)
  - Config backward compatibility
  - CI/CD file existence
  - K8s manifest existence
  - API endpoint smoke tests

All tests use in-memory backends — no external services required.
"""

import os
import time
import pytest
from pathlib import Path
from fastapi.testclient import TestClient


# ══════════════════════════════════════════════════════════════════════════════
#  DISTRIBUTED CACHE LAYER
# ══════════════════════════════════════════════════════════════════════════════

class TestDistributedCacheLayer:
    """Multi-tier distributed cache."""

    def _make_cache(self):
        from app.performance.cache_layer import DistributedCacheLayer
        return DistributedCacheLayer(l1_capacity=5, l1_ttl=60, l2_ttl=300)

    def test_put_get_l1(self):
        cache = self._make_cache()
        cache.put("k1", {"data": [1, 2, 3]})
        assert cache.get("k1") == {"data": [1, 2, 3]}

    def test_cache_miss(self):
        cache = self._make_cache()
        assert cache.get("nonexistent") is None

    def test_invalidate(self):
        cache = self._make_cache()
        cache.put("k1", "value")
        cache.invalidate("k1")
        assert cache.get("k1") is None

    def test_lru_eviction(self):
        cache = self._make_cache()  # capacity=5
        for i in range(6):
            cache.put(f"k{i}", i)
        assert cache.get("k0") is None  # evicted
        assert cache.get("k5") == 5     # still present

    def test_clear(self):
        cache = self._make_cache()
        cache.put("k1", 1)
        cache.put("k2", 2)
        cache.clear()
        assert cache.get("k1") is None
        assert cache.get("k2") is None

    def test_stats(self):
        cache = self._make_cache()
        cache.put("k1", 1)
        cache.get("k1")  # hit
        cache.get("k2")  # miss
        s = cache.stats()
        assert s["l1_hits"] >= 1
        assert s["misses"] >= 1
        assert s["total_requests"] >= 2
        assert 0 <= s["overall_hit_rate"] <= 1

    def test_with_redis_l2(self):
        from app.redis.client import InMemoryRedisClient
        from app.performance.cache_layer import DistributedCacheLayer
        redis = InMemoryRedisClient()
        cache = DistributedCacheLayer(redis_client=redis, l1_capacity=2)
        cache.put("k1", {"x": 1})
        # Evict from L1 by filling it
        cache.put("k2", 2)
        cache.put("k3", 3)  # k1 evicted from L1
        # Should still find k1 in L2 (Redis)
        val = cache.get("k1")
        assert val == {"x": 1}
        s = cache.stats()
        assert s["l2_hits"] >= 1


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

class TestBatchProcessor:
    """Batch processing with size and time triggers."""

    def test_flush_at_max_size(self):
        from app.performance.batch_processor import BatchProcessor, BatchConfig
        collected = []
        bp = BatchProcessor(
            handler=lambda items: collected.extend(items),
            config=BatchConfig(max_size=3),
        )
        bp.add(1)
        bp.add(2)
        assert len(collected) == 0  # not yet
        bp.add(3)                   # triggers flush
        assert len(collected) == 3

    def test_manual_flush(self):
        from app.performance.batch_processor import BatchProcessor, BatchConfig
        collected = []
        bp = BatchProcessor(
            handler=lambda items: collected.extend(items),
            config=BatchConfig(max_size=100),
        )
        bp.add("a")
        bp.add("b")
        count = bp.flush()
        assert count == 2
        assert collected == ["a", "b"]

    def test_flush_empty(self):
        from app.performance.batch_processor import BatchProcessor
        bp = BatchProcessor(handler=lambda items: None)
        assert bp.flush() == 0

    def test_add_many(self):
        from app.performance.batch_processor import BatchProcessor, BatchConfig
        collected = []
        bp = BatchProcessor(
            handler=lambda items: collected.extend(items),
            config=BatchConfig(max_size=100),
        )
        bp.add_many([1, 2, 3, 4, 5])
        assert bp.pending() == 5
        bp.flush()
        assert len(collected) == 5

    def test_stats(self):
        from app.performance.batch_processor import BatchProcessor, BatchConfig
        bp = BatchProcessor(
            handler=lambda items: None,
            config=BatchConfig(max_size=2),
            name="test-batch",
        )
        bp.add(1)
        bp.add(2)  # triggers flush
        s = bp.stats()
        assert s["name"] == "test-batch"
        assert s["total_items"] == 2
        assert s["total_flushes"] == 1

    def test_start_stop(self):
        from app.performance.batch_processor import BatchProcessor, BatchConfig
        collected = []
        bp = BatchProcessor(
            handler=lambda items: collected.extend(items),
            config=BatchConfig(max_size=100, flush_interval=0.1),
        )
        bp.start()
        bp.add("item")
        time.sleep(0.3)  # wait for background flush
        bp.stop()
        assert "item" in collected

    def test_handler_failure_retry(self):
        from app.performance.batch_processor import BatchProcessor, BatchConfig
        attempts = []
        def flaky_handler(items):
            attempts.append(1)
            if len(attempts) < 2:
                raise RuntimeError("transient")
        bp = BatchProcessor(
            handler=flaky_handler,
            config=BatchConfig(max_size=100, max_retries=2),
        )
        bp.add(1)
        bp.flush()
        assert len(attempts) == 2  # retried once


# ══════════════════════════════════════════════════════════════════════════════
#  PERFORMANCE OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

class TestPerformanceOptimizer:
    """Performance profiling and optimization."""

    def _make_opt(self):
        from app.performance.optimizer import PerformanceOptimizer
        return PerformanceOptimizer()

    def test_profile_context_manager(self):
        opt = self._make_opt()
        with opt.profile("search") as ctx:
            _ = sum(range(1000))
        assert ctx.result is not None
        assert ctx.result.success is True
        assert ctx.result.duration_ms >= 0

    def test_profile_fn(self):
        opt = self._make_opt()
        result = opt.profile_fn("compute", lambda: 42)
        assert result == 42

    def test_profile_error(self):
        opt = self._make_opt()
        with pytest.raises(ValueError):
            opt.profile_fn("fail", lambda: (_ for _ in ()).throw(ValueError("x")))
        s = opt.summary()
        ops = s["operations"]
        assert "fail" in ops

    def test_dedup_query(self):
        opt = self._make_opt()
        assert opt.dedup_query("q1") is True   # new
        assert opt.dedup_query("q1") is False  # duplicate
        opt.complete_query("q1")
        assert opt.dedup_query("q1") is True   # new again

    def test_summary(self):
        opt = self._make_opt()
        with opt.profile("op1"):
            pass
        with opt.profile("op2"):
            pass
        s = opt.summary()
        assert s["total_profiled"] == 2
        assert "op1" in s["operations"]
        assert "op2" in s["operations"]
        assert "mean_ms" in s["operations"]["op1"]

    def test_get_history(self):
        opt = self._make_opt()
        with opt.profile("a"):
            pass
        with opt.profile("b"):
            pass
        h = opt.get_history(name="a")
        assert len(h) == 1
        assert h[0]["name"] == "a"


# ══════════════════════════════════════════════════════════════════════════════
#  BENCHMARK RESULT
# ══════════════════════════════════════════════════════════════════════════════

class TestBenchmarkResult:
    """Benchmark result data model."""

    def test_throughput(self):
        from load_tests.benchmark import BenchmarkResult
        r = BenchmarkResult(endpoint="/search")
        r.requests_total = 100
        r.successes = 95
        r.failures = 5
        r.latencies_ms = [10.0] * 100  # 10ms each = 1s total
        assert r.throughput_rps == pytest.approx(100.0, rel=0.01)

    def test_percentiles(self):
        from load_tests.benchmark import BenchmarkResult
        r = BenchmarkResult(endpoint="/test")
        r.latencies_ms = list(range(1, 101))  # 1ms to 100ms
        r.requests_total = 100
        r.successes = 100
        assert r.p50 == 50.0 or r.p50 == 51.0  # median
        assert r.p95 >= 95.0
        assert r.p99 >= 99.0

    def test_to_dict(self):
        from load_tests.benchmark import BenchmarkResult
        r = BenchmarkResult(endpoint="/test")
        r.latencies_ms = [5.0, 10.0]
        r.requests_total = 2
        r.successes = 2
        d = r.to_dict()
        assert d["endpoint"] == "/test"
        assert "throughput_rps" in d
        assert "latency_p50_ms" in d

    def test_empty_latencies(self):
        from load_tests.benchmark import BenchmarkResult
        r = BenchmarkResult(endpoint="/empty")
        assert r.p50 == 0.0
        assert r.throughput_rps == 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  FILE EXISTENCE CHECKS
# ══════════════════════════════════════════════════════════════════════════════

class TestFileExistence:
    """Verify CI/CD and K8s files exist."""

    PROJECT_ROOT = Path(__file__).parent.parent

    def test_ci_pipeline_exists(self):
        assert (self.PROJECT_ROOT / ".github" / "workflows" / "ci.yml").exists()

    def test_deploy_pipeline_exists(self):
        assert (self.PROJECT_ROOT / ".github" / "workflows" / "deploy.yml").exists()

    def test_k8s_namespace(self):
        assert (self.PROJECT_ROOT / "k8s" / "namespace.yml").exists()

    def test_k8s_app_deployment(self):
        assert (self.PROJECT_ROOT / "k8s" / "app-deployment.yml").exists()

    def test_k8s_app_service(self):
        assert (self.PROJECT_ROOT / "k8s" / "app-service.yml").exists()

    def test_k8s_hpa(self):
        assert (self.PROJECT_ROOT / "k8s" / "app-hpa.yml").exists()

    def test_k8s_ingress(self):
        assert (self.PROJECT_ROOT / "k8s" / "ingress.yml").exists()

    def test_k8s_configmap(self):
        assert (self.PROJECT_ROOT / "k8s" / "configmap.yml").exists()

    def test_k8s_secrets(self):
        assert (self.PROJECT_ROOT / "k8s" / "secrets.yml").exists()

    def test_dockerfile_exists(self):
        assert (self.PROJECT_ROOT / "Dockerfile").exists()

    def test_docker_compose_exists(self):
        assert (self.PROJECT_ROOT / "docker-compose.yml").exists()

    def test_locustfile_exists(self):
        assert (self.PROJECT_ROOT / "load_tests" / "locustfile.py").exists()


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG COMPATIBILITY
# ══════════════════════════════════════════════════════════════════════════════

class TestBatch5ConfigCompat:
    """Config backward compatibility — final check."""

    def test_engine_config_zero_arg(self):
        from app.config import EngineConfig
        cfg = EngineConfig()
        # Phase 1-7 defaults
        assert cfg.database.backend == "sqlite"
        assert cfg.bm25.k1 == 1.5
        assert cfg.rag.llm.provider == "mock"
        assert cfg.research.agent.max_retries == 3
        # Phase 8 Batch 1
        assert cfg.events.backend == "memory"
        assert cfg.redis.host == "localhost"
        # Phase 8 Batch 2
        assert cfg.kafka.bootstrap_servers == "localhost:9092"
        assert cfg.qdrant.collection_name == "search_engine"
        # Phase 8 Batch 3
        assert cfg.tenancy.enabled is False
        assert cfg.agent_execution.max_workers == 8
        # Phase 8 Batch 4
        assert cfg.security.enabled is False
        assert cfg.resilience.circuit_breaker_enabled is True
        assert cfg.cost.enabled is True


# ══════════════════════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

class TestBatch5APIEndpoints:
    """Batch 5 API endpoint smoke tests."""

    @pytest.fixture(scope="class")
    def client(self, tmp_path_factory):
        from app.api.routes import create_app
        from app.config import EngineConfig, DatabaseConfig
        import tempfile, pathlib
        db_path = pathlib.Path(tempfile.mkdtemp()) / "test.db"
        app = create_app(EngineConfig(database=DatabaseConfig(db_path=db_path)))
        with TestClient(app) as c:
            yield c

    def test_perf_cache_stats(self, client):
        r = client.get("/performance/cache/stats")
        assert r.status_code == 200
        assert "l1_hits" in r.json()

    def test_perf_profiling(self, client):
        r = client.get("/performance/profiling")
        assert r.status_code == 200
        assert "total_profiled" in r.json()

    def test_perf_history(self, client):
        r = client.get("/performance/profiling/history")
        assert r.status_code == 200
        assert "history" in r.json()
