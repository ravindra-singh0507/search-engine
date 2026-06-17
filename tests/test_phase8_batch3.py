"""
Phase 8 Batch 3 Test Suite — Platform Services, Multi-Tenancy, Distributed Agents & Workflows

Tests cover:
  - ServiceRegistry (register, deregister, heartbeat, get_instances, get_healthy,
    cleanup_stale, stats, register_duplicate/max capacity)
  - HealthCheck (liveness, readiness all pass, readiness one fails, add_check, is_ready)
  - ServiceDiscovery (discover, round-robin, no instances, discover_all)
  - Tenant models (Tenant creation, TenantQuotas defaults, TenantUsage defaults)
  - TenantManager (create, get, list, delete, suspend, get_usage, check_quota, increment_usage)
  - TenantContext (set/get, clear, require raises, tenant_scope context manager, thread isolation)
  - TenantIsolation (scope_key, validate_access, get_tenant_prefix)
  - AgentTaskQueue (enqueue/dequeue, priority ordering, max size, is_empty, peek,
    cancel, get_by_id, stats)
  - AgentScheduler (submit, schedule_next, complete, fail, stats)
  - AgentWorkerPool (creation, get_workers, stats, start/stop)
  - DistributedAgentExecutor (creation, setup, stats, is_running)
  - WorkflowExecution (creation, progress, is_terminal, to_dict)
  - InMemoryCheckpointStore (save/load, delete, list, bounded size)
  - WorkflowScheduler (add_schedule, remove, list, stats)
  - DistributedWorkflowEngine (creation, list_executions, recover, stats)
  - ExecutionTracker (record_start, record_complete, get_recent, stats)
  - Config compatibility (EngineConfig zero-arg, batch 3 config defaults)
  - API endpoints (services list, services health, tenants list,
    agent distributed stats, workflow distributed stats)
"""

import threading
import time
import uuid

import pytest
from unittest.mock import MagicMock


# ==============================================================================
#  SERVICE REGISTRY TESTS
# ==============================================================================


class TestServiceRegistry:
    """Service registry: register, deregister, heartbeat, lookup, cleanup."""

    def test_register(self):
        from app.config import ServiceRegistryConfig
        from app.services.registry import ServiceRegistry

        config = ServiceRegistryConfig()
        registry = ServiceRegistry(config)
        instance_id = registry.register("indexer", "localhost", 8000)
        assert isinstance(instance_id, str)
        assert len(instance_id) == 36  # UUID format

    def test_deregister(self):
        from app.config import ServiceRegistryConfig
        from app.services.registry import ServiceRegistry

        config = ServiceRegistryConfig()
        registry = ServiceRegistry(config)
        iid = registry.register("indexer", "localhost", 8000)
        assert registry.deregister(iid) is True
        assert registry.deregister(iid) is False  # already removed

    def test_heartbeat(self):
        from app.config import ServiceRegistryConfig
        from app.services.registry import ServiceRegistry, ServiceStatus

        config = ServiceRegistryConfig()
        registry = ServiceRegistry(config)
        iid = registry.register("indexer", "localhost", 8000)
        # heartbeat should not raise
        registry.heartbeat(iid)
        instances = registry.get_instances("indexer")
        assert instances[0].status == ServiceStatus.HEALTHY

    def test_get_instances(self):
        from app.config import ServiceRegistryConfig
        from app.services.registry import ServiceRegistry

        config = ServiceRegistryConfig()
        registry = ServiceRegistry(config)
        registry.register("svc-a", "host1", 8001)
        registry.register("svc-a", "host2", 8002)
        registry.register("svc-b", "host3", 8003)
        assert len(registry.get_instances("svc-a")) == 2
        assert len(registry.get_instances("svc-b")) == 1
        assert len(registry.get_instances("unknown")) == 0

    def test_get_healthy(self):
        from app.config import ServiceRegistryConfig
        from app.services.registry import ServiceRegistry

        config = ServiceRegistryConfig()
        registry = ServiceRegistry(config)
        iid1 = registry.register("svc", "h1", 8001)
        iid2 = registry.register("svc", "h2", 8002)
        registry.mark_unhealthy(iid2)
        healthy = registry.get_healthy("svc")
        assert len(healthy) == 1
        assert healthy[0].instance_id == iid1

    def test_cleanup_stale(self):
        from app.config import ServiceRegistryConfig
        from app.services.registry import ServiceRegistry

        config = ServiceRegistryConfig(stale_threshold_sec=0.01)
        registry = ServiceRegistry(config)
        registry.register("svc", "h1", 8001)
        time.sleep(0.05)
        removed = registry.cleanup_stale()
        assert removed == 1
        assert len(registry.get_instances("svc")) == 0

    def test_stats(self):
        from app.config import ServiceRegistryConfig
        from app.services.registry import ServiceRegistry

        config = ServiceRegistryConfig()
        registry = ServiceRegistry(config)
        registry.register("svc-a", "h1", 8001)
        registry.register("svc-b", "h2", 8002)
        s = registry.stats()
        assert s["total_instances"] == 2
        assert s["total_services"] == 2
        assert s["healthy_count"] == 2

    def test_register_duplicate(self):
        from app.config import ServiceRegistryConfig
        from app.services.registry import ServiceRegistry

        config = ServiceRegistryConfig(max_instances=2)
        registry = ServiceRegistry(config)
        registry.register("svc", "h1", 8001)
        registry.register("svc", "h2", 8002)
        with pytest.raises(ValueError, match="Registry full"):
            registry.register("svc", "h3", 8003)


# ==============================================================================
#  HEALTH CHECK TESTS
# ==============================================================================


class TestHealthCheck:
    """Health check system: liveness, readiness, check registration."""

    def test_liveness(self):
        from app.services.health import HealthCheck

        hc = HealthCheck()
        result = hc.liveness()
        assert result["status"] == "alive"
        assert "uptime_seconds" in result
        assert result["uptime_seconds"] >= 0

    def test_readiness_all_pass(self):
        from app.services.health import HealthCheck

        hc = HealthCheck()
        hc.add_check("db", lambda: True)
        hc.add_check("cache", lambda: True)
        result = hc.readiness()
        assert result["status"] == "ready"
        assert result["healthy_checks"] == 2
        assert result["total_checks"] == 2

    def test_readiness_one_fails(self):
        from app.services.health import HealthCheck

        hc = HealthCheck()
        hc.add_check("db", lambda: True)
        hc.add_check("broken", lambda: False)
        result = hc.readiness()
        assert result["status"] == "not_ready"
        assert result["healthy_checks"] == 1
        assert result["total_checks"] == 2

    def test_add_check(self):
        from app.services.health import HealthCheck

        hc = HealthCheck()
        hc.add_check("test_check", lambda: True)
        checks = hc.run_checks()
        assert "test_check" in checks
        assert checks["test_check"] is True

    def test_is_ready(self):
        from app.services.health import HealthCheck

        hc = HealthCheck()
        assert hc.is_ready() is True  # no checks = ready
        hc.add_check("ok", lambda: True)
        assert hc.is_ready() is True
        hc.add_check("fail", lambda: False)
        assert hc.is_ready() is False


# ==============================================================================
#  SERVICE DISCOVERY TESTS
# ==============================================================================


class TestServiceDiscovery:
    """Client-side service discovery with round-robin load balancing."""

    def test_discover(self):
        from app.config import ServiceRegistryConfig
        from app.services.registry import ServiceRegistry
        from app.services.discovery import ServiceDiscovery

        config = ServiceRegistryConfig()
        registry = ServiceRegistry(config)
        registry.register("svc", "h1", 8001)
        discovery = ServiceDiscovery(registry)
        instance = discovery.discover("svc")
        assert instance is not None
        assert instance.host == "h1"
        assert instance.port == 8001

    def test_discover_round_robin(self):
        from app.config import ServiceRegistryConfig
        from app.services.registry import ServiceRegistry
        from app.services.discovery import ServiceDiscovery

        config = ServiceRegistryConfig()
        registry = ServiceRegistry(config)
        registry.register("svc", "h1", 8001)
        registry.register("svc", "h2", 8002)
        discovery = ServiceDiscovery(registry)
        first = discovery.discover("svc")
        second = discovery.discover("svc")
        # Round-robin should alternate between instances
        assert first.instance_id != second.instance_id

    def test_discover_no_instances(self):
        from app.config import ServiceRegistryConfig
        from app.services.registry import ServiceRegistry
        from app.services.discovery import ServiceDiscovery

        config = ServiceRegistryConfig()
        registry = ServiceRegistry(config)
        discovery = ServiceDiscovery(registry)
        assert discovery.discover("nonexistent") is None

    def test_discover_all(self):
        from app.config import ServiceRegistryConfig
        from app.services.registry import ServiceRegistry
        from app.services.discovery import ServiceDiscovery

        config = ServiceRegistryConfig()
        registry = ServiceRegistry(config)
        registry.register("svc", "h1", 8001)
        registry.register("svc", "h2", 8002)
        discovery = ServiceDiscovery(registry)
        all_instances = discovery.discover_all("svc")
        assert len(all_instances) == 2


# ==============================================================================
#  TENANT MODELS TESTS
# ==============================================================================


class TestTenantModels:
    """Tenant data models: Tenant, TenantQuotas, TenantUsage."""

    def test_tenant_creation(self):
        from app.tenancy.models import Tenant

        t = Tenant(tenant_id="acme", name="Acme Corp")
        assert t.tenant_id == "acme"
        assert t.name == "Acme Corp"
        assert t.status == "active"
        assert t.is_active() is True
        assert t.created_at > 0

    def test_quotas_defaults(self):
        from app.tenancy.models import TenantQuotas

        q = TenantQuotas()
        assert q.max_documents == 100000
        assert q.max_sessions == 1000
        assert q.max_agents == 50
        assert q.max_queries_per_minute == 120
        assert q.max_storage_mb == 10000

    def test_usage_defaults(self):
        from app.tenancy.models import TenantUsage

        u = TenantUsage(tenant_id="test")
        assert u.tenant_id == "test"
        assert u.document_count == 0
        assert u.session_count == 0
        assert u.agent_count == 0
        assert u.queries_today == 0
        assert u.storage_mb == 0.0


# ==============================================================================
#  TENANT MANAGER TESTS
# ==============================================================================


class TestTenantManager:
    """Tenant lifecycle management: create, get, list, delete, suspend, quota."""

    def test_create_tenant(self):
        from app.config import TenancyConfig
        from app.tenancy.manager import TenantManager

        config = TenancyConfig()
        mgr = TenantManager(config)
        t = mgr.create_tenant("acme", "Acme Corp")
        assert t.tenant_id == "acme"
        assert t.name == "Acme Corp"
        assert t.status == "active"

    def test_get_tenant(self):
        from app.config import TenancyConfig
        from app.tenancy.manager import TenantManager

        config = TenancyConfig()
        mgr = TenantManager(config)
        mgr.create_tenant("acme", "Acme Corp")
        t = mgr.get_tenant("acme")
        assert t is not None
        assert t.name == "Acme Corp"
        assert mgr.get_tenant("nonexistent") is None

    def test_list_tenants(self):
        from app.config import TenancyConfig
        from app.tenancy.manager import TenantManager

        config = TenancyConfig()
        mgr = TenantManager(config)
        mgr.create_tenant("a", "Tenant A")
        mgr.create_tenant("b", "Tenant B")
        all_tenants = mgr.list_tenants()
        assert len(all_tenants) == 2

    def test_delete_tenant(self):
        from app.config import TenancyConfig
        from app.tenancy.manager import TenantManager

        config = TenancyConfig()
        mgr = TenantManager(config)
        mgr.create_tenant("acme", "Acme Corp")
        assert mgr.delete_tenant("acme") is True
        t = mgr.get_tenant("acme")
        assert t.status == "deleted"
        assert mgr.delete_tenant("nonexistent") is False

    def test_suspend_tenant(self):
        from app.config import TenancyConfig
        from app.tenancy.manager import TenantManager

        config = TenancyConfig()
        mgr = TenantManager(config)
        mgr.create_tenant("acme", "Acme Corp")
        assert mgr.suspend_tenant("acme") is True
        t = mgr.get_tenant("acme")
        assert t.status == "suspended"
        assert mgr.suspend_tenant("nonexistent") is False

    def test_get_usage(self):
        from app.config import TenancyConfig
        from app.tenancy.manager import TenantManager

        config = TenancyConfig()
        mgr = TenantManager(config)
        mgr.create_tenant("acme", "Acme Corp")
        usage = mgr.get_usage("acme")
        assert usage.tenant_id == "acme"
        assert usage.document_count == 0

    def test_check_quota(self):
        from app.config import TenancyConfig
        from app.tenancy.manager import TenantManager

        config = TenancyConfig()
        mgr = TenantManager(config)
        mgr.create_tenant("acme", "Acme Corp", quotas={"max_documents": 2})
        assert mgr.check_quota("acme", "documents") is True
        mgr.increment_usage("acme", "documents", 2)
        assert mgr.check_quota("acme", "documents") is False

    def test_increment_usage(self):
        from app.config import TenancyConfig
        from app.tenancy.manager import TenantManager

        config = TenancyConfig()
        mgr = TenantManager(config)
        mgr.create_tenant("acme", "Acme Corp")
        mgr.increment_usage("acme", "documents", 5)
        usage = mgr.get_usage("acme")
        assert usage.document_count == 5
        mgr.increment_usage("acme", "queries", 3)
        usage = mgr.get_usage("acme")
        assert usage.queries_today == 3


# ==============================================================================
#  TENANT CONTEXT TESTS
# ==============================================================================


class TestTenantContext:
    """Thread-local tenant context: set, get, clear, require, scope."""

    def test_set_get(self):
        from app.tenancy.context import TenantContext

        TenantContext.set("acme")
        assert TenantContext.get() == "acme"
        TenantContext.clear()

    def test_clear(self):
        from app.tenancy.context import TenantContext

        TenantContext.set("acme")
        TenantContext.clear()
        assert TenantContext.get() is None

    def test_require_raises(self):
        from app.tenancy.context import TenantContext

        TenantContext.clear()
        with pytest.raises(RuntimeError, match="No tenant context set"):
            TenantContext.require()

    def test_tenant_scope_context_manager(self):
        from app.tenancy.context import TenantContext, tenant_scope

        TenantContext.clear()
        with tenant_scope("acme") as tid:
            assert tid == "acme"
            assert TenantContext.get() == "acme"
        assert TenantContext.get() is None

    def test_thread_isolation(self):
        from app.tenancy.context import TenantContext

        results = {}

        def set_and_read(name, value):
            TenantContext.set(value)
            time.sleep(0.01)
            results[name] = TenantContext.get()

        t1 = threading.Thread(target=set_and_read, args=("t1", "tenant-a"))
        t2 = threading.Thread(target=set_and_read, args=("t2", "tenant-b"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["t1"] == "tenant-a"
        assert results["t2"] == "tenant-b"


# ==============================================================================
#  TENANT ISOLATION TESTS
# ==============================================================================


class TestTenantIsolation:
    """Tenant data isolation: key scoping, access validation, prefix generation."""

    def test_scope_key(self):
        from app.config import TenancyConfig
        from app.tenancy.manager import TenantManager
        from app.tenancy.isolation import TenantIsolation

        config = TenancyConfig()
        mgr = TenantManager(config)
        isolation = TenantIsolation(mgr)
        scoped = isolation.scope_key("documents:123", tenant_id="acme")
        assert scoped == "tenant:acme:documents:123"

    def test_validate_access(self):
        from app.config import TenancyConfig
        from app.tenancy.manager import TenantManager
        from app.tenancy.isolation import TenantIsolation

        config = TenancyConfig()
        mgr = TenantManager(config)
        isolation = TenantIsolation(mgr)
        assert isolation.validate_access("acme", "acme") is True
        assert isolation.validate_access("acme", "other") is False

    def test_get_tenant_prefix(self):
        from app.config import TenancyConfig
        from app.tenancy.manager import TenantManager
        from app.tenancy.isolation import TenantIsolation

        config = TenancyConfig()
        mgr = TenantManager(config)
        isolation = TenantIsolation(mgr)
        prefix = isolation.get_tenant_prefix(tenant_id="acme")
        assert prefix == "tenant:acme:"
        # With no tenant should return empty
        assert isolation.get_tenant_prefix(tenant_id=None) == ""


# ==============================================================================
#  AGENT TASK QUEUE TESTS
# ==============================================================================


class TestAgentTaskQueue:
    """Agent task queue: priority-based enqueue/dequeue, cancellation, stats."""

    def _make_task(self, goal="test"):
        from app.agents.base import AgentTask
        return AgentTask(goal=goal)

    def test_enqueue_dequeue(self):
        from app.distributed.agents.queue import AgentTaskQueue

        q = AgentTaskQueue(max_size=10)
        task = self._make_task("do stuff")
        assert q.enqueue(task, priority=5) is True
        dequeued = q.dequeue()
        assert dequeued is not None
        assert dequeued.goal == "do stuff"

    def test_priority_ordering(self):
        from app.distributed.agents.queue import AgentTaskQueue

        q = AgentTaskQueue(max_size=10)
        low = self._make_task("low")
        high = self._make_task("high")
        q.enqueue(low, priority=1)
        q.enqueue(high, priority=10)
        first = q.dequeue()
        assert first.goal == "high"
        second = q.dequeue()
        assert second.goal == "low"

    def test_max_size(self):
        from app.distributed.agents.queue import AgentTaskQueue

        q = AgentTaskQueue(max_size=2)
        assert q.enqueue(self._make_task("a"), priority=5) is True
        assert q.enqueue(self._make_task("b"), priority=5) is True
        assert q.enqueue(self._make_task("c"), priority=5) is False  # full

    def test_is_empty(self):
        from app.distributed.agents.queue import AgentTaskQueue

        q = AgentTaskQueue(max_size=10)
        assert q.is_empty() is True
        q.enqueue(self._make_task(), priority=5)
        assert q.is_empty() is False

    def test_peek(self):
        from app.distributed.agents.queue import AgentTaskQueue

        q = AgentTaskQueue(max_size=10)
        task = self._make_task("peek-me")
        q.enqueue(task, priority=5)
        peeked = q.peek()
        assert peeked is not None
        assert peeked.goal == "peek-me"
        # peek should not remove the task
        assert q.is_empty() is False

    def test_cancel(self):
        from app.distributed.agents.queue import AgentTaskQueue

        q = AgentTaskQueue(max_size=10)
        task = self._make_task("cancel-me")
        q.enqueue(task, priority=5)
        assert q.cancel(task.task_id) is True
        assert q.dequeue() is None  # cancelled, nothing left

    def test_get_by_id(self):
        from app.distributed.agents.queue import AgentTaskQueue

        q = AgentTaskQueue(max_size=10)
        task = self._make_task("find-me")
        q.enqueue(task, priority=5)
        found = q.get_by_id(task.task_id)
        assert found is not None
        assert found.goal == "find-me"
        assert q.get_by_id("nonexistent") is None

    def test_stats(self):
        from app.distributed.agents.queue import AgentTaskQueue

        q = AgentTaskQueue(max_size=100)
        q.enqueue(self._make_task(), priority=5)
        q.enqueue(self._make_task(), priority=5)
        q.dequeue()
        s = q.stats()
        assert s["enqueued"] == 2
        assert s["dequeued"] == 1
        assert s["current_size"] == 1
        assert s["max_size"] == 100


# ==============================================================================
#  AGENT SCHEDULER TESTS
# ==============================================================================


class TestAgentScheduler:
    """Agent scheduler: submit, schedule_next, complete, fail, stats."""

    def _make_task(self, goal="test"):
        from app.agents.base import AgentTask, TaskPriority
        return AgentTask(goal=goal, priority=TaskPriority.NORMAL)

    def _make_scheduler(self):
        from app.config import AgentExecutionConfig
        from app.distributed.agents.queue import AgentTaskQueue
        from app.distributed.agents.scheduler import AgentScheduler

        config = AgentExecutionConfig()
        queue = AgentTaskQueue(max_size=100)
        return AgentScheduler(config=config, queue=queue)

    def test_submit(self):
        scheduler = self._make_scheduler()
        task = self._make_task("plan research")
        task_id = scheduler.submit(task)
        assert task_id == task.task_id

    def test_schedule_next(self):
        scheduler = self._make_scheduler()
        task = self._make_task("research")
        scheduler.submit(task)
        assigned = scheduler.schedule_next("worker-1")
        assert assigned is not None
        assert assigned.task_id == task.task_id

    def test_complete(self):
        from app.agents.base import AgentResult, AgentStatus, AgentType

        scheduler = self._make_scheduler()
        task = self._make_task("research")
        scheduler.submit(task)
        scheduler.schedule_next("worker-1")
        result = AgentResult(
            task_id=task.task_id,
            agent_type=AgentType.RETRIEVAL,
            status=AgentStatus.DONE,
            output="done",
        )
        scheduler.complete(task.task_id, result)
        s = scheduler.stats()
        assert s["total_completed"] == 1

    def test_fail(self):
        scheduler = self._make_scheduler()
        task = self._make_task("will-fail")
        scheduler.submit(task)
        scheduler.schedule_next("worker-1")
        scheduler.fail(task.task_id, "timeout error")
        s = scheduler.stats()
        assert s["total_failed"] == 1

    def test_stats(self):
        scheduler = self._make_scheduler()
        s = scheduler.stats()
        assert "strategy" in s
        assert s["total_submitted"] == 0
        assert s["total_completed"] == 0
        assert s["total_failed"] == 0
        assert s["pending_count"] == 0
        assert s["running_count"] == 0


# ==============================================================================
#  AGENT WORKER POOL TESTS
# ==============================================================================


class TestAgentWorkerPool:
    """Agent worker pool: creation, worker listing, stats, start/stop lifecycle."""

    def _make_pool(self):
        from app.agents.base import AgentContext, AgentType
        from app.config import AgentExecutionConfig
        from app.distributed.agents.worker_pool import AgentWorkerPool

        config = AgentExecutionConfig(max_workers=2, max_queue_size=10)
        agents = {}  # no agents needed for structural tests
        context = AgentContext()
        return AgentWorkerPool(config=config, agents=agents, context=context)

    def test_creation(self):
        pool = self._make_pool()
        assert pool is not None

    def test_get_workers(self):
        pool = self._make_pool()
        pool.start()
        workers = pool.get_workers()
        assert len(workers) == 2
        for w in workers:
            assert "worker_id" in w
            assert "state" in w
        pool.stop(graceful=False)

    def test_stats(self):
        pool = self._make_pool()
        pool.start()
        s = pool.stats()
        assert s["total_workers"] == 2
        assert s["is_running"] is True
        assert "queue_size" in s
        pool.stop(graceful=False)

    def test_start_stop(self):
        pool = self._make_pool()
        pool.start()
        assert pool.stats()["is_running"] is True
        pool.stop(graceful=False)
        assert pool.stats()["is_running"] is False


# ==============================================================================
#  DISTRIBUTED AGENT EXECUTOR TESTS
# ==============================================================================


class TestDistributedAgentExecutor:
    """Distributed agent executor: creation, setup, stats, running state."""

    def _make_executor(self):
        from app.agents.base import AgentContext, AgentType
        from app.config import AgentExecutionConfig
        from app.distributed.agents.executor import DistributedAgentExecutor

        config = AgentExecutionConfig(max_workers=2, max_queue_size=10)
        agents = {}
        context = AgentContext()
        return DistributedAgentExecutor(
            config=config, agents=agents, context=context,
        )

    def test_creation(self):
        executor = self._make_executor()
        assert executor is not None
        assert executor.is_running() is False

    def test_setup(self):
        executor = self._make_executor()
        executor.setup()
        s = executor.stats()
        assert s["running"] is False
        assert "pool" in s
        assert "scheduler" in s

    def test_stats(self):
        executor = self._make_executor()
        s = executor.stats()
        assert "running" in s
        assert s["running"] is False

    def test_is_running(self):
        executor = self._make_executor()
        assert executor.is_running() is False
        executor.start()
        assert executor.is_running() is True
        executor.stop()
        assert executor.is_running() is False


# ==============================================================================
#  WORKFLOW EXECUTION STATE TESTS
# ==============================================================================


class TestWorkflowExecution:
    """Workflow execution state: creation, progress, terminal check, serialization."""

    def test_creation(self):
        from app.distributed.workflows.state import WorkflowExecution, WorkflowState

        ex = WorkflowExecution(workflow_name="research", goal="find papers")
        assert ex.workflow_name == "research"
        assert ex.goal == "find papers"
        assert ex.state == WorkflowState.CREATED
        assert len(ex.execution_id) == 36

    def test_progress(self):
        from app.distributed.workflows.state import WorkflowExecution

        ex = WorkflowExecution(steps=["s1", "s2", "s3", "s4"])
        assert ex.progress() == 0.0
        ex.completed_steps = ["s1", "s2"]
        assert ex.progress() == 0.5

    def test_is_terminal(self):
        from app.distributed.workflows.state import WorkflowExecution, WorkflowState

        ex = WorkflowExecution()
        assert ex.is_terminal() is False  # CREATED is not terminal
        ex.state = WorkflowState.COMPLETED
        assert ex.is_terminal() is True
        ex.state = WorkflowState.FAILED
        assert ex.is_terminal() is True
        ex.state = WorkflowState.CANCELLED
        assert ex.is_terminal() is True

    def test_to_dict(self):
        from app.distributed.workflows.state import WorkflowExecution

        ex = WorkflowExecution(workflow_name="test", goal="g")
        d = ex.to_dict()
        assert d["workflow_name"] == "test"
        assert d["goal"] == "g"
        assert d["state"] == "created"
        assert "execution_id" in d
        assert "completed_steps" in d


# ==============================================================================
#  CHECKPOINT STORE TESTS
# ==============================================================================


class TestInMemoryCheckpointStore:
    """In-memory checkpoint store: save/load, delete, list, eviction."""

    def test_save_load(self):
        from app.distributed.workflows.checkpoint import InMemoryCheckpointStore

        store = InMemoryCheckpointStore(max_size=10)
        store.save("exec-1", {"step": 3, "data": "hello"})
        loaded = store.load("exec-1")
        assert loaded is not None
        assert loaded["step"] == 3
        assert loaded["data"] == "hello"
        assert store.load("nonexistent") is None

    def test_delete(self):
        from app.distributed.workflows.checkpoint import InMemoryCheckpointStore

        store = InMemoryCheckpointStore(max_size=10)
        store.save("exec-1", {"step": 1})
        store.delete("exec-1")
        assert store.load("exec-1") is None

    def test_list(self):
        from app.distributed.workflows.checkpoint import InMemoryCheckpointStore

        store = InMemoryCheckpointStore(max_size=10)
        store.save("exec-1", {"step": 1})
        store.save("exec-2", {"step": 2})
        ids = store.list_checkpoints()
        assert set(ids) == {"exec-1", "exec-2"}

    def test_bounded_size(self):
        from app.distributed.workflows.checkpoint import InMemoryCheckpointStore

        store = InMemoryCheckpointStore(max_size=3)
        store.save("a", {"v": 1})
        store.save("b", {"v": 2})
        store.save("c", {"v": 3})
        store.save("d", {"v": 4})  # should evict "a"
        assert store.load("a") is None
        assert store.load("d") is not None
        assert len(store) == 3


# ==============================================================================
#  WORKFLOW SCHEDULER TESTS
# ==============================================================================


class TestWorkflowScheduler:
    """Workflow scheduler: add schedule, remove, list, stats."""

    def _make_scheduler(self):
        from app.config import DistributedWorkflowConfig
        from app.distributed.workflows.scheduler import WorkflowScheduler

        config = DistributedWorkflowConfig()
        return WorkflowScheduler(config)

    def test_add_schedule(self):
        scheduler = self._make_scheduler()
        sid = scheduler.add_schedule("research", interval_seconds=3600)
        assert isinstance(sid, str)
        assert len(sid) == 36

    def test_remove(self):
        scheduler = self._make_scheduler()
        sid = scheduler.add_schedule("research", interval_seconds=60)
        assert scheduler.remove_schedule(sid) is True
        assert scheduler.remove_schedule(sid) is False  # already removed

    def test_list(self):
        scheduler = self._make_scheduler()
        scheduler.add_schedule("wf1", interval_seconds=60)
        scheduler.add_schedule("wf2", interval_seconds=120)
        schedules = scheduler.list_schedules()
        assert len(schedules) == 2

    def test_stats(self):
        scheduler = self._make_scheduler()
        scheduler.add_schedule("wf1", interval_seconds=60)
        s = scheduler.stats()
        assert s["total_schedules"] == 1
        assert s["enabled_schedules"] == 1
        assert s["total_executions"] == 0


# ==============================================================================
#  DISTRIBUTED WORKFLOW ENGINE TESTS
# ==============================================================================


class TestDistributedWorkflowEngine:
    """Distributed workflow engine: creation, list executions, recover, stats."""

    def _make_engine(self):
        from app.config import DistributedWorkflowConfig
        from app.distributed.workflows.engine import DistributedWorkflowEngine

        config = DistributedWorkflowConfig(checkpoint_enabled=True)
        return DistributedWorkflowEngine(config=config)

    def test_creation(self):
        engine = self._make_engine()
        assert engine is not None

    def test_list_executions(self):
        engine = self._make_engine()
        execs = engine.list_executions()
        assert isinstance(execs, list)
        assert len(execs) == 0

    def test_recover(self):
        engine = self._make_engine()
        recovered = engine.recover()
        assert recovered == 0  # no checkpoints to recover

    def test_stats(self):
        engine = self._make_engine()
        s = engine.stats()
        assert s["active"] == 0
        assert s["completed"] == 0
        assert s["failed"] == 0
        assert s["total_started"] == 0
        assert "scheduler" in s
        assert "tracker" in s


# ==============================================================================
#  EXECUTION TRACKER TESTS
# ==============================================================================


class TestExecutionTracker:
    """Execution tracker: record events, get recent, stats."""

    def _make_tracker(self):
        from app.distributed.workflows.tracker import ExecutionTracker
        return ExecutionTracker(max_history=100)

    def test_record_start(self):
        from app.distributed.workflows.state import WorkflowExecution

        tracker = self._make_tracker()
        ex = WorkflowExecution(workflow_name="test", goal="g", steps=["s1"])
        tracker.record_start(ex)
        history = tracker.get_history(ex.execution_id)
        assert len(history) == 1
        assert history[0]["type"] == "start"

    def test_record_complete(self):
        tracker = self._make_tracker()
        tracker.record_complete("exec-123")
        history = tracker.get_history("exec-123")
        assert len(history) == 1
        assert history[0]["type"] == "complete"

    def test_get_recent(self):
        tracker = self._make_tracker()
        tracker.record_complete("exec-1")
        tracker.record_complete("exec-2")
        recent = tracker.get_recent(limit=10)
        assert len(recent) == 2
        # Most recent first
        assert recent[0]["execution_id"] == "exec-2"

    def test_stats(self):
        from app.distributed.workflows.state import WorkflowExecution

        tracker = self._make_tracker()
        ex = WorkflowExecution(workflow_name="test", goal="g")
        tracker.record_start(ex)
        tracker.record_complete(ex.execution_id)
        s = tracker.stats()
        assert s["total_executions"] == 1
        assert s["total_events"] == 2
        assert s["total_recorded"] == 2


# ==============================================================================
#  CONFIG COMPAT TESTS
# ==============================================================================


class TestBatch3ConfigCompat:
    """Config compatibility: EngineConfig zero-arg and batch 3 config defaults."""

    def test_engine_config_zero_arg(self):
        from app.config import EngineConfig
        cfg = EngineConfig()
        assert cfg.service_registry is not None
        assert cfg.agent_execution is not None
        assert cfg.distributed_workflow is not None
        assert cfg.tenancy is not None

    def test_batch3_configs_defaults(self):
        from app.config import (
            ServiceRegistryConfig,
            AgentExecutionConfig,
            DistributedWorkflowConfig,
            TenancyConfig,
        )

        sr = ServiceRegistryConfig()
        assert sr.heartbeat_interval == 10.0
        assert sr.stale_threshold_sec == 30.0
        assert sr.max_instances == 100

        ae = AgentExecutionConfig()
        assert ae.max_workers == 8
        assert ae.max_queue_size == 1000
        assert ae.scheduling_strategy == "priority"

        dw = DistributedWorkflowConfig()
        assert dw.max_concurrent_workflows == 20
        assert dw.checkpoint_enabled is True
        assert dw.schedule_enabled is False

        tc = TenancyConfig()
        assert tc.max_tenants == 100
        assert tc.isolation_level == "logical"
        assert tc.default_tenant == "default"


# ==============================================================================
#  API ENDPOINT TESTS
# ==============================================================================


class TestBatch3APIEndpoints:
    """Phase 8 Batch 3 API endpoints using FastAPI TestClient."""

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

    def test_services_list(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.get("/services")
            assert resp.status_code == 200

    def test_services_health(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.get("/services/health")
            assert resp.status_code == 200
            data = resp.json()
            assert "status" in data
            assert "checks" in data

    def test_tenants_list(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.get("/tenants")
            assert resp.status_code == 200
            data = resp.json()
            assert "tenants" in data
            assert isinstance(data["tenants"], list)

    def test_agent_distributed_stats(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.get("/agents/distributed/stats")
            assert resp.status_code == 200
            data = resp.json()
            assert "config" in data
            assert data["status"] == "available"
            assert "max_workers" in data["config"]

    def test_workflow_distributed_stats(self, tmp_path):
        with self._make_client(tmp_path) as c:
            resp = c.get("/workflows/distributed/stats")
            assert resp.status_code == 200
            data = resp.json()
            assert "config" in data
            assert data["status"] == "available"
            assert "max_concurrent" in data["config"]
