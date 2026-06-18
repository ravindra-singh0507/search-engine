"""
Load Testing Suite — Phase 8 Batch 5

Locust-based load tests benchmarking search, retrieval, RAG, agent,
and workflow throughput.

Run: locust -f load_tests/locustfile.py --host http://localhost:8000
Web UI: http://localhost:8089

=== THEORY ===
Load testing measures system behavior under concurrent user load.
Key metrics: throughput (req/s), latency (p50/p95/p99), error rate.

=== PRODUCTION EQUIVALENTS ===
Google: Borgmon + load testing infra
Netflix: Chaos Monkey + load testing
Uber: Ballast (automated load testing)
"""

from locust import HttpUser, task, between, tag

class SearchUser(HttpUser):
    """Simulates users performing keyword searches."""
    wait_time = between(1, 3)
    weight = 5  # most common user type

    QUERIES = ["python programming", "machine learning", "web development",
               "data science", "artificial intelligence", "search engine",
               "distributed systems", "cloud computing", "database design",
               "neural networks"]

    @task(10)
    @tag("search")
    def search(self):
        q = self.QUERIES[hash(str(id(self))) % len(self.QUERIES)]
        self.client.get(f"/search?q={q}&top_k=10", name="/search")

    @task(3)
    @tag("autocomplete")
    def autocomplete(self):
        self.client.get("/autocomplete?q=py", name="/autocomplete")

    @task(1)
    @tag("stats")
    def stats(self):
        self.client.get("/stats", name="/stats")

class SemanticSearchUser(HttpUser):
    """Simulates semantic search users."""
    wait_time = between(2, 5)
    weight = 3

    @task(10)
    @tag("semantic")
    def semantic_search(self):
        self.client.get("/semantic-search?q=how+do+neural+networks+work&top_k=5",
                       name="/semantic-search")

    @task(5)
    @tag("hybrid")
    def hybrid_search(self):
        self.client.get("/hybrid-search?q=python+frameworks&top_k=10",
                       name="/hybrid-search")

class RAGUser(HttpUser):
    """Simulates RAG/chat users."""
    wait_time = between(3, 8)
    weight = 2

    @task(10)
    @tag("rag")
    def chat(self):
        self.client.post("/chat", json={
            "message": "What are the best practices for building search engines?",
            "top_k": 3, "template": "qa",
        }, name="/chat")

    @task(3)
    @tag("rag")
    def rag_query(self):
        self.client.post("/rag/query", json={
            "query": "Explain BM25 ranking algorithm",
            "top_k": 5,
        }, name="/rag/query")

class GatewayUser(HttpUser):
    """Simulates gateway search users."""
    wait_time = between(1, 4)
    weight = 3

    @task(10)
    @tag("gateway")
    def gateway_search(self):
        self.client.post("/gateway/search?q=distributed+systems&mode=hybrid&top_k=10",
                        name="/gateway/search")

class AgentUser(HttpUser):
    """Simulates research agent users."""
    wait_time = between(5, 15)
    weight = 1

    @task(5)
    @tag("agent")
    def research(self):
        self.client.post("/research", json={
            "goal": "Compare Python web frameworks",
            "workflow": "investigation",
        }, name="/research")

    @task(3)
    @tag("agent")
    def plan(self):
        self.client.post("/research/plan", json={
            "goal": "Analyze search quality metrics",
        }, name="/research/plan")

class InfraUser(HttpUser):
    """Simulates monitoring/health check requests."""
    wait_time = between(5, 10)
    weight = 1

    @task(5)
    @tag("health")
    def health(self):
        self.client.get("/health", name="/health")

    @task(3)
    @tag("metrics")
    def metrics(self):
        self.client.get("/metrics", name="/metrics")

    @task(2)
    @tag("events")
    def events(self):
        self.client.get("/events?limit=10", name="/events")

    @task(1)
    @tag("cost")
    def cost(self):
        self.client.get("/cost/stats", name="/cost/stats")
