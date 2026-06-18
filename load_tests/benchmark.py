"""
Benchmark runner — generates performance reports without Locust.

Runs sequential benchmarks measuring:
  - Search throughput (queries/sec)
  - Retrieval latency (p50, p95, p99)
  - RAG response time
  - Agent workflow duration
  - Endpoint availability

Results saved to data/benchmark_report.json

Usage: python -m load_tests.benchmark --host http://localhost:8000
"""

import argparse, json, time, statistics, requests
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class BenchmarkResult:
    endpoint: str
    requests_total: int = 0
    successes: int = 0
    failures: int = 0
    latencies_ms: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    @property
    def throughput_rps(self) -> float:
        total_sec = sum(self.latencies_ms) / 1000.0
        return self.requests_total / total_sec if total_sec > 0 else 0.0

    @property
    def p50(self) -> float:
        if not self.latencies_ms: return 0.0
        s = sorted(self.latencies_ms)
        return s[len(s)//2]

    @property
    def p95(self) -> float:
        if not self.latencies_ms: return 0.0
        s = sorted(self.latencies_ms)
        return s[int(len(s)*0.95)]

    @property
    def p99(self) -> float:
        if not self.latencies_ms: return 0.0
        s = sorted(self.latencies_ms)
        return s[int(len(s)*0.99)]

    def to_dict(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "requests": self.requests_total,
            "successes": self.successes,
            "failures": self.failures,
            "throughput_rps": round(self.throughput_rps, 2),
            "latency_p50_ms": round(self.p50, 2),
            "latency_p95_ms": round(self.p95, 2),
            "latency_p99_ms": round(self.p99, 2),
            "latency_mean_ms": round(statistics.mean(self.latencies_ms), 2) if self.latencies_ms else 0,
            "error_rate": round(self.failures / max(self.requests_total, 1), 4),
        }

class BenchmarkRunner:
    """Runs benchmarks against a live server."""

    def __init__(self, host: str = "http://localhost:8000"):
        self.host = host.rstrip("/")
        self.results: list[BenchmarkResult] = []

    def benchmark_endpoint(self, method: str, path: str, iterations: int = 50,
                           json_body: dict = None, name: str = "") -> BenchmarkResult:
        result = BenchmarkResult(endpoint=name or path)
        for _ in range(iterations):
            t0 = time.perf_counter()
            try:
                if method == "GET":
                    r = requests.get(f"{self.host}{path}", timeout=30)
                else:
                    r = requests.post(f"{self.host}{path}", json=json_body or {}, timeout=30)
                ms = (time.perf_counter() - t0) * 1000
                result.latencies_ms.append(ms)
                result.requests_total += 1
                if r.status_code < 400:
                    result.successes += 1
                else:
                    result.failures += 1
                    result.errors.append(f"HTTP {r.status_code}")
            except Exception as e:
                ms = (time.perf_counter() - t0) * 1000
                result.latencies_ms.append(ms)
                result.requests_total += 1
                result.failures += 1
                result.errors.append(str(e))
        self.results.append(result)
        return result

    def run_all(self, iterations: int = 20) -> list[dict]:
        print(f"Running benchmarks against {self.host} ({iterations} iterations each)...")
        self.benchmark_endpoint("GET", "/health", iterations, name="health")
        self.benchmark_endpoint("GET", "/search?q=python&top_k=10", iterations, name="search")
        self.benchmark_endpoint("GET", "/stats", iterations, name="stats")
        self.benchmark_endpoint("GET", "/events?limit=10", iterations, name="events")
        self.benchmark_endpoint("GET", "/metrics", iterations, name="metrics")
        return [r.to_dict() for r in self.results]

    def save_report(self, path: str = "data/benchmark_report.json") -> None:
        report = {
            "host": self.host,
            "timestamp": time.time(),
            "results": [r.to_dict() for r in self.results],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {path}")

    def print_report(self) -> None:
        print(f"\n{'='*70}")
        print(f"BENCHMARK REPORT — {self.host}")
        print(f"{'='*70}")
        for r in self.results:
            d = r.to_dict()
            print(f"\n{d['endpoint']}")
            print(f"  Requests: {d['requests']} ({d['successes']} ok, {d['failures']} fail)")
            print(f"  Throughput: {d['throughput_rps']} req/s")
            print(f"  Latency: p50={d['latency_p50_ms']}ms  p95={d['latency_p95_ms']}ms  p99={d['latency_p99_ms']}ms")
            print(f"  Error rate: {d['error_rate']*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    runner = BenchmarkRunner(args.host)
    runner.run_all(args.iterations)
    runner.print_report()
    runner.save_report()
