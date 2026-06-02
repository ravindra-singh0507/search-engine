"""
Benchmarking Suite

Measures and compares performance of all search engine components.

Metrics:
  - BM25 vs TF-IDF latency comparison
  - Query throughput (queries/second)
  - Index size (documents, terms, postings)
  - Memory usage (RSS)
  - Cache hit rate under repeated queries

Usage:
    from app.benchmarks.benchmarker import Benchmarker
    bench = Benchmarker(search_service, db, tfidf_ranker, bm25_ranker)
    report = bench.run_all(sample_queries=["python", "machine learning"])
    bench.print_report(report)
"""

import time
import statistics
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Any

logger = logging.getLogger(__name__)


@dataclass
class LatencyStats:
    n:      int
    mean:   float
    median: float
    p95:    float
    p99:    float
    min:    float
    max:    float

    @classmethod
    def from_samples(cls, samples: list[float]) -> "LatencyStats":
        if not samples:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        s = sorted(samples)
        n = len(s)
        return cls(
            n=n,
            mean=round(statistics.mean(s), 3),
            median=round(statistics.median(s), 3),
            p95=round(s[int(0.95 * n)], 3),
            p99=round(s[int(0.99 * n)], 3),
            min=round(s[0], 3),
            max=round(s[-1], 3),
        )

    def as_dict(self) -> dict:
        return self.__dict__


@dataclass
class BenchmarkReport:
    timestamp: str
    index_stats: dict
    search_latency: dict          = field(default_factory=dict)
    tfidf_latency: dict           = field(default_factory=dict)
    bm25_latency: dict            = field(default_factory=dict)
    throughput_qps: float         = 0.0
    cache_hit_rate: float         = 0.0
    memory_mb: float              = 0.0
    notes: list[str]              = field(default_factory=list)


def _measure(fn: Callable, iterations: int = 100) -> LatencyStats:
    """Run fn `iterations` times and return latency statistics."""
    samples: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return LatencyStats.from_samples(samples)


def _get_memory_mb() -> float:
    """Return current process RSS in MB (best-effort)."""
    try:
        import resource  # Unix only
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        # Windows fallback
        try:
            import psutil
            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0


class Benchmarker:
    """
    Runs benchmarks over the search engine components.
    Designed to be called from a test or standalone script.
    """

    def __init__(self, search_service: Any, db: Any,
                 tfidf_ranker: Any | None = None):
        self.search = search_service
        self.db     = db
        self.tfidf  = tfidf_ranker   # optional, for comparison

    def benchmark_search(
        self,
        queries: list[str],
        iterations: int = 50,
    ) -> LatencyStats:
        """Benchmark the full search pipeline (includes cache)."""
        if not queries:
            return LatencyStats.from_samples([])

        idx = 0
        def run():
            nonlocal idx
            self.search.search(queries[idx % len(queries)], top_k=10)
            idx += 1

        return _measure(run, iterations * len(queries))

    def benchmark_bm25_vs_tfidf(
        self,
        query_terms: list[str],
        candidate_ids: set[int],
        iterations: int = 200,
    ) -> dict:
        """Direct comparison of BM25 and TF-IDF ranking latency."""
        if not candidate_ids:
            return {"error": "No candidate documents — index something first"}

        bm25_samples: list[float] = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            self.search.bm25.rank_documents(query_terms, candidate_ids, top_k=10)
            bm25_samples.append((time.perf_counter() - t0) * 1000)

        tfidf_samples: list[float] = []
        if self.tfidf:
            for _ in range(iterations):
                t0 = time.perf_counter()
                self.tfidf.rank_documents(query_terms, candidate_ids, top_k=10)
                tfidf_samples.append((time.perf_counter() - t0) * 1000)

        return {
            "bm25":  LatencyStats.from_samples(bm25_samples).as_dict(),
            "tfidf": LatencyStats.from_samples(tfidf_samples).as_dict() if tfidf_samples else None,
        }

    def benchmark_throughput(
        self, queries: list[str], duration_seconds: float = 5.0
    ) -> float:
        """Measure sustained query throughput (queries/second)."""
        if not queries:
            return 0.0
        count = 0
        start = time.perf_counter()
        while (time.perf_counter() - start) < duration_seconds:
            self.search.search(queries[count % len(queries)], top_k=10)
            count += 1
        elapsed = time.perf_counter() - start
        qps = count / elapsed
        logger.info("Throughput: %.1f QPS over %.1f s (%d queries)", qps, elapsed, count)
        return round(qps, 2)

    def run_all(
        self,
        sample_queries: list[str] | None = None,
        iterations: int = 50,
    ) -> BenchmarkReport:
        """Run the full benchmark suite and return a consolidated report."""
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()

        queries = sample_queries or ["python", "search engine", "machine learning"]
        stats   = self.db.get_stats()

        logger.info("Running benchmarks — %d queries × %d iterations …", len(queries), iterations)

        # Full search pipeline
        search_lat = self.benchmark_search(queries, iterations=iterations)

        # BM25 vs TF-IDF (need some candidate IDs)
        all_docs = self.db.get_all_documents()
        cand_ids = {d.doc_id for d in all_docs[:50]}
        parsed   = self.search._simple_parser.parse(queries[0])
        bm25_comparison = self.benchmark_bm25_vs_tfidf(
            parsed.terms, cand_ids, iterations=iterations
        )

        # Throughput
        qps = self.benchmark_throughput(queries, duration_seconds=3.0)

        # Cache hit rate
        cache_stats = self.search.cache.stats()

        # Memory
        mem_mb = _get_memory_mb()

        report = BenchmarkReport(
            timestamp=ts,
            index_stats=stats,
            search_latency=search_lat.as_dict(),
            bm25_latency=bm25_comparison.get("bm25", {}),
            tfidf_latency=bm25_comparison.get("tfidf") or {},
            throughput_qps=qps,
            cache_hit_rate=cache_stats.get("hit_rate", 0.0),
            memory_mb=mem_mb,
        )
        return report

    @staticmethod
    def print_report(report: BenchmarkReport) -> None:
        print("\n" + "═" * 60)
        print("  SEARCH ENGINE BENCHMARK REPORT")
        print("═" * 60)
        print(f"  Timestamp:       {report.timestamp}")
        print(f"  Documents:       {report.index_stats.get('total_documents', '?')}")
        print(f"  Terms:           {report.index_stats.get('total_terms', '?')}")
        print(f"  Postings:        {report.index_stats.get('total_postings', '?')}")
        print(f"  Memory (RSS):    {report.memory_mb:.1f} MB")
        print()
        print("  ── Search Latency (full pipeline) ──")
        for k, v in report.search_latency.items():
            print(f"    {k:8s}:  {v} ms")
        print()
        print("  ── BM25 Ranking Latency ──")
        for k, v in report.bm25_latency.items():
            print(f"    {k:8s}:  {v} ms")
        if report.tfidf_latency:
            print("  ── TF-IDF Ranking Latency ──")
            for k, v in report.tfidf_latency.items():
                print(f"    {k:8s}:  {v} ms")
        print()
        print(f"  Throughput:      {report.throughput_qps} QPS")
        print(f"  Cache hit rate:  {report.cache_hit_rate:.1%}")
        print("═" * 60 + "\n")
