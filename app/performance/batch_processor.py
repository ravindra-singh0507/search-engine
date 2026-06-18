"""
Batch Processor — Phase 8 Batch 5

=== THEORY ===

Batch processing amortises per-item overhead by grouping operations:
  - Database: 100 individual INSERTs → 1 batch INSERT (10-100x faster)
  - Embeddings: 32 texts at once vs 32 individual model.encode() calls
  - Network: 1 bulk request vs N individual round-trips

The batch processor accumulates items until either:
  1. The batch reaches max_size (flush by size)
  2. The flush_interval expires (flush by time)

This ensures bounded latency (no item waits longer than flush_interval)
while still benefiting from batching.

=== COMPLEXITY ===

  add:            O(1) — append to buffer
  flush:          O(B) — B = batch size
  Throughput:     up to max_size / flush_interval items/sec

=== PRODUCTION EQUIVALENTS ===

  Kafka:    Producer batching (batch.size + linger.ms)
  Kinesis:  PutRecords (up to 500 records per call)
  BigQuery: Streaming insert batch (insertAll)
  FAISS:    Batch add (add_with_ids accepts arrays)
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """
    Batch processing parameters.

    max_size:        Flush when batch reaches this many items.
    flush_interval:  Flush after this many seconds regardless of size.
    max_retries:     Retry count on flush failure.
    """
    max_size:        int   = 32
    flush_interval:  float = 5.0
    max_retries:     int   = 2


class BatchProcessor:
    """
    Accumulates items and flushes in batches.

    Thread-safe: items can be added from multiple threads.
    The processor can run a background flusher thread or be flushed
    manually.

    Usage:
        def handle_batch(items):
            db.executemany("INSERT INTO t VALUES (?)", items)

        bp = BatchProcessor(handler=handle_batch, config=BatchConfig(max_size=100))
        bp.start()   # starts background flusher
        bp.add(item1)
        bp.add(item2)
        ...
        bp.stop()    # flushes remaining + stops background thread
    """

    def __init__(
        self,
        handler: Callable[[List[Any]], None],
        config: Optional[BatchConfig] = None,
        name: str = "batch",
    ):
        self._handler = handler
        self._config = config or BatchConfig()
        self._name = name
        self._buffer: List[Any] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._total_items = 0
        self._total_flushes = 0
        self._total_failures = 0
        self._last_flush_time = time.time()

    def add(self, item: Any) -> None:
        """Add an item to the batch buffer. Flushes if max_size reached."""
        with self._lock:
            self._buffer.append(item)
            self._total_items += 1
            if len(self._buffer) >= self._config.max_size:
                self._flush_locked()

    def add_many(self, items: List[Any]) -> None:
        """Add multiple items at once."""
        with self._lock:
            self._buffer.extend(items)
            self._total_items += len(items)
            if len(self._buffer) >= self._config.max_size:
                self._flush_locked()

    def flush(self) -> int:
        """Manually flush the current buffer. Returns count flushed."""
        with self._lock:
            return self._flush_locked()

    def start(self) -> None:
        """Start the background flusher thread."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._background_flusher, daemon=True,
            name=f"batch-flusher-{self._name}",
        )
        self._thread.start()
        logger.info("BatchProcessor '%s' started (max_size=%d, interval=%.1fs)",
                     self._name, self._config.max_size, self._config.flush_interval)

    def stop(self) -> None:
        """Stop background flusher and flush remaining items."""
        if not self._running:
            self.flush()
            return
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10.0)
        self.flush()
        logger.info("BatchProcessor '%s' stopped", self._name)

    def pending(self) -> int:
        """Number of items waiting to be flushed."""
        with self._lock:
            return len(self._buffer)

    def stats(self) -> dict:
        with self._lock:
            pending = len(self._buffer)
        return {
            "name": self._name,
            "running": self._running,
            "pending": pending,
            "total_items": self._total_items,
            "total_flushes": self._total_flushes,
            "total_failures": self._total_failures,
            "config": {
                "max_size": self._config.max_size,
                "flush_interval": self._config.flush_interval,
            },
        }

    def _flush_locked(self) -> int:
        """Flush buffer while holding lock. Returns count flushed."""
        if not self._buffer:
            return 0
        batch = list(self._buffer)
        self._buffer.clear()
        count = len(batch)

        for attempt in range(self._config.max_retries + 1):
            try:
                self._handler(batch)
                self._total_flushes += 1
                self._last_flush_time = time.time()
                return count
            except Exception as exc:
                if attempt < self._config.max_retries:
                    logger.warning(
                        "BatchProcessor '%s' flush attempt %d/%d failed: %s",
                        self._name, attempt + 1, self._config.max_retries + 1, exc,
                    )
                else:
                    logger.error(
                        "BatchProcessor '%s' flush failed after %d attempts: %s",
                        self._name, self._config.max_retries + 1, exc,
                    )
                    self._total_failures += 1

        return 0

    def _background_flusher(self) -> None:
        """Periodically flush based on interval."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self._config.flush_interval)
            if not self._stop_event.is_set():
                self.flush()
