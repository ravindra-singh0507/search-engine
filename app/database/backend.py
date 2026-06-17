"""
Database Backend Abstraction — Phase 8

=== THEORY ===

The Repository / Backend pattern decouples the storage engine from the
domain logic.  By defining a Protocol (structural subtype), any database
engine — SQLite, PostgreSQL, MySQL, CockroachDB — can be swapped without
modifying the 90+ methods in the Database class.

=== ARCHITECTURE ===

  DatabaseBackend (Protocol)
    ├── SQLiteBackend      — wraps sqlite3, used for dev/tests (default)
    └── PostgreSQLBackend  — wraps psycopg2 + connection pool, production

The Database class (db.py) owns a `_backend` attribute of type
DatabaseBackend.  All raw SQL goes through the backend, which handles:
  - Parameter placeholder style (? vs %s)
  - Row factory (dict-like rows)
  - Connection lifecycle (pool vs single connection)
  - Dialect-specific DDL (AUTOINCREMENT vs SERIAL)

=== COMPLEXITY ===

  SQLiteBackend:
    connect:    O(1)
    execute:    O(query) — same as raw sqlite3
    fetchone:   O(1)
    fetchall:   O(rows)

  PostgreSQLBackend:
    connect:    O(1) — pool.getconn()
    execute:    O(query) — same as raw psycopg2
    fetchone:   O(1)
    fetchall:   O(rows)
    Pool management: O(1) amortised

=== PRODUCTION EQUIVALENTS ===

  Google Cloud Spanner: connection-pooled gRPC client
  Amazon RDS:           connection pool via pgBouncer or application-level
  Django:               database backend abstraction (django.db.backends)
  SQLAlchemy:           Engine + Connection + Dialect pattern
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class DatabaseBackend(Protocol):
    """
    Structural interface for database backends.

    Implementations must handle:
      - SQL execution with parameterised queries
      - Dict-like row results (column name → value)
      - Connection lifecycle
      - Dialect-specific placeholder style
    """

    @property
    def placeholder(self) -> str:
        """Return '?' for SQLite or '%s' for PostgreSQL."""
        ...

    @property
    def is_connected(self) -> bool:
        ...

    def connect(self) -> None:
        ...

    def close(self) -> None:
        ...

    def execute(self, sql: str, params: tuple = ()) -> Any:
        ...

    def executemany(self, sql: str, params_list: list[tuple]) -> None:
        ...

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[dict]:
        ...

    def fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        ...

    def commit(self) -> None:
        ...

    def begin(self) -> Any:
        """Return a context manager for transactions."""
        ...


class SQLiteBackend:
    """
    SQLite backend — wraps sqlite3.Connection.

    Preserves all existing behaviour: WAL mode, foreign keys, 32 MB cache,
    dict-like Row factory.  This is the default backend for development
    and testing.
    """

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def placeholder(self) -> str:
        return "?"

    @property
    def is_connected(self) -> bool:
        return self._conn is not None

    @property
    def conn(self) -> Optional[sqlite3.Connection]:
        return self._conn

    def connect(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA cache_size=-32000")
        logger.info("SQLiteBackend connected at %s", self._db_path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def execute(self, sql: str, params: tuple = ()) -> Any:
        return self._conn.execute(sql, params)

    def executemany(self, sql: str, params_list: list[tuple]) -> None:
        self._conn.executemany(sql, params_list)

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[dict]:
        row = self._conn.execute(sql, params).fetchone()
        if row is None:
            return None
        return {k: row[k] for k in row.keys()}

    def fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        rows = self._conn.execute(sql, params).fetchall()
        return [{k: r[k] for k in r.keys()} for r in rows]

    def commit(self) -> None:
        if self._conn:
            self._conn.commit()

    def begin(self) -> sqlite3.Connection:
        return self._conn


class PostgreSQLBackend:
    """
    PostgreSQL backend using psycopg2 with connection pooling.

    Uses ThreadedConnectionPool for multi-threaded access.
    Graceful fallback: if psycopg2 is not installed or connection
    fails, the caller (Database.connect()) falls back to SQLiteBackend.

    === CONNECTION POOL ===

    ThreadedConnectionPool manages a pool of connections:
      - minconn: minimum connections kept alive
      - maxconn: maximum connections allowed
      - getconn(): borrow a connection from the pool
      - putconn(): return a connection to the pool

    This avoids the overhead of creating a new TCP connection per query
    (TCP handshake + TLS + PostgreSQL auth = 5-50ms per connection).
    """

    def __init__(self, config):
        self._config = config
        self._pool = None

    @property
    def placeholder(self) -> str:
        return "%s"

    @property
    def is_connected(self) -> bool:
        return self._pool is not None

    def connect(self) -> None:
        try:
            import psycopg2
            import psycopg2.pool
            import psycopg2.extras
        except ImportError:
            raise ImportError(
                "psycopg2 is required for PostgreSQL backend. "
                "Install with: pip install psycopg2-binary"
            )

        self._pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=self._config.pool_min,
            maxconn=self._config.pool_size,
            host=self._config.host,
            port=self._config.port,
            database=self._config.database,
            user=self._config.user,
            password=self._config.password,
            sslmode=self._config.ssl_mode,
        )
        logger.info(
            "PostgreSQLBackend connected to %s:%d/%s (pool=%d-%d)",
            self._config.host, self._config.port, self._config.database,
            self._config.pool_min, self._config.pool_size,
        )

    def close(self) -> None:
        if self._pool:
            self._pool.closeall()
            self._pool = None

    def _get_conn(self):
        conn = self._pool.getconn()
        conn.autocommit = False
        return conn

    def _put_conn(self, conn):
        self._pool.putconn(conn)

    def execute(self, sql: str, params: tuple = ()) -> Any:
        conn = self._get_conn()
        try:
            import psycopg2.extras
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cur.execute(sql, params)
            conn.commit()
            self._put_conn(conn)
            return cur
        except Exception:
            conn.rollback()
            self._put_conn(conn)
            raise

    def executemany(self, sql: str, params_list: list[tuple]) -> None:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.executemany(sql, params_list)
            conn.commit()
            self._put_conn(conn)
        except Exception:
            conn.rollback()
            self._put_conn(conn)
            raise

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[dict]:
        conn = self._get_conn()
        try:
            import psycopg2.extras
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(sql, params)
            row = cur.fetchone()
            self._put_conn(conn)
            return dict(row) if row else None
        except Exception:
            conn.rollback()
            self._put_conn(conn)
            raise

    def fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        conn = self._get_conn()
        try:
            import psycopg2.extras
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(sql, params)
            rows = cur.fetchall()
            self._put_conn(conn)
            return [dict(r) for r in rows]
        except Exception:
            conn.rollback()
            self._put_conn(conn)
            raise

    def commit(self) -> None:
        pass

    def begin(self):
        return self._get_conn()

    def stats(self) -> dict:
        if not self._pool:
            return {"status": "disconnected"}
        return {
            "status": "connected",
            "min_connections": self._config.pool_min,
            "max_connections": self._config.pool_size,
        }
