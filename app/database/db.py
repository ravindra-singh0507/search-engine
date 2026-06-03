"""
Database Layer — Phase 3

Extends the Phase 2 schema with:

  search_logs   — every search query with latency and result count
  click_logs    — every result click with position
  query_stats   — per-query aggregate counters (for autocomplete / trending)

The postings table gains a `field` column ('title' | 'body') so the
ranker can boost title matches independently.

=== SCHEMA ===

  documents     doc_id, title, content, source, doc_type, word_count, created_at
  terms         term_id, term, document_frequency
  postings      term_id, doc_id, term_frequency, positions, field
  crawled_pages page_id, url, title, content, html, status_code, depth, crawled_at, doc_id
  search_logs   log_id, query, results_count, latency_ms, timestamp, session_id
  click_logs    click_id, log_id, doc_id, position, timestamp
  query_stats   query_id, query (UNIQUE), total_searches, avg_latency_ms,
                zero_result_searches, last_searched

=== AT GOOGLE SCALE ===

  Google keeps these in separate services:
  - Bigtable / Spanner for documents / index
  - Kafka + BigQuery for click-stream analytics
  - Memcache / Redis for query stats / trending
  SQLite is fine for a single-node learning project.
"""

import sqlite3
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass
class DocumentRecord:
    doc_id: int
    title: str
    content: str
    source: str
    doc_type: str
    word_count: int
    created_at: str


@dataclass
class TermRecord:
    term_id: int
    term: str
    document_frequency: int


@dataclass
class PostingRecord:
    term_id: int
    doc_id: int
    term_frequency: int
    positions: list[int]
    field: str = "body"


@dataclass
class CrawledPageRecord:
    page_id: int
    url: str
    title: str
    content: str
    html: str
    status_code: int
    crawl_depth: int
    crawled_at: str
    doc_id: Optional[int]


# ── Phase 4 dataclasses ────────────────────────────────────────────────────────

@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: int
    chunk_index: int
    text: str
    start_offset: int
    end_offset: int
    word_count: int
    created_at: str


@dataclass
class EmbeddingJobRecord:
    job_id: int
    doc_id: Optional[int]
    status: str           # pending | running | done | error
    model_name: str
    chunks_total: int
    chunks_processed: int
    error_message: Optional[str]
    created_at: str
    completed_at: Optional[str]


@dataclass
class SearchLogRecord:
    log_id: int
    query: str
    results_count: int
    latency_ms: float
    timestamp: str
    session_id: Optional[str]


@dataclass
class ClickLogRecord:
    click_id: int
    log_id: int
    doc_id: int
    position: int
    timestamp: str


@dataclass
class QueryStatRecord:
    query_id: int
    query: str
    total_searches: int
    avg_latency_ms: float
    zero_result_searches: int
    last_searched: str


# ── Database ───────────────────────────────────────────────────────────────


class Database:
    """
    SQLite-backed storage.  Single persistent connection in WAL mode.
    Thread-safe for reads; indexing serialises through the GIL.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA cache_size=-32000")  # 32 MB page cache
        self._create_tables()
        self._migrate()
        logger.info("Database connected at %s", self.db_path)

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    # ── Schema ─────────────────────────────────────────────────────────────

    def _create_tables(self) -> None:
        c = self.conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT NOT NULL,
                content     TEXT NOT NULL,
                source      TEXT NOT NULL DEFAULT 'local',
                doc_type    TEXT NOT NULL DEFAULT 'text',
                word_count  INTEGER NOT NULL DEFAULT 0,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS terms (
                term_id            INTEGER PRIMARY KEY AUTOINCREMENT,
                term               TEXT NOT NULL UNIQUE,
                document_frequency INTEGER NOT NULL DEFAULT 0
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS postings (
                term_id        INTEGER NOT NULL,
                doc_id         INTEGER NOT NULL,
                term_frequency INTEGER NOT NULL DEFAULT 0,
                positions      TEXT NOT NULL DEFAULT '[]',
                field          TEXT NOT NULL DEFAULT 'body',
                PRIMARY KEY (term_id, doc_id, field),
                FOREIGN KEY (term_id) REFERENCES terms(term_id),
                FOREIGN KEY (doc_id)  REFERENCES documents(doc_id)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS crawled_pages (
                page_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                url         TEXT NOT NULL UNIQUE,
                title       TEXT,
                content     TEXT,
                html        TEXT,
                status_code INTEGER,
                crawl_depth INTEGER NOT NULL DEFAULT 0,
                crawled_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                doc_id      INTEGER,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        """)

        # ── Phase 3: Analytics tables ───────────────────────────────────────

        c.execute("""
            CREATE TABLE IF NOT EXISTS search_logs (
                log_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                query         TEXT NOT NULL,
                results_count INTEGER NOT NULL DEFAULT 0,
                latency_ms    REAL NOT NULL DEFAULT 0,
                timestamp     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_id    TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS click_logs (
                click_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                log_id    INTEGER,
                doc_id    INTEGER NOT NULL,
                position  INTEGER NOT NULL DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (log_id) REFERENCES search_logs(log_id)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS query_stats (
                query_id             INTEGER PRIMARY KEY AUTOINCREMENT,
                query                TEXT NOT NULL UNIQUE,
                total_searches       INTEGER NOT NULL DEFAULT 0,
                avg_latency_ms       REAL NOT NULL DEFAULT 0,
                zero_result_searches INTEGER NOT NULL DEFAULT 0,
                last_searched        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ── Indexes (Phase 1-3 tables only) ────────────────────────────────

        for stmt in [
            "CREATE INDEX IF NOT EXISTS idx_terms_term         ON terms(term)",
            "CREATE INDEX IF NOT EXISTS idx_postings_doc       ON postings(doc_id)",
            "CREATE INDEX IF NOT EXISTS idx_postings_field     ON postings(field)",
            "CREATE INDEX IF NOT EXISTS idx_crawled_url        ON crawled_pages(url)",
            "CREATE INDEX IF NOT EXISTS idx_search_logs_ts     ON search_logs(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_search_logs_q      ON search_logs(query)",
            "CREATE INDEX IF NOT EXISTS idx_click_logs_log     ON click_logs(log_id)",
            "CREATE INDEX IF NOT EXISTS idx_query_stats_q      ON query_stats(query)",
        ]:
            c.execute(stmt)

        # ── Phase 4 tables ─────────────────────────────────────────────────

        c.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                chunk_id     TEXT PRIMARY KEY,
                doc_id       INTEGER NOT NULL,
                chunk_index  INTEGER NOT NULL,
                text         TEXT NOT NULL,
                start_offset INTEGER NOT NULL DEFAULT 0,
                end_offset   INTEGER NOT NULL DEFAULT 0,
                word_count   INTEGER NOT NULL DEFAULT 0,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS document_embeddings (
                embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id     TEXT NOT NULL UNIQUE,
                doc_id       INTEGER NOT NULL,
                model_name   TEXT NOT NULL,
                vector_dim   INTEGER NOT NULL DEFAULT 384,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chunk_id) REFERENCES document_chunks(chunk_id),
                FOREIGN KEY (doc_id)   REFERENCES documents(doc_id)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS embedding_jobs (
                job_id           INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id           INTEGER,
                status           TEXT NOT NULL DEFAULT 'pending',
                model_name       TEXT NOT NULL,
                chunks_total     INTEGER DEFAULT 0,
                chunks_processed INTEGER DEFAULT 0,
                error_message    TEXT,
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at     TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS vector_index_metadata (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name    TEXT NOT NULL,
                dimension     INTEGER NOT NULL,
                total_vectors INTEGER NOT NULL DEFAULT 0,
                index_path    TEXT,
                updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                content_hash TEXT PRIMARY KEY,
                model_name   TEXT NOT NULL,
                vector_json  TEXT NOT NULL,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ── Phase 4 indexes (created AFTER their tables) ───────────────────
        for stmt in [
            "CREATE INDEX IF NOT EXISTS idx_chunks_doc       ON document_chunks(doc_id)",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_chunk ON document_embeddings(chunk_id)",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_doc   ON document_embeddings(doc_id)",
            "CREATE INDEX IF NOT EXISTS idx_emb_cache_model  ON embedding_cache(model_name)",
        ]:
            c.execute(stmt)

        self.conn.commit()

    def _migrate(self) -> None:
        """Add `field` column to postings if upgrading from Phase 2."""
        c = self.conn.cursor()
        c.execute("PRAGMA table_info(postings)")
        cols = {row["name"] for row in c.fetchall()}
        if "field" not in cols:
            c.execute("ALTER TABLE postings ADD COLUMN field TEXT NOT NULL DEFAULT 'body'")
            self.conn.commit()
            logger.info("Migrated postings table: added `field` column")

    # ── Document Operations ─────────────────────────────────────────────────

    def insert_document(self, title: str, content: str, source: str = "local",
                        doc_type: str = "text", word_count: int = 0) -> int:
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO documents (title, content, source, doc_type, word_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (title, content, source, doc_type, word_count),
        )
        self.conn.commit()
        return c.lastrowid

    def get_document(self, doc_id: int) -> Optional[DocumentRecord]:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            return None
        return DocumentRecord(**{k: row[k] for k in row.keys()})

    def get_all_documents(self) -> list[DocumentRecord]:
        rows = self.conn.execute(
            "SELECT * FROM documents ORDER BY doc_id"
        ).fetchall()
        return [DocumentRecord(**{k: r[k] for k in r.keys()}) for r in rows]

    def get_document_count(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM documents"
        ).fetchone()["cnt"]

    def get_average_document_length(self) -> float:
        """Corpus-wide average document length — needed by BM25."""
        row = self.conn.execute(
            "SELECT AVG(word_count) AS avg FROM documents"
        ).fetchone()
        return float(row["avg"] or 0.0)

    def document_exists_by_source(self, source: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM documents WHERE source = ? LIMIT 1", (source,)
        ).fetchone() is not None

    def delete_document(self, doc_id: int) -> bool:
        c = self.conn.cursor()
        # Cascade manually: clean up analytics rows that reference this doc
        c.execute("UPDATE click_logs SET doc_id = NULL WHERE doc_id = ?", (doc_id,))
        c.execute("DELETE FROM postings  WHERE doc_id = ?", (doc_id,))
        c.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self.conn.commit()
        deleted = c.rowcount > 0
        if deleted:
            self._recalculate_document_frequencies()
        return deleted

    def get_recent_documents(self, limit: int = 100) -> list[DocumentRecord]:
        rows = self.conn.execute(
            "SELECT * FROM documents ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [DocumentRecord(**{k: r[k] for k in r.keys()}) for r in rows]

    # ── Term Operations ─────────────────────────────────────────────────────

    def clear_postings_for_doc(self, doc_id: int) -> None:
        """Delete all postings for a document (used by reindex_document)."""
        self.conn.execute("DELETE FROM postings WHERE doc_id = ?", (doc_id,))
        self.conn.commit()

    def update_document_content(self, doc_id: int, content: str,
                                word_count: int) -> None:
        """Update content and word_count for an existing document."""
        self.conn.execute(
            "UPDATE documents SET content = ?, word_count = ? WHERE doc_id = ?",
            (content, word_count, doc_id),
        )
        self.conn.commit()

    def batch_update_document_frequencies(self, terms: list[str]) -> None:
        """
        Recalculate document_frequency for all given terms in a SINGLE SQL
        statement, replacing the O(N · DB_queries) loop in the old
        _update_document_frequencies implementation.
        """
        if not terms:
            return
        placeholders = ",".join("?" for _ in terms)
        self.conn.execute(
            f"""
            UPDATE terms
            SET document_frequency = (
                SELECT COUNT(DISTINCT doc_id) FROM postings
                WHERE postings.term_id = terms.term_id
            )
            WHERE term IN ({placeholders})
            """,
            list(terms),
        )
        self.conn.commit()

    def get_inverted_index_snapshot(self) -> dict[str, list[int]]:
        """
        Readable snapshot: term → [doc_ids].  Useful for debugging.
        Moved here from Indexer to eliminate db.conn direct access in callers.
        """
        rows = self.conn.execute("""
            SELECT t.term, GROUP_CONCAT(DISTINCT p.doc_id) AS doc_ids
            FROM terms t
            JOIN postings p ON t.term_id = p.term_id
            GROUP BY t.term
            ORDER BY t.term
        """).fetchall()
        return {
            row["term"]: [int(d) for d in row["doc_ids"].split(",")]
            for row in rows
        }

    def get_or_create_term(self, term: str) -> int:
        row = self.conn.execute(
            "SELECT term_id FROM terms WHERE term = ?", (term,)
        ).fetchone()
        if row:
            return row["term_id"]
        c = self.conn.cursor()
        c.execute("INSERT INTO terms (term, document_frequency) VALUES (?, 0)", (term,))
        self.conn.commit()
        return c.lastrowid

    def get_term(self, term: str) -> Optional[TermRecord]:
        row = self.conn.execute(
            "SELECT * FROM terms WHERE term = ?", (term,)
        ).fetchone()
        if row is None:
            return None
        return TermRecord(
            term_id=row["term_id"],
            term=row["term"],
            document_frequency=row["document_frequency"],
        )

    def get_term_count(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM terms"
        ).fetchone()["cnt"]

    def update_document_frequency(self, term_id: int, df: int) -> None:
        self.conn.execute(
            "UPDATE terms SET document_frequency = ? WHERE term_id = ?", (df, term_id)
        )
        self.conn.commit()

    def get_all_terms(self) -> list[str]:
        """Return every known term — used to build the spell-check vocabulary."""
        rows = self.conn.execute("SELECT term FROM terms ORDER BY term").fetchall()
        return [r["term"] for r in rows]

    def get_term_by_id(self, term_id: int) -> Optional[TermRecord]:
        """Look up a term by its integer ID (used by TFIDFRanker)."""
        row = self.conn.execute(
            "SELECT * FROM terms WHERE term_id = ?", (term_id,)
        ).fetchone()
        if row is None:
            return None
        return TermRecord(
            term_id=row["term_id"],
            term=row["term"],
            document_frequency=row["document_frequency"],
        )

    # ── Posting Operations ──────────────────────────────────────────────────

    def insert_posting(self, term_id: int, doc_id: int,
                       term_frequency: int, positions: list[int],
                       field: str = "body") -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO postings "
            "(term_id, doc_id, term_frequency, positions, field) VALUES (?,?,?,?,?)",
            (term_id, doc_id, term_frequency, json.dumps(positions), field),
        )
        self.conn.commit()

    def batch_insert_postings(self, postings: list[tuple]) -> None:
        """Each tuple: (term_id, doc_id, tf, positions_json, field)."""
        self.conn.executemany(
            "INSERT OR REPLACE INTO postings "
            "(term_id, doc_id, term_frequency, positions, field) VALUES (?,?,?,?,?)",
            postings,
        )
        self.conn.commit()

    def get_postings_for_term(self, term: str,
                              field: Optional[str] = None) -> list[PostingRecord]:
        if field:
            rows = self.conn.execute("""
                SELECT p.term_id, p.doc_id, p.term_frequency, p.positions, p.field
                FROM postings p
                JOIN terms t ON p.term_id = t.term_id
                WHERE t.term = ? AND p.field = ?
                ORDER BY p.doc_id
            """, (term, field)).fetchall()
        else:
            rows = self.conn.execute("""
                SELECT p.term_id, p.doc_id, p.term_frequency, p.positions, p.field
                FROM postings p
                JOIN terms t ON p.term_id = t.term_id
                WHERE t.term = ?
                ORDER BY p.doc_id
            """, (term,)).fetchall()
        return [
            PostingRecord(
                term_id=r["term_id"], doc_id=r["doc_id"],
                term_frequency=r["term_frequency"],
                positions=json.loads(r["positions"]),
                field=r["field"],
            )
            for r in rows
        ]

    def get_postings_for_doc(self, doc_id: int) -> list[PostingRecord]:
        rows = self.conn.execute(
            "SELECT * FROM postings WHERE doc_id = ? ORDER BY term_id", (doc_id,)
        ).fetchall()
        return [
            PostingRecord(
                term_id=r["term_id"], doc_id=r["doc_id"],
                term_frequency=r["term_frequency"],
                positions=json.loads(r["positions"]),
                field=r["field"],
            )
            for r in rows
        ]

    def get_posting_count(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM postings"
        ).fetchone()["cnt"]

    def get_postings_for_terms_batch(
        self, terms: list[str], field: str = "body"
    ) -> dict[str, dict[int, int]]:
        """
        Fetch TF data for multiple terms in a single SQL query.

        Returns: {term: {doc_id: term_frequency}}

        This eliminates the O(candidates × terms) N+1 query pattern in
        BM25Ranker by fetching all necessary data in one round-trip.
        """
        if not terms:
            return {}
        unique_terms = list(set(terms))
        placeholders = ",".join("?" for _ in unique_terms)
        rows = self.conn.execute(
            f"""
            SELECT t.term, p.doc_id, p.term_frequency
            FROM postings p
            JOIN terms t ON p.term_id = t.term_id
            WHERE t.term IN ({placeholders}) AND p.field = ?
            ORDER BY t.term, p.doc_id
            """,
            (*unique_terms, field),
        ).fetchall()

        result: dict[str, dict[int, int]] = {t: {} for t in unique_terms}
        for row in rows:
            result[row["term"]][row["doc_id"]] = row["term_frequency"]
        return result

    def _recalculate_document_frequencies(self) -> None:
        self.conn.execute("""
            UPDATE terms SET document_frequency = (
                SELECT COUNT(DISTINCT doc_id) FROM postings
                WHERE postings.term_id = terms.term_id
            )
        """)
        self.conn.commit()

    # ── Crawled Pages ───────────────────────────────────────────────────────

    def insert_crawled_page(self, url: str, title: str, content: str,
                            html: str, status_code: int, crawl_depth: int,
                            doc_id: Optional[int] = None) -> int:
        c = self.conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO crawled_pages "
            "(url, title, content, html, status_code, crawl_depth, doc_id) "
            "VALUES (?,?,?,?,?,?,?)",
            (url, title, content, html, status_code, crawl_depth, doc_id),
        )
        self.conn.commit()
        return c.lastrowid

    def url_already_crawled(self, url: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM crawled_pages WHERE url = ? LIMIT 1", (url,)
        ).fetchone() is not None

    def get_crawled_page_count(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM crawled_pages"
        ).fetchone()["cnt"]

    def get_crawl_stats(self) -> dict:
        total   = self.conn.execute("SELECT COUNT(*) AS n FROM crawled_pages").fetchone()["n"]
        indexed = self.conn.execute(
            "SELECT COUNT(*) AS n FROM crawled_pages WHERE doc_id IS NOT NULL"
        ).fetchone()["n"]
        row = self.conn.execute(
            "SELECT AVG(status_code) AS avg FROM crawled_pages"
        ).fetchone()
        return {
            "total_pages_crawled": total,
            "pages_indexed": indexed,
            "average_status_code": round(row["avg"], 1) if row["avg"] else 0,
        }

    # ── Analytics: Search Logs ──────────────────────────────────────────────

    def log_search(self, query: str, results_count: int,
                   latency_ms: float, session_id: Optional[str] = None) -> int:
        """
        Insert one search event and upsert the per-query aggregate.
        Both writes share a single explicit transaction so they are either
        both committed or both rolled back (no partial state on crash).
        """
        with self.conn:           # BEGIN … COMMIT / ROLLBACK on exception
            c = self.conn.cursor()
            c.execute(
                "INSERT INTO search_logs (query, results_count, latency_ms, session_id) "
                "VALUES (?,?,?,?)",
                (query, results_count, latency_ms, session_id),
            )
            log_id = c.lastrowid

            existing = c.execute(
                "SELECT query_id, total_searches, avg_latency_ms, zero_result_searches "
                "FROM query_stats WHERE query = ?", (query,)
            ).fetchone()

            if existing:
                n       = existing["total_searches"]
                new_avg = (existing["avg_latency_ms"] * n + latency_ms) / (n + 1)
                zeros   = existing["zero_result_searches"] + (1 if results_count == 0 else 0)
                c.execute(
                    "UPDATE query_stats SET total_searches=?, avg_latency_ms=?, "
                    "zero_result_searches=?, last_searched=CURRENT_TIMESTAMP "
                    "WHERE query=?",
                    (n + 1, new_avg, zeros, query),
                )
            else:
                c.execute(
                    "INSERT INTO query_stats (query, total_searches, avg_latency_ms, "
                    "zero_result_searches) VALUES (?,1,?,?)",
                    (query, latency_ms, 1 if results_count == 0 else 0),
                )

        return log_id

    def log_click(self, log_id: int, doc_id: int, position: int) -> None:
        self.conn.execute(
            "INSERT INTO click_logs (log_id, doc_id, position) VALUES (?,?,?)",
            (log_id, doc_id, position),
        )
        self.conn.commit()

    def get_top_queries(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT query, total_searches, avg_latency_ms, zero_result_searches, "
            "last_searched FROM query_stats "
            "ORDER BY total_searches DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_failed_queries(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT query, zero_result_searches, total_searches, "
            "CAST(zero_result_searches AS REAL)/total_searches AS failure_rate "
            "FROM query_stats WHERE zero_result_searches > 0 "
            "ORDER BY zero_result_searches DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_search_volume(self, hours: int = 24) -> list[dict]:
        rows = self.conn.execute("""
            SELECT strftime('%Y-%m-%d %H:00', timestamp) AS hour,
                   COUNT(*) AS searches
            FROM search_logs
            WHERE timestamp >= datetime('now', ? || ' hours')
            GROUP BY hour
            ORDER BY hour
        """, (f"-{hours}",)).fetchall()
        return [dict(r) for r in rows]

    def get_click_through_rate(self) -> dict:
        total_searches = self.conn.execute(
            "SELECT COUNT(*) AS n FROM search_logs"
        ).fetchone()["n"]
        searches_with_clicks = self.conn.execute(
            "SELECT COUNT(DISTINCT log_id) AS n FROM click_logs"
        ).fetchone()["n"]
        ctr = (searches_with_clicks / total_searches) if total_searches else 0.0
        return {
            "total_searches": total_searches,
            "searches_with_clicks": searches_with_clicks,
            "click_through_rate": round(ctr, 4),
        }

    def get_avg_click_position(self) -> float:
        row = self.conn.execute(
            "SELECT AVG(position) AS avg FROM click_logs"
        ).fetchone()
        return round(row["avg"] or 0.0, 2)

    # ── Stats (combined) ────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "total_documents":    self.get_document_count(),
            "total_terms":        self.get_term_count(),
            "total_postings":     self.get_posting_count(),
            "total_crawled_pages": self.get_crawled_page_count(),
            "avg_document_length": round(self.get_average_document_length(), 1),
        }

    # ── Phase 4: Chunks ─────────────────────────────────────────────────────

    def insert_chunk(self, chunk_id: str, doc_id: int, chunk_index: int,
                     text: str, start_offset: int, end_offset: int,
                     word_count: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO document_chunks "
            "(chunk_id, doc_id, chunk_index, text, start_offset, end_offset, word_count) "
            "VALUES (?,?,?,?,?,?,?)",
            (chunk_id, doc_id, chunk_index, text, start_offset, end_offset, word_count),
        )
        self.conn.commit()

    def get_chunks_for_doc(self, doc_id: int) -> list[ChunkRecord]:
        rows = self.conn.execute(
            "SELECT * FROM document_chunks WHERE doc_id = ? ORDER BY chunk_index",
            (doc_id,),
        ).fetchall()
        return [
            ChunkRecord(
                chunk_id=r["chunk_id"], doc_id=r["doc_id"],
                chunk_index=r["chunk_index"], text=r["text"],
                start_offset=r["start_offset"], end_offset=r["end_offset"],
                word_count=r["word_count"], created_at=r["created_at"],
            )
            for r in rows
        ]

    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        row = self.conn.execute(
            "SELECT * FROM document_chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            return None
        return ChunkRecord(
            chunk_id=row["chunk_id"], doc_id=row["doc_id"],
            chunk_index=row["chunk_index"], text=row["text"],
            start_offset=row["start_offset"], end_offset=row["end_offset"],
            word_count=row["word_count"], created_at=row["created_at"],
        )

    def delete_chunks_for_doc(self, doc_id: int) -> None:
        self.conn.execute(
            "DELETE FROM document_chunks WHERE doc_id = ?", (doc_id,)
        )
        self.conn.commit()

    def chunk_count(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) AS n FROM document_chunks"
        ).fetchone()["n"]

    # ── Phase 4: Embedding records ──────────────────────────────────────────

    def insert_embedding_record(self, chunk_id: str, doc_id: int,
                                model_name: str, vector_dim: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO document_embeddings "
            "(chunk_id, doc_id, model_name, vector_dim) VALUES (?,?,?,?)",
            (chunk_id, doc_id, model_name, vector_dim),
        )
        self.conn.commit()

    def is_doc_embedded(self, doc_id: int, model_name: str) -> bool:
        """Return True if at least one chunk of this doc has been embedded."""
        row = self.conn.execute(
            "SELECT 1 FROM document_embeddings "
            "WHERE doc_id = ? AND model_name = ? LIMIT 1",
            (doc_id, model_name),
        ).fetchone()
        return row is not None

    def get_unembedded_doc_ids(self, model_name: str) -> list[int]:
        """Return doc IDs that have no embedding record for this model."""
        rows = self.conn.execute(
            """
            SELECT d.doc_id FROM documents d
            WHERE NOT EXISTS (
                SELECT 1 FROM document_embeddings e
                WHERE e.doc_id = d.doc_id AND e.model_name = ?
            )
            ORDER BY d.doc_id
            """,
            (model_name,),
        ).fetchall()
        return [r["doc_id"] for r in rows]

    def delete_embeddings_for_doc(self, doc_id: int) -> None:
        self.conn.execute(
            "DELETE FROM document_embeddings WHERE doc_id = ?", (doc_id,)
        )
        self.conn.commit()

    def embedding_count(self, model_name: str | None = None) -> int:
        if model_name:
            return self.conn.execute(
                "SELECT COUNT(*) AS n FROM document_embeddings WHERE model_name = ?",
                (model_name,),
            ).fetchone()["n"]
        return self.conn.execute(
            "SELECT COUNT(*) AS n FROM document_embeddings"
        ).fetchone()["n"]

    def get_embedded_doc_ids(self, model_name: str) -> list[int]:
        rows = self.conn.execute(
            "SELECT DISTINCT doc_id FROM document_embeddings WHERE model_name = ?",
            (model_name,),
        ).fetchall()
        return [r["doc_id"] for r in rows]

    # ── Phase 4: Embedding cache ────────────────────────────────────────────

    def get_cached_embedding(self, content_hash: str,
                             model_name: str) -> Optional[list[float]]:
        if self.conn is None:
            return None
        row = self.conn.execute(
            "SELECT vector_json FROM embedding_cache "
            "WHERE content_hash = ? AND model_name = ?",
            (content_hash, model_name),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["vector_json"])

    def cache_embedding(self, content_hash: str, model_name: str,
                        vector: list[float]) -> None:
        if self.conn is None:
            return
        self.conn.execute(
            "INSERT OR REPLACE INTO embedding_cache "
            "(content_hash, model_name, vector_json) VALUES (?,?,?)",
            (content_hash, model_name, json.dumps(vector)),
        )
        self.conn.commit()

    def clear_embedding_cache(self) -> None:
        self.conn.execute("DELETE FROM embedding_cache")
        self.conn.commit()

    def embedding_cache_size(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) AS n FROM embedding_cache"
        ).fetchone()["n"]

    # ── Phase 4: Embedding jobs ─────────────────────────────────────────────

    def create_embedding_job(self, doc_id: Optional[int], model_name: str,
                             chunks_total: int) -> int:
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO embedding_jobs (doc_id, model_name, chunks_total, status) "
            "VALUES (?,?,?,'running')",
            (doc_id, model_name, chunks_total),
        )
        self.conn.commit()
        return c.lastrowid

    def update_embedding_job(self, job_id: int, status: str,
                             chunks_processed: int = 0,
                             error: Optional[str] = None) -> None:
        self.conn.execute(
            "UPDATE embedding_jobs SET status=?, chunks_processed=?, "
            "error_message=?, completed_at=CURRENT_TIMESTAMP "
            "WHERE job_id=?",
            (status, chunks_processed, error, job_id),
        )
        self.conn.commit()

    def get_embedding_jobs(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM embedding_jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Phase 4: Vector index metadata ─────────────────────────────────────

    def upsert_vector_index_metadata(self, model_name: str, dimension: int,
                                     total_vectors: int,
                                     index_path: str) -> None:
        existing = self.conn.execute(
            "SELECT id FROM vector_index_metadata WHERE model_name = ?", (model_name,)
        ).fetchone()
        if existing:
            self.conn.execute(
                "UPDATE vector_index_metadata SET dimension=?, total_vectors=?, "
                "index_path=?, updated_at=CURRENT_TIMESTAMP WHERE model_name=?",
                (dimension, total_vectors, index_path, model_name),
            )
        else:
            self.conn.execute(
                "INSERT INTO vector_index_metadata (model_name, dimension, "
                "total_vectors, index_path) VALUES (?,?,?,?)",
                (model_name, dimension, total_vectors, index_path),
            )
        self.conn.commit()

    def get_vector_index_metadata(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM vector_index_metadata ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Phase 4: Combined semantic stats ───────────────────────────────────

    def get_semantic_stats(self, model_name: str) -> dict:
        return {
            "total_chunks": self.chunk_count(),
            "embedded_chunks": self.embedding_count(model_name),
            "unembedded_docs": len(self.get_unembedded_doc_ids(model_name)),
            "cache_entries": self.embedding_cache_size(),
        }

