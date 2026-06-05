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

        # ── Phase 5 tables ─────────────────────────────────────────────────

        c.execute("""
            CREATE TABLE IF NOT EXISTS reranking_logs (
                log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                query           TEXT NOT NULL,
                doc_id          INTEGER NOT NULL,
                bm25_score      REAL,
                semantic_score  REAL,
                reranker_score  REAL,
                final_score     REAL,
                final_rank      INTEGER,
                model_name      TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_experiments (
                experiment_id  TEXT PRIMARY KEY,
                name           TEXT NOT NULL,
                description    TEXT,
                config_json    TEXT,
                status         TEXT DEFAULT 'pending',
                created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS experiment_results (
                result_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id  TEXT NOT NULL,
                run_id         TEXT,
                metrics_json   TEXT NOT NULL,
                latency_ms     REAL,
                query_count    INTEGER,
                created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES retrieval_experiments(experiment_id)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS ranking_features (
                feature_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                query         TEXT NOT NULL,
                doc_id        INTEGER NOT NULL,
                features_json TEXT NOT NULL,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS query_intents (
                intent_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                query         TEXT NOT NULL,
                intent        TEXT NOT NULL,
                confidence    REAL,
                metadata_json TEXT,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_reports (
                report_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT NOT NULL,
                experiment_id TEXT,
                metrics_json  TEXT NOT NULL,
                config_json   TEXT,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS personalization_profiles (
                user_id               TEXT PRIMARY KEY,
                search_history_json   TEXT DEFAULT '[]',
                click_history_json    TEXT DEFAULT '[]',
                preferences_json      TEXT DEFAULT '{}',
                created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Phase 5 indexes
        for stmt in [
            "CREATE INDEX IF NOT EXISTS idx_reranking_query  ON reranking_logs(query)",
            "CREATE INDEX IF NOT EXISTS idx_reranking_doc    ON reranking_logs(doc_id)",
            "CREATE INDEX IF NOT EXISTS idx_exp_results_exp  ON experiment_results(experiment_id)",
            "CREATE INDEX IF NOT EXISTS idx_features_query   ON ranking_features(query)",
            "CREATE INDEX IF NOT EXISTS idx_intents_query    ON query_intents(query)",
        ]:
            c.execute(stmt)

        # ── Phase 6 tables ─────────────────────────────────────────────────

        c.execute("""
            CREATE TABLE IF NOT EXISTS conversation_sessions (
                session_id    TEXT PRIMARY KEY,
                user_id       TEXT NOT NULL DEFAULT '',
                message_count INTEGER NOT NULL DEFAULT 0,
                is_active     INTEGER NOT NULL DEFAULT 1,
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS conversation_messages (
                message_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT NOT NULL REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,
                role          TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
                content       TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at    TEXT NOT NULL
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS citations (
                citation_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT,
                query           TEXT NOT NULL,
                doc_id          INTEGER,
                chunk_id        TEXT,
                citation_index  INTEGER NOT NULL DEFAULT 1,
                snippet         TEXT,
                relevance_score REAL,
                created_at      TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS grounding_reports (
                report_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id         TEXT,
                query              TEXT NOT NULL,
                grounding_score    REAL NOT NULL DEFAULT 0,
                support_score      REAL NOT NULL DEFAULT 0,
                hallucination_risk TEXT NOT NULL DEFAULT 'medium',
                report_json        TEXT,
                created_at         TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS rag_evaluations (
                eval_id               INTEGER PRIMARY KEY AUTOINCREMENT,
                query                 TEXT NOT NULL,
                faithfulness          REAL,
                groundedness          REAL,
                answer_relevance      REAL,
                context_precision     REAL,
                context_recall        REAL,
                citation_accuracy     REAL,
                response_completeness REAL,
                overall_score         REAL,
                eval_json             TEXT,
                created_at            TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS answer_confidence (
                confidence_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id           TEXT,
                query                TEXT NOT NULL,
                retrieval_confidence REAL,
                context_confidence   REAL,
                grounding_confidence REAL,
                citation_confidence  REAL,
                overall_confidence   REAL,
                tier                 TEXT,
                created_at           TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS memory_snapshots (
                snapshot_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT NOT NULL,
                snapshot_type TEXT NOT NULL DEFAULT 'full',
                content_json  TEXT NOT NULL,
                message_count INTEGER NOT NULL DEFAULT 0,
                created_at    TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        # Phase 6 indexes
        for stmt in [
            "CREATE INDEX IF NOT EXISTS idx_conv_sessions_user    ON conversation_sessions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_conv_messages_session ON conversation_messages(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_citations_session     ON citations(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_grounding_session     ON grounding_reports(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_rag_eval_score        ON rag_evaluations(overall_score)",
            "CREATE INDEX IF NOT EXISTS idx_confidence_session    ON answer_confidence(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_memory_session        ON memory_snapshots(session_id)",
        ]:
            c.execute(stmt)

        # ── Phase 7 tables ─────────────────────────────────────────────────

        c.execute("""
            CREATE TABLE IF NOT EXISTS agent_tasks (
                task_id       TEXT PRIMARY KEY,
                task_type     TEXT NOT NULL,
                agent_type    TEXT NOT NULL,
                goal          TEXT NOT NULL,
                status        TEXT NOT NULL DEFAULT 'pending',
                priority      INTEGER NOT NULL DEFAULT 5,
                params_json   TEXT DEFAULT '{}',
                parent_id     TEXT,
                created_at    TEXT NOT NULL DEFAULT (datetime('now')),
                completed_at  TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS agent_runs (
                run_id        TEXT PRIMARY KEY,
                task_id       TEXT NOT NULL,
                agent_type    TEXT NOT NULL,
                status        TEXT NOT NULL,
                output_json   TEXT,
                error         TEXT,
                confidence    REAL DEFAULT 0,
                latency_ms    REAL DEFAULT 0,
                attempts      INTEGER DEFAULT 1,
                created_at    TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (task_id) REFERENCES agent_tasks(task_id)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS workflow_runs (
                run_id          TEXT PRIMARY KEY,
                workflow_name   TEXT NOT NULL,
                goal            TEXT NOT NULL,
                status          TEXT NOT NULL DEFAULT 'pending',
                total_steps     INTEGER DEFAULT 0,
                completed_steps INTEGER DEFAULT 0,
                failed_steps    INTEGER DEFAULT 0,
                total_latency_ms REAL DEFAULT 0,
                metadata_json   TEXT DEFAULT '{}',
                created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                finished_at     TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS evidence_records (
                evidence_id   TEXT PRIMARY KEY,
                doc_id        INTEGER,
                chunk_id      TEXT,
                claim         TEXT,
                content       TEXT,
                score         REAL DEFAULT 0,
                confidence    REAL DEFAULT 0,
                source_title  TEXT,
                validated     INTEGER DEFAULT 0,
                tags_json     TEXT DEFAULT '[]',
                session_id    TEXT,
                created_at    TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS research_sessions (
                session_id    TEXT PRIMARY KEY,
                user_id       TEXT DEFAULT '',
                goal          TEXT NOT NULL DEFAULT '',
                status        TEXT DEFAULT 'active',
                task_count    INTEGER DEFAULT 0,
                evidence_count INTEGER DEFAULT 0,
                event_count   INTEGER DEFAULT 0,
                snapshot_json TEXT DEFAULT '{}',
                created_at    TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS citation_validation_reports (
                report_id           TEXT PRIMARY KEY,
                session_id          TEXT,
                citation_accuracy   REAL DEFAULT 0,
                total_citations     INTEGER DEFAULT 0,
                supported_count     INTEGER DEFAULT 0,
                unsupported_count   INTEGER DEFAULT 0,
                drift_count         INTEGER DEFAULT 0,
                report_json         TEXT DEFAULT '{}',
                created_at          TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS research_reports (
                report_id       TEXT PRIMARY KEY,
                session_id      TEXT,
                workflow_run_id TEXT,
                goal            TEXT,
                strategy        TEXT,
                format          TEXT DEFAULT 'markdown',
                report_text     TEXT,
                evidence_used   INTEGER DEFAULT 0,
                quality_json    TEXT DEFAULT '{}',
                created_at      TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS agent_metrics (
                metric_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_type    TEXT NOT NULL,
                task_id       TEXT,
                latency_ms    REAL DEFAULT 0,
                success       INTEGER DEFAULT 0,
                token_count   INTEGER DEFAULT 0,
                metadata_json TEXT DEFAULT '{}',
                created_at    TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        # Phase 7 indexes
        for stmt in [
            "CREATE INDEX IF NOT EXISTS idx_agent_tasks_type    ON agent_tasks(agent_type)",
            "CREATE INDEX IF NOT EXISTS idx_agent_tasks_status  ON agent_tasks(status)",
            "CREATE INDEX IF NOT EXISTS idx_agent_runs_task     ON agent_runs(task_id)",
            "CREATE INDEX IF NOT EXISTS idx_agent_runs_type     ON agent_runs(agent_type)",
            "CREATE INDEX IF NOT EXISTS idx_workflow_runs_name  ON workflow_runs(workflow_name)",
            "CREATE INDEX IF NOT EXISTS idx_evidence_doc        ON evidence_records(doc_id)",
            "CREATE INDEX IF NOT EXISTS idx_evidence_session    ON evidence_records(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_research_sess_user  ON research_sessions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_citval_session      ON citation_validation_reports(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_reports_session     ON research_reports(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_agent_metrics_type  ON agent_metrics(agent_type)",
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

    # ── Phase 5: Reranking logs ─────────────────────────────────────────────

    def log_reranking(self, query: str, doc_id: int, bm25_score: float,
                      semantic_score: float, reranker_score: float,
                      final_score: float, final_rank: int,
                      model_name: str) -> int:
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO reranking_logs "
            "(query, doc_id, bm25_score, semantic_score, reranker_score, "
            "final_score, final_rank, model_name) VALUES (?,?,?,?,?,?,?,?)",
            (query, doc_id, bm25_score, semantic_score, reranker_score,
             final_score, final_rank, model_name),
        )
        self.conn.commit()
        return c.lastrowid

    def get_reranking_stats(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT query, AVG(reranker_score) as avg_score, COUNT(*) as cnt "
            "FROM reranking_logs GROUP BY query ORDER BY cnt DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Phase 5: Experiments ────────────────────────────────────────────────

    def upsert_experiment(self, experiment_id: str, name: str,
                          description: str, config_json: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO retrieval_experiments "
            "(experiment_id, name, description, config_json) VALUES (?,?,?,?)",
            (experiment_id, name, description, config_json),
        )
        self.conn.commit()

    def insert_experiment_result(self, experiment_id: str, run_id: str,
                                  metrics_json: str, latency_ms: float,
                                  query_count: int) -> int:
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO experiment_results "
            "(experiment_id, run_id, metrics_json, latency_ms, query_count) "
            "VALUES (?,?,?,?,?)",
            (experiment_id, run_id, metrics_json, latency_ms, query_count),
        )
        self.conn.commit()
        return c.lastrowid

    def get_experiments(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM retrieval_experiments ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_experiment_results(self, experiment_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM experiment_results WHERE experiment_id = ? "
            "ORDER BY created_at DESC", (experiment_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Phase 5: Ranking features ───────────────────────────────────────────

    def insert_ranking_features(self, query: str, doc_id: int,
                                 features_json: str) -> None:
        self.conn.execute(
            "INSERT INTO ranking_features (query, doc_id, features_json) VALUES (?,?,?)",
            (query, doc_id, features_json),
        )
        self.conn.commit()

    # ── Phase 5: Query intents ──────────────────────────────────────────────

    def log_query_intent(self, query: str, intent: str, confidence: float,
                          metadata_json: str = "{}") -> None:
        self.conn.execute(
            "INSERT INTO query_intents (query, intent, confidence, metadata_json) "
            "VALUES (?,?,?,?)",
            (query, intent, confidence, metadata_json),
        )
        self.conn.commit()

    def get_intent_distribution(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT intent, COUNT(*) as cnt, AVG(confidence) as avg_confidence "
            "FROM query_intents GROUP BY intent ORDER BY cnt DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Phase 5: Evaluation reports ─────────────────────────────────────────

    def save_evaluation_report(self, name: str, metrics_json: str,
                                config_json: str = "{}",
                                experiment_id: str | None = None) -> int:
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO evaluation_reports (name, experiment_id, metrics_json, config_json) "
            "VALUES (?,?,?,?)",
            (name, experiment_id, metrics_json, config_json),
        )
        self.conn.commit()
        return c.lastrowid

    def get_evaluation_reports(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM evaluation_reports ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Phase 5: Personalization ────────────────────────────────────────────

    def get_user_profile(self, user_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM personalization_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        return dict(row) if row else None

    def upsert_user_profile(self, user_id: str, search_history_json: str,
                             click_history_json: str, preferences_json: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO personalization_profiles "
            "(user_id, search_history_json, click_history_json, preferences_json, "
            "updated_at) VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
            (user_id, search_history_json, click_history_json, preferences_json),
        )
        self.conn.commit()

    def get_all_user_profiles(self, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            "SELECT user_id, created_at, updated_at FROM personalization_profiles "
            "ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Phase 6: Conversation Sessions ─────────────────────────────────────────

    def create_conversation_session(self, session_id: str, user_id: str,
                                     created_at: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO conversation_sessions "
            "(session_id, user_id, created_at, updated_at) VALUES (?,?,?,?)",
            (session_id, user_id, created_at, created_at),
        )
        self.conn.commit()

    def get_conversation_session(self, session_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM conversation_sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_session_timestamp(self, session_id: str, updated_at: str) -> None:
        self.conn.execute(
            "UPDATE conversation_sessions SET updated_at=?, "
            "message_count = (SELECT COUNT(*) FROM conversation_messages WHERE session_id=?) "
            "WHERE session_id=?",
            (updated_at, session_id, session_id),
        )
        self.conn.commit()

    def delete_conversation_session(self, session_id: str) -> bool:
        c = self.conn.cursor()
        c.execute("DELETE FROM conversation_messages WHERE session_id=?", (session_id,))
        c.execute("DELETE FROM conversation_sessions WHERE session_id=?", (session_id,))
        self.conn.commit()
        return c.rowcount > 0

    def list_conversation_sessions(self, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            "SELECT session_id, user_id, message_count, is_active, created_at, updated_at "
            "FROM conversation_sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Phase 6: Conversation Messages ─────────────────────────────────────────

    def insert_conversation_message(self, session_id: str, role: str,
                                     content: str, metadata_json: str,
                                     created_at: str) -> int:
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO conversation_messages "
            "(session_id, role, content, metadata_json, created_at) VALUES (?,?,?,?,?)",
            (session_id, role, content, metadata_json, created_at),
        )
        self.conn.commit()
        return c.lastrowid

    def get_conversation_messages(self, session_id: str,
                                   limit: int = 200) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM conversation_messages WHERE session_id=? "
            "ORDER BY message_id ASC LIMIT ?",
            (session_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Phase 6: Citations ──────────────────────────────────────────────────────

    def insert_citation(self, session_id: str, query: str, doc_id: int,
                         chunk_id: str, citation_index: int,
                         snippet: str, relevance_score: float) -> int:
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO citations "
            "(session_id, query, doc_id, chunk_id, citation_index, snippet, relevance_score) "
            "VALUES (?,?,?,?,?,?,?)",
            (session_id, query, doc_id, chunk_id, citation_index, snippet, relevance_score),
        )
        self.conn.commit()
        return c.lastrowid

    def get_citations_for_session(self, session_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM citations WHERE session_id=? ORDER BY created_at DESC",
            (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Phase 6: Grounding Reports ─────────────────────────────────────────────

    def insert_grounding_report(self, session_id: str, query: str,
                                 grounding_score: float, support_score: float,
                                 hallucination_risk: str, report_json: str) -> int:
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO grounding_reports "
            "(session_id, query, grounding_score, support_score, "
            "hallucination_risk, report_json) VALUES (?,?,?,?,?,?)",
            (session_id, query, grounding_score, support_score,
             hallucination_risk, report_json),
        )
        self.conn.commit()
        return c.lastrowid

    def get_grounding_stats(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) AS total, AVG(grounding_score) AS avg_score, "
            "SUM(CASE WHEN hallucination_risk='high' THEN 1 ELSE 0 END) AS high_risk "
            "FROM grounding_reports"
        ).fetchone()
        return dict(row) if row else {}

    # ── Phase 6: RAG Evaluations ───────────────────────────────────────────────

    def insert_rag_evaluation(self, query: str, faithfulness: float,
                               groundedness: float, answer_relevance: float,
                               context_precision: float, context_recall: float,
                               citation_accuracy: float,
                               response_completeness: float,
                               overall_score: float, eval_json: str) -> int:
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO rag_evaluations (query, faithfulness, groundedness, "
            "answer_relevance, context_precision, context_recall, "
            "citation_accuracy, response_completeness, overall_score, eval_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (query, faithfulness, groundedness, answer_relevance,
             context_precision, context_recall, citation_accuracy,
             response_completeness, overall_score, eval_json),
        )
        self.conn.commit()
        return c.lastrowid

    def get_rag_eval_stats(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) AS total, AVG(overall_score) AS avg_overall, "
            "AVG(faithfulness) AS avg_faithfulness, "
            "AVG(groundedness) AS avg_groundedness "
            "FROM rag_evaluations"
        ).fetchone()
        return dict(row) if row else {}

    # ── Phase 6: Answer Confidence ─────────────────────────────────────────────

    def insert_answer_confidence(self, session_id: str, query: str,
                                  retrieval_conf: float, context_conf: float,
                                  grounding_conf: float, citation_conf: float,
                                  overall_conf: float, tier: str) -> int:
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO answer_confidence "
            "(session_id, query, retrieval_confidence, context_confidence, "
            "grounding_confidence, citation_confidence, overall_confidence, tier) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (session_id, query, retrieval_conf, context_conf,
             grounding_conf, citation_conf, overall_conf, tier),
        )
        self.conn.commit()
        return c.lastrowid

    def get_confidence_stats(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) AS total, AVG(overall_confidence) AS avg_confidence, "
            "SUM(CASE WHEN tier='high' THEN 1 ELSE 0 END) AS high_count, "
            "SUM(CASE WHEN tier='low'  THEN 1 ELSE 0 END) AS low_count "
            "FROM answer_confidence"
        ).fetchone()
        return dict(row) if row else {}

    # ── Phase 6: Memory Snapshots ──────────────────────────────────────────────

    def insert_memory_snapshot(self, session_id: str, snapshot_type: str,
                                content_json: str, message_count: int) -> int:
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO memory_snapshots "
            "(session_id, snapshot_type, content_json, message_count) "
            "VALUES (?,?,?,?)",
            (session_id, snapshot_type, content_json, message_count),
        )
        self.conn.commit()
        return c.lastrowid

    # ── Phase 7: Agent Tasks ──────────────────────────────────────────────────

    def insert_agent_task(self, task_id: str, task_type: str, agent_type: str,
                          goal: str, status: str = "pending", priority: int = 5,
                          params_json: str = "{}", parent_id: str | None = None) -> str:
        self.conn.execute(
            "INSERT OR REPLACE INTO agent_tasks "
            "(task_id, task_type, agent_type, goal, status, priority, params_json, parent_id) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (task_id, task_type, agent_type, goal, status, priority, params_json, parent_id),
        )
        self.conn.commit()
        return task_id

    def update_agent_task_status(self, task_id: str, status: str) -> None:
        self.conn.execute(
            "UPDATE agent_tasks SET status = ?, completed_at = datetime('now') WHERE task_id = ?",
            (status, task_id),
        )
        self.conn.commit()

    def get_agent_tasks(self, status: str | None = None, limit: int = 50) -> list[dict]:
        if status:
            rows = self.conn.execute(
                "SELECT * FROM agent_tasks WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM agent_tasks ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Phase 7: Agent Runs ───────────────────────────────────────────────────

    def insert_agent_run(self, run_id: str, task_id: str, agent_type: str,
                         status: str, output_json: str = None,
                         error: str = None, confidence: float = 0,
                         latency_ms: float = 0, attempts: int = 1) -> str:
        self.conn.execute(
            "INSERT INTO agent_runs "
            "(run_id, task_id, agent_type, status, output_json, error, confidence, latency_ms, attempts) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, task_id, agent_type, status, output_json, error, confidence, latency_ms, attempts),
        )
        self.conn.commit()
        return run_id

    def get_agent_runs(self, agent_type: str | None = None, limit: int = 50) -> list[dict]:
        if agent_type:
            rows = self.conn.execute(
                "SELECT * FROM agent_runs WHERE agent_type = ? ORDER BY created_at DESC LIMIT ?",
                (agent_type, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM agent_runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Phase 7: Workflow Runs ────────────────────────────────────────────────

    def insert_workflow_run(self, run_id: str, workflow_name: str, goal: str,
                            status: str = "pending", total_steps: int = 0,
                            metadata_json: str = "{}") -> str:
        self.conn.execute(
            "INSERT INTO workflow_runs "
            "(run_id, workflow_name, goal, status, total_steps, metadata_json) "
            "VALUES (?,?,?,?,?,?)",
            (run_id, workflow_name, goal, status, total_steps, metadata_json),
        )
        self.conn.commit()
        return run_id

    def update_workflow_run(self, run_id: str, status: str,
                            completed_steps: int = 0, failed_steps: int = 0,
                            total_latency_ms: float = 0) -> None:
        self.conn.execute(
            "UPDATE workflow_runs SET status = ?, completed_steps = ?, "
            "failed_steps = ?, total_latency_ms = ?, finished_at = datetime('now') "
            "WHERE run_id = ?",
            (status, completed_steps, failed_steps, total_latency_ms, run_id),
        )
        self.conn.commit()

    def get_workflow_runs(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM workflow_runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Phase 7: Evidence Records ─────────────────────────────────────────────

    def insert_evidence_record(self, evidence_id: str, doc_id: int,
                                chunk_id: str, claim: str, content: str,
                                score: float, confidence: float,
                                source_title: str, validated: bool,
                                tags_json: str, session_id: str = "") -> str:
        self.conn.execute(
            "INSERT OR REPLACE INTO evidence_records "
            "(evidence_id, doc_id, chunk_id, claim, content, score, confidence, "
            "source_title, validated, tags_json, session_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (evidence_id, doc_id, chunk_id, claim, content, score, confidence,
             source_title, int(validated), tags_json, session_id),
        )
        self.conn.commit()
        return evidence_id

    def get_evidence_by_session(self, session_id: str, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM evidence_records WHERE session_id = ? ORDER BY score DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Phase 7: Research Sessions ────────────────────────────────────────────

    def insert_research_session(self, session_id: str, user_id: str = "",
                                 goal: str = "") -> str:
        self.conn.execute(
            "INSERT OR REPLACE INTO research_sessions "
            "(session_id, user_id, goal) VALUES (?,?,?)",
            (session_id, user_id, goal),
        )
        self.conn.commit()
        return session_id

    def update_research_session(self, session_id: str, status: str = "active",
                                 task_count: int = 0, evidence_count: int = 0,
                                 event_count: int = 0, snapshot_json: str = "{}") -> None:
        self.conn.execute(
            "UPDATE research_sessions SET status = ?, task_count = ?, "
            "evidence_count = ?, event_count = ?, snapshot_json = ?, "
            "updated_at = datetime('now') WHERE session_id = ?",
            (status, task_count, evidence_count, event_count, snapshot_json, session_id),
        )
        self.conn.commit()

    def get_research_sessions(self, user_id: str | None = None, limit: int = 20) -> list[dict]:
        if user_id:
            rows = self.conn.execute(
                "SELECT * FROM research_sessions WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM research_sessions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Phase 7: Research Reports ─────────────────────────────────────────────

    def insert_research_report(self, report_id: str, session_id: str = "",
                                workflow_run_id: str = "", goal: str = "",
                                strategy: str = "", fmt: str = "markdown",
                                report_text: str = "", evidence_used: int = 0,
                                quality_json: str = "{}") -> str:
        self.conn.execute(
            "INSERT INTO research_reports "
            "(report_id, session_id, workflow_run_id, goal, strategy, format, "
            "report_text, evidence_used, quality_json) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (report_id, session_id, workflow_run_id, goal, strategy, fmt,
             report_text, evidence_used, quality_json),
        )
        self.conn.commit()
        return report_id

    def get_research_reports(self, session_id: str | None = None, limit: int = 20) -> list[dict]:
        if session_id:
            rows = self.conn.execute(
                "SELECT * FROM research_reports WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM research_reports ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Phase 7: Agent Metrics ────────────────────────────────────────────────

    def insert_agent_metric(self, agent_type: str, task_id: str = "",
                             latency_ms: float = 0, success: bool = True,
                             token_count: int = 0, metadata_json: str = "{}") -> int:
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO agent_metrics "
            "(agent_type, task_id, latency_ms, success, token_count, metadata_json) "
            "VALUES (?,?,?,?,?,?)",
            (agent_type, task_id, latency_ms, int(success), token_count, metadata_json),
        )
        self.conn.commit()
        return c.lastrowid

    def get_agent_metrics_summary(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) AS total, "
            "AVG(latency_ms) AS avg_latency, "
            "SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) AS success_count, "
            "SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) AS failure_count "
            "FROM agent_metrics"
        ).fetchone()
        return dict(row) if row else {}
