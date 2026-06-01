"""
Database Layer

=== THEORY ===

Every search engine needs persistent storage for two things:
1. The original documents (so we can return them in results)
2. The inverted index (so we can look up which documents contain a term)

We use SQLite because it's a single-file embedded database — no server needed.
The schema mirrors what a production search engine stores, just at smaller scale.

=== TABLES ===

documents: Stores raw document content and metadata.
    - Each document gets a unique integer ID (doc_id).
    - We store the original content so we can return snippets in search results.

terms: A dictionary of every unique term we've seen after tokenization.
    - Maps each term string to a unique integer ID (term_id).
    - This saves space in the postings table: we store integer IDs instead of
      repeating the full term string millions of times.

postings: The core of the inverted index.
    - Each row says: "term T appears in document D with frequency F at positions P."
    - This is what makes search fast: instead of scanning every document for a term,
      we look up the term and immediately get all documents containing it.

crawled_pages: Stores raw HTML and metadata from the web crawler.
    - Keeps crawl state separate from indexed content.

=== AT GOOGLE SCALE ===

Google doesn't use SQLite. They use:
- Bigtable / Spanner for distributed storage
- SSTable files for the inverted index on disk
- Custom columnar formats optimized for sequential reads
- The index is sharded across thousands of machines

But the logical structure is the same: documents table + inverted index.

=== COMPLEXITY ===

- Insert document: O(1) amortized
- Lookup term -> postings: O(log N) where N = number of terms (B-tree index)
- Lookup document by ID: O(1) (primary key)
"""

import sqlite3
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


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


class Database:
    """
    SQLite-backed storage for documents, terms, postings, and crawled pages.

    Uses connection pooling via a single persistent connection with WAL mode
    for better concurrent read performance.
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
        self._create_tables()
        logger.info("Database connected at %s", self.db_path)

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")

    def _create_tables(self) -> None:
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'local',
                doc_type TEXT NOT NULL DEFAULT 'text',
                word_count INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS terms (
                term_id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT NOT NULL UNIQUE,
                document_frequency INTEGER NOT NULL DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS postings (
                term_id INTEGER NOT NULL,
                doc_id INTEGER NOT NULL,
                term_frequency INTEGER NOT NULL DEFAULT 0,
                positions TEXT NOT NULL DEFAULT '[]',
                PRIMARY KEY (term_id, doc_id),
                FOREIGN KEY (term_id) REFERENCES terms(term_id),
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crawled_pages (
                page_id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL UNIQUE,
                title TEXT,
                content TEXT,
                html TEXT,
                status_code INTEGER,
                crawl_depth INTEGER NOT NULL DEFAULT 0,
                crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                doc_id INTEGER,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_terms_term ON terms(term)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_postings_doc ON postings(doc_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_crawled_url ON crawled_pages(url)
        """)

        self.conn.commit()
        logger.info("Database tables created")

    # ── Document Operations ──

    def insert_document(self, title: str, content: str, source: str = "local",
                        doc_type: str = "text", word_count: int = 0) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO documents (title, content, source, doc_type, word_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (title, content, source, doc_type, word_count)
        )
        self.conn.commit()
        doc_id = cursor.lastrowid
        logger.debug("Inserted document %d: %s", doc_id, title)
        return doc_id

    def get_document(self, doc_id: int) -> Optional[DocumentRecord]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return DocumentRecord(
            doc_id=row["doc_id"], title=row["title"], content=row["content"],
            source=row["source"], doc_type=row["doc_type"],
            word_count=row["word_count"], created_at=row["created_at"]
        )

    def get_all_documents(self) -> list[DocumentRecord]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents ORDER BY doc_id")
        return [
            DocumentRecord(
                doc_id=row["doc_id"], title=row["title"], content=row["content"],
                source=row["source"], doc_type=row["doc_type"],
                word_count=row["word_count"], created_at=row["created_at"]
            )
            for row in cursor.fetchall()
        ]

    def get_document_count(self) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM documents")
        return cursor.fetchone()["cnt"]

    def document_exists_by_source(self, source: str) -> bool:
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM documents WHERE source = ? LIMIT 1", (source,))
        return cursor.fetchone() is not None

    def delete_document(self, doc_id: int) -> bool:
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM postings WHERE doc_id = ?", (doc_id,))
        cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self.conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            self._recalculate_document_frequencies()
        return deleted

    # ── Term Operations ──

    def get_or_create_term(self, term: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT term_id FROM terms WHERE term = ?", (term,))
        row = cursor.fetchone()
        if row:
            return row["term_id"]
        cursor.execute(
            "INSERT INTO terms (term, document_frequency) VALUES (?, 0)", (term,)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_term(self, term: str) -> Optional[TermRecord]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM terms WHERE term = ?", (term,))
        row = cursor.fetchone()
        if row is None:
            return None
        return TermRecord(
            term_id=row["term_id"], term=row["term"],
            document_frequency=row["document_frequency"]
        )

    def get_term_count(self) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM terms")
        return cursor.fetchone()["cnt"]

    def update_document_frequency(self, term_id: int, df: int) -> None:
        self.conn.execute(
            "UPDATE terms SET document_frequency = ? WHERE term_id = ?",
            (df, term_id)
        )
        self.conn.commit()

    # ── Posting Operations ──

    def insert_posting(self, term_id: int, doc_id: int,
                       term_frequency: int, positions: list[int]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO postings (term_id, doc_id, term_frequency, positions) "
            "VALUES (?, ?, ?, ?)",
            (term_id, doc_id, term_frequency, json.dumps(positions))
        )
        self.conn.commit()

    def get_postings_for_term(self, term: str) -> list[PostingRecord]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT p.term_id, p.doc_id, p.term_frequency, p.positions
            FROM postings p
            JOIN terms t ON p.term_id = t.term_id
            WHERE t.term = ?
            ORDER BY p.doc_id
        """, (term,))
        return [
            PostingRecord(
                term_id=row["term_id"], doc_id=row["doc_id"],
                term_frequency=row["term_frequency"],
                positions=json.loads(row["positions"])
            )
            for row in cursor.fetchall()
        ]

    def get_postings_for_doc(self, doc_id: int) -> list[PostingRecord]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM postings WHERE doc_id = ? ORDER BY term_id",
            (doc_id,)
        )
        return [
            PostingRecord(
                term_id=row["term_id"], doc_id=row["doc_id"],
                term_frequency=row["term_frequency"],
                positions=json.loads(row["positions"])
            )
            for row in cursor.fetchall()
        ]

    def get_posting_count(self) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM postings")
        return cursor.fetchone()["cnt"]

    def batch_insert_postings(self, postings: list[tuple]) -> None:
        """Insert multiple postings at once. Each tuple: (term_id, doc_id, tf, positions_json)."""
        self.conn.executemany(
            "INSERT OR REPLACE INTO postings (term_id, doc_id, term_frequency, positions) "
            "VALUES (?, ?, ?, ?)",
            postings
        )
        self.conn.commit()

    def _recalculate_document_frequencies(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE terms SET document_frequency = (
                SELECT COUNT(DISTINCT doc_id) FROM postings WHERE postings.term_id = terms.term_id
            )
        """)
        self.conn.commit()

    # ── Crawled Page Operations ──

    def insert_crawled_page(self, url: str, title: str, content: str,
                            html: str, status_code: int,
                            crawl_depth: int, doc_id: Optional[int] = None) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO crawled_pages "
            "(url, title, content, html, status_code, crawl_depth, doc_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (url, title, content, html, status_code, crawl_depth, doc_id)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_crawled_page_by_url(self, url: str) -> Optional[CrawledPageRecord]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM crawled_pages WHERE url = ?", (url,))
        row = cursor.fetchone()
        if row is None:
            return None
        return CrawledPageRecord(
            page_id=row["page_id"], url=row["url"], title=row["title"],
            content=row["content"], html=row["html"],
            status_code=row["status_code"], crawl_depth=row["crawl_depth"],
            crawled_at=row["crawled_at"], doc_id=row["doc_id"]
        )

    def get_crawled_page_count(self) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM crawled_pages")
        return cursor.fetchone()["cnt"]

    def url_already_crawled(self, url: str) -> bool:
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM crawled_pages WHERE url = ? LIMIT 1", (url,))
        return cursor.fetchone() is not None

    def get_crawl_stats(self) -> dict:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as total FROM crawled_pages")
        total = cursor.fetchone()["total"]
        cursor.execute("SELECT COUNT(*) as indexed FROM crawled_pages WHERE doc_id IS NOT NULL")
        indexed = cursor.fetchone()["indexed"]
        cursor.execute("SELECT AVG(status_code) as avg_status FROM crawled_pages")
        avg_status = cursor.fetchone()["avg_status"]
        return {
            "total_pages_crawled": total,
            "pages_indexed": indexed,
            "average_status_code": round(avg_status, 1) if avg_status else 0
        }

    # ── Stats ──

    def get_stats(self) -> dict:
        return {
            "total_documents": self.get_document_count(),
            "total_terms": self.get_term_count(),
            "total_postings": self.get_posting_count(),
            "total_crawled_pages": self.get_crawled_page_count(),
        }
