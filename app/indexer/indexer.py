"""
Indexer

=== THEORY ===

The inverted index is the most important data structure in information retrieval.

A "forward index" maps:  document → list of terms it contains
An "inverted index" maps: term → list of documents containing it

Example inverted index:
    "python"  → [doc1, doc4, doc7]
    "search"  → [doc2, doc5]
    "engine"  → [doc2, doc5, doc7]

WHY inverted (not forward)?
Forward index requires scanning every document for every query — O(N * D).
Inverted index jumps directly to the relevant documents — O(1) lookup + O(k) results.

=== POSTING LISTS ===

Each entry in the inverted index is a "posting list." Each posting stores:
- doc_id: which document
- term_frequency: how many times the term appears (needed for TF-IDF)
- positions: where in the document the term appears (needed for phrase queries)

Example posting list for "python":
    [
        {doc_id: 1, tf: 3, positions: [0, 15, 42]},
        {doc_id: 4, tf: 1, positions: [7]},
        {doc_id: 7, tf: 5, positions: [0, 3, 12, 28, 55]}
    ]

=== INDEXING PROCESS ===

For each document:
1. Tokenize the text → get terms and their positions
2. For each unique term:
   a. Look up or create the term in the terms table
   b. Create a posting: (term_id, doc_id, frequency, positions)
   c. Update the term's document_frequency (how many docs contain it)

=== COMPLEXITY ===

- Index one document of length L: O(L) for tokenization + O(U * log T)
  for U unique terms looked up in T total terms
- Index N documents: O(N * avg_doc_length)
- Space: O(total_postings) ≈ O(N * avg_unique_terms_per_doc)

=== AT GOOGLE SCALE ===

Google's indexing:
- Processes billions of pages using MapReduce / Flume pipelines
- The index is partitioned ("sharded") by document ID across thousands of machines
- Each shard has its own inverted index
- Posting lists are compressed using variable-byte or PForDelta encoding
- Index updates are tiered: a small real-time index handles fresh content,
  periodically merged into the main index
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass

from app.database.db import Database
from app.tokenizer.tokenizer import Tokenizer, TokenizerConfig

logger = logging.getLogger(__name__)


@dataclass
class IndexResult:
    doc_id: int
    title: str
    terms_indexed: int
    total_tokens: int


class Indexer:
    """
    Builds and maintains the inverted index.

    Takes raw text, tokenizes it, and stores the resulting terms and postings
    in the database. Supports both individual document indexing and batch
    operations for initial bulk loading.
    """

    def __init__(self, db: Database, tokenizer: Tokenizer):
        self.db = db
        self.tokenizer = tokenizer

    def index_document(self, title: str, content: str,
                       source: str = "local", doc_type: str = "text") -> IndexResult:
        """
        Index a single document: tokenize → store document → build postings.
        """
        result = self.tokenizer.tokenize(content)

        doc_id = self.db.insert_document(
            title=title, content=content, source=source,
            doc_type=doc_type, word_count=result.token_count
        )

        term_frequencies = self.tokenizer.get_term_frequencies(result.tokens)

        postings_batch: list[tuple] = []
        for term, freq in term_frequencies.items():
            term_id = self.db.get_or_create_term(term)
            positions = result.positions.get(term, [])
            postings_batch.append((term_id, doc_id, freq, json.dumps(positions)))

        if postings_batch:
            self.db.batch_insert_postings(postings_batch)

        self._update_document_frequencies(term_frequencies.keys())

        logger.info(
            "Indexed document %d (%s): %d terms, %d tokens",
            doc_id, title, len(term_frequencies), result.token_count
        )

        return IndexResult(
            doc_id=doc_id, title=title,
            terms_indexed=len(term_frequencies),
            total_tokens=result.token_count
        )

    def index_directory(self, directory: Path) -> list[IndexResult]:
        """
        Index all .txt files in a directory.
        Skips files that are already indexed (by source path).
        """
        results: list[IndexResult] = []
        if not directory.exists():
            logger.warning("Directory does not exist: %s", directory)
            return results

        txt_files = sorted(directory.glob("*.txt"))
        logger.info("Found %d text files in %s", len(txt_files), directory)

        for filepath in txt_files:
            source = str(filepath)
            if self.db.document_exists_by_source(source):
                logger.debug("Skipping already indexed: %s", source)
                continue

            content = filepath.read_text(encoding="utf-8", errors="replace")
            if not content.strip():
                logger.debug("Skipping empty file: %s", source)
                continue

            result = self.index_document(
                title=filepath.stem,
                content=content,
                source=source,
                doc_type="text"
            )
            results.append(result)

        logger.info("Indexed %d new documents from %s", len(results), directory)
        return results

    def reindex_document(self, doc_id: int, content: str) -> IndexResult:
        """
        Re-index an existing document (for updates / recrawl).
        Deletes old postings, recomputes, and stores new ones.
        """
        doc = self.db.get_document(doc_id)
        if doc is None:
            raise ValueError(f"Document {doc_id} not found")

        cursor = self.db.conn.cursor()
        cursor.execute("DELETE FROM postings WHERE doc_id = ?", (doc_id,))

        cursor.execute(
            "UPDATE documents SET content = ?, word_count = ? WHERE doc_id = ?",
            (content, 0, doc_id)
        )
        self.db.conn.commit()

        result = self.tokenizer.tokenize(content)

        cursor.execute(
            "UPDATE documents SET word_count = ? WHERE doc_id = ?",
            (result.token_count, doc_id)
        )
        self.db.conn.commit()

        term_frequencies = self.tokenizer.get_term_frequencies(result.tokens)

        postings_batch: list[tuple] = []
        for term, freq in term_frequencies.items():
            term_id = self.db.get_or_create_term(term)
            positions = result.positions.get(term, [])
            postings_batch.append((term_id, doc_id, freq, json.dumps(positions)))

        if postings_batch:
            self.db.batch_insert_postings(postings_batch)

        self._update_document_frequencies(term_frequencies.keys())

        logger.info("Reindexed document %d: %d terms", doc_id, len(term_frequencies))

        return IndexResult(
            doc_id=doc_id, title=doc.title,
            terms_indexed=len(term_frequencies),
            total_tokens=result.token_count
        )

    def _update_document_frequencies(self, terms: object) -> None:
        """Recalculate document frequency for each term."""
        for term_str in terms:
            term_record = self.db.get_term(term_str)
            if term_record is None:
                continue
            postings = self.db.get_postings_for_term(term_str)
            self.db.update_document_frequency(term_record.term_id, len(postings))

    def get_inverted_index_snapshot(self) -> dict[str, list[int]]:
        """
        Return a readable snapshot of the inverted index.
        Maps term → list of doc_ids. Useful for debugging and education.
        """
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT t.term, GROUP_CONCAT(p.doc_id) as doc_ids
            FROM terms t
            JOIN postings p ON t.term_id = p.term_id
            GROUP BY t.term
            ORDER BY t.term
        """)
        index: dict[str, list[int]] = {}
        for row in cursor.fetchall():
            index[row["term"]] = [int(d) for d in row["doc_ids"].split(",")]
        return index
