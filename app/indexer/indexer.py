"""
Indexer — Phase 3: Field-Aware

Now builds separate postings for 'title' and 'body' fields.
The BM25 ranker and the relevance-tuning framework can boost
title matches independently of body matches.

All Phase 2 behaviour is preserved — the only new behaviour is:
  1. index_document() accepts an optional title_weight parameter
  2. Postings now carry a `field` column ('title' | 'body')
  3. The BM25 ranker (and relevance tuner) read field-specific postings
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

    Field-aware indexing stores separate postings for 'title' and 'body'
    so downstream rankers can weight field matches differently.
    """

    def __init__(self, db: Database, tokenizer: Tokenizer):
        self.db = db
        self.tokenizer = tokenizer

    def index_document(
        self,
        title: str,
        content: str,
        source: str = "local",
        doc_type: str = "text",
    ) -> IndexResult:
        """
        Index a document with field-aware postings:
          - 'title' field from the document title
          - 'body'  field from the document content
        """
        # Tokenise body
        body_result = self.tokenizer.tokenize(content)

        doc_id = self.db.insert_document(
            title=title, content=content, source=source,
            doc_type=doc_type, word_count=body_result.token_count,
        )

        postings_batch: list[tuple] = []

        # ── Body postings ──────────────────────────────────────────────────
        body_freqs = self.tokenizer.get_term_frequencies(body_result.tokens)
        for term, freq in body_freqs.items():
            term_id = self.db.get_or_create_term(term)
            positions = body_result.positions.get(term, [])
            postings_batch.append(
                (term_id, doc_id, freq, json.dumps(positions), "body")
            )

        # ── Title postings ─────────────────────────────────────────────────
        title_result = self.tokenizer.tokenize(title)
        title_freqs = self.tokenizer.get_term_frequencies(title_result.tokens)
        for term, freq in title_freqs.items():
            term_id = self.db.get_or_create_term(term)
            positions = title_result.positions.get(term, [])
            postings_batch.append(
                (term_id, doc_id, freq, json.dumps(positions), "title")
            )

        if postings_batch:
            self.db.batch_insert_postings(postings_batch)

        all_terms = set(body_freqs) | set(title_freqs)
        self._update_document_frequencies(all_terms)

        logger.info(
            "Indexed doc %d (%s): body=%d terms, title=%d terms, %d tokens",
            doc_id, title, len(body_freqs), len(title_freqs), body_result.token_count,
        )

        return IndexResult(
            doc_id=doc_id,
            title=title,
            terms_indexed=len(all_terms),
            total_tokens=body_result.token_count,
        )

    def index_directory(self, directory: Path) -> list[IndexResult]:
        """Index all .txt files in a directory, skipping already-indexed ones."""
        results: list[IndexResult] = []
        if not directory.exists():
            logger.warning("Directory does not exist: %s", directory)
            return results

        for filepath in sorted(directory.glob("*.txt")):
            source = str(filepath)
            if self.db.document_exists_by_source(source):
                logger.debug("Skipping already indexed: %s", source)
                continue
            content = filepath.read_text(encoding="utf-8", errors="replace")
            if not content.strip():
                continue
            result = self.index_document(
                title=filepath.stem,
                content=content,
                source=source,
                doc_type="text",
            )
            results.append(result)

        logger.info("Indexed %d new documents from %s", len(results), directory)
        return results

    def reindex_document(self, doc_id: int, content: str) -> IndexResult:
        """Delete old postings and rebuild for an existing document (in-place)."""
        doc = self.db.get_document(doc_id)
        if doc is None:
            raise ValueError(f"Document {doc_id} not found")

        # Use Database abstraction instead of accessing db.conn directly
        self.db.clear_postings_for_doc(doc_id)

        body_result = self.tokenizer.tokenize(content)
        self.db.update_document_content(doc_id, content, body_result.token_count)

        postings_batch: list[tuple] = []
        body_freqs = self.tokenizer.get_term_frequencies(body_result.tokens)
        for term, freq in body_freqs.items():
            term_id = self.db.get_or_create_term(term)
            positions = body_result.positions.get(term, [])
            postings_batch.append((term_id, doc_id, freq, json.dumps(positions), "body"))

        title_result = self.tokenizer.tokenize(doc.title)
        title_freqs = self.tokenizer.get_term_frequencies(title_result.tokens)
        for term, freq in title_freqs.items():
            term_id = self.db.get_or_create_term(term)
            positions = title_result.positions.get(term, [])
            postings_batch.append((term_id, doc_id, freq, json.dumps(positions), "title"))
            postings_batch.append((term_id, doc_id, freq, json.dumps(positions), "title"))

        if postings_batch:
            self.db.batch_insert_postings(postings_batch)

        all_terms = set(body_freqs) | set(title_freqs)
        self._update_document_frequencies(all_terms)

        return IndexResult(
            doc_id=doc_id,
            title=doc.title,
            terms_indexed=len(all_terms),
            total_tokens=body_result.token_count,
        )

    def _update_document_frequencies(self, terms: set[str]) -> None:
        """
        Update document_frequency for all affected terms.
        Delegates to a single-SQL batch update instead of N individual queries.
        """
        self.db.batch_update_document_frequencies(list(terms))

    def get_inverted_index_snapshot(self) -> dict[str, list[int]]:
        """Readable snapshot: term → [doc_ids].  Delegates to Database."""
        return self.db.get_inverted_index_snapshot()
