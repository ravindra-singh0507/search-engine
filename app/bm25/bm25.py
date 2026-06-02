"""
BM25 Ranking Engine

=== THEORY ===

BM25 ("Best Matching 25") is the de-facto standard retrieval model used by
every major production search engine including Elasticsearch, Lucene, Solr,
and historically Google.

It improves upon TF-IDF in two fundamental ways:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROBLEM 1: TF is linear — BM25 saturates it
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TF-IDF raw TF:   score ∝ f(t,d)
→ a term appearing 100 times scores 100× more than one appearing once.
→ a 1000-word document can always beat a focused 100-word document.

BM25 applies a saturating transformation:

         f(t,d) · (k1 + 1)
tf* = ──────────────────────────────────────────
       f(t,d) + k1 · (1 − b + b · |d| / avgdl)

As f(t,d) → ∞, tf* → (k1+1), so the contribution is bounded.

k1 = saturation speed  (1.2 = fast saturation, 2.0 = slower)
b  = length penalty    (0 = none, 1 = full normalization)
|d| = document length in tokens
avgdl = corpus average document length

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROBLEM 2: Classic IDF can go negative — BM25 fixes that
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Classic IDF:  log(N / df)
→ when df > N/2 (term appears in majority of docs) the result is < 0.
→ a very common term can subtract from the score — counterintuitive.

Robertson–Spärck Jones IDF:

   idf(t) = log( (N − df + 0.5) / (df + 0.5) + 1 )

Always ≥ 0. Smoothed with 0.5 to avoid edge cases at df=0 or df=N.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FULL BM25 FORMULA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  score(d, Q) = Σ_{t∈Q}  idf(t) · f(t,d)·(k1+1) / (f(t,d) + k1·(1−b+b·|d|/avgdl))

=== BM25 vs TF-IDF COMPARISON ===

┌───────────────────┬─────────────────────────┬──────────────────────────────┐
│ Aspect            │ TF-IDF                  │ BM25                         │
├───────────────────┼─────────────────────────┼──────────────────────────────┤
│ TF Saturation     │ Linear (unbounded)      │ Saturating (k1 controls)     │
│ Length Norm       │ Cosine (implicit)       │ Explicit b parameter         │
│ IDF sign          │ Can be negative         │ Always ≥ 0                   │
│ Tunable params    │ None                    │ k1, b                        │
│ Empirical perf    │ Good baseline           │ Consistently outperforms     │
│ Corpus tuning     │ N/A                     │ k1=1.2/b=0.75 is a good start│
└───────────────────┴─────────────────────────┴──────────────────────────────┘

=== COMPLEXITY ===

  Build corpus stats (avgdl):    O(N) — single pass over all documents
  Score one query term:          O(Q) — Q = number of query terms
  Rank M candidates:             O(M·Q)
  Sort:                          O(M log M)
  Total:                         O(N + M·Q + M log M)

=== AT GOOGLE SCALE ===

Google uses BM25F — an extension that computes separate TF scores for
distinct document fields (title, body, anchor text, URL) and then
combines them with field weights before applying the IDF factor.
BM25F is wrapped inside a learning-to-rank model as one feature
alongside hundreds of others (PageRank, click-through, freshness…).
"""

import math
import logging
from dataclasses import dataclass

from app.database.db import Database
from app.tokenizer.tokenizer import Tokenizer
from app.config import BM25Config

logger = logging.getLogger(__name__)


@dataclass
class BM25ScoredDocument:
    doc_id: int
    score: float
    title: str
    snippet: str
    term_scores: dict[str, float]
    field: str = "body"


class BM25Ranker:
    """
    BM25 ranker that reads posting statistics from the SQLite database.

    Corpus statistics (avgdl, total documents) are computed lazily on the
    first ranking call and cached so we don't hit the DB on every query.
    """

    def __init__(self, db: Database, tokenizer: Tokenizer,
                 config: BM25Config | None = None):
        self.db = db
        self.tokenizer = tokenizer
        self.config = config or BM25Config()
        self._avgdl_cache: float | None = None
        self._total_docs_cache: int | None = None

    # ── Public API ────────────────────────────────────────────────────────

    def rank_documents(
        self,
        query_terms: list[str],
        candidate_doc_ids: set[int],
        top_k: int = 10,
    ) -> list[BM25ScoredDocument]:
        """
        Score every candidate document against the query terms using BM25.

        Performance fix: use a single batch DB query for all term TFs instead
        of O(candidates × terms) individual round-trips.

        Correctness fix: score using 'body' field only to avoid double-counting
        TF when a term appears in both the title and body postings.
        Title-field boosting is handled separately by RelevanceTuner.
        """
        total_docs = self._get_total_docs()
        avgdl      = self._get_avgdl()
        if total_docs == 0 or avgdl == 0:
            return []

        unique_terms = list(set(query_terms))

        # ── 1. Pre-fetch IDF (1 DB query per unique term) ─────────────────
        idf: dict[str, float] = {}
        for term in unique_terms:
            term_rec = self.db.get_term(term)
            df = term_rec.document_frequency if term_rec else 0
            idf[term] = self._compute_idf(df, total_docs)

        # ── 2. Batch-fetch body TF for all terms in one SQL query ─────────
        # Returns {term: {doc_id: tf}}  — body field only (no double-TF)
        term_tf_map: dict[str, dict[int, int]] = self.db.get_postings_for_terms_batch(
            unique_terms, field="body"
        )

        # Pre-fetch document lengths for all candidates in one query
        doc_records = {
            doc_id: self.db.get_document(doc_id)
            for doc_id in candidate_doc_ids
        }

        scored: list[BM25ScoredDocument] = []

        for doc_id in candidate_doc_ids:
            doc = doc_records.get(doc_id)
            if doc is None:
                continue

            doc_len = doc.word_count or 1
            score   = 0.0
            term_scores: dict[str, float] = {}

            for term in unique_terms:
                tf = term_tf_map.get(term, {}).get(doc_id, 0)
                if tf == 0:
                    term_scores[term] = 0.0
                    continue
                ts = idf[term] * self._compute_tf_bm25(tf, doc_len, avgdl)
                term_scores[term] = ts
                score += ts

            snippet = (
                doc.content[:200] + "..."
                if len(doc.content) > 200
                else doc.content
            )
            scored.append(BM25ScoredDocument(
                doc_id=doc_id, score=score,
                title=doc.title, snippet=snippet,
                term_scores=term_scores,
            ))

        scored.sort(key=lambda x: x.score, reverse=True)
        logger.debug(
            "BM25 ranked %d docs for %d terms → top %d",
            len(scored), len(query_terms), top_k,
        )
        return scored[:top_k]

    # ── BM25 Math ─────────────────────────────────────────────────────────

    def _compute_idf(self, df: int, total_docs: int) -> float:
        """
        Robertson–Spärck Jones IDF — always non-negative.

        idf(t) = log( (N − df + 0.5) / (df + 0.5) + 1 )
        """
        numerator = total_docs - df + 0.5
        denominator = df + 0.5
        return math.log(numerator / denominator + 1)

    def _compute_tf_bm25(self, tf: int, doc_len: int, avgdl: float) -> float:
        """
        Saturating TF function.

                  tf · (k1 + 1)
        tf* = ──────────────────────────────────────
               tf + k1 · (1 − b + b · |d| / avgdl)
        """
        k1 = self.config.k1
        b = self.config.b
        denominator = tf + k1 * (1.0 - b + b * doc_len / avgdl)
        return (tf * (k1 + 1.0)) / denominator

    # ── Corpus Statistics ─────────────────────────────────────────────────

    def _get_total_docs(self) -> int:
        if self._total_docs_cache is None:
            self._total_docs_cache = self.db.get_document_count()
        return self._total_docs_cache

    def _get_avgdl(self) -> float:
        """Average document length across the corpus (cached)."""
        if self._avgdl_cache is None:
            self._avgdl_cache = self.db.get_average_document_length()
        return self._avgdl_cache or 1.0

    def invalidate_cache(self) -> None:
        """Call after indexing new documents to reset corpus statistics."""
        self._avgdl_cache = None
        self._total_docs_cache = None

    # ── Diagnostic utilities ──────────────────────────────────────────────

    def explain(self, query_terms: list[str], doc_id: int) -> dict:
        """
        Return a human-readable breakdown of the BM25 score for one document.
        Useful for debugging and understanding why a document ranked where it did.
        """
        total_docs = self._get_total_docs()
        avgdl = self._get_avgdl()
        doc = self.db.get_document(doc_id)
        if doc is None:
            return {"error": f"Document {doc_id} not found"}

        doc_len = doc.word_count or 1
        breakdown = {
            "doc_id": doc_id,
            "title": doc.title,
            "doc_len": doc_len,
            "avgdl": round(avgdl, 2),
            "k1": self.config.k1,
            "b": self.config.b,
            "total_docs": total_docs,
            "terms": {},
        }
        total = 0.0
        for term in set(query_terms):
            term_rec = self.db.get_term(term)
            df = term_rec.document_frequency if term_rec else 0
            postings = self.db.get_postings_for_term(term)
            tf = next((p.term_frequency for p in postings if p.doc_id == doc_id), 0)
            idf_val = self._compute_idf(df, total_docs)
            tf_bm25_val = self._compute_tf_bm25(tf, doc_len, avgdl) if tf else 0.0
            contribution = idf_val * tf_bm25_val
            total += contribution
            breakdown["terms"][term] = {
                "df": df,
                "tf_raw": tf,
                "idf": round(idf_val, 6),
                "tf_bm25": round(tf_bm25_val, 6),
                "contribution": round(contribution, 6),
            }
        breakdown["total_score"] = round(total, 6)
        return breakdown
