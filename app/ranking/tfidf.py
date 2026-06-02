"""
TF-IDF Ranking

=== THEORY ===

TF-IDF (Term Frequency - Inverse Document Frequency) is the foundational
ranking algorithm in information retrieval. It answers: "How relevant is
document D to query Q?"

The intuition: A term is important to a document if it appears frequently
in that document (high TF) BUT rarely across all documents (high IDF).

=== FORMULAS ===

1. Term Frequency (TF):
   How often does term t appear in document d?

   tf(t, d) = count(t in d) / total_terms_in_d

   Why normalize by document length? A 10,000-word document mentioning
   "python" 5 times is less focused on Python than a 100-word document
   mentioning it 5 times.

2. Document Frequency (DF):
   In how many documents does term t appear?

   df(t) = count(documents containing t)

3. Inverse Document Frequency (IDF):
   How rare/specific is this term across all documents?

   idf(t) = log(N / df(t))

   where N = total number of documents

   Why logarithm? Without it, a term appearing in 1 document vs 1000
   would have IDF ratio of 1000:1, which is too extreme. Log compresses
   this to about 7:1.

   Why inverse? Common terms (appearing in many docs) get LOW scores.
   Rare terms get HIGH scores. "The" appears everywhere → low IDF.
   "Kubernetes" appears in few docs → high IDF.

4. TF-IDF Score:
   tfidf(t, d) = tf(t, d) * idf(t)

5. For a multi-term query, we score each document as a vector and compute
   cosine similarity between the query vector and document vector.

=== COSINE SIMILARITY ===

Documents and queries are represented as vectors in term-space:
- Each dimension = one term
- Each value = TF-IDF weight for that term

Cosine similarity measures the angle between two vectors:

   cos(q, d) = (q · d) / (|q| * |d|)

   where q · d = sum of (q_i * d_i) for each dimension
         |q|   = sqrt(sum of q_i²)
         |d|   = sqrt(sum of d_i²)

Why cosine over dot product? Cosine normalizes for vector length,
so long documents aren't unfairly boosted.

=== COMPLEXITY ===

- Compute TF for one document: O(L) where L = document length
- Compute IDF for one term: O(1) lookup
- Score one document against a query: O(Q) where Q = query terms
- Rank all matching documents: O(M * Q) where M = matching docs
- Sort by score: O(M * log M)

=== AT GOOGLE SCALE ===

Google moved beyond pure TF-IDF decades ago, but TF-IDF is still a
component signal. Modern ranking uses:
- BM25 (a TF-IDF variant with saturation — diminishing returns for
  high term frequency)
- PageRank (link-based authority)
- BERT/neural models for semantic understanding
- Hundreds of other signals (freshness, click-through rate, etc.)
- Learning-to-rank models that combine all signals

But TF-IDF remains the foundation that all of these build upon.
"""

import math
import logging
from dataclasses import dataclass

from app.database.db import Database
from app.tokenizer.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


@dataclass
class ScoredDocument:
    doc_id: int
    score: float
    title: str
    snippet: str
    term_scores: dict[str, float]


class TFIDFRanker:
    """
    Ranks documents against a query using TF-IDF with cosine similarity.

    All math is implemented from scratch — no sklearn or numpy.
    """

    def __init__(self, db: Database, tokenizer: Tokenizer):
        self.db = db
        self.tokenizer = tokenizer

    def compute_tf(self, term_frequency: int, total_terms: int) -> float:
        """
        Term Frequency: what fraction of the document is this term?
        tf(t,d) = count(t in d) / total_terms_in_d
        """
        if total_terms == 0:
            return 0.0
        return term_frequency / total_terms

    def compute_idf(self, document_frequency: int, total_documents: int) -> float:
        """
        Inverse Document Frequency: how rare is this term?
        idf(t) = log(N / df(t))

        We add 1 to df to avoid division by zero for unknown terms.
        """
        if document_frequency == 0:
            return 0.0
        return math.log(total_documents / document_frequency)

    def compute_tfidf(self, tf: float, idf: float) -> float:
        """TF-IDF score for a single term in a single document."""
        return tf * idf

    def _dot_product(self, vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
        """Dot product of two sparse vectors represented as dicts."""
        result = 0.0
        for key in vec_a:
            if key in vec_b:
                result += vec_a[key] * vec_b[key]
        return result

    def _magnitude(self, vec: dict[str, float]) -> float:
        """Euclidean magnitude (L2 norm) of a sparse vector."""
        return math.sqrt(sum(v * v for v in vec.values()))

    def cosine_similarity(self, vec_a: dict[str, float],
                          vec_b: dict[str, float]) -> float:
        """
        Cosine similarity between two TF-IDF vectors.
        cos(a, b) = (a · b) / (|a| * |b|)
        Returns 0.0 if either vector has zero magnitude.
        """
        dot = self._dot_product(vec_a, vec_b)
        mag_a = self._magnitude(vec_a)
        mag_b = self._magnitude(vec_b)

        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0

        return dot / (mag_a * mag_b)

    def build_document_vector(self, doc_id: int,
                              total_documents: int) -> dict[str, float]:
        """
        Build a TF-IDF vector for a document.
        Returns: {term: tfidf_score, ...}
        Uses Database.get_term_by_id() instead of accessing db.conn directly.
        """
        postings = self.db.get_postings_for_doc(doc_id)
        doc = self.db.get_document(doc_id)
        if doc is None:
            return {}

        total_terms = doc.word_count or 1
        vector: dict[str, float] = {}

        for posting in postings:
            term_record = self.db.get_term_by_id(posting.term_id)
            if term_record is None:
                continue

            tf  = self.compute_tf(posting.term_frequency, total_terms)
            idf = self.compute_idf(term_record.document_frequency, total_documents)
            vector[term_record.term] = self.compute_tfidf(tf, idf)

        return vector

    def build_query_vector(self, query_terms: list[str],
                           total_documents: int) -> dict[str, float]:
        """
        Build a TF-IDF vector for the query.
        Query TF = count / len (treating the query as a tiny document).
        """
        if not query_terms:
            return {}

        term_counts: dict[str, int] = {}
        for term in query_terms:
            term_counts[term] = term_counts.get(term, 0) + 1

        query_len = len(query_terms)
        vector: dict[str, float] = {}

        for term, count in term_counts.items():
            term_record = self.db.get_term(term)
            df = term_record.document_frequency if term_record else 0

            tf = count / query_len
            idf = self.compute_idf(df, total_documents)
            vector[term] = self.compute_tfidf(tf, idf)

        return vector

    def rank_documents(self, query_terms: list[str],
                       candidate_doc_ids: set[int],
                       top_k: int = 10) -> list[ScoredDocument]:
        """
        Rank candidate documents against a query using cosine similarity
        of TF-IDF vectors.

        Steps:
        1. Build query vector
        2. Build document vector for each candidate
        3. Compute cosine similarity
        4. Sort by score descending
        5. Return top-k
        """
        total_documents = self.db.get_document_count()
        if total_documents == 0:
            return []

        query_vector = self.build_query_vector(query_terms, total_documents)
        if not query_vector:
            return []

        scored: list[ScoredDocument] = []

        for doc_id in candidate_doc_ids:
            doc = self.db.get_document(doc_id)
            if doc is None:
                continue

            doc_vector = self.build_document_vector(doc_id, total_documents)
            score = self.cosine_similarity(query_vector, doc_vector)

            term_scores = {
                term: doc_vector.get(term, 0.0) for term in query_terms
            }

            snippet = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content

            scored.append(ScoredDocument(
                doc_id=doc_id,
                score=score,
                title=doc.title,
                snippet=snippet,
                term_scores=term_scores
            ))

        scored.sort(key=lambda x: x.score, reverse=True)

        logger.debug(
            "Ranked %d documents for %d query terms, returning top %d",
            len(scored), len(query_terms), top_k
        )

        return scored[:top_k]
