"""
Learning to Rank — Feature Extraction

=== THEORY ===

Learning to Rank (LtR) treats ranking as a supervised learning problem.
A model is trained to predict relevance given a feature vector that
describes the (query, document) pair.

Phase 5 implements the FEATURE EXTRACTION layer only.  No model is trained.
This prepares the infrastructure for Phase 6 where click logs will provide
the training signal.

=== FEATURE CATEGORIES ===

  1. Term matching features (BM25, TF-IDF)
  2. Semantic features (dense embedding similarity)
  3. Neural re-ranking features (cross-encoder score)
  4. Document quality features (length, recency)
  5. Personalisation features (click history, session signals)

=== MODELS THAT USE THESE FEATURES ===

  LambdaMART (gradient-boosted trees for ranking)  — prod @ Bing, LinkedIn
  RankNet (pairwise neural)                         — prod @ old Bing
  ListNet / ListMLE (listwise)                      — research
  LambdaRank                                        — prod @ Yahoo
  Direct LLM scoring (GPT-4 as judge)               — emerging

Our implementation follows the LambdaMART feature schema used in
the MSN Learning to Rank dataset (30 features, similar to ours).

=== COMPLEXITY ===

  Extract all features for M candidates:  O(M × F)
  where F = number of features ≈ constant (8 here)
  Dominated by BM25 and semantic score lookups (already computed upstream).
"""

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ── Feature Protocol ──────────────────────────────────────────────────────────

@runtime_checkable
class RankingFeature(Protocol):
    """
    Structural interface for a single ranking feature.
    Implement this to add new signals (PageRank, freshness, social, etc.)
    without modifying the extractor.
    """
    name: str

    def compute(self, query: str, doc_id: int, context: dict) -> float:
        """
        Compute the feature value for a (query, doc) pair.

        context: dict carrying pre-computed values (bm25_score, semantic_score,
                 doc_record, click_counts, etc.) to avoid redundant DB reads.
        """
        ...


# ── Feature Vector dataclass ──────────────────────────────────────────────────

@dataclass
class FeatureVector:
    query:    str
    doc_id:   int
    features: dict[str, float]   # feature_name → value

    def to_list(self, feature_names: list[str]) -> list[float]:
        """Return features as an ordered float list for ML frameworks."""
        return [self.features.get(n, 0.0) for n in feature_names]


# ── Concrete features ─────────────────────────────────────────────────────────

class BM25ScoreFeature:
    name = "bm25_score"
    def compute(self, query: str, doc_id: int, context: dict) -> float:
        return float(context.get("bm25_score", 0.0))


class SemanticScoreFeature:
    name = "semantic_score"
    def compute(self, query: str, doc_id: int, context: dict) -> float:
        return float(context.get("semantic_score", 0.0))


class RerankerScoreFeature:
    name = "reranker_score"
    def compute(self, query: str, doc_id: int, context: dict) -> float:
        return float(context.get("reranker_score", 0.0))


class TitleMatchFeature:
    """Fraction of query terms that appear in the document title."""
    name = "title_match"
    def compute(self, query: str, doc_id: int, context: dict) -> float:
        doc = context.get("doc_record")
        if doc is None:
            return 0.0
        q_terms    = set(query.lower().split())
        title_terms = set(doc.title.lower().split())
        if not q_terms:
            return 0.0
        return len(q_terms & title_terms) / len(q_terms)


class ClickScoreFeature:
    """Normalised click count for this document from analytics."""
    name = "click_score"
    def compute(self, query: str, doc_id: int, context: dict) -> float:
        click_counts: dict[int, int] = context.get("click_counts", {})
        max_clicks = max(click_counts.values()) if click_counts else 1
        return click_counts.get(doc_id, 0) / max_clicks if max_clicks else 0.0


class FreshnessScoreFeature:
    """Exponential decay based on document age (days since created_at)."""
    name = "freshness_score"
    def compute(self, query: str, doc_id: int, context: dict) -> float:
        import math
        doc = context.get("doc_record")
        if doc is None or not getattr(doc, "created_at", None):
            return 0.5
        try:
            from datetime import datetime, timezone
            created = datetime.fromisoformat(
                doc.created_at.replace("Z", "+00:00")
            )
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_days = max(0, (datetime.now(timezone.utc) - created).days)
            return math.exp(-0.01 * age_days)
        except Exception:
            return 0.5


class DocumentLengthFeature:
    """Log-normalised document length.  Longer docs get lower scores."""
    name = "document_length"
    def compute(self, query: str, doc_id: int, context: dict) -> float:
        import math
        doc = context.get("doc_record")
        if doc is None:
            return 0.5
        wc = getattr(doc, "word_count", 0) or 0
        return 1.0 / (1.0 + math.log1p(wc))


class QueryTermCountFeature:
    """Number of query terms (normalised)."""
    name = "query_term_count"
    def compute(self, query: str, doc_id: int, context: dict) -> float:
        n = len(query.split())
        return min(n / 10.0, 1.0)   # cap at 10 terms = 1.0


# ── Default feature set ───────────────────────────────────────────────────────

DEFAULT_FEATURES: list[RankingFeature] = [
    BM25ScoreFeature(),
    SemanticScoreFeature(),
    RerankerScoreFeature(),
    TitleMatchFeature(),
    ClickScoreFeature(),
    FreshnessScoreFeature(),
    DocumentLengthFeature(),
    QueryTermCountFeature(),
]


# ── Feature extractor ─────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Extracts a feature vector for each (query, doc_id) candidate.

    Usage
    -----
    extractor = FeatureExtractor(db, features=DEFAULT_FEATURES)
    vectors   = extractor.extract("python web framework", [1, 3, 5], context)
    """

    def __init__(self, db, features: list[RankingFeature] | None = None):
        self.db       = db
        self.features = features or DEFAULT_FEATURES

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    def extract(
        self,
        query:   str,
        doc_ids: list[int],
        context: dict | None = None,
    ) -> list[FeatureVector]:
        """
        Extract features for all (query, doc_id) pairs.

        context: optional dict with pre-computed values.  If a doc_record
                 is not in context it is fetched from the DB.
        """
        context = context or {}
        vectors: list[FeatureVector] = []

        for doc_id in doc_ids:
            # Ensure doc_record is in context
            local_ctx = {**context}
            if "doc_record" not in local_ctx:
                local_ctx["doc_record"] = self.db.get_document(doc_id)

            feat_vals: dict[str, float] = {}
            for feat in self.features:
                try:
                    feat_vals[feat.name] = round(
                        feat.compute(query, doc_id, local_ctx), 6
                    )
                except Exception as exc:
                    logger.warning("Feature %s failed: %s", feat.name, exc)
                    feat_vals[feat.name] = 0.0

            vectors.append(FeatureVector(
                query=query, doc_id=doc_id, features=feat_vals
            ))

        return vectors

    def extract_dict(
        self, query: str, doc_ids: list[int], context: dict | None = None
    ) -> dict[int, dict[str, float]]:
        """Return {doc_id: {feature_name: value}} for easy lookup."""
        vecs = self.extract(query, doc_ids, context)
        return {v.doc_id: v.features for v in vecs}
