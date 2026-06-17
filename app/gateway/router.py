"""
Query Router — Intent-Based Retrieval Backend Selection

=== THEORY ===

Query routing selects the optimal retrieval backend for a given query
based on its characteristics.  This is a form of ADAPTIVE RETRIEVAL
where the system dynamically chooses the best strategy rather than
using a one-size-fits-all approach.

The key insight is that different query types benefit from different
retrieval methods:

  Navigational queries ("python requests github"):
    Best served by BM25 — the user wants a specific resource and exact
    keyword matching is most precise.

  Informational queries ("how does gradient descent work"):
    Best served by hybrid search — combines keyword precision with
    semantic understanding for broad coverage.

  Semantic queries (natural language questions, paraphrases):
    Best served by semantic search — dense retrieval captures meaning
    even when query words don't match document words.

  Complex queries (multi-faceted, requiring synthesis):
    Best served by the full pipeline — multi-stage retrieval with
    fusion and reranking for maximum quality.

=== ROUTING ALGORITHM ===

  1. If the caller provides a hint (explicit mode), use it directly.
  2. Otherwise, classify the query using QueryClassifier (Phase 5).
  3. Map the classified intent to a retrieval backend:
       navigational    -> "bm25"      (keyword precision)
       transactional   -> "bm25"      (specific resource lookup)
       informational   -> "hybrid"    (broad coverage)
       documentation   -> "hybrid"    (mix of exact + conceptual)
       troubleshooting -> "hybrid"    (error terms + context)
       research        -> "pipeline"  (deep multi-stage retrieval)

=== PRODUCTION EQUIVALENTS ===

  Google:     Query classification -> different index tiers
  Bing:       Intent-based routing to specialised verticals
  Vespa:      Query profiles with different rank expressions
  Elastic:    Query-time index routing + query rewriting

=== COMPLEXITY ===

  route: O(Q * P) where Q = query tokens, P = classifier patterns
  Practical: < 1 ms per query
"""

import logging

logger = logging.getLogger(__name__)

# Intent -> retrieval mode mapping
_INTENT_TO_MODE: dict[str, str] = {
    "navigational":    "bm25",
    "transactional":   "bm25",
    "informational":   "hybrid",
    "documentation":   "hybrid",
    "troubleshooting": "hybrid",
    "research":        "pipeline",
}

# Valid retrieval modes that callers can request
VALID_MODES = frozenset({"bm25", "semantic", "hybrid", "pipeline"})


class QueryRouter:
    """
    Routes queries to the optimal retrieval backend based on query analysis.

    Uses QueryClassifier (Phase 5) to determine intent, then maps the
    intent to one of: "bm25", "semantic", "hybrid", or "pipeline".

    If no classifier is provided, defaults to "hybrid" for all queries
    (safe default that combines keyword and semantic retrieval).
    """

    def __init__(self, classifier=None):
        """
        Parameters
        ----------
        classifier : QueryClassifier instance from Phase 5, or None.
                     If None, all queries are routed to "hybrid".
        """
        self._classifier = classifier

    def route(self, query: str, hint: str | None = None) -> str:
        """
        Determine the retrieval mode for a query.

        Parameters
        ----------
        query : raw query string
        hint  : explicit mode override from the caller.  If provided and
                valid, it is returned directly without classification.

        Returns
        -------
        One of: "bm25", "semantic", "hybrid", "pipeline"
        """
        # Explicit hint overrides classification
        if hint and hint.lower() in VALID_MODES:
            logger.debug("Router: using caller hint %r for query %r", hint, query)
            return hint.lower()

        # No classifier available — safe default
        if self._classifier is None:
            logger.debug("Router: no classifier, defaulting to hybrid")
            return "hybrid"

        # Classify and map
        try:
            intent = self._classifier.classify(query)
            mode = _INTENT_TO_MODE.get(intent.intent, "hybrid")
            logger.debug(
                "Router: query=%r -> intent=%s (conf=%.2f) -> mode=%s",
                query, intent.intent, intent.confidence, mode,
            )
            return mode
        except Exception as exc:
            logger.warning(
                "Router: classification failed (%s), defaulting to hybrid", exc,
            )
            return "hybrid"
