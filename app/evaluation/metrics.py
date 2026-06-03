"""
Retrieval Evaluation Metrics

=== THEORY ===

Information Retrieval evaluation requires a ground-truth set of
(query, relevant_documents) pairs called a TEST COLLECTION or QREL.

We compute:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRECISION@K
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

P@K = |{retrieved[:K] ∩ relevant}| / K

Measures: of the K documents we showed the user, what fraction
          were actually relevant?

Example:
  relevant = {1, 3, 5}
  retrieved = [1, 2, 3, 4, 5]  (K=5)
  P@5 = 3/5 = 0.6

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECALL@K
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

R@K = |{retrieved[:K] ∩ relevant}| / |relevant|

Measures: of all relevant documents, what fraction did we find
          in the top K positions?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MRR (Mean Reciprocal Rank)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MRR = (1/|Q|) Σ_{q∈Q} 1/rank_q

rank_q = position of the FIRST relevant document for query q.
         0 if no relevant document is retrieved.

Measures: on average, how far down the list do users need to scroll
          to find the FIRST relevant document?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MAP (Mean Average Precision)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AP(q) = (1/|R_q|) Σ_{k: retrieved[k]∈R_q} P@k
MAP   = (1/|Q|)   Σ_{q∈Q} AP(q)

where R_q = relevant docs for query q.

Measures: quality of the full ranked list — rewards both precision
          AND early placement of relevant documents.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NDCG@K (Normalised Discounted Cumulative Gain)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Uses GRADED relevance (e.g. 0 = not relevant, 1 = relevant, 2 = highly).

DCG@K  = Σ_{i=1}^{K} rel_i / log2(i + 1)
IDCG@K = DCG@K of the ideal (perfect) ranking
NDCG@K = DCG@K / IDCG@K    (normalised to [0, 1])

Measures: quality of graded ranking — highly relevant docs early gets
          the best score.

=== COMPLEXITY ===

All metrics: O(K) per query, O(|Q| · K) total.

=== PRODUCTION USE ===

  Elasticsearch:   Uses NDCG@K and MRR for Learning-to-Rank evaluation
  Bing/Google:     Internal A/B testing on live query logs with human judgements
  Kaggle:          NDCG is commonly used as the competition metric
  Cohere Reranker: MAP@5 / NDCG@10 on BEIR benchmark
"""

import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def precision_at_k(
    retrieved:  list[int],
    relevant:   set[int],
    k:          int,
) -> float:
    """
    P@K = |retrieved[:K] ∩ relevant| / K

    Returns 0.0 if K=0 or no documents retrieved.
    """
    if k <= 0 or not retrieved:
        return 0.0
    top_k = retrieved[:k]
    hits  = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def recall_at_k(
    retrieved:  list[int],
    relevant:   set[int],
    k:          int,
) -> float:
    """
    R@K = |retrieved[:K] ∩ relevant| / |relevant|

    Returns 0.0 if no relevant documents exist.
    """
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits  = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def reciprocal_rank(
    retrieved: list[int],
    relevant:  set[int],
) -> float:
    """
    RR = 1 / rank_of_first_relevant_doc

    Returns 0.0 if no relevant doc is found in retrieved.
    """
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    retrieved_lists: list[list[int]],
    relevant_sets:   list[set[int]],
) -> float:
    """
    MRR = mean of RR across all queries.

    Args:
        retrieved_lists: one ranked list per query
        relevant_sets:   one set of relevant doc IDs per query

    Returns float in [0, 1].
    """
    if not retrieved_lists:
        return 0.0
    rr_scores = [
        reciprocal_rank(ret, rel)
        for ret, rel in zip(retrieved_lists, relevant_sets)
    ]
    return sum(rr_scores) / len(rr_scores)


def average_precision(
    retrieved: list[int],
    relevant:  set[int],
) -> float:
    """
    AP = (1/|R|) Σ_{k: retrieved[k]∈R} P@k

    Returns 0.0 if no relevant documents exist.
    """
    if not relevant:
        return 0.0
    hits   = 0
    ap_sum = 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits   += 1
            ap_sum += hits / rank    # P@rank
    return ap_sum / len(relevant)


def mean_average_precision(
    retrieved_lists: list[list[int]],
    relevant_sets:   list[set[int]],
) -> float:
    """MAP = mean of AP across all queries."""
    if not retrieved_lists:
        return 0.0
    ap_scores = [
        average_precision(ret, rel)
        for ret, rel in zip(retrieved_lists, relevant_sets)
    ]
    return sum(ap_scores) / len(ap_scores)


def dcg_at_k(
    retrieved:         list[int],
    relevance_scores:  dict[int, float],   # doc_id → relevance grade (0, 1, 2, ...)
    k:                 int,
) -> float:
    """
    DCG@K = Σ_{i=1}^{K} rel_i / log2(i + 1)

    Uses graded relevance; binary relevance = {0, 1} is a special case.
    """
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        rel   = relevance_scores.get(doc_id, 0.0)
        dcg  += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(
    retrieved:         list[int],
    relevance_scores:  dict[int, float],
    k:                 int,
) -> float:
    """
    NDCG@K = DCG@K / IDCG@K

    IDCG@K = DCG of the ideal (perfect) ranking of relevant documents.
    Returns 0.0 if no relevant documents exist.
    """
    # Build ideal ranking: sort doc IDs by descending relevance score
    ideal_ranked = sorted(
        (doc_id for doc_id in relevance_scores if relevance_scores[doc_id] > 0),
        key=lambda d: relevance_scores[d],
        reverse=True,
    )
    idcg = dcg_at_k(ideal_ranked, relevance_scores, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(retrieved, relevance_scores, k) / idcg


def compute_all_metrics(
    retrieved_lists:     list[list[int]],
    relevant_sets:       list[set[int]],
    relevance_scores_list: list[dict[int, float]] | None = None,
    k_values:            list[int] | None = None,
) -> dict:
    """
    Compute P@K, R@K, MRR, MAP, and NDCG@K for each K in k_values.

    Returns a dict of metric name → float value.
    """
    k_values = k_values or [1, 3, 5, 10]
    if relevance_scores_list is None:
        # Convert binary relevance sets to graded dicts
        relevance_scores_list = [
            {doc_id: 1.0 for doc_id in rel}
            for rel in relevant_sets
        ]

    result: dict[str, float] = {}

    for k in k_values:
        p_k = [precision_at_k(ret, rel, k)
               for ret, rel in zip(retrieved_lists, relevant_sets)]
        r_k = [recall_at_k(ret, rel, k)
               for ret, rel in zip(retrieved_lists, relevant_sets)]
        n_k = [ndcg_at_k(ret, rscores, k)
               for ret, rscores in zip(retrieved_lists, relevance_scores_list)]

        result[f"P@{k}"]    = round(sum(p_k) / max(len(p_k), 1), 4)
        result[f"R@{k}"]    = round(sum(r_k) / max(len(r_k), 1), 4)
        result[f"NDCG@{k}"] = round(sum(n_k) / max(len(n_k), 1), 4)

    result["MRR"] = round(mean_reciprocal_rank(retrieved_lists, relevant_sets), 4)
    result["MAP"] = round(mean_average_precision(retrieved_lists, relevant_sets), 4)

    return result
