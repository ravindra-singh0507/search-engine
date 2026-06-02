"""
Retrieval Evaluator

Runs evaluation across BM25, semantic, and hybrid retrieval systems
and produces a comparative report.

=== EVALUATION DATASET FORMAT ===

JSON file: list of EvalQuery objects.

  [
    {
      "query_id":          "q1",
      "query":             "python web framework",
      "relevant_doc_ids":  [3, 7, 12],
      "relevance_scores":  {"3": 3, "7": 2, "12": 1}
    },
    ...
  ]

  relevant_doc_ids: list of doc_ids that are relevant (binary or graded)
  relevance_scores: optional — maps doc_id (as string) to grade 1-3
                    If omitted, binary relevance (grade=1) is assumed.

=== USAGE ===

  evaluator = RetrievalEvaluator(...)
  report    = evaluator.evaluate(k_values=[1, 5, 10])
  print(report["bm25"]["MAP"])    # e.g. 0.42
  print(report["hybrid"]["NDCG@10"])
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from app.evaluation.metrics import compute_all_metrics
from app.config import EvaluationConfig

logger = logging.getLogger(__name__)


# ── Eval dataset dataclass ────────────────────────────────────────────────────

@dataclass
class EvalQuery:
    query_id:        str
    query:           str
    relevant_doc_ids: list[int]
    relevance_scores: dict[int, float]    # doc_id → grade (1.0 if binary)


def load_eval_dataset(path: Path) -> list[EvalQuery]:
    """
    Load evaluation queries from a JSON file.
    Returns empty list if file doesn't exist.
    """
    if not path.exists():
        logger.warning("Eval dataset not found at %s", path)
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    queries: list[EvalQuery] = []
    for item in data:
        rel_ids = [int(x) for x in item["relevant_doc_ids"]]
        raw_scores = item.get("relevance_scores", {})
        rel_scores = {int(k): float(v) for k, v in raw_scores.items()} \
            if raw_scores else {did: 1.0 for did in rel_ids}
        queries.append(EvalQuery(
            query_id         = item["query_id"],
            query            = item["query"],
            relevant_doc_ids = rel_ids,
            relevance_scores = rel_scores,
        ))
    logger.info("Loaded %d eval queries from %s", len(queries), path)
    return queries


# ── Evaluator ─────────────────────────────────────────────────────────────────

# A retrieval function takes (query: str, top_k: int) and returns list[int] of doc_ids
RetrievalFn = Callable[[str, int], list[int]]


class RetrievalEvaluator:
    """
    Evaluates one or more retrieval systems against a ground-truth dataset.

    Usage:
        evaluator = RetrievalEvaluator(config)
        evaluator.add_system("bm25", bm25_retrieval_fn)
        evaluator.add_system("hybrid", hybrid_retrieval_fn)
        report = evaluator.run()
    """

    def __init__(self, config: EvaluationConfig | None = None):
        self.config  = config or EvaluationConfig()
        self._systems: dict[str, RetrievalFn] = {}

    def add_system(self, name: str, retrieval_fn: RetrievalFn) -> None:
        """Register a retrieval system.  retrieval_fn(query, top_k) → [doc_ids]."""
        self._systems[name] = retrieval_fn

    def run(
        self,
        dataset: list[EvalQuery] | None = None,
        top_k:   int = 10,
    ) -> dict:
        """
        Run all registered systems against the dataset and return a report.

        Returns:
          {
            "system_name": {
              "P@1": 0.4, "P@5": 0.3, ..., "MRR": 0.5, "MAP": 0.4,
              "NDCG@10": 0.6, "per_query": [...]
            },
            ...
          }
        """
        if dataset is None:
            dataset = load_eval_dataset(self.config.eval_dataset_path)

        if not dataset:
            logger.warning("Empty evaluation dataset — returning empty report")
            return {}

        if not self._systems:
            logger.warning("No retrieval systems registered")
            return {}

        k_vals  = self.config.k_values
        max_k   = max(k_vals)
        report  = {}

        for sys_name, fn in self._systems.items():
            logger.info("Evaluating %s over %d queries …", sys_name, len(dataset))
            retrieved_lists       = []
            relevant_sets         = []
            relevance_scores_list = []
            per_query             = []

            for eq in dataset:
                try:
                    retrieved = fn(eq.query, max(max_k, top_k))
                except Exception as exc:
                    logger.error("Retrieval error for %s/%s: %s",
                                 sys_name, eq.query_id, exc)
                    retrieved = []

                retrieved_lists.append(retrieved)
                relevant_sets.append(set(eq.relevant_doc_ids))
                relevance_scores_list.append(eq.relevance_scores)

                q_metrics = compute_all_metrics(
                    [retrieved], [set(eq.relevant_doc_ids)],
                    [eq.relevance_scores], k_values=k_vals,
                )
                per_query.append({"query_id": eq.query_id,
                                   "query": eq.query, **q_metrics})

            agg = compute_all_metrics(
                retrieved_lists, relevant_sets,
                relevance_scores_list, k_values=k_vals,
            )
            agg["per_query"] = per_query
            report[sys_name] = agg

        return report

    @staticmethod
    def comparison_table(report: dict) -> str:
        """
        Render a human-readable ASCII comparison table from a report dict.
        """
        if not report:
            return "No evaluation data."

        systems = list(report.keys())
        metrics = [k for k in next(iter(report.values())) if k != "per_query"]

        col_w  = max(len(m) for m in metrics) + 2
        sys_w  = max(max(len(s) for s in systems), 10) + 2
        header = f"{'Metric':<{col_w}}" + "".join(f"{s:>{sys_w}}" for s in systems)
        sep    = "─" * len(header)
        rows   = [header, sep]
        for m in metrics:
            row = f"{m:<{col_w}}"
            for s in systems:
                val = report[s].get(m, "—")
                row += f"{val:>{sys_w}}" if isinstance(val, str) else f"{val:>{sys_w}.4f}"
            rows.append(row)
        return "\n".join(rows)
