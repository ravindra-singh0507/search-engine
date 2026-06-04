"""
Retrieval Experiment Framework

Supports A/B experiments comparing retrieval configurations:

  Experiment A: BM25 only
  Experiment B: BM25 + Semantic + RRF
  Experiment C: BM25 + Semantic + RRF + CrossEncoder reranker

Experiments are stored in SQLite and can be replayed.

=== DESIGN ===

  ExperimentConfig  — defines WHAT to run (retrieval system + params)
  ExperimentRun     — one execution of one config over a dataset
  ExperimentResults — aggregated metrics + per-query breakdown
  ExperimentRunner  — executes configs, stores results, compares

=== AT SCALE ===

  Production A/B:   Traffic split at query level (50/50 coin flip)
  Offline eval:     Run on TREC / MS-MARCO judged datasets
  Online metrics:   CTR, dwell time, session success rate
  Our approach:     Offline evaluation on dev/test datasets
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.evaluation.metrics import compute_all_metrics
from app.evaluation.evaluator import EvalQuery, load_eval_dataset
from app.config import ExperimentConfig

logger = logging.getLogger(__name__)

RetrievalFn = Callable[[str, int], list[int]]   # (query, top_k) → [doc_ids]


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class Experiment:
    experiment_id:  str
    name:           str
    description:    str = ""
    config:         dict = field(default_factory=dict)   # retrieval params
    created_at:     str  = ""

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "name":          self.name,
            "description":   self.description,
            "config":        self.config,
            "created_at":    self.created_at,
        }


@dataclass
class ExperimentRun:
    experiment_id:  str
    run_id:         str
    metrics:        dict[str, float]
    latency_ms:     float
    query_count:    int
    per_query:      list[dict]   = field(default_factory=list)
    status:         str           = "done"
    created_at:     str           = ""


# ── ExperimentRunner ──────────────────────────────────────────────────────────

class ExperimentRunner:
    """
    Runs retrieval experiments and stores results for comparison.
    Results are persisted to disk as JSON + to SQLite via the DB layer.
    """

    def __init__(
        self,
        db=None,          # optional Database reference for persistence
        config: ExperimentConfig | None = None,
    ):
        self.db     = db
        self.config = config or ExperimentConfig()
        self._systems: dict[str, RetrievalFn] = {}
        self._results: dict[str, ExperimentRun] = {}
        self.config.storage_path.mkdir(parents=True, exist_ok=True)

    # ── System registration ───────────────────────────────────────────────

    def register_system(self, name: str, fn: RetrievalFn) -> None:
        """Register a named retrieval function for experiment runs."""
        self._systems[name] = fn
        logger.debug("Experiment system registered: %s", name)

    # ── Run experiments ───────────────────────────────────────────────────

    def run(
        self,
        experiment: Experiment,
        dataset:    list[EvalQuery] | None = None,
        top_k:      int = 10,
    ) -> ExperimentRun:
        """
        Run one experiment over the dataset and return metrics.
        Uses all registered systems unless experiment.config["systems"] is set.
        """
        if dataset is None and self.db:
            from app.config import EvaluationConfig
            from pathlib import Path
            ds_path = Path(self.db.db_path).parent / "eval_dataset.json"
            dataset = load_eval_dataset(ds_path)

        dataset = dataset or []
        if not dataset:
            logger.warning("Empty dataset — experiment %s has no queries", experiment.name)
            return ExperimentRun(
                experiment_id=experiment.experiment_id,
                run_id=f"{experiment.experiment_id}_empty",
                metrics={}, latency_ms=0, query_count=0,
            )

        systems_to_run = experiment.config.get("systems", list(self._systems.keys()))
        run_id         = f"{experiment.experiment_id}_{int(time.time())}"
        start          = time.perf_counter()

        all_retrieved:  dict[str, list[list[int]]]     = {s: [] for s in systems_to_run}
        all_relevant:   list[set[int]]                 = []
        all_rel_scores: list[dict[int, float]]         = []
        per_query:      list[dict]                     = []

        for eq in dataset:
            all_relevant.append(set(eq.relevant_doc_ids))
            all_rel_scores.append(eq.relevance_scores)
            q_row: dict = {"query_id": eq.query_id, "query": eq.query}

            for sys_name in systems_to_run:
                fn = self._systems.get(sys_name)
                if fn is None:
                    all_retrieved[sys_name].append([])
                    continue
                try:
                    retrieved = fn(eq.query, top_k)
                except Exception as exc:
                    logger.error("System %s failed on %r: %s", sys_name, eq.query, exc)
                    retrieved = []
                all_retrieved[sys_name].append(retrieved)
                q_metrics = compute_all_metrics(
                    [retrieved], [set(eq.relevant_doc_ids)],
                    [eq.relevance_scores],
                    k_values=[1, 3, 5, 10],
                )
                q_row[sys_name] = q_metrics

            per_query.append(q_row)

        # Aggregate metrics per system
        agg_metrics: dict[str, float] = {}
        for sys_name in systems_to_run:
            sys_metrics = compute_all_metrics(
                all_retrieved[sys_name], all_relevant, all_rel_scores,
                k_values=[1, 3, 5, 10],
            )
            for k, v in sys_metrics.items():
                agg_metrics[f"{sys_name}.{k}"] = v

        elapsed = (time.perf_counter() - start) * 1000
        run = ExperimentRun(
            experiment_id = experiment.experiment_id,
            run_id        = run_id,
            metrics       = agg_metrics,
            latency_ms    = round(elapsed, 2),
            query_count   = len(dataset),
            per_query     = per_query,
        )
        self._results[run_id] = run
        self._persist_run(experiment, run)

        if self.db:
            self._save_to_db(experiment, run)

        logger.info(
            "Experiment %r done: %d queries, %d systems, %.1f ms",
            experiment.name, len(dataset), len(systems_to_run), elapsed,
        )
        return run

    def compare(self, run_ids: list[str]) -> dict:
        """Return side-by-side metrics for a list of run IDs."""
        comparison = {}
        for rid in run_ids:
            run = self._results.get(rid)
            if run:
                comparison[rid] = {
                    "experiment_id": run.experiment_id,
                    "metrics":       run.metrics,
                    "query_count":   run.query_count,
                    "latency_ms":    run.latency_ms,
                }
        return comparison

    def list_runs(self) -> list[dict]:
        return [
            {
                "run_id":        r.run_id,
                "experiment_id": r.experiment_id,
                "query_count":   r.query_count,
                "latency_ms":    r.latency_ms,
                "created_at":    r.created_at,
            }
            for r in self._results.values()
        ]

    # ── Persistence ───────────────────────────────────────────────────────

    def _persist_run(self, exp: Experiment, run: ExperimentRun) -> None:
        path = self.config.storage_path / f"{run.run_id}.json"
        data = {
            "experiment": exp.to_dict(),
            "run_id":     run.run_id,
            "metrics":    run.metrics,
            "latency_ms": run.latency_ms,
            "query_count": run.query_count,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug("Experiment run saved to %s", path)

    def _save_to_db(self, exp: Experiment, run: ExperimentRun) -> None:
        try:
            self.db.upsert_experiment(exp.experiment_id, exp.name, exp.description,
                                      json.dumps(exp.config))
            self.db.insert_experiment_result(
                exp.experiment_id, run.run_id,
                json.dumps(run.metrics), run.latency_ms, run.query_count,
            )
        except Exception as exc:
            logger.warning("Failed to save experiment to DB: %s", exc)

    def load_runs_from_disk(self) -> None:
        """Reload previously saved experiment runs from the storage path."""
        for f in self.config.storage_path.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                run = ExperimentRun(
                    experiment_id = data["experiment"]["experiment_id"],
                    run_id        = data["run_id"],
                    metrics       = data["metrics"],
                    latency_ms    = data["latency_ms"],
                    query_count   = data["query_count"],
                )
                self._results[run.run_id] = run
            except Exception as exc:
                logger.warning("Failed to load %s: %s", f, exc)
