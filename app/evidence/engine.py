"""
Evidence Engine — Phase 7

=== THEORY ===

Evidence is the atomic unit of factual support in agentic retrieval.
Unlike a raw search result (doc_id + score), an evidence record tracks:

  - Provenance:   which document and chunk it came from
  - Claim:        the specific factual statement it supports
  - Confidence:   how strongly the source supports the claim
  - Validation:   whether a critic or validator has reviewed it
  - Relationships: links to other evidence (supports, contradicts, extends)

This is inspired by knowledge graphs in evidence-based medicine and
legal reasoning systems where every claim must be traceable to source.

=== DATA STRUCTURES ===

  EvidenceRecord
    evidence_id:  unique identifier
    doc_id:       source document
    chunk_id:     source chunk within document
    claim:        the factual statement
    content:      supporting text from the source
    score:        retrieval relevance score
    confidence:   agent-assessed confidence in [0, 1]
    source_title: human-readable source name
    validated:    bool (has a critic reviewed this?)
    tags:         list[str] (topic labels)

  EvidenceStore
    In-memory store backed by a dict[evidence_id → EvidenceRecord].
    Supports CRUD, filtering, and bulk operations.
    SQLite persistence is handled at the DB layer.

  EvidenceGraph
    Tracks relationships between evidence records:
      supports, contradicts, extends
    Implemented as an adjacency list.

  EvidenceExtractor
    Converts raw retrieval results into EvidenceRecord objects.

  EvidenceValidator
    Rules-based quality gate: checks content length, score threshold,
    and duplication.

=== COMPLEXITY ===

  EvidenceStore.add:            O(1) dict insert
  EvidenceStore.filter_by_tag:  O(N) linear scan
  EvidenceGraph.add_relation:   O(1) list append
  EvidenceGraph.get_relations:  O(degree) of the node
  EvidenceExtractor.extract:    O(K) where K = results count
  EvidenceValidator.validate:   O(N) where N = evidence count

=== SPACE ===

  Store: O(N) where N = evidence records
  Graph: O(N + E) where E = relationships

=== PRODUCTION EQUIVALENTS ===

  Semantic Scholar:  evidence linking for paper claims
  Glean:             source attribution graph
  Elicit:            claim extraction + evidence mapping
  LangSmith:         trace-level evidence tracking
"""

import logging
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class EvidenceRecord:
    """One atomic piece of evidence."""
    evidence_id:  str                     = field(default_factory=lambda: str(uuid.uuid4())[:12])
    doc_id:       int                     = 0
    chunk_id:     str                     = ""
    claim:        str                     = ""
    content:      str                     = ""
    score:        float                   = 0.0
    confidence:   float                   = 0.0
    source_title: str                     = ""
    source_url:   str                     = ""
    validated:    bool                    = False
    tags:         list[str]               = field(default_factory=list)
    metadata:     dict[str, Any]          = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "evidence_id":  self.evidence_id,
            "doc_id":       self.doc_id,
            "chunk_id":     self.chunk_id,
            "claim":        self.claim,
            "content":      self.content[:300],
            "score":        round(self.score, 4),
            "confidence":   round(self.confidence, 4),
            "source_title": self.source_title,
            "validated":    self.validated,
            "tags":         self.tags,
        }


class EvidenceRelationType:
    SUPPORTS     = "supports"
    CONTRADICTS  = "contradicts"
    EXTENDS      = "extends"


@dataclass
class EvidenceRelation:
    from_id:       str
    to_id:         str
    relation_type: str
    confidence:    float = 0.5


# ── Evidence Store ─────────────────────────────────────────────────────────────

class EvidenceStore:
    """
    In-memory evidence repository with CRUD and filtering.

    Backed by dict[evidence_id → EvidenceRecord].  Supports:
      add, get, remove, filter_by_tag, filter_by_doc, all, count.
    """

    def __init__(self) -> None:
        self._records: dict[str, EvidenceRecord] = {}

    def add(self, record: EvidenceRecord) -> str:
        self._records[record.evidence_id] = record
        return record.evidence_id

    def add_many(self, records: list[EvidenceRecord]) -> list[str]:
        return [self.add(r) for r in records]

    def get(self, evidence_id: str) -> Optional[EvidenceRecord]:
        return self._records.get(evidence_id)

    def remove(self, evidence_id: str) -> bool:
        return self._records.pop(evidence_id, None) is not None

    def all(self) -> list[EvidenceRecord]:
        return list(self._records.values())

    def count(self) -> int:
        return len(self._records)

    def filter_by_tag(self, tag: str) -> list[EvidenceRecord]:
        return [r for r in self._records.values() if tag in r.tags]

    def filter_by_doc(self, doc_id: int) -> list[EvidenceRecord]:
        return [r for r in self._records.values() if r.doc_id == doc_id]

    def filter_validated(self, validated: bool = True) -> list[EvidenceRecord]:
        return [r for r in self._records.values() if r.validated == validated]

    def filter_by_score(self, min_score: float) -> list[EvidenceRecord]:
        return [r for r in self._records.values() if r.score >= min_score]

    def clear(self) -> None:
        self._records.clear()


# ── Evidence Graph ─────────────────────────────────────────────────────────────

class EvidenceGraph:
    """
    Tracks relationships between evidence records.

    Adjacency list implementation:
      _adj[evidence_id] = list[EvidenceRelation]

    Supports directional queries:
      get_relations(id)       → all outgoing edges
      get_supporters(id)      → evidence that supports id
      get_contradictions(id)  → evidence that contradicts id
      get_extensions(id)      → evidence that extends id
    """

    def __init__(self) -> None:
        self._adj: dict[str, list[EvidenceRelation]] = defaultdict(list)

    def add_relation(self, relation: EvidenceRelation) -> None:
        self._adj[relation.from_id].append(relation)

    def get_relations(self, evidence_id: str) -> list[EvidenceRelation]:
        return self._adj.get(evidence_id, [])

    def get_by_type(self, evidence_id: str, relation_type: str) -> list[EvidenceRelation]:
        return [
            r for r in self._adj.get(evidence_id, [])
            if r.relation_type == relation_type
        ]

    def get_supporters(self, evidence_id: str) -> list[str]:
        return [r.to_id for r in self.get_by_type(evidence_id, EvidenceRelationType.SUPPORTS)]

    def get_contradictions(self, evidence_id: str) -> list[str]:
        return [r.to_id for r in self.get_by_type(evidence_id, EvidenceRelationType.CONTRADICTS)]

    def get_extensions(self, evidence_id: str) -> list[str]:
        return [r.to_id for r in self.get_by_type(evidence_id, EvidenceRelationType.EXTENDS)]

    def node_count(self) -> int:
        return len(self._adj)

    def edge_count(self) -> int:
        return sum(len(edges) for edges in self._adj.values())

    def clear(self) -> None:
        self._adj.clear()


# ── Evidence Extractor ─────────────────────────────────────────────────────────

class EvidenceExtractor:
    """
    Converts raw retrieval results or agent evidence dicts into
    EvidenceRecord objects.

    Handles:
      - agent evidence dicts (from RetrievalAgent output)
      - raw search result objects (Phase 4/5 results)
    """

    def extract_from_agent_output(
        self, evidence_dicts: list[dict], tags: list[str] | None = None,
    ) -> list[EvidenceRecord]:
        records: list[EvidenceRecord] = []
        for ev in evidence_dicts:
            records.append(EvidenceRecord(
                doc_id       = ev.get("doc_id", 0),
                chunk_id     = ev.get("chunk_id", ""),
                claim        = ev.get("content", "")[:200],
                content      = ev.get("content", ""),
                score        = ev.get("score", 0.0),
                confidence   = ev.get("score", 0.0),
                source_title = ev.get("title", ""),
                tags         = tags or [],
            ))
        return records

    def extract_from_search_result(self, result: Any) -> list[EvidenceRecord]:
        records: list[EvidenceRecord] = []
        if hasattr(result, "results"):
            for d in result.results:
                records.append(EvidenceRecord(
                    doc_id       = d.doc_id,
                    chunk_id     = f"doc_{d.doc_id}_0",
                    claim        = getattr(d, "snippet", "")[:200],
                    content      = getattr(d, "snippet", str(d))[:500],
                    score        = getattr(d, "score", getattr(d, "hybrid_score", 0.5)),
                    confidence   = getattr(d, "score", 0.5),
                    source_title = getattr(d, "title", f"Document {d.doc_id}"),
                ))
        return records


# ── Evidence Validator ─────────────────────────────────────────────────────────

class EvidenceValidator:
    """
    Rules-based quality gate for evidence records.

    Checks:
      - Minimum content length (default: 20 chars)
      - Minimum relevance score (default: 0.1)
      - Duplication detection (Jaccard overlap ≥ 0.8)

    Returns (valid_records, rejected_records, report).
    """

    def __init__(
        self,
        min_content_length: int   = 20,
        min_score:          float = 0.1,
        dedup_threshold:    float = 0.80,
    ) -> None:
        self._min_len   = min_content_length
        self._min_score = min_score
        self._dedup     = dedup_threshold

    def validate(
        self, records: list[EvidenceRecord],
    ) -> tuple[list[EvidenceRecord], list[EvidenceRecord], dict]:
        valid:    list[EvidenceRecord] = []
        rejected: list[EvidenceRecord] = []
        reasons:  dict[str, int] = defaultdict(int)

        seen_tokens: list[set[str]] = []

        for rec in records:
            if len(rec.content.strip()) < self._min_len:
                rejected.append(rec)
                reasons["too_short"] += 1
                continue

            if rec.score < self._min_score:
                rejected.append(rec)
                reasons["low_score"] += 1
                continue

            tokens = set(rec.content.lower().split())
            is_dup = False
            for seen in seen_tokens:
                if not tokens or not seen:
                    continue
                jaccard = len(tokens & seen) / len(tokens | seen)
                if jaccard >= self._dedup:
                    is_dup = True
                    break

            if is_dup:
                rejected.append(rec)
                reasons["duplicate"] += 1
                continue

            rec.validated = True
            valid.append(rec)
            seen_tokens.append(tokens)

        report = {
            "total":    len(records),
            "valid":    len(valid),
            "rejected": len(rejected),
            "reasons":  dict(reasons),
        }

        return valid, rejected, report
