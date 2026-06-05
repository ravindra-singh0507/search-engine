"""
Retrieval Agent — Phase 7

=== THEORY ===

The retrieval agent is the primary evidence-gathering unit.  It wraps
the Phase 5 multi-stage retrieval pipeline (BM25 → semantic → fusion →
rerank) and the Phase 6 RAG pipeline (context builder → LLM → cite →
ground) into a single agent interface.

Unlike a raw retrieval call, the agent:
  1. Validates the query (non-empty, reasonable length)
  2. Executes retrieval through the injected pipeline
  3. Extracts structured evidence from the results
  4. Computes a retrieval quality score
  5. Records what it found in agent memory (for downstream agents)

=== EVIDENCE EXTRACTION ===

Each retrieved document is converted to an evidence record:
  {
    "doc_id":     int,
    "title":      str,
    "content":    str (first 500 chars),
    "score":      float,
    "source":     str,
    "chunk_id":   str,
  }

This uniform structure is what the critic and synthesis agents consume.

=== COMPLEXITY ===

  _execute():  dominated by retrieval pipeline O(C) where C = corpus
  Evidence extraction: O(K) where K = top_k results

=== PRODUCTION EQUIVALENTS ===

  Perplexity:      search agent that queries multiple backends
  OpenAI Deep Research: retrieval step in the research loop
  Glean Agents:    enterprise search agent with connector abstraction
  LangGraph:       retriever tool node
"""

import logging
from typing import Any

from app.agents.base import (
    Agent, AgentContext, AgentMemory, AgentResult, AgentStatus,
    AgentTask, AgentType,
)

logger = logging.getLogger(__name__)


class RetrievalAgent(Agent):
    """
    Gathers evidence from the search platform for a given query.

    Reads from context:
      context.retriever — Phase 5 RetrievalPipeline or Phase 4 HybridSearchService
      context.rag_pipeline — Phase 6 RAGPipeline (optional, for answer gen)

    Returns:
      AgentResult.output = {
        "query":          str,
        "evidence":       list[dict],
        "evidence_count": int,
        "quality_score":  float,
        "answer":         str | None,  (if RAG pipeline available)
      }
    """

    @property
    def agent_type(self) -> AgentType:
        return AgentType.RETRIEVAL

    def _execute(
        self,
        task:    AgentTask,
        context: AgentContext,
        memory:  AgentMemory,
    ) -> AgentResult:
        query = task.params.get("query", task.goal)
        top_k = task.params.get("top_k", 5)
        use_rag = task.params.get("use_rag", False)

        if not query or not query.strip():
            return AgentResult(
                task_id    = task.task_id,
                agent_type = AgentType.RETRIEVAL,
                status     = AgentStatus.FAILED,
                output     = None,
                error      = "Empty query",
            )

        evidence: list[dict] = []
        answer:   str | None = None

        # Use RAG pipeline if available and requested
        if use_rag and context.rag_pipeline:
            from app.rag.pipeline import RAGRequest
            rag_req = RAGRequest(query=query, top_k=top_k)
            rag_resp = context.rag_pipeline.query(rag_req)
            answer = rag_resp.answer
            for cit in rag_resp.citations:
                evidence.append({
                    "doc_id":   cit.doc_id,
                    "title":    cit.title,
                    "content":  cit.snippet[:500] if cit.snippet else "",
                    "score":    cit.relevance_score,
                    "source":   "rag",
                    "chunk_id": cit.chunk_id,
                })

        # Always run retrieval for raw evidence
        if context.retriever:
            try:
                result = context.retriever.search(query, top_k=top_k)
                raw_docs = self._extract_docs(result)
                for doc in raw_docs:
                    if not any(e["doc_id"] == doc["doc_id"] for e in evidence):
                        evidence.append(doc)
            except Exception as exc:
                logger.warning("Retrieval failed for %r: %s", query[:50], exc)

        quality = self._compute_quality(evidence)

        memory.remember("retrieval", {
            "query": query,
            "evidence_count": len(evidence),
            "quality": quality,
        })

        return AgentResult(
            task_id    = task.task_id,
            agent_type = AgentType.RETRIEVAL,
            status     = AgentStatus.DONE,
            output     = {
                "query":          query,
                "evidence":       evidence,
                "evidence_count": len(evidence),
                "quality_score":  quality,
                "answer":         answer,
            },
            evidence   = evidence,
            confidence = quality,
        )

    def _extract_docs(self, result: Any) -> list[dict]:
        """Convert retriever result to uniform evidence dicts."""
        docs = []

        # Phase 5 PipelineResult
        if hasattr(result, "results") and hasattr(result, "stage_latencies"):
            for c in result.results:
                if c.doc_id:
                    docs.append({
                        "doc_id":   c.doc_id,
                        "title":    c.title,
                        "content":  c.content[:500] if c.content else "",
                        "score":    c.final_score,
                        "source":   "pipeline",
                        "chunk_id": f"doc_{c.doc_id}_0",
                    })
            return docs

        # Phase 4 SearchResult / HybridResult
        if hasattr(result, "results"):
            for d in result.results:
                docs.append({
                    "doc_id":   d.doc_id,
                    "title":    getattr(d, "title", f"Document {d.doc_id}"),
                    "content":  getattr(d, "snippet", str(d))[:500],
                    "score":    getattr(d, "score", getattr(d, "hybrid_score",
                                getattr(d, "bm25_score", 0.5))),
                    "source":   "search",
                    "chunk_id": f"doc_{d.doc_id}_0",
                })
            return docs

        return docs

    @staticmethod
    def _compute_quality(evidence: list[dict]) -> float:
        """
        Heuristic quality score for a retrieval set.

        Factors:
          - Count: more evidence = better (up to a point)
          - Score distribution: higher average score = better
          - Diversity: unique titles = broader coverage
        """
        if not evidence:
            return 0.0

        count_score = min(1.0, len(evidence) / 5)
        avg_score   = sum(e.get("score", 0) for e in evidence) / len(evidence)
        avg_score   = min(1.0, avg_score)
        titles      = {e.get("title", "") for e in evidence}
        diversity   = min(1.0, len(titles) / max(len(evidence), 1))

        return round(0.3 * count_score + 0.4 * avg_score + 0.3 * diversity, 4)
