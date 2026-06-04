"""
RAG Orchestration Pipeline

=== THEORY ===

Retrieval-Augmented Generation (RAG) was introduced by Lewis et al. (2020)
as a way to ground LLM outputs in external knowledge.  The key insight:
LLMs have broad reasoning ability but stale/unreliable factual knowledge.
Retrieval provides up-to-date, verifiable facts.  The LLM's job is to READ
and REASON, not to recall.

=== PIPELINE ARCHITECTURE ===

  Query
    │
    ├── [Optional] Memory load     — inject prior turns as history
    ├── [Optional] Multi-step      — decompose complex queries
    │
    ├── Stage 1: Retrieval         — Phase 5 RetrievalPipeline or HybridSearch
    │             → list of RankedDocument / PipelineCandidate
    │
    ├── Stage 2: Context Build     — ContextBuilder → Context
    │             filter, dedup, MMR, token budget
    │
    ├── Stage 3: Prompt Assembly   — PromptRegistry.render()
    │             system + context + history + question
    │
    ├── Stage 4: LLM Generation    — LLMProvider.generate() or .stream()
    │
    ├── Stage 5: Citation          — CitationEngine.annotate()
    │
    ├── Stage 6: Grounding         — GroundingVerifier.verify()
    │
    ├── Stage 7: Confidence        — ConfidenceEngine.score()
    │
    ├── [Optional] Memory write    — persist messages
    │
    └── Stage 8: Observability     — record all latencies + token counts

=== MULTI-STEP QUESTION HANDLING ===

Complex queries like "Compare FastAPI, Flask, and Django for scalability"
are decomposed into sub-queries:
  1. "FastAPI scalability performance"
  2. "Flask scalability performance"
  3. "Django scalability performance"

Each sub-query is retrieved independently; results are merged before context
construction.  The final answer synthesizes across all sub-results.

Query decomposition uses a lightweight keyword split heuristic — no LLM
call needed for decomposition (avoids latency + cost).  For Phase 7
(Agentic workflows), this can be upgraded to LLM-based planning.

=== STREAMING ===

stream() is a generator that yields:
  - data: {"type": "token", "content": "..."}\n\n
  - data: {"type": "done", "citations": [...], "grounding_score": 0.7}\n\n

The caller wraps this in FastAPI's StreamingResponse with
  media_type="text/event-stream"

=== DEPENDENCY INJECTION ===

RAGPipeline accepts all components via constructor.  This makes it fully
testable with mocks and swappable in production without code changes.

=== COMPLEXITY ===

  Stage 1 (retrieval):  O(C)  — from Phase 5
  Stage 2 (context):    O(C²) — dedup; C ≤ 100
  Stage 3 (prompt):     O(T)  — T = token count
  Stage 4 (LLM):        O(T · D²) — transformer inference
  Stage 5-7 (post):     O(S × C) — S sentences, C chunks
  Total: dominated by LLM inference

=== PRODUCTION EQUIVALENTS ===

  LangChain:   RetrievalQA chain, ConversationalRetrievalChain
  LlamaIndex:  QueryEngine, CitationQueryEngine
  Perplexity:  custom pipeline (retrieval → reading → answer with citations)
  Glean:       enterprise search + generative answers with citations
  ChatGPT:     GPT-4o with Browse — retrieval → grounded generation
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Iterator

from app.config import RAGConfig, ContextConfig
from app.context_builder.builder import ContextBuilder, ContextChunk, Context
from app.prompts.templates import PromptRegistry, get_registry
from app.llm.provider import LLMProvider, LLMResponse, MockLLMProvider
from app.citations.engine import CitationEngine, CitedAnswer, Citation
from app.grounding.verifier import GroundingVerifier, GroundingReport
from app.confidence.engine import ConfidenceEngine, ConfidenceScores
from app.memory.memory import MemoryService

logger = logging.getLogger(__name__)


# ── Request / Response dataclasses ────────────────────────────────────────────

@dataclass
class RAGRequest:
    query:           str
    session_id:      str | None = None
    top_k:           int        = 5
    max_tokens:      int        = 2048
    template:        str        = "qa"
    use_reranker:    bool       = True
    stream:          bool       = False
    user_id:         str | None = None
    multi_step:      bool       = False
    override_config: dict | None = None


@dataclass
class RAGResponse:
    query:            str
    answer:           str
    citations:        list[Citation]
    grounding:        GroundingReport
    confidence:       ConfidenceScores
    context_metadata: dict
    retrieval_count:  int
    stage_latencies:  dict[str, float]
    total_latency_ms: float
    tokens_used:      int
    session_id:       str | None = None
    subqueries:       list[str]  = field(default_factory=list)
    formatted_answer: str        = ""   # answer + references block

    def __post_init__(self) -> None:
        if not self.formatted_answer:
            cited = next(
                (c for c in [self.answer] if c), ""
            )
            self.formatted_answer = cited


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Orchestrates retrieval → context → prompt → LLM → citation → grounding.

    All components are injected; the pipeline itself contains no business
    logic outside of orchestration and latency tracking.
    """

    def __init__(
        self,
        retriever,                        # SearchService or RetrievalPipeline (Phase 5)
        context_builder:  ContextBuilder,
        prompt_registry:  PromptRegistry,
        llm:              LLMProvider,
        citation_engine:  CitationEngine,
        grounding_verifier: GroundingVerifier,
        confidence_engine:  ConfidenceEngine,
        memory:           MemoryService | None = None,
        metrics           = None,         # MetricsCollector (optional)
        config:           RAGConfig | None = None,
    ):
        self.retriever    = retriever
        self.ctx_builder  = context_builder
        self.prompts      = prompt_registry
        self.llm          = llm
        self.citations    = citation_engine
        self.grounding    = grounding_verifier
        self.confidence   = confidence_engine
        self.memory       = memory
        self.metrics      = metrics
        self.config       = config or RAGConfig()

    # ── Main entry-points ─────────────────────────────────────────────────

    def query(self, request: RAGRequest) -> RAGResponse:
        """Execute the full RAG pipeline synchronously."""
        t_total = time.perf_counter()
        latencies: dict[str, float] = {}

        # Optionally decompose complex query into sub-queries
        subqueries: list[str] = []
        if request.multi_step and self.config.enable_multi_step:
            subqueries = self._decompose_query(request.query)

        queries_to_run = subqueries if subqueries else [request.query]

        # Stage 1: Retrieval (all sub-queries merged)
        t0 = time.perf_counter()
        chunks = self._retrieve_all(queries_to_run, request)
        latencies["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Stage 2: Context construction
        t0 = time.perf_counter()
        context = self.ctx_builder.build(
            chunks, query=request.query,
            max_tokens=self.config.context.max_tokens,
        )
        latencies["context_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Stage 3: Prompt assembly
        t0 = time.perf_counter()
        history = ""
        if self.memory and request.session_id:
            history = self.memory.format_history(request.session_id)
        rendered = self.prompts.render(
            name=request.template,
            context=context.text,
            question=request.query,
            history=history,
        )
        latencies["prompt_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Prompt injection detection
        safe_query = _sanitize_query(request.query)

        # Stage 4: LLM generation
        t0 = time.perf_counter()
        llm_resp: LLMResponse = self.llm.generate(
            prompt=rendered["user"],
            system=rendered.get("system", ""),
            max_tokens=request.max_tokens,
        )
        latencies["llm_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Stage 5: Citation
        t0 = time.perf_counter()
        cited: CitedAnswer = self.citations.annotate(llm_resp.text, context.chunks)
        latencies["citation_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Stage 6: Grounding verification
        t0 = time.perf_counter()
        grounding_report: GroundingReport = self.grounding.verify(cited.answer, context)
        latencies["grounding_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Stage 7: Confidence scoring
        t0 = time.perf_counter()
        conf: ConfidenceScores = self.confidence.score(
            context=context,
            grounding_report=grounding_report,
            citations=cited.citations,
            answer=cited.answer,
        )
        latencies["confidence_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Memory write
        if self.memory and request.session_id:
            self.memory.add_message(request.session_id, "user", request.query)
            self.memory.add_message(
                request.session_id, "assistant", cited.answer,
                metadata={
                    "grounding_score": grounding_report.grounding_score,
                    "confidence":      conf.overall_confidence,
                    "citation_count":  cited.citation_count,
                },
            )

        total_ms = round((time.perf_counter() - t_total) * 1000, 2)

        # Metrics
        if self.metrics:
            self.metrics.record_rag_query(total_ms, llm_resp.total_tokens)

        formatted = cited.answer
        if cited.formatted_references:
            formatted = cited.answer + "\n" + cited.formatted_references

        return RAGResponse(
            query            = request.query,
            answer           = cited.answer,
            citations        = cited.citations,
            grounding        = grounding_report,
            confidence       = conf,
            context_metadata = {
                "total_chunks":     context.metadata.total_chunks,
                "total_tokens":     context.metadata.total_tokens,
                "source_count":     context.metadata.source_count,
                "redundancy_score": context.metadata.redundancy_score,
                "diversity_score":  context.metadata.diversity_score,
                "sources":          context.metadata.sources,
            },
            retrieval_count  = len(chunks),
            stage_latencies  = latencies,
            total_latency_ms = total_ms,
            tokens_used      = llm_resp.total_tokens,
            session_id       = request.session_id,
            subqueries       = subqueries,
            formatted_answer = formatted,
        )

    def stream(self, request: RAGRequest) -> Iterator[str]:
        """
        Streaming version of query().

        Yields SSE-formatted strings:
          data: {"type":"token","content":"..."}\n\n
          data: {"type":"done","grounding_score":0.7,"confidence":"high"}\n\n
        """
        # Run retrieval + context + prompt synchronously
        chunks  = self._retrieve_all([request.query], request)
        context = self.ctx_builder.build(
            chunks, query=request.query,
            max_tokens=self.config.context.max_tokens,
        )
        history = ""
        if self.memory and request.session_id:
            history = self.memory.format_history(request.session_id)
        rendered = self.prompts.render(
            name=request.template,
            context=context.text,
            question=request.query,
            history=history,
        )

        full_text = ""
        chunk_size = self.config.stream_chunk_size

        # Stream tokens
        for token in self.llm.stream(
            prompt=rendered["user"],
            system=rendered.get("system", ""),
            max_tokens=request.max_tokens,
            chunk_size=chunk_size,
        ):
            full_text += token
            payload = json.dumps({"type": "token", "content": token})
            yield f"data: {payload}\n\n"

        # Post-process
        cited    = self.citations.annotate(full_text, context.chunks)
        grounding = self.grounding.verify(cited.answer, context)
        conf     = self.confidence.score(context, grounding, cited.citations, full_text)

        if self.memory and request.session_id:
            self.memory.add_message(request.session_id, "user", request.query)
            self.memory.add_message(request.session_id, "assistant", full_text)

        done_payload = json.dumps({
            "type":            "done",
            "citations":       [
                {"index": c.index, "title": c.title, "snippet": c.snippet}
                for c in cited.citations
            ],
            "grounding_score":    grounding.grounding_score,
            "hallucination_risk": grounding.hallucination_risk,
            "confidence":         conf.overall_confidence,
            "tier":               conf.tier,
            "references":         cited.formatted_references,
        })
        yield f"data: {done_payload}\n\n"

    # ── Stage implementations ─────────────────────────────────────────────

    def _retrieve_all(
        self, queries: list[str], request: RAGRequest
    ) -> list[ContextChunk]:
        """Run retrieval for all queries and merge results."""
        seen: set[int] = set()
        all_chunks: list[ContextChunk] = []

        for q in queries:
            chunks = self._retrieve_one(q, request)
            for c in chunks:
                if c.doc_id not in seen:
                    seen.add(c.doc_id)
                    all_chunks.append(c)

        return all_chunks

    def _retrieve_one(
        self, query: str, request: RAGRequest
    ) -> list[ContextChunk]:
        """Retrieve from the injected retriever, converting to ContextChunk."""
        top_k = request.top_k

        # Support both Phase 5 RetrievalPipeline and Phase 4 HybridSearchService
        result = self.retriever.search(query, top_k=top_k)

        # Phase 5 RetrievalPipeline returns PipelineResult
        if hasattr(result, "results") and hasattr(result, "stage_latencies"):
            candidates = result.results
            return [
                ContextChunk(
                    chunk_id     = f"doc_{c.doc_id}_0",
                    doc_id       = c.doc_id,
                    text         = c.content[:2000],
                    score        = c.final_score,
                    source_title = c.title,
                )
                for c in candidates if c.doc_id
            ]

        # Phase 4 HybridSearchService / SearchService returns SearchResult
        if hasattr(result, "results"):
            docs = result.results
            return [
                ContextChunk(
                    chunk_id     = f"doc_{d.doc_id}_0",
                    doc_id       = d.doc_id,
                    text         = getattr(d, "snippet", str(d))[:2000],
                    score        = getattr(d, "score", getattr(d, "hybrid_score",
                                   getattr(d, "bm25_score", 0.5))),
                    source_title = getattr(d, "title", f"Document {d.doc_id}"),
                )
                for d in docs
            ]

        return []

    def _decompose_query(self, query: str) -> list[str]:
        """
        Lightweight rule-based query decomposition for multi-step retrieval.

        Patterns handled:
          - "Compare A, B and C" → ["A", "B", "C"]
          - "Difference between A and B" → ["A", "B"]
          - "A vs B vs C" → ["A", "B", "C"]
          - Queries with multiple question marks → split on '?'
          - Everything else → [query] (no decomposition)

        Phase 7 will upgrade this to LLM-based planning.
        """
        q = query.strip()

        # Multi-question: "What is X? How does Y work?"
        questions = [p.strip() + "?" for p in q.split("?") if p.strip()]
        if len(questions) > 1:
            return questions[: self.config.max_subqueries]

        # "Compare A, B and C" / "A vs B vs C"
        compare_match = re.search(
            r"compare\s+(.+)|(.+)\s+vs\.?\s+(.+)|difference between\s+(.+)\s+and\s+(.+)",
            q, re.IGNORECASE,
        )
        if compare_match:
            text = compare_match.group(0)
            # Split on commas, "and", "vs"
            parts = re.split(r"\s*,\s*|\s+and\s+|\s+vs\.?\s+", text, flags=re.IGNORECASE)
            # Clean leading "compare" keyword
            parts = [
                re.sub(r"^(compare|difference between)\s+", "", p, flags=re.IGNORECASE).strip()
                for p in parts if p.strip()
            ]
            if len(parts) > 1:
                return parts[: self.config.max_subqueries]

        return [q]


# ── Prompt injection protection ───────────────────────────────────────────────

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous\s+|prior\s+|above\s+)?(instructions?|prompts?|rules?)",
    r"you are now",
    r"pretend (you are|to be)",
    r"disregard",
    r"system:\s",
    r"<\|im_start\|>",
    r"\[INST\]",
]
_INJECTION_RE = re.compile(
    "|".join(_INJECTION_PATTERNS), re.IGNORECASE
)


def _sanitize_query(query: str) -> str:
    """
    Detect and neutralise prompt injection attempts.
    Returns the original query if safe; returns a sanitized version if injection
    is detected (logs a warning — caller may choose to reject or proceed).
    """
    if _INJECTION_RE.search(query):
        logger.warning("Potential prompt injection detected: %r", query[:80])
        # Strip the offending text; safe fallback
        sanitized = _INJECTION_RE.sub("[REDACTED]", query)
        return sanitized
    return query
