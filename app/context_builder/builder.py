"""
Context Construction Engine

=== THEORY ===

After retrieval we have a ranked list of chunks.  Before passing them to the
LLM we must:

  1. Filter   — discard low-score chunks (noise)
  2. Deduplicate — remove chunks that convey the same information
  3. Diversify — prefer chunks from different source documents (MMR)
  4. Budget   — fit within the LLM's context window

=== MAXIMAL MARGINAL RELEVANCE (MMR) ===

MMR (Carbonell & Goldstein 1998) is a classic diversification algorithm used
by every modern RAG system:

  MMR(d) = λ · Sim(d, query) − (1−λ) · max_{s ∈ Selected} Sim(d, s)

  λ=1 → pure relevance (no diversification)
  λ=0 → pure novelty (ignore relevance)
  λ=0.6 → balanced (our default)

We approximate Sim(a, b) with token-Jaccard similarity (no vector needed).

=== TOKEN BUDGETING ===

We estimate tokens as words × 1.3 (same heuristic as LLMProvider).
Chunks are added in MMR order until the budget is exhausted.  A small
safety margin (10%) is reserved so the LLM can always produce output.

=== DEDUPLICATION ===

Two chunks are considered duplicates if their token Jaccard overlap
exceeds `dedup_threshold` (default 0.80).  Only the higher-scoring one
is kept.  This prevents the model from reading "Python is a programming
language" five times from five different chunks of the same document.

=== CONTEXT FORMAT ===

The assembled context is formatted as a numbered list:

  [1] <title> (doc_id=3)
  <chunk text>

  [2] <title> (doc_id=7)
  <chunk text>

Numbered references allow the citation engine to match [1], [2] etc.

=== DATA STRUCTURES ===

  ContextChunk — lightweight value object for one chunk of retrieved text
  Context      — assembled context ready to inject into a prompt
  ContextMetadata — statistics logged for observability

=== COMPLEXITY ===

  filter:     O(C)               C = candidate chunks
  dedup:      O(C²) worst case   (in practice small C ≤ 100)
  MMR:        O(C × S)           S = selected so far
  assemble:   O(C × len)
  Total:      O(C²) — acceptable for C ≤ 100

=== PRODUCTION EQUIVALENTS ===

  LangChain:   ContextualCompressionRetriever + EmbeddingsFilter
  LlamaIndex:  SentenceWindowNodeParser + MetadataReplacementNodePostProcessor
  Perplexity:  internal context scoring + deduplication pipeline
  Glean:       passage ranking with BM25 + neural re-scoring
"""

import logging
import re
from dataclasses import dataclass, field

from app.config import ContextConfig

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ContextChunk:
    """A single retrieved chunk flowing through the context builder."""
    chunk_id:     str
    doc_id:       int
    text:         str
    score:        float             # retrieval / reranker score
    source_title: str
    source_url:   str  = ""
    token_count:  int  = 0

    def __post_init__(self) -> None:
        if self.token_count == 0:
            self.token_count = max(1, int(len(self.text.split()) * 1.3))


@dataclass
class ContextMetadata:
    """Statistics about the assembled context — logged for observability."""
    total_chunks:     int
    total_tokens:     int
    source_count:     int
    redundancy_score: float         # fraction of candidates that were deduped
    diversity_score:  float         # fraction of chunks from distinct sources
    sources:          list[dict] = field(default_factory=list)


@dataclass
class Context:
    """Ready-to-inject context with numbered references."""
    text:     str
    chunks:   list[ContextChunk]
    metadata: ContextMetadata

    def is_empty(self) -> bool:
        return not self.chunks


# ── Helper: token-Jaccard similarity ─────────────────────────────────────────

def _token_jaccard(a: str, b: str) -> float:
    """Fast token-Jaccard similarity.  O(|a| + |b|)."""
    ta = set(re.sub(r"[^\w]", " ", a.lower()).split())
    tb = set(re.sub(r"[^\w]", " ", b.lower()).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ── Context builder ───────────────────────────────────────────────────────────

class ContextBuilder:
    """
    Transforms a list of retrieved ContextChunks into a compact, diverse
    Context object that fits within the LLM's token budget.

    Injection example (DI in RAGPipeline):
        builder = ContextBuilder(config.context)
        context = builder.build(chunks, query=request.query)
    """

    def __init__(self, config: ContextConfig | None = None):
        self.config = config or ContextConfig()

    def build(
        self,
        chunks:     list[ContextChunk],
        query:      str,
        max_tokens: int | None = None,
    ) -> Context:
        """
        Full pipeline: filter → deduplicate → MMR-diversify → budget → assemble.

        Parameters
        ----------
        chunks      Raw retrieved chunks (may be unsorted).
        query       The user query (used for MMR relevance scoring).
        max_tokens  Override config.max_tokens for this call.
        """
        budget = max_tokens or self.config.max_tokens

        if not chunks:
            return Context(
                text="", chunks=[],
                metadata=ContextMetadata(0, 0, 0, 0.0, 0.0),
            )

        # 1. Filter by minimum score
        pre_filter = len(chunks)
        chunks = [c for c in chunks if c.score >= self.config.min_score]

        # 2. Sort by score descending
        chunks = sorted(chunks, key=lambda c: c.score, reverse=True)

        # 3. Deduplicate
        chunks, deduped_count = self._deduplicate(chunks)

        # 4. MMR diversification (or score-only if use_mmr=False)
        if self.config.use_mmr:
            selected = self._mmr(chunks, query, self.config.max_chunks)
        else:
            selected = chunks[: self.config.max_chunks]

        # 5. Token budget enforcement
        selected = self._apply_budget(selected, budget)

        redundancy = deduped_count / pre_filter if pre_filter else 0.0
        source_ids = {c.doc_id for c in selected}
        diversity  = len(source_ids) / len(selected) if selected else 0.0

        metadata = ContextMetadata(
            total_chunks     = len(selected),
            total_tokens     = sum(c.token_count for c in selected),
            source_count     = len(source_ids),
            redundancy_score = round(redundancy, 3),
            diversity_score  = round(diversity, 3),
            sources          = [
                {"doc_id": c.doc_id, "title": c.source_title, "url": c.source_url}
                for c in selected
            ],
        )

        text = self._assemble(selected)
        logger.debug(
            "ContextBuilder: %d chunks → %d selected, %d tokens, %d sources",
            pre_filter, len(selected), metadata.total_tokens, metadata.source_count,
        )
        return Context(text=text, chunks=selected, metadata=metadata)

    # ── Internal stages ───────────────────────────────────────────────────

    def _deduplicate(
        self, chunks: list[ContextChunk]
    ) -> tuple[list[ContextChunk], int]:
        """
        Remove chunks that are near-duplicates of a higher-ranked chunk.
        O(C²) but C is typically ≤ 100.
        """
        kept:   list[ContextChunk] = []
        deduped = 0
        for candidate in chunks:
            is_dup = False
            for existing in kept:
                if _token_jaccard(candidate.text, existing.text) >= self.config.dedup_threshold:
                    is_dup = True
                    deduped += 1
                    break
            if not is_dup:
                kept.append(candidate)
        return kept, deduped

    def _mmr(
        self,
        chunks: list[ContextChunk],
        query:  str,
        k:      int,
    ) -> list[ContextChunk]:
        """
        Maximal Marginal Relevance selection.
        Balances relevance (score) against diversity (novelty vs. selected set).
        """
        if not chunks:
            return []

        λ = self.config.diversity_lambda
        selected:   list[ContextChunk] = []
        candidates: list[ContextChunk] = list(chunks)

        while candidates and len(selected) < k:
            best_idx   = 0
            best_score = -float("inf")

            for i, cand in enumerate(candidates):
                # Relevance term: normalised original score
                rel = cand.score

                # Redundancy term: max similarity to any selected chunk
                if selected:
                    red = max(
                        _token_jaccard(cand.text, s.text)
                        for s in selected
                    )
                else:
                    red = 0.0

                mmr_score = λ * rel - (1 - λ) * red
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx   = i

            selected.append(candidates.pop(best_idx))

        return selected

    def _apply_budget(
        self, chunks: list[ContextChunk], budget: int
    ) -> list[ContextChunk]:
        """Trim chunk list to fit within token budget (90 % safety margin)."""
        safe_budget = int(budget * 0.90)
        used = 0
        result: list[ContextChunk] = []
        for c in chunks:
            if used + c.token_count > safe_budget:
                break
            result.append(c)
            used += c.token_count
        return result

    def _assemble(self, chunks: list[ContextChunk]) -> str:
        """Format selected chunks as a numbered reference list."""
        parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            header = f"[{i}] {chunk.source_title} (doc_id={chunk.doc_id})"
            parts.append(f"{header}\n{chunk.text.strip()}")
        return "\n\n".join(parts)
