"""
Citation Engine

=== THEORY ===

Citations answer: "Where does this claim come from?"

In a RAG system, every claim in the generated answer should be traceable to a
specific source chunk.  Without citations, the user cannot verify correctness,
and the system is indistinguishable from hallucination.

=== CITATION STRATEGIES ===

  Numbered  [1][2]:   Each source chunk gets an index.  Claims are annotated
                      inline with the bracket index.
                      → Used by Perplexity, Bing Chat, ChatGPT Browse

  Inline    (Source): Source title inserted directly after the claim.
                      More readable; harder to parse programmatically.

=== ATTRIBUTION ALGORITHM ===

  1. Extract the source chunks used in context (we have them from ContextBuilder).
  2. Split the answer into sentences.
  3. For each sentence, find the most overlapping source chunk.
  4. If overlap ≥ threshold, annotate the sentence with [idx].
  5. Build the reference list.

Token-Jaccard overlap is used as a lightweight similarity proxy.  This does
NOT require re-ranking or embedding.

=== DATA STRUCTURES ===

  Citation    — one source reference with index, doc, snippet, score
  CitedAnswer — the annotated answer text + list of citations + formatted refs

=== COMPLEXITY ===

  Annotate(A sentences, C chunks):  O(A × C × avg_len)
  Practical: A ≤ 20, C ≤ 10 → negligible

=== PRODUCTION EQUIVALENTS ===

  Perplexity: LLM is explicitly instructed to emit [1]...[N] tags;
              post-processing maps tags back to sources.
  Bing Chat:  Superscript citations; each source card links to the webpage.
  Llama Index: CitationQueryEngine — forces the LLM to output JSON with
              {answer, sources} and post-processes into rich citations.
"""

import logging
import re
from dataclasses import dataclass, field

from app.context_builder.builder import ContextChunk
from app.config import CitationConfig

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Citation:
    """A single source reference."""
    index:           int
    doc_id:          int
    chunk_id:        str
    title:           str
    url:             str
    snippet:         str           # short excerpt from the chunk
    relevance_score: float


@dataclass
class CitedAnswer:
    """The answer with inline citations and a formatted reference block."""
    answer:               str
    citations:            list[Citation]
    formatted_references: str          # human-readable reference list
    citation_count:       int = 0

    def __post_init__(self) -> None:
        self.citation_count = len(self.citations)


# ── Citation engine ───────────────────────────────────────────────────────────

class CitationEngine:
    """
    Annotates generated answers with source citations.

    Injection:
        engine = CitationEngine(config.citation)
        cited  = engine.annotate(answer, context.chunks)
    """

    def __init__(self, config: CitationConfig | None = None):
        self.config = config or CitationConfig()

    def annotate(
        self,
        answer:        str,
        context_chunks: list[ContextChunk],
    ) -> CitedAnswer:
        """
        Main entry-point.  Split the answer into sentences, attribute each
        sentence to the best-matching context chunk, and build the reference list.
        """
        if not context_chunks:
            return CitedAnswer(
                answer=answer, citations=[],
                formatted_references="", citation_count=0,
            )

        # Build citation objects (index 1…N)
        citations: list[Citation] = []
        for i, chunk in enumerate(context_chunks, 1):
            snippet = self._make_snippet(chunk.text)
            citations.append(Citation(
                index=i, doc_id=chunk.doc_id, chunk_id=chunk.chunk_id,
                title=chunk.source_title, url=chunk.source_url,
                snippet=snippet, relevance_score=round(chunk.score, 4),
            ))

        # Annotate the answer text
        if self.config.style == "numbered":
            annotated = self._annotate_numbered(answer, citations, context_chunks)
        else:
            annotated = self._annotate_inline(answer, citations, context_chunks)

        # Format the reference section
        refs = self._format_references(citations)

        return CitedAnswer(
            answer=annotated,
            citations=citations,
            formatted_references=refs,
        )

    # ── Annotation strategies ─────────────────────────────────────────────

    def _annotate_numbered(
        self,
        answer:         str,
        citations:      list[Citation],
        chunks:         list[ContextChunk],
    ) -> str:
        """
        Insert [N] tags at the end of sentences that are supported by a source.
        """
        sentences = self._split_sentences(answer)
        result_parts: list[str] = []

        for sentence in sentences:
            best_idx   = None
            best_score = 0.0
            for i, chunk in enumerate(chunks):
                sim = _token_jaccard(sentence, chunk.text)
                if sim > best_score:
                    best_score = sim
                    best_idx   = i + 1   # 1-indexed

            if best_idx is not None and best_score >= 0.08:
                sentence = sentence.rstrip() + f" [{best_idx}]"
            result_parts.append(sentence)

        return " ".join(result_parts)

    def _annotate_inline(
        self,
        answer:    str,
        citations: list[Citation],
        chunks:    list[ContextChunk],
    ) -> str:
        """Insert (Source: title) after supported sentences."""
        sentences = self._split_sentences(answer)
        result_parts: list[str] = []

        for sentence in sentences:
            best_chunk = None
            best_score = 0.0
            for chunk in chunks:
                sim = _token_jaccard(sentence, chunk.text)
                if sim > best_score:
                    best_score = sim
                    best_chunk = chunk

            if best_chunk is not None and best_score >= 0.08:
                sentence = sentence.rstrip() + f" (Source: {best_chunk.source_title})"
            result_parts.append(sentence)

        return " ".join(result_parts)

    # ── Reference formatting ──────────────────────────────────────────────

    def _format_references(self, citations: list[Citation]) -> str:
        """
        Format the reference list at the bottom of the answer.

        Example:
          References:
          [1] FastAPI Documentation — "FastAPI is a modern, fast web framework..."
          [2] Python Docs (doc_id=5) — "Python is a high-level programming language..."
        """
        if not citations:
            return ""
        lines = ["", "**References:**"]
        for c in citations:
            if self.config.include_snippet and c.snippet:
                line = f"[{c.index}] {c.title} — \"{c.snippet}\""
            else:
                line = f"[{c.index}] {c.title}"
            if c.url:
                line += f"  ({c.url})"
            lines.append(line)
        return "\n".join(lines)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Naive sentence splitter — sufficient for attribution."""
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p for p in parts if p]

    def _make_snippet(self, text: str) -> str:
        """Return the first N chars of the chunk text."""
        text = text.strip()
        if len(text) <= self.config.max_snippet_len:
            return text
        return text[: self.config.max_snippet_len].rsplit(" ", 1)[0] + "…"


# ── Token-Jaccard helper (duplicated from context_builder to avoid circular) ─

def _token_jaccard(a: str, b: str) -> float:
    ta = set(re.sub(r"[^\w]", " ", a.lower()).split())
    tb = set(re.sub(r"[^\w]", " ", b.lower()).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
