"""
Document Chunking

=== THEORY ===

Transformer models (the backbone of all modern embedding models) have a
maximum context window — typically 512 tokens for bi-encoder models like
BGE/E5.  A 2 000-word document tokenises to ~2 500 tokens: more than the
limit.  We must split ("chunk") long documents before embedding.

Beyond the technical limit, smaller chunks improve retrieval precision.
If a relevant passage is buried in a 5 000-word document, embedding the
whole document produces a vector that is an average of all topics in the
document.  A targeted chunk embedding focuses on the relevant passage.

=== TWO STRATEGIES ===

1. FixedSizeChunker
   Non-overlapping windows of chunk_size words.
   Pro: Simple, no repeated text.
   Con: A phrase that straddles a boundary is split across two chunks;
        neither chunk contains the full phrase.

   Example (chunk_size=4):
   "the quick brown fox jumps over the lazy dog"
   → ["the quick brown fox", "jumps over the lazy", "dog"]

2. SlidingWindowChunker
   Windows of chunk_size words, advanced by (chunk_size - chunk_overlap) words.
   Pro: Every phrase is fully contained in at least one chunk.
   Con: Chunks overlap → more chunks → more embeddings → larger index.

   Example (chunk_size=4, overlap=2):
   "the quick brown fox jumps over the lazy dog"
   → ["the quick brown fox", "brown fox jumps over", "jumps over the lazy", ...]

In production: chunk_size=256 words, overlap=32 words is a good default for
bge-small-en-v1.5 (512 token limit, ~1.3 tokens per word → 332 tokens max).

=== COMPLEXITY ===

  Chunk a document of L words:
    Fixed:   O(L),  produces ceil(L / C) chunks
    Sliding: O(L),  produces ceil((L - C) / (C - O)) + 1 chunks
    where C = chunk_size, O = chunk_overlap

=== PRODUCTION EQUIVALENTS ===

  LangChain:    RecursiveCharacterTextSplitter (token-aware)
  LlamaIndex:   SentenceSplitter (semantic sentence boundaries)
  Haystack:     PreProcessor (sentence-aware chunking)

Our implementation splits by word count.  A production implementation would
use a tokenizer to stay precisely within the model's token budget.
"""

import logging
from dataclasses import dataclass

from app.config import ChunkingConfig

logger = logging.getLogger(__name__)


# ── Chunk dataclass ────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """One passage extracted from a document."""
    chunk_id:     str       # "{doc_id}_{chunk_index}"
    doc_id:       int
    chunk_index:  int
    text:         str
    start_offset: int       # word offset from document start
    end_offset:   int       # exclusive word offset
    word_count:   int


# ── Chunker base ───────────────────────────────────────────────────────────────

class Chunker:
    """Base class for all chunking strategies."""

    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()

    def chunk(self, text: str, doc_id: int) -> list[Chunk]:
        raise NotImplementedError

    @staticmethod
    def _words(text: str) -> list[str]:
        return text.split()

    @staticmethod
    def _join(words: list[str]) -> str:
        return " ".join(words)


# ── Fixed-size chunker ────────────────────────────────────────────────────────

class FixedSizeChunker(Chunker):
    """
    Non-overlapping fixed-size windows.

    Every word belongs to exactly one chunk.  Very simple and cache-friendly
    (deterministic offsets).
    """

    def chunk(self, text: str, doc_id: int) -> list[Chunk]:
        words     = self._words(text)
        size      = self.config.chunk_size
        min_words = self.config.min_chunk_words
        chunks: list[Chunk] = []
        idx = 0

        for start in range(0, len(words), size):
            end   = min(start + size, len(words))
            w_slice = words[start:end]
            if len(w_slice) < min_words:
                break
            chunk_text = self._join(w_slice)
            chunks.append(Chunk(
                chunk_id     = f"{doc_id}_{idx}",
                doc_id       = doc_id,
                chunk_index  = idx,
                text         = chunk_text,
                start_offset = start,
                end_offset   = end,
                word_count   = len(w_slice),
            ))
            idx += 1

        if not chunks and words:
            # Document is shorter than chunk_size — one chunk for the whole doc
            chunks.append(Chunk(
                chunk_id     = f"{doc_id}_0",
                doc_id       = doc_id,
                chunk_index  = 0,
                text         = text.strip(),
                start_offset = 0,
                end_offset   = len(words),
                word_count   = len(words),
            ))

        logger.debug("FixedSizeChunker: doc=%d → %d chunks", doc_id, len(chunks))
        return chunks


# ── Sliding-window chunker ────────────────────────────────────────────────────

class SlidingWindowChunker(Chunker):
    """
    Overlapping sliding-window chunks.

    stride = chunk_size - chunk_overlap
    Ensures every phrase of length ≤ chunk_size appears in at least one chunk.
    Recommended over fixed-size for most retrieval use cases.
    """

    def chunk(self, text: str, doc_id: int) -> list[Chunk]:
        words     = self._words(text)
        size      = self.config.chunk_size
        overlap   = min(self.config.chunk_overlap, size - 1)
        stride    = max(1, size - overlap)
        min_words = self.config.min_chunk_words
        chunks: list[Chunk] = []

        if len(words) <= size:
            # Short document → single chunk
            return [Chunk(
                chunk_id     = f"{doc_id}_0",
                doc_id       = doc_id,
                chunk_index  = 0,
                text         = text.strip(),
                start_offset = 0,
                end_offset   = len(words),
                word_count   = len(words),
            )]

        idx   = 0
        start = 0
        while start < len(words):
            end      = min(start + size, len(words))
            w_slice  = words[start:end]
            if len(w_slice) >= min_words:
                chunks.append(Chunk(
                    chunk_id     = f"{doc_id}_{idx}",
                    doc_id       = doc_id,
                    chunk_index  = idx,
                    text         = self._join(w_slice),
                    start_offset = start,
                    end_offset   = end,
                    word_count   = len(w_slice),
                ))
                idx += 1
            if end == len(words):
                break
            start += stride

        logger.debug("SlidingWindowChunker: doc=%d → %d chunks", doc_id, len(chunks))
        return chunks


# ── Factory ────────────────────────────────────────────────────────────────────

def make_chunker(config: ChunkingConfig | None = None) -> Chunker:
    """Return the appropriate Chunker for the configured strategy."""
    cfg = config or ChunkingConfig()
    if cfg.strategy == "fixed":
        return FixedSizeChunker(cfg)
    return SlidingWindowChunker(cfg)
