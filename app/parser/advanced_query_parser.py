"""
Advanced Query Parser — Phase 3

Supports:
  1. Parentheses grouping   — (python OR java) AND backend
  2. Phrase search          — "machine learning"  [POSITIONAL — fixed]
  3. Field search           — title:python  url:github  site:example.com
  4. Wildcard / prefix      — py*  (expands against known vocabulary)
  5. Complex nesting        — (title:python OR title:java) AND "web framework"

=== THREAD-SAFETY FIX ===

The original implementation stored _tokens and _pos as instance variables,
causing data corruption when multiple threads called parse() concurrently.

Fix: all mutable parse state lives in a _ParseState dataclass that is
created fresh on every parse() call.  The parser instance itself is now
fully stateless and safe for shared use across threads.

=== PHRASE SEARCH FIX ===

PhraseNode now evaluates with real positional intersection instead of
treating it as a plain AND.  "machine learning" will only match documents
where "machine" appears immediately before "learning" in the token stream.
Position values include stop-word gaps (the tokenizer increments the
position counter even for filtered tokens) so proximity is correctly
measured against absolute token offsets.

=== WILDCARD CACHE FIX ===

ASTEvaluator.invalidate_vocab_cache() must be called after every indexing
or deletion operation so new terms are visible to subsequent wildcard queries.
"""

import re
import logging
from dataclasses import dataclass, field as dc_field
from enum import Enum, auto
from typing import Optional, Union

from app.database.db import Database
from app.tokenizer.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

# ── Token types ───────────────────────────────────────────────────────────────


class TokType(Enum):
    TERM     = auto()
    PHRASE   = auto()
    FIELD    = auto()
    WILDCARD = auto()
    AND      = auto()
    OR       = auto()
    NOT      = auto()
    LPAREN   = auto()
    RPAREN   = auto()
    EOF      = auto()


@dataclass
class Token:
    type: TokType
    value: str = ""


# ── AST Node types ────────────────────────────────────────────────────────────


@dataclass
class TermNode:
    term: str


@dataclass
class PhraseNode:
    terms: list[str]


@dataclass
class FieldNode:
    field: str
    term: str


@dataclass
class WildcardNode:
    prefix: str


@dataclass
class AndNode:
    left: "ASTNode"
    right: "ASTNode"


@dataclass
class OrNode:
    left: "ASTNode"
    right: "ASTNode"


@dataclass
class NotNode:
    operand: "ASTNode"


ASTNode = Union[
    TermNode, PhraseNode, FieldNode, WildcardNode,
    AndNode, OrNode, NotNode,
]

KNOWN_FIELDS = frozenset({"title", "body", "url", "site", "source"})


# ── Parse state (one per parse() call — makes the parser thread-safe) ────────


@dataclass
class _ParseState:
    """
    All mutable state for a single parse invocation.
    Creating a new instance per call eliminates shared mutable state and
    makes AdvancedQueryParser safe for concurrent use.
    """
    tokens: list[Token]
    pos: int = 0

    def current(self) -> Token:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else Token(TokType.EOF)

    def advance(self) -> None:
        self.pos += 1


# ── Lexer ─────────────────────────────────────────────────────────────────────


class Lexer:
    _FIELD_RE    = re.compile(r"^(\w+):(.*)", re.IGNORECASE)
    _WILDCARD_RE = re.compile(r"^(\w+)\*$")

    def tokenise(self, query: str) -> list[Token]:
        tokens: list[Token] = []
        i = 0
        chars = query.strip()

        while i < len(chars):
            ch = chars[i]

            if ch.isspace():
                i += 1
                continue

            if ch == '"':
                j = chars.find('"', i + 1)
                if j == -1:
                    j = len(chars)
                tokens.append(Token(TokType.PHRASE, chars[i + 1: j]))
                i = j + 1
                continue

            if ch == '(':
                tokens.append(Token(TokType.LPAREN, "("))
                i += 1
                continue

            if ch == ')':
                tokens.append(Token(TokType.RPAREN, ")"))
                i += 1
                continue

            j = i
            while j < len(chars) and chars[j] not in (' ', '\t', '\n', '(', ')'):
                j += 1
            word = chars[i:j]
            i = j

            upper = word.upper()
            if upper == "AND":
                tokens.append(Token(TokType.AND, "AND"))
            elif upper == "OR":
                tokens.append(Token(TokType.OR, "OR"))
            elif upper == "NOT":
                tokens.append(Token(TokType.NOT, "NOT"))
            else:
                m_field = self._FIELD_RE.match(word)
                m_wild  = self._WILDCARD_RE.match(word)
                if m_field and m_field.group(1).lower() in KNOWN_FIELDS:
                    tokens.append(Token(TokType.FIELD, word))
                elif m_wild:
                    tokens.append(Token(TokType.WILDCARD, word[:-1]))
                else:
                    tokens.append(Token(TokType.TERM, word))

        tokens.append(Token(TokType.EOF))
        return tokens


# ── Parser ────────────────────────────────────────────────────────────────────


class AdvancedQueryParser:
    """
    Recursive-descent parser.  Thread-safe: all state is in _ParseState
    which is created fresh for each parse() call.

    Grammar (AND binds tighter than OR — correct precedence):
      query    := expr EOF
      expr     := or_expr
      or_expr  := and_expr ( OR and_expr )*
      and_expr := not_expr ( AND? not_expr )*
      not_expr := NOT? primary
      primary  := '(' expr ')' | phrase | field | wildcard | term
    """

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self._lexer    = Lexer()

    def parse(self, query: str) -> Optional[ASTNode]:
        if not query or not query.strip():
            return None
        state = _ParseState(tokens=self._lexer.tokenise(query))
        return self._parse_expr(state)

    # ── Grammar rules (all receive _ParseState) ───────────────────────────

    def _parse_expr(self, state: _ParseState) -> Optional[ASTNode]:
        return self._parse_or(state)

    def _parse_or(self, state: _ParseState) -> Optional[ASTNode]:
        left = self._parse_and(state)
        while state.current().type == TokType.OR:
            state.advance()
            right = self._parse_and(state)
            if left is None:
                left = right
            elif right is not None:
                left = OrNode(left, right)
        return left

    def _parse_and(self, state: _ParseState) -> Optional[ASTNode]:
        left = self._parse_not(state)
        while True:
            tok = state.current()
            if tok.type == TokType.AND:
                state.advance()
                right = self._parse_not(state)
                if left is None:
                    left = right
                elif right is not None:
                    left = AndNode(left, right)
            elif tok.type not in (TokType.OR, TokType.RPAREN, TokType.EOF):
                right = self._parse_not(state)
                if right is None:
                    break
                left = AndNode(left, right) if left else right
            else:
                break
        return left

    def _parse_not(self, state: _ParseState) -> Optional[ASTNode]:
        if state.current().type == TokType.NOT:
            state.advance()
            operand = self._parse_primary(state)
            if operand is None:
                return None
            return NotNode(operand)
        return self._parse_primary(state)

    def _parse_primary(self, state: _ParseState) -> Optional[ASTNode]:
        tok = state.current()

        if tok.type == TokType.LPAREN:
            state.advance()
            inner = self._parse_expr(state)
            if state.current().type == TokType.RPAREN:
                state.advance()
            return inner

        if tok.type == TokType.PHRASE:
            state.advance()
            terms = self.tokenizer.tokenize(tok.value).tokens
            return PhraseNode(terms) if terms else None

        if tok.type == TokType.FIELD:
            state.advance()
            parts      = tok.value.split(":", 1)
            field_name = parts[0].lower()
            raw_term   = parts[1] if len(parts) > 1 else ""
            tok_terms  = self.tokenizer.tokenize(raw_term).tokens
            term       = tok_terms[0] if tok_terms else raw_term.lower()
            return FieldNode(field=field_name, term=term)

        if tok.type == TokType.WILDCARD:
            state.advance()
            return WildcardNode(prefix=tok.value.lower())

        if tok.type == TokType.TERM:
            state.advance()
            tok_terms = self.tokenizer.tokenize(tok.value).tokens
            if not tok_terms:
                return None
            return TermNode(term=tok_terms[0])

        return None


# ── AST Evaluator ─────────────────────────────────────────────────────────────


class ASTEvaluator:
    """
    Walk the AST and return a set of matching document IDs.

    Changes from Phase 3.0:
      - PhraseNode uses real positional intersection (BUG-005 fix).
      - Wildcard vocab cache is exposed via invalidate_vocab_cache() so
        it is reset after each indexing/deletion (BUG wildcard stale-cache).
    """

    def __init__(self, db: Database):
        self.db = db
        self._vocab_cache: set[str] | None = None

    def evaluate(self, node: Optional[ASTNode]) -> set[int]:
        if node is None:
            return set()
        return self._eval(node)

    def invalidate_vocab_cache(self) -> None:
        """Call after indexing or deleting documents."""
        self._vocab_cache = None

    def _eval(self, node: ASTNode) -> set[int]:
        if isinstance(node, TermNode):
            return self._posting_ids(node.term)

        if isinstance(node, PhraseNode):
            # Real positional phrase search (BUG-005 fix)
            return self._eval_phrase(node.terms)

        if isinstance(node, FieldNode):
            postings = self.db.get_postings_for_term(node.term, field=node.field)
            return {p.doc_id for p in postings}

        if isinstance(node, WildcardNode):
            return self._expand_wildcard(node.prefix)

        if isinstance(node, AndNode):
            left = self._eval(node.left)
            if not left:              # short-circuit: AND with empty = empty
                return set()
            return left & self._eval(node.right)

        if isinstance(node, OrNode):
            return self._eval(node.left) | self._eval(node.right)

        if isinstance(node, NotNode):
            all_ids = {d.doc_id for d in self.db.get_all_documents()}
            return all_ids - self._eval(node.operand)

        return set()

    # ── Phrase search ─────────────────────────────────────────────────────

    def _eval_phrase(self, terms: list[str]) -> set[int]:
        """
        Positional phrase search.

        For a phrase ["machine", "learning"] we require that in some
        candidate document d, "learning" appears at position p+1 wherever
        "machine" appears at position p.

        Position values in the posting list are absolute token offsets from
        the start of the document, incremented for every token including
        stop words.  This means stop words inside a phrase correctly prevent
        a match (e.g. "machine to learning" ≠ "machine learning").

        Complexity: O(Σ|posting lists|) — dominated by the fetch step.
        """
        if not terms:
            return set()
        if len(terms) == 1:
            return self._posting_ids(terms[0])

        # ── Fetch postings with positions for every term ──────────────────
        term_doc_positions: dict[str, dict[int, list[int]]] = {}
        for term in terms:
            postings = self.db.get_postings_for_term(term)
            doc_pos: dict[int, list[int]] = {}
            for p in postings:
                if p.positions:
                    doc_pos[p.doc_id] = p.positions
            term_doc_positions[term] = doc_pos

        # ── Candidates must contain ALL terms ─────────────────────────────
        candidate_docs: set[int] | None = None
        for term in terms:
            doc_ids = set(term_doc_positions[term].keys())
            candidate_docs = doc_ids if candidate_docs is None else candidate_docs & doc_ids

        if not candidate_docs:
            return set()

        # ── For each candidate verify positional adjacency ────────────────
        result: set[int] = set()
        for doc_id in candidate_docs:
            if self._positions_match(terms, term_doc_positions, doc_id):
                result.add(doc_id)
        return result

    def _positions_match(
        self,
        terms: list[str],
        term_doc_positions: dict[str, dict[int, list[int]]],
        doc_id: int,
    ) -> bool:
        """
        Check that terms appear at consecutive positions in doc_id.

        Uses a forward set-intersection: start with possible anchor
        positions for terms[0], then for each subsequent term keep only
        anchors where the next term appears at anchor+offset.
        """
        anchor_positions = set(term_doc_positions[terms[0]].get(doc_id, []))
        for offset, term in enumerate(terms[1:], start=1):
            next_pos = set(term_doc_positions[term].get(doc_id, []))
            # Keep anchors for which terms[offset] is at anchor + offset
            anchor_positions = {
                p for p in anchor_positions if (p + offset) in next_pos
            }
            if not anchor_positions:
                return False
        return True

    # ── Helpers ───────────────────────────────────────────────────────────

    def _posting_ids(self, term: str) -> set[int]:
        return {p.doc_id for p in self.db.get_postings_for_term(term)}

    def _expand_wildcard(self, prefix: str) -> set[int]:
        """Union posting lists for all vocabulary terms that start with prefix."""
        if self._vocab_cache is None:
            self._vocab_cache = set(self.db.get_all_terms())

        result: set[int] = set()
        for term in self._vocab_cache:
            if term.startswith(prefix):
                result |= self._posting_ids(term)
        return result
