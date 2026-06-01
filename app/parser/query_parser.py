"""
Query Parser

=== THEORY ===

A query parser translates a user's text query into a structured representation
that the search engine can execute. It's the bridge between human intent and
machine operations.

=== BOOLEAN RETRIEVAL MODEL ===

The simplest retrieval model treats queries as Boolean expressions:
- AND: intersection of posting lists (both terms must appear)
- OR:  union of posting lists (either term can appear)
- NOT: set difference (exclude documents containing a term)

Example: "python AND backend NOT java"
→ (docs with "python") ∩ (docs with "backend") - (docs with "java")

WHY Boolean retrieval?
- It's exact: documents either match or they don't
- It's fast: set operations on sorted posting lists are O(n + m)
- It gives users precise control over results

LIMITATIONS:
- No ranking within results (a document either matches or doesn't)
- Users must know Boolean logic
- Hard to express "find documents mostly about X"
→ That's why we add TF-IDF ranking on top (see ranking module)

=== QUERY TYPES ===

1. Simple query: "python" → search for single term
2. Multi-term (implicit AND): "python backend" → python AND backend
3. Explicit Boolean: "python AND backend OR java"
4. Negation: "python NOT java"

=== COMPLEXITY ===

- Parse query: O(Q) where Q = query length
- AND (intersection): O(min(|A|, |B|)) with sorted lists
- OR (union): O(|A| + |B|)
- NOT (difference): O(|A| + |B|)

=== AT GOOGLE SCALE ===

Google's query processing:
- Parses natural language queries, not Boolean expressions
- Uses query expansion (synonyms, spelling correction)
- Applies personalization based on user history
- Rewrites queries internally for better recall
- Processes queries across thousands of index shards in parallel
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto

from app.database.db import Database
from app.tokenizer.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class Operator(Enum):
    AND = auto()
    OR = auto()
    NOT = auto()


@dataclass
class QueryToken:
    term: str | None = None
    operator: Operator | None = None

    @property
    def is_operator(self) -> bool:
        return self.operator is not None


@dataclass
class ParsedQuery:
    raw_query: str
    tokens: list[QueryToken]
    terms: list[str]

    @property
    def is_boolean(self) -> bool:
        return any(t.is_operator for t in self.tokens)


class QueryParser:
    """
    Parses search queries into structured representations.

    Supports:
    - Simple queries: "python"
    - Multi-term (implicit AND): "python backend"
    - Boolean: "python AND backend", "python OR java", "python NOT java"
    - Mixed: "python AND backend NOT java"
    """

    OPERATORS = {"AND": Operator.AND, "OR": Operator.OR, "NOT": Operator.NOT}

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def parse(self, query: str) -> ParsedQuery:
        if not query or not query.strip():
            return ParsedQuery(raw_query=query, tokens=[], terms=[])

        raw_words = query.strip().split()
        tokens: list[QueryToken] = []
        terms: list[str] = []

        for word in raw_words:
            upper = word.upper()
            if upper in self.OPERATORS:
                tokens.append(QueryToken(operator=self.OPERATORS[upper]))
            else:
                result = self.tokenizer.tokenize(word)
                for term in result.tokens:
                    tokens.append(QueryToken(term=term))
                    terms.append(term)

        logger.debug("Parsed query '%s' → %d tokens, %d terms", query, len(tokens), len(terms))
        return ParsedQuery(raw_query=query, tokens=tokens, terms=terms)


class BooleanRetriever:
    """
    Executes Boolean queries against the inverted index.

    Takes a parsed query and returns a set of matching document IDs
    by performing set operations on posting lists.
    """

    def __init__(self, db: Database):
        self.db = db

    def retrieve(self, parsed_query: ParsedQuery) -> set[int]:
        if not parsed_query.tokens:
            return set()

        if not parsed_query.is_boolean:
            return self._implicit_and(parsed_query.terms)

        return self._execute_boolean(parsed_query.tokens)

    def _get_doc_ids_for_term(self, term: str) -> set[int]:
        postings = self.db.get_postings_for_term(term)
        return {p.doc_id for p in postings}

    def _implicit_and(self, terms: list[str]) -> set[int]:
        """Multiple terms without operators → AND them together."""
        if not terms:
            return set()

        result = self._get_doc_ids_for_term(terms[0])
        for term in terms[1:]:
            result &= self._get_doc_ids_for_term(term)
        return result

    def _execute_boolean(self, tokens: list[QueryToken]) -> set[int]:
        """
        Execute a Boolean query left-to-right.

        Algorithm:
        1. Start with the first term's posting list
        2. Walk through tokens:
           - If we see AND, intersect with the next term
           - If we see OR, union with the next term
           - If we see NOT, subtract the next term
        3. If two terms appear with no operator between them, default to AND
        """
        result: set[int] | None = None
        pending_op: Operator | None = None

        for token in tokens:
            if token.is_operator:
                pending_op = token.operator
                continue

            if token.term is None:
                continue

            term_docs = self._get_doc_ids_for_term(token.term)

            if result is None:
                if pending_op == Operator.NOT:
                    all_docs = self._get_all_doc_ids()
                    result = all_docs - term_docs
                else:
                    result = term_docs
                pending_op = None
                continue

            op = pending_op or Operator.AND
            pending_op = None

            if op == Operator.AND:
                result &= term_docs
            elif op == Operator.OR:
                result |= term_docs
            elif op == Operator.NOT:
                result -= term_docs

        return result or set()

    def _get_all_doc_ids(self) -> set[int]:
        docs = self.db.get_all_documents()
        return {d.doc_id for d in docs}
