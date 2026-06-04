"""
Prompt Engineering System

=== THEORY ===

A prompt is the complete text sent to an LLM.  In a RAG system a prompt has
three parts:

  1. System prompt  — role and behaviour instructions (Claude / OpenAI)
  2. Context block  — retrieved passages, numbered [1] … [N]
  3. User query     — the question to answer

Well-designed prompts dramatically improve answer quality:
  - Explicit "only use the context" → reduces hallucination
  - "Cite [N]" instruction → enables citation attribution
  - Role framing → improves answer style and depth
  - Conversation history → enables multi-turn coherence

=== TEMPLATE VERSIONING ===

Each PromptTemplate carries a `version` field (semver string).  The registry
keeps the latest version per name.  When templates are persisted to disk for
evaluation, the version is recorded so we can compare prompt A vs prompt B.

=== TEMPLATE VARIABLES ===

All templates use Python str.format()-style placeholders:
  {context}    — assembled context from ContextBuilder
  {question}   — the user's question / query
  {history}    — formatted conversation history (empty string if first turn)

=== TEMPLATES ===

  qa              — General question answering
  research        — Deep research assistant
  summarization   — Summarize documents
  documentation   — Technical documentation assistant
  comparison      — Compare multiple items
  troubleshooting — Debug errors and problems

=== PROMPT EVALUATION ===

Templates can be evaluated offline by calling evaluate() on a set of
(question, context, ground_truth) triples.  This is integrated with the
RAGEvaluator in Phase 6.

=== PRODUCTION EQUIVALENTS ===

  LangChain:  PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
  LlamaIndex: PromptTemplate, SelectorPromptTemplate
  DSPy:       Signature, ChainOfThought, ReAct
  Guidance:   constraint-based templating
"""

import logging
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


# ── Core dataclass ────────────────────────────────────────────────────────────

@dataclass
class PromptTemplate:
    """
    A versioned prompt template.

    system     : system-level instructions passed as the 'system' argument to LLM.
    user_tmpl  : user-facing template with {context}, {question}, {history}.
    """
    name:      str
    version:   str
    system:    str
    user_tmpl: str
    tags:      list[str] = field(default_factory=list)

    def render(
        self,
        context:  str,
        question: str,
        history:  str = "",
    ) -> dict[str, str]:
        """
        Return {"system": ..., "user": ...} ready to pass to LLMProvider.generate().
        """
        user = self.user_tmpl.format(
            context=context,
            question=question,
            history=history,
        )
        return {"system": self.system, "user": user}

    def render_full(
        self,
        context:  str,
        question: str,
        history:  str = "",
    ) -> str:
        """
        Single-string prompt for providers that don't support a separate system turn.
        """
        rendered = self.render(context, question, history)
        if rendered["system"]:
            return f"System: {rendered['system']}\n\n{rendered['user']}"
        return rendered["user"]


# ── Built-in templates ────────────────────────────────────────────────────────

_QA_SYSTEM = """\
You are a precise and helpful knowledge assistant.
Your task is to answer the user's question STRICTLY based on the provided context.

Rules:
- If the answer is in the context, provide it clearly and concisely.
- Cite your sources using inline numbers like [1], [2] that correspond to the context blocks.
- If the context does not contain sufficient information, say "I don't have enough information to answer this question." Do NOT make up facts.
- Do not reference your training knowledge. Only use the provided context.
- Be concise. Prefer bullet points for lists."""

_QA_USER = """\
{history}### Context
{context}

### Question
{question}

### Answer (cite sources using [1], [2], etc.)"""


_RESEARCH_SYSTEM = """\
You are a senior research analyst.
Synthesize information from multiple sources to provide a comprehensive, well-structured answer.
Always cite the sources that support each claim using [1], [2], etc.
Identify agreements and contradictions across sources.
Acknowledge gaps or uncertainty when the context is incomplete."""

_RESEARCH_USER = """\
{history}### Retrieved Sources
{context}

### Research Question
{question}

### Analysis (cite all sources; note any contradictions or gaps)"""


_SUMMARIZATION_SYSTEM = """\
You are a document summarization expert.
Produce a clear, structured summary of the provided text.
Preserve key facts, figures, and conclusions.
Do not add information not present in the text."""

_SUMMARIZATION_USER = """\
### Documents to Summarize
{context}

### Summarization Task
{question}

### Summary"""


_DOCS_SYSTEM = """\
You are a technical documentation assistant.
Answer questions about APIs, libraries, tools, and frameworks using only the provided documentation.
Include code examples if they appear in the context.
Use precise technical language.
Cite the relevant documentation sections using [1], [2], etc."""

_DOCS_USER = """\
{history}### Documentation Context
{context}

### Developer Question
{question}

### Answer"""


_COMPARISON_SYSTEM = """\
You are an expert technical analyst.
Compare the items mentioned in the question using evidence from the provided context.
Structure your answer as a comparison table or bullet-point breakdown.
Be balanced and objective.
Cite sources using [1], [2], etc."""

_COMPARISON_USER = """\
{history}### Context
{context}

### Comparison Question
{question}

### Comparative Analysis"""


_TROUBLESHOOTING_SYSTEM = """\
You are a senior software engineer and debugging expert.
Diagnose the problem described and provide actionable solutions based on the provided context.
Structure your answer as:
1. Root cause (if identifiable)
2. Solution steps (numbered)
3. Prevention (if applicable)
Cite sources using [1], [2], etc."""

_TROUBLESHOOTING_USER = """\
{history}### Relevant Documentation / Error Reports
{context}

### Problem Description
{question}

### Diagnosis and Solution"""


# ── Registry ──────────────────────────────────────────────────────────────────

class PromptRegistry:
    """
    Central store for all prompt templates.

    Templates are keyed by name.  Multiple versions of the same template can
    coexist; `get()` returns the latest (highest semver) unless a version is
    specified.

    Usage
    -----
    registry = PromptRegistry()
    rendered = registry.render("qa", context="...", question="...")
    """

    def __init__(self) -> None:
        self._store: dict[str, list[PromptTemplate]] = {}
        self._load_defaults()

    # ── Public API ────────────────────────────────────────────────────────

    def register(self, template: PromptTemplate) -> None:
        """Register a template.  Replaces older version with same name+version."""
        name = template.name
        if name not in self._store:
            self._store[name] = []
        # Remove old entry with same version
        self._store[name] = [
            t for t in self._store[name] if t.version != template.version
        ]
        self._store[name].append(template)
        self._store[name].sort(key=lambda t: t.version)
        logger.debug("Registered prompt template: %s v%s", name, template.version)

    def get(self, name: str, version: str | None = None) -> PromptTemplate:
        """
        Retrieve a template by name (and optionally version).
        Raises KeyError if the template or version is not found.
        """
        versions = self._store.get(name)
        if not versions:
            raise KeyError(
                f"Prompt template {name!r} not found. "
                f"Available: {sorted(self._store.keys())}"
            )
        if version:
            for t in versions:
                if t.version == version:
                    return t
            raise KeyError(f"Template {name!r} version {version!r} not found.")
        return versions[-1]   # latest version

    def render(
        self,
        name:     str,
        context:  str,
        question: str,
        history:  str = "",
        version:  str | None = None,
    ) -> dict[str, str]:
        """Convenience: get + render in one call."""
        return self.get(name, version).render(context, question, history)

    def list_templates(self) -> list[dict]:
        """Return metadata for all registered templates."""
        result = []
        for name, versions in self._store.items():
            latest = versions[-1]
            result.append({
                "name":    name,
                "version": latest.version,
                "tags":    latest.tags,
                "versions": [t.version for t in versions],
            })
        return result

    # ── Defaults ──────────────────────────────────────────────────────────

    def _load_defaults(self) -> None:
        defaults = [
            PromptTemplate("qa",              "1.0", _QA_SYSTEM,              _QA_USER,              ["general", "default"]),
            PromptTemplate("research",        "1.0", _RESEARCH_SYSTEM,        _RESEARCH_USER,        ["research", "analysis"]),
            PromptTemplate("summarization",   "1.0", _SUMMARIZATION_SYSTEM,   _SUMMARIZATION_USER,   ["summary"]),
            PromptTemplate("documentation",   "1.0", _DOCS_SYSTEM,            _DOCS_USER,            ["docs", "api"]),
            PromptTemplate("comparison",      "1.0", _COMPARISON_SYSTEM,      _COMPARISON_USER,      ["compare", "analysis"]),
            PromptTemplate("troubleshooting", "1.0", _TROUBLESHOOTING_SYSTEM, _TROUBLESHOOTING_USER, ["debug", "error"]),
        ]
        for t in defaults:
            self.register(t)


# ── Module-level singleton ────────────────────────────────────────────────────

_default_registry: PromptRegistry | None = None


def get_registry() -> PromptRegistry:
    """Return the module-level singleton registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = PromptRegistry()
    return _default_registry
