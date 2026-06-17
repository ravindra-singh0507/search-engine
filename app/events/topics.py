"""
Event Topics — Phase 8

=== THEORY ===

Well-known topic constants define the canonical event vocabulary of the
search engine platform.  Every event published through the EventBus
carries a topic string that identifies *what happened*.

Using module-level constants instead of raw strings provides:
  1. Autocomplete and static-analysis support
  2. Single source of truth (rename once, fix everywhere)
  3. A discoverable catalogue of all system events

=== NAMING CONVENTION ===

  <domain>.<verb_past_tense>

  domain  — bounded context (document, crawl, search, agent, ...)
  verb    — past-tense action (indexed, started, completed, ...)

This mirrors the CloudEvents `type` attribute convention and aligns
with Apache Kafka topic naming best practices.

=== ARCHITECTURE ===

Topics are grouped by domain:

  Document lifecycle   — index / update / delete
  Crawler              — started / page_fetched / completed
  Search               — executed / click
  Embeddings           — started / completed
  Agent                — task created / completed / failed
  Workflow             — started / completed
  Research             — completed
  Evaluation           — completed
  RAG                  — query completed
  Memory               — session created

=== PRODUCTION EQUIVALENTS ===

  Apache Kafka:       topic names are first-class resources
  AWS EventBridge:    detail-type strings
  Google Pub/Sub:     topic resource names
  CloudEvents spec:   `type` attribute (reverse-DNS recommended)
"""

# ── Document lifecycle ───────────────────────────────────────────────────────

DOCUMENT_INDEXED = "document.indexed"
DOCUMENT_DELETED = "document.deleted"
DOCUMENT_UPDATED = "document.updated"

# ── Crawler ──────────────────────────────────────────────────────────────────

CRAWL_STARTED      = "crawl.started"
CRAWL_PAGE_FETCHED = "crawl.page_fetched"
CRAWL_COMPLETED    = "crawl.completed"

# ── Search ───────────────────────────────────────────────────────────────────

SEARCH_EXECUTED = "search.executed"
SEARCH_CLICK    = "search.click"

# ── Embeddings ───────────────────────────────────────────────────────────────

EMBEDDING_STARTED   = "embedding.started"
EMBEDDING_COMPLETED = "embedding.completed"

# ── Chunking ────────────────────────────────────────────────────────────────

CHUNKING_STARTED   = "chunking.started"
CHUNKING_COMPLETED = "chunking.completed"

# ── Agent ────────────────────────────────────────────────────────────────────

AGENT_TASK_CREATED   = "agent.task.created"
AGENT_TASK_COMPLETED = "agent.task.completed"
AGENT_TASK_FAILED    = "agent.task.failed"

# ── Workflow ─────────────────────────────────────────────────────────────────

WORKFLOW_STARTED   = "workflow.started"
WORKFLOW_COMPLETED = "workflow.completed"

# ── Research ─────────────────────────────────────────────────────────────────

RESEARCH_COMPLETED = "research.completed"

# ── Evaluation ───────────────────────────────────────────────────────────────

EVALUATION_COMPLETED = "evaluation.completed"

# ── RAG ──────────────────────────────────────────────────────────────────────

RAG_QUERY_COMPLETED = "rag.query.completed"

# ── Memory ───────────────────────────────────────────────────────────────────

MEMORY_SESSION_CREATED = "memory.session.created"
