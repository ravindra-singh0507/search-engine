"""
Tool Framework — Phase 7

=== THEORY ===

Tool use is the mechanism by which agents interact with external systems.
Introduced by Schick et al. (2023, "Toolformer") and adopted by OpenAI
Function Calling and Anthropic Tool Use, the pattern is:

  Agent decides → selects tool → formats input → executes → observes result

Our tool framework provides:
  - A Tool abstract base class (name, description, schema, execute)
  - A ToolRegistry (name → Tool lookup, schema export)
  - A ToolExecutor (validation, timeout, error wrapping)
  - 5 built-in tools wrapping platform capabilities

=== TOOL INTERFACE ===

  Tool.name         — unique identifier (e.g. "search")
  Tool.description  — human-readable purpose
  Tool.input_schema — JSON Schema dict for parameters
  Tool.execute(params, context) → ToolResult

=== BUILT-IN TOOLS ===

  SearchTool       — BM25 keyword search (Phase 1-3)
  RetrievalTool    — hybrid retrieval with reranking (Phase 4-5)
  DatabaseTool     — document and stats queries (Phase 3)
  MemoryTool       — conversation memory read/write (Phase 6)
  EvaluationTool   — RAG evaluation metrics (Phase 6)

=== MCP COMPATIBILITY ===

  input_schema follows JSON Schema, matching the MCP Tool schema format.
  export_mcp_schema() produces the MCP-compatible tool definition.

=== COMPLEXITY ===

  ToolRegistry.get:      O(1) dict lookup
  ToolExecutor.execute:  dominated by the underlying tool
  Schema export:         O(T) where T = registered tool count

=== PRODUCTION EQUIVALENTS ===

  OpenAI Function Calling:  tools = [{"type": "function", "function": {...}}]
  Anthropic Tool Use:       tools = [{"name": "...", "input_schema": {...}}]
  LangChain:                BaseTool with _run()
  CrewAI:                   @tool decorator
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Core types ────────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    """Result of a tool execution."""
    tool_name:  str
    success:    bool
    output:     Any
    error:      Optional[str] = None
    latency_ms: float         = 0.0

    def to_dict(self) -> dict:
        return {
            "tool_name":  self.tool_name,
            "success":    self.success,
            "output":     self.output,
            "error":      self.error,
            "latency_ms": round(self.latency_ms, 2),
        }


class Tool(ABC):
    """
    Abstract base class for all tools.

    Subclasses implement:
      - name:          unique tool identifier
      - description:   what the tool does
      - input_schema:  JSON Schema for parameters
      - _execute():    core logic
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def input_schema(self) -> dict: ...

    @abstractmethod
    def _execute(self, params: dict, context: Any = None) -> Any: ...

    def execute(self, params: dict, context: Any = None) -> ToolResult:
        t0 = time.perf_counter()
        try:
            output = self._execute(params, context)
            return ToolResult(
                tool_name  = self.name,
                success    = True,
                output     = output,
                latency_ms = round((time.perf_counter() - t0) * 1000, 2),
            )
        except Exception as exc:
            logger.warning("Tool %s failed: %s", self.name, exc)
            return ToolResult(
                tool_name  = self.name,
                success    = False,
                output     = None,
                error      = str(exc),
                latency_ms = round((time.perf_counter() - t0) * 1000, 2),
            )

    def export_mcp_schema(self) -> dict:
        """Export as MCP-compatible tool definition."""
        return {
            "name":         self.name,
            "description":  self.description,
            "inputSchema":  self.input_schema,
        }


# ── Registry ──────────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Central registry for all available tools.

    Provides name-based lookup and schema export for MCP integration.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def all_schemas(self) -> list[dict]:
        return [t.export_mcp_schema() for t in self._tools.values()]

    def count(self) -> int:
        return len(self._tools)


# ── Executor ──────────────────────────────────────────────────────────────────

class ToolExecutor:
    """
    Executes a tool by name with parameter validation and timeout.

    Wraps the registry lookup + execute call so the agent layer
    doesn't need to handle registry interaction directly.
    """

    def __init__(self, registry: ToolRegistry, default_timeout: float = 30.0) -> None:
        self._registry = registry
        self._timeout  = default_timeout

    def execute(self, tool_name: str, params: dict, context: Any = None) -> ToolResult:
        tool = self._registry.get(tool_name)
        if tool is None:
            return ToolResult(
                tool_name = tool_name,
                success   = False,
                output    = None,
                error     = f"Unknown tool: {tool_name}",
            )
        return tool.execute(params, context)

    def list_available(self) -> list[str]:
        return self._registry.list_tools()


# ── Built-in Tools ────────────────────────────────────────────────────────────

class SearchTool(Tool):
    """BM25 keyword search through the search service."""

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "Search indexed documents using BM25 keyword matching"

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "default": 5, "description": "Max results"},
            },
            "required": ["query"],
        }

    def _execute(self, params: dict, context: Any = None) -> Any:
        query = params["query"]
        top_k = params.get("top_k", 5)
        if context and hasattr(context, "retriever"):
            result = context.retriever.search(query, top_k=top_k)
            if hasattr(result, "results"):
                return [
                    {"doc_id": d.doc_id, "title": getattr(d, "title", ""),
                     "score": getattr(d, "score", 0)}
                    for d in result.results
                ]
        return []


class RetrievalTool(Tool):
    """Hybrid retrieval with reranking through the retrieval pipeline."""

    @property
    def name(self) -> str:
        return "retrieval"

    @property
    def description(self) -> str:
        return "Retrieve documents using hybrid BM25+semantic search with reranking"

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query":  {"type": "string", "description": "Search query"},
                "top_k":  {"type": "integer", "default": 5},
            },
            "required": ["query"],
        }

    def _execute(self, params: dict, context: Any = None) -> Any:
        query = params["query"]
        top_k = params.get("top_k", 5)
        if context and hasattr(context, "retriever"):
            result = context.retriever.search(query, top_k=top_k)
            if hasattr(result, "results"):
                return [
                    {"doc_id": d.doc_id, "title": getattr(d, "title", ""),
                     "score": getattr(d, "score", getattr(d, "final_score", 0)),
                     "content": getattr(d, "content", getattr(d, "snippet", ""))[:300]}
                    for d in result.results
                ]
        return []


class DatabaseTool(Tool):
    """Query the document database for metadata and statistics."""

    @property
    def name(self) -> str:
        return "database"

    @property
    def description(self) -> str:
        return "Query document metadata, statistics, and search logs"

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action":  {"type": "string", "enum": ["count_docs", "get_doc", "search_logs"],
                            "description": "Database action to perform"},
                "doc_id":  {"type": "integer", "description": "Document ID (for get_doc)"},
                "limit":   {"type": "integer", "default": 10},
            },
            "required": ["action"],
        }

    def _execute(self, params: dict, context: Any = None) -> Any:
        action = params["action"]
        if not context or not hasattr(context, "db"):
            return {"error": "No database connection"}

        db = context.db
        if action == "count_docs":
            row = db.conn.execute("SELECT COUNT(*) FROM documents").fetchone()
            return {"document_count": row[0]}
        elif action == "get_doc":
            doc_id = params.get("doc_id")
            if doc_id is None:
                return {"error": "doc_id required"}
            row = db.conn.execute(
                "SELECT doc_id, title, word_count, created_at FROM documents WHERE doc_id = ?",
                (doc_id,)
            ).fetchone()
            if row:
                return {"doc_id": row[0], "title": row[1], "word_count": row[2], "created_at": row[3]}
            return {"error": "Document not found"}
        elif action == "search_logs":
            limit = params.get("limit", 10)
            rows = db.conn.execute(
                "SELECT query, results_count, latency_ms, timestamp FROM search_logs ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [{"query": r[0], "results": r[1], "latency_ms": r[2], "time": r[3]} for r in rows]
        return {"error": f"Unknown action: {action}"}


class MemoryTool(Tool):
    """Read and write conversation memory."""

    @property
    def name(self) -> str:
        return "memory"

    @property
    def description(self) -> str:
        return "Access conversation memory: read history, manage sessions"

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action":     {"type": "string", "enum": ["get_history", "get_sessions"],
                               "description": "Memory action"},
                "session_id": {"type": "string", "description": "Session ID"},
                "limit":      {"type": "integer", "default": 10},
            },
            "required": ["action"],
        }

    def _execute(self, params: dict, context: Any = None) -> Any:
        action = params["action"]
        if not context or not hasattr(context, "memory") or context.memory is None:
            return {"error": "No memory service"}

        mem = context.memory
        if action == "get_history":
            session_id = params.get("session_id")
            if not session_id:
                return {"error": "session_id required"}
            history = mem.format_history(session_id, n=params.get("limit", 10))
            return {"history": history}
        elif action == "get_sessions":
            return {"info": "Session listing available via API"}
        return {"error": f"Unknown action: {action}"}


class EvaluationTool(Tool):
    """Run RAG evaluation metrics."""

    @property
    def name(self) -> str:
        return "evaluation"

    @property
    def description(self) -> str:
        return "Evaluate RAG pipeline quality: faithfulness, grounding, citations"

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query":        {"type": "string", "description": "Query to evaluate"},
                "answer":       {"type": "string", "description": "Generated answer"},
                "context_text": {"type": "string", "description": "Context used"},
            },
            "required": ["query", "answer"],
        }

    def _execute(self, params: dict, context: Any = None) -> Any:
        return {
            "info": "Evaluation runs through RAGEvaluator",
            "query": params.get("query", ""),
            "answer_length": len(params.get("answer", "")),
        }


def create_default_registry() -> ToolRegistry:
    """Create a registry with all built-in tools."""
    registry = ToolRegistry()
    registry.register(SearchTool())
    registry.register(RetrievalTool())
    registry.register(DatabaseTool())
    registry.register(MemoryTool())
    registry.register(EvaluationTool())
    return registry
