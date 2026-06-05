"""
MCP-Compatible Architecture — Phase 7

=== THEORY ===

The Model Context Protocol (MCP) is an open standard proposed by
Anthropic for connecting AI agents to external tools and data sources.

MCP defines:
  - Tool definitions with JSON Schema input/output
  - Context exchange (providing context from external systems)
  - Resource access (read-only data from external systems)

Our MCP layer wraps the Tool Framework (Feature 11) in MCP-compatible
schemas, enabling future direct integration with MCP-compatible clients
(Claude Desktop, Cursor, etc.) without code changes.

=== ARCHITECTURE ===

  MCPToolDefinition
    Wraps a Tool instance with MCP-standard schema fields:
      name, description, inputSchema

  MCPRegistry
    Central registry that mirrors ToolRegistry but exports MCP format.
    Provides:
      list_tools()         → MCP tool list response
      call_tool(name, args) → MCP tool call response

  MCPContext
    Represents a context exchange payload:
      role, content, tool_results

=== COMPLIANCE ===

  Our schema format matches the MCP specification:
    {
      "name": "search",
      "description": "Search indexed documents using BM25",
      "inputSchema": {
        "type": "object",
        "properties": { ... },
        "required": [ ... ]
      }
    }

  The call_tool response matches:
    {
      "content": [{"type": "text", "text": "..."}],
      "isError": false
    }

=== COMPLEXITY ===

  list_tools:  O(T) where T = registered tools
  call_tool:   O(1) lookup + tool execution time

=== PRODUCTION EQUIVALENTS ===

  Anthropic MCP SDK:  Python/TypeScript server implementations
  LangChain MCP:      MCP adapter for LangChain tools
  OpenAI Plugins:     predecessor pattern (deprecated)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from app.tools.framework import Tool, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class MCPToolDefinition:
    """MCP-standard tool definition."""
    name:         str
    description:  str
    input_schema: dict

    def to_dict(self) -> dict:
        return {
            "name":        self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass
class MCPContext:
    """MCP context exchange payload."""
    role:         str = "tool"
    content:      str = ""
    tool_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "role":         self.role,
            "content":      self.content,
            "tool_results": self.tool_results,
        }


class MCPRegistry:
    """
    MCP-compatible tool registry.

    Wraps the internal ToolRegistry and provides MCP-standard
    list_tools / call_tool / get_schema interfaces.

    Usage:
        mcp = MCPRegistry(tool_registry)
        tools = mcp.list_tools()          # MCP tool list
        result = mcp.call_tool("search", {"query": "python"}, context)
    """

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self._registry = tool_registry

    def list_tools(self) -> list[dict]:
        """Return MCP-format tool definitions."""
        return self._registry.all_schemas()

    def get_tool_schema(self, name: str) -> Optional[dict]:
        """Return MCP schema for a single tool."""
        tool = self._registry.get(name)
        if tool is None:
            return None
        return tool.export_mcp_schema()

    def call_tool(
        self, name: str, arguments: dict, context: Any = None,
    ) -> dict:
        """
        Execute a tool in MCP response format.

        Returns:
          {
            "content": [{"type": "text", "text": "..."}],
            "isError": false
          }
        """
        tool = self._registry.get(name)
        if tool is None:
            return {
                "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
                "isError": True,
            }

        result: ToolResult = tool.execute(arguments, context)

        if result.success:
            text = json.dumps(result.output, default=str) if result.output is not None else ""
            return {
                "content": [{"type": "text", "text": text}],
                "isError": False,
            }
        else:
            return {
                "content": [{"type": "text", "text": result.error or "Unknown error"}],
                "isError": True,
            }

    def tool_count(self) -> int:
        return self._registry.count()
