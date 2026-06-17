"""
Retrieval Gateway Package

=== THEORY ===

The gateway pattern centralises cross-cutting concerns (caching, rate
limiting, routing, metrics) into a single entry point.  This decouples
clients from backend service topology and provides a uniform API
regardless of which retrieval backend handles the request.

Exports:
  RetrievalGateway  — main gateway service
  QueryRouter       — intent-based query routing
  GatewayCache      — two-tier (L1 + L2) result cache
  GatewayRequest    — inbound request DTO
  GatewayResponse   — outbound response DTO
"""

from app.gateway.service import RetrievalGateway
from app.gateway.router import QueryRouter
from app.gateway.cache import GatewayCache
from app.gateway.models import GatewayRequest, GatewayResponse

__all__ = [
    "RetrievalGateway",
    "QueryRouter",
    "GatewayCache",
    "GatewayRequest",
    "GatewayResponse",
]
