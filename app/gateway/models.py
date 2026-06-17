"""
Gateway Response / Request Models

=== THEORY ===

Data Transfer Objects (DTOs) decouple the gateway's internal processing
from its external API contract.  By using dataclasses rather than raw
dicts, we get:

  - Type safety at development time (IDE autocompletion, mypy)
  - Documented field names and defaults
  - Immutable-ish contracts that are easy to version

GatewayRequest captures everything the caller can specify for a single
retrieval request.  GatewayResponse carries the results plus metadata
that clients use for debugging, caching decisions, and latency tracking.

=== PRODUCTION EQUIVALENTS ===

  Google:     SearchRequest / SearchResponse protobufs
  Elastic:    SearchRequest / SearchResponse JSON schemas
  Vespa:      Query + Result objects
"""

from dataclasses import dataclass, field


@dataclass
class GatewayRequest:
    """
    Inbound retrieval request to the gateway.

    Fields
    ------
    query       : raw search query string
    mode        : retrieval mode — "bm25", "semantic", "hybrid", "pipeline"
    top_k       : number of results to return
    fusion      : fusion strategy name (rrf, combsum, combmnz, weighted, borda)
    rerank      : whether to apply cross-encoder reranking
    client_id   : caller identifier for rate limiting
    timeout_sec : per-request timeout (overrides gateway default)
    """
    query:       str
    mode:        str   = "hybrid"
    top_k:       int   = 10
    fusion:      str   = "rrf"
    rerank:      bool  = True
    client_id:   str   = ""
    timeout_sec: float = 30.0


@dataclass
class GatewayResponse:
    """
    Outbound retrieval response from the gateway.

    Fields
    ------
    query          : echoed query for correlation
    mode           : retrieval mode that was used
    results        : list of result dicts (doc_id, title, snippet, score, ...)
    total_results  : number of results returned
    latency_ms     : end-to-end gateway latency in milliseconds
    cache_hit      : True if the response was served from cache
    fusion_strategy: which fusion was applied (empty if N/A)
    reranked       : whether reranking was applied
    metadata       : extensible metadata dict (gateway stats, debug info)
    """
    query:            str
    mode:             str
    results:          list[dict]
    total_results:    int
    latency_ms:       float
    cache_hit:        bool  = False
    fusion_strategy:  str   = ""
    reranked:         bool  = False
    metadata:         dict  = field(default_factory=dict)
