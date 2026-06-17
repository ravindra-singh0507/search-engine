"""
Vector Store Factory

=== THEORY ===

The Factory pattern centralises object creation behind a single function,
decoupling the caller from the concrete implementation class.  The caller
requests "a VectorStore" and the factory decides whether to return a
QdrantVectorStore or FaissVectorStore based on configuration.

Selection logic:

  1. If EngineConfig.qdrant is configured and qdrant-client is installed,
     try to connect to Qdrant.  If the connection succeeds, return a
     QdrantVectorStore.

  2. If Qdrant is unavailable (library missing, server unreachable),
     fall back to FaissVectorStore.

This mirrors how production systems handle backend selection:
  - Elasticsearch clients try the configured cluster, fall back to local
  - Database ORMs support multiple backends (SQLite/PostgreSQL) via config

=== COMPLEXITY ===

  create_vector_store: O(1) — one connection attempt + object creation
"""

import logging

from app.config import EngineConfig
from app.vector_store.store import VectorStore, FaissVectorStore

logger = logging.getLogger(__name__)


def create_vector_store(config: EngineConfig) -> VectorStore:
    """
    Factory that returns QdrantVectorStore or FaissVectorStore based on config.

    Tries Qdrant first (if configured and importable), falls back to FAISS.

    Parameters
    ----------
    config : EngineConfig with .qdrant (QdrantConfig) and .vector_store
             (VectorStoreConfig) sub-configs.

    Returns
    -------
    A VectorStore protocol implementor — either QdrantVectorStore or
    FaissVectorStore.
    """
    # Attempt Qdrant if the config looks intentional (non-default host or
    # explicit collection name change)
    qdrant_cfg = config.qdrant
    try:
        from app.vector_store.qdrant import QdrantVectorStore

        store = QdrantVectorStore(config=qdrant_cfg)
        # Verify connectivity by triggering lazy connection + collection check
        store.create_collection()
        logger.info(
            "Vector store: Qdrant (%s:%d, collection=%s)",
            qdrant_cfg.host, qdrant_cfg.port, qdrant_cfg.collection_name,
        )
        return store

    except ImportError:
        logger.info(
            "qdrant-client not installed — falling back to FAISS"
        )
    except Exception as exc:
        logger.warning(
            "Qdrant unavailable (%s) — falling back to FAISS", exc,
        )

    # Fallback: FAISS
    store = FaissVectorStore(config=config.vector_store)
    logger.info("Vector store: FAISS (dim=%d)", config.vector_store.dimension)
    return store
