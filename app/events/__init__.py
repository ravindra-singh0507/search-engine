"""Phase 8 Event-Driven Architecture."""
from app.events.models import Event, EventMetadata, EventEnvelope, EventStatus
from app.events.bus import EventBus, InMemoryEventBus
from app.events.producer import EventProducer
from app.events.consumer import EventConsumer
from app.events.router import EventRouter
from app.events.store import EventStore, InMemoryEventStore
from app.events.retry import EventRetryPolicy, DeadLetterQueue
from app.events.topics import (
    DOCUMENT_INDEXED, DOCUMENT_DELETED, DOCUMENT_UPDATED,
    CRAWL_STARTED, CRAWL_PAGE_FETCHED, CRAWL_COMPLETED,
    SEARCH_EXECUTED, SEARCH_CLICK,
    EMBEDDING_STARTED, EMBEDDING_COMPLETED,
    CHUNKING_STARTED, CHUNKING_COMPLETED,
    AGENT_TASK_CREATED, AGENT_TASK_COMPLETED, AGENT_TASK_FAILED,
    WORKFLOW_STARTED, WORKFLOW_COMPLETED,
    RESEARCH_COMPLETED, EVALUATION_COMPLETED,
    RAG_QUERY_COMPLETED, MEMORY_SESSION_CREATED,
)

__all__ = [
    # Models
    "Event", "EventMetadata", "EventEnvelope", "EventStatus",
    # Bus
    "EventBus", "InMemoryEventBus",
    # Producer / Consumer
    "EventProducer", "EventConsumer",
    # Router
    "EventRouter",
    # Store
    "EventStore", "InMemoryEventStore",
    # Retry / DLQ
    "EventRetryPolicy", "DeadLetterQueue",
    # Topics
    "DOCUMENT_INDEXED", "DOCUMENT_DELETED", "DOCUMENT_UPDATED",
    "CRAWL_STARTED", "CRAWL_PAGE_FETCHED", "CRAWL_COMPLETED",
    "SEARCH_EXECUTED", "SEARCH_CLICK",
    "EMBEDDING_STARTED", "EMBEDDING_COMPLETED",
    "CHUNKING_STARTED", "CHUNKING_COMPLETED",
    "AGENT_TASK_CREATED", "AGENT_TASK_COMPLETED", "AGENT_TASK_FAILED",
    "WORKFLOW_STARTED", "WORKFLOW_COMPLETED",
    "RESEARCH_COMPLETED", "EVALUATION_COMPLETED",
    "RAG_QUERY_COMPLETED", "MEMORY_SESSION_CREATED",
]
