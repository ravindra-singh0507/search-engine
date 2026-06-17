"""
Event Router — Phase 8

=== THEORY ===

The Event Router implements content-based routing (Hohpe & Woolf,
*Enterprise Integration Patterns*, 2003, Chapter 7).

Unlike direct topic subscription (exact match only), the router
supports *wildcard patterns* using fnmatch-style globbing:

  "document.*"       — matches document.indexed, document.deleted, ...
  "*.completed"      — matches crawl.completed, workflow.completed, ...
  "agent.task.*"     — matches agent.task.created, agent.task.failed, ...
  "*"                — matches everything (catch-all)

This enables cross-cutting concerns like logging, metrics, and auditing
to subscribe once with a pattern instead of subscribing to each topic.

=== ARCHITECTURE ===

  EventBus
    │  publish(event)
    ▼
  EventRouter (subscribed to ALL topics via internal dispatch)
    │
    ├── pattern "document.*"   → handler_A
    ├── pattern "*.completed"  → handler_B
    └── pattern "*"            → audit_logger

The router subscribes a single internal handler to the bus for each
unique topic it has seen.  When an event arrives, it checks all
registered patterns and dispatches to matching handlers.

=== COMPLEXITY ===

  route (on publish):    O(R) where R = number of registered routes
  add_route():           O(1) amortised
  remove_route():        O(R) worst-case (linear scan)
  matches():             O(len(pattern) + len(topic)) via fnmatch

=== SPACE COMPLEXITY ===

  O(R) where R = number of active routes

=== TRADEOFFS ===

  + Wildcard patterns reduce subscription boilerplate
  + Centralised routing logic for cross-cutting concerns
  + Pattern matching via stdlib fnmatch (no regex overhead)
  - O(R) per event (acceptable for moderate route counts)
  - No topic hierarchy / prefix trees (KISS for now)

=== PRODUCTION EQUIVALENTS ===

  RabbitMQ:       topic exchange with routing key patterns (*.stock.#)
  Apache Camel:   Content-Based Router EIP
  AWS EventBridge: event pattern matching with prefix/suffix/wildcard
  NATS:           subject-based wildcarding (foo.*, foo.>)
"""

import logging
import threading
import uuid
from fnmatch import fnmatch
from typing import Callable

from app.events.bus import EventBus
from app.events.models import Event

logger = logging.getLogger(__name__)


class EventRouter:
    """
    Routes events to handlers based on topic patterns.

    Patterns use fnmatch-style globbing:
      "*"           — match everything
      "document.*"  — match any document event
      "*.completed" — match any completion event

    The router subscribes an internal dispatch handler to the bus.
    When an event arrives, all matching routes are invoked.

    Usage:
        bus = InMemoryEventBus()
        router = EventRouter(bus)
        router.add_route("document.*", handle_doc_events)
        router.add_route("*.completed", log_completion)
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._routes: list[tuple[str, str, Callable[[Event], None]]] = []  # (route_id, pattern, handler)
        self._lock = threading.Lock()
        self._subscribed_topics: dict[str, str] = {}  # topic -> subscription_id

    def _ensure_subscribed(self, topic: str) -> None:
        """Subscribe an internal dispatcher to the bus for a given topic."""
        if topic not in self._subscribed_topics:
            sub_id = self._bus.subscribe(topic, self._dispatch)
            self._subscribed_topics[topic] = sub_id

    def _dispatch(self, event: Event) -> None:
        """
        Internal handler called by the bus for every event.

        Iterates all routes and invokes handlers whose pattern matches
        the event's topic.
        """
        with self._lock:
            routes = list(self._routes)

        for route_id, pattern, handler in routes:
            if self.matches(pattern, event.topic):
                try:
                    handler(event)
                except Exception as exc:
                    logger.error(
                        "Route %s (pattern='%s') handler raised: %s",
                        route_id[:8], pattern, exc,
                    )

    def add_route(self, pattern: str, handler: Callable[[Event], None]) -> str:
        """
        Register a handler for a topic pattern.

        Args:
            pattern: fnmatch-style glob pattern (e.g. "document.*")
            handler: callable invoked for matching events

        Returns:
            A unique route_id for later removal.
        """
        route_id = str(uuid.uuid4())
        with self._lock:
            self._routes.append((route_id, pattern, handler))
        logger.debug("Added route %s for pattern '%s'", route_id[:8], pattern)
        return route_id

    def remove_route(self, route_id: str) -> None:
        """Remove a route by its ID."""
        with self._lock:
            self._routes = [
                (rid, pat, h) for rid, pat, h in self._routes if rid != route_id
            ]
        logger.debug("Removed route %s", route_id[:8])

    @staticmethod
    def matches(pattern: str, topic: str) -> bool:
        """
        Check whether a topic matches an fnmatch-style pattern.

        Examples:
            matches("document.*", "document.indexed")   -> True
            matches("*.completed", "crawl.completed")    -> True
            matches("agent.task.*", "agent.task.failed") -> True
            matches("*", "anything.at.all")              -> True
            matches("document.*", "search.executed")     -> False
        """
        return fnmatch(topic, pattern)

    def route_count(self) -> int:
        """Return the number of active routes."""
        with self._lock:
            return len(self._routes)
