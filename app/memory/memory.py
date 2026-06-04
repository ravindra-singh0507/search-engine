"""
Conversation Memory

=== THEORY ===

Multi-turn dialogue requires maintaining state across requests.  In a stateless
HTTP API this means serialising the conversation history to a durable store
(SQLite) and reloading it on each request.

=== MESSAGE ROLES ===

We use the OpenAI/Anthropic convention:
  "system"    — injected once at the start; describes the assistant's role
  "user"      — the human's message
  "assistant" — the LLM's reply

=== CONTEXT WINDOW MANAGEMENT ===

LLMs have fixed context windows (8k–200k tokens).  Naive approaches just
append every message, eventually overflowing the window.  We use:

  1. Context window trimming — include only the last N messages in the prompt.
  2. Summarization — when the session exceeds `summarize_at` messages, the
     older messages are summarized and replaced with a single summary message.
     This preserves the gist while freeing context space.
     (The full history is kept in SQLite for auditing.)

=== RETRIEVAL MEMORY ===

Each message stores `metadata_json` which may contain:
  - retrieved_doc_ids: which documents were cited in this turn
  - query_intent:      the classified intent
  - grounding_score:   how grounded the answer was

This allows "which documents did I tell you about last turn?" lookups.

=== SPACE COMPLEXITY ===

  Per session: O(M) messages in SQLite, O(W) messages in the context window
  M = total messages, W = context_window (default 8)

=== PRODUCTION EQUIVALENTS ===

  LangChain:   ConversationBufferMemory, ConversationSummaryMemory
  LlamaIndex:  ChatMemoryBuffer, VectorMemory
  ChatGPT:     rolling window + gpt-4o summary every ~40k tokens
  Claude:      per-project conversation memory
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.config import MemoryConfig

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Message:
    role:      str                           # "user" | "assistant" | "system"
    content:   str
    timestamp: str
    metadata:  dict = field(default_factory=dict)
    message_id: int  = 0                     # set after DB insert


@dataclass
class ConversationSession:
    session_id:    str
    messages:      list[Message]
    created_at:    str
    updated_at:    str
    user_id:       str = ""
    is_active:     bool = True

    @property
    def message_count(self) -> int:
        return len(self.messages)

    def last_user_message(self) -> Message | None:
        for m in reversed(self.messages):
            if m.role == "user":
                return m
        return None


# ── Memory service ────────────────────────────────────────────────────────────

class MemoryService:
    """
    Manages conversation sessions and their message history.

    All session state is persisted to the Database via the db layer.
    An in-memory dict acts as a write-through cache for active sessions.
    """

    def __init__(self, db, config: MemoryConfig | None = None):
        self.db     = db
        self.config = config or MemoryConfig()
        self._cache: dict[str, ConversationSession] = {}

    # ── Session lifecycle ─────────────────────────────────────────────────

    def create_session(
        self,
        session_id: str | None = None,
        user_id:    str        = "",
    ) -> ConversationSession:
        """Create a new conversation session and persist it."""
        sid = session_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        session = ConversationSession(
            session_id=sid, messages=[],
            created_at=now, updated_at=now,
            user_id=user_id,
        )
        if self.config.persist:
            self.db.create_conversation_session(sid, user_id, now)
        self._cache[sid] = session
        logger.debug("Created session %s", sid)
        return session

    def get_session(self, session_id: str) -> ConversationSession | None:
        """Return a session, loading from DB if not in cache."""
        if session_id in self._cache:
            return self._cache[session_id]
        if self.config.persist:
            row = self.db.get_conversation_session(session_id)
            if row:
                messages = self._load_messages(session_id)
                session = ConversationSession(
                    session_id=session_id,
                    messages=messages,
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    user_id=row.get("user_id", ""),
                )
                self._cache[session_id] = session
                return session
        return None

    def get_or_create(
        self,
        session_id: str | None = None,
        user_id:    str        = "",
    ) -> ConversationSession:
        """Fetch existing session or create a new one."""
        if session_id:
            existing = self.get_session(session_id)
            if existing:
                return existing
        return self.create_session(session_id, user_id)

    def delete_session(self, session_id: str) -> bool:
        """Remove a session from cache and DB."""
        self._cache.pop(session_id, None)
        if self.config.persist:
            return self.db.delete_conversation_session(session_id)
        return True

    # ── Message management ────────────────────────────────────────────────

    def add_message(
        self,
        session_id: str,
        role:       str,
        content:    str,
        metadata:   dict | None = None,
    ) -> Message:
        """Append a message to the session and persist it."""
        session = self.get_or_create(session_id)
        now     = datetime.now(timezone.utc).isoformat()
        msg     = Message(role=role, content=content, timestamp=now,
                          metadata=metadata or {})

        if self.config.persist:
            mid = self.db.insert_conversation_message(
                session_id=session_id,
                role=role,
                content=content,
                metadata_json=json.dumps(metadata or {}),
                created_at=now,
            )
            msg.message_id = mid
            self.db.update_session_timestamp(session_id, now)

        session.messages.append(msg)
        session.updated_at = now

        # Enforce max_messages
        if len(session.messages) > self.config.max_messages:
            session.messages = session.messages[-self.config.max_messages:]

        # Auto-summarize when threshold is exceeded
        if len(session.messages) >= self.config.summarize_at:
            self._summarize_old_messages(session)

        return msg

    def get_context_window(
        self, session_id: str, n: int | None = None
    ) -> list[Message]:
        """Return the last N messages for injection into a prompt."""
        session = self.get_session(session_id)
        if not session:
            return []
        window = n or self.config.context_window
        return session.messages[-window:]

    def format_history(self, session_id: str, n: int | None = None) -> str:
        """
        Format recent conversation history as a string for prompt injection.
        Returns empty string for a new session.
        """
        messages = self.get_context_window(session_id, n)
        if not messages:
            return ""
        lines: list[str] = ["### Conversation History"]
        for m in messages:
            prefix = "User" if m.role == "user" else "Assistant"
            lines.append(f"{prefix}: {m.content}")
        lines.append("")
        return "\n".join(lines) + "\n"

    def get_all_sessions(self, limit: int = 100) -> list[dict]:
        """Return metadata for all sessions (for admin API)."""
        if self.config.persist:
            return self.db.list_conversation_sessions(limit)
        return [
            {"session_id": s.session_id, "message_count": s.message_count,
             "created_at": s.created_at, "updated_at": s.updated_at}
            for s in self._cache.values()
        ]

    # ── Private helpers ───────────────────────────────────────────────────

    def _load_messages(self, session_id: str) -> list[Message]:
        rows = self.db.get_conversation_messages(session_id)
        return [
            Message(
                role=r["role"],
                content=r["content"],
                timestamp=r["created_at"],
                metadata=json.loads(r.get("metadata_json", "{}")),
                message_id=r["message_id"],
            )
            for r in rows
        ]

    def _summarize_old_messages(self, session: ConversationSession) -> None:
        """
        Replace messages older than the context window with a brief summary.
        The summary is a simple extractive join — no LLM needed here.
        This keeps the session history bounded without losing the gist.
        """
        keep_n = self.config.context_window
        old    = session.messages[:-keep_n]
        recent = session.messages[-keep_n:]

        if not old:
            return

        summary_lines = [f"[Earlier in this conversation ({len(old)} messages)]"]
        user_msgs = [m.content for m in old if m.role == "user"][:3]
        for q in user_msgs:
            summary_lines.append(f"  User asked: {q[:100]}")

        now = datetime.now(timezone.utc).isoformat()
        summary_msg = Message(
            role="system",
            content="\n".join(summary_lines),
            timestamp=now,
            metadata={"type": "summary", "summarized_count": len(old)},
        )
        session.messages = [summary_msg] + recent
        logger.debug("Summarized %d old messages for session %s", len(old), session.session_id)
