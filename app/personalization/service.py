"""
Personalization Infrastructure

Stores user profiles (search history, click history, preferences).
Full personalization (re-ranking based on user signals) is disabled by
default — it requires enough click data to avoid cold-start problems.

=== DESIGN ===

  UserProfile  — search history, click history, inferred topics
  PersonalizationService — CRUD for profiles, preference computation

=== PERSONALIZATION BOOST FORMULA ===

  boost(doc, user) = Σ_{c in clicks} similarity(doc, clicked_doc) × decay(c.age)

  In practice we use a simple proxy:
    - If doc was previously clicked by user → large boost (0.5)
    - If doc's topics overlap with user preferences → small boost (0.1-0.3)
    - Otherwise → 0

=== COLD START ===

  New users have no history → no boost applied (disabled flag helps here).
  After ~20 clicks the signal is meaningful.

=== PRIVACY ===

  In production: user IDs should be pseudonymous.  History should be
  retention-limited (e.g. 90 days).  GDPR: provide user data export +
  right to deletion.  We log no PII beyond the user_id passed in.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.config import PersonalizationConfig

logger = logging.getLogger(__name__)


@dataclass
class ClickEvent:
    doc_id:     int
    query:      str
    timestamp:  str


@dataclass
class UserProfile:
    user_id:       str
    search_history: list[str]              = field(default_factory=list)
    click_history:  list[ClickEvent]       = field(default_factory=list)
    preferences:    dict[str, float]       = field(default_factory=dict)  # topic → weight
    created_at:     str                    = ""
    updated_at:     str                    = ""


class PersonalizationService:
    """
    Manages user profiles and computes relevance boosts.
    Requires a Database reference to persist profiles.
    """

    def __init__(self, db, config: PersonalizationConfig | None = None):
        self.db     = db
        self.config = config or PersonalizationConfig()

    # ── Profile management ────────────────────────────────────────────────

    def get_or_create(self, user_id: str) -> UserProfile:
        """Return existing profile or create a blank one."""
        raw = self.db.get_user_profile(user_id)
        if raw:
            return UserProfile(
                user_id       = user_id,
                search_history = json.loads(raw["search_history_json"]),
                click_history  = [
                    ClickEvent(**c) for c in json.loads(raw["click_history_json"])
                ],
                preferences   = json.loads(raw["preferences_json"]),
                created_at    = raw["created_at"],
                updated_at    = raw["updated_at"],
            )
        profile = UserProfile(
            user_id    = user_id,
            created_at = datetime.now(timezone.utc).isoformat(),
            updated_at = datetime.now(timezone.utc).isoformat(),
        )
        self._save(profile)
        return profile

    def record_search(self, user_id: str, query: str) -> None:
        """Add a query to the user's search history (capped at max_history)."""
        profile = self.get_or_create(user_id)
        profile.search_history.append(query)
        if len(profile.search_history) > self.config.max_search_history:
            profile.search_history = profile.search_history[-self.config.max_search_history:]
        profile.updated_at = datetime.now(timezone.utc).isoformat()
        self._save(profile)

    def record_click(self, user_id: str, doc_id: int, query: str) -> None:
        """Record a click event and update preference signals."""
        profile = self.get_or_create(user_id)
        event   = ClickEvent(
            doc_id    = doc_id,
            query     = query,
            timestamp = datetime.now(timezone.utc).isoformat(),
        )
        profile.click_history.append(event)
        if len(profile.click_history) > self.config.max_click_history:
            profile.click_history = profile.click_history[-self.config.max_click_history:]
        profile.updated_at = datetime.now(timezone.utc).isoformat()
        self._save(profile)

    def get_boost_map(self, user_id: str, doc_ids: list[int]) -> dict[int, float]:
        """
        Return a {doc_id: boost} map for the given user.
        boost ∈ [0, 1]; 0 = no history, 1 = strongly preferred.
        Returns all zeros if personalization is disabled.
        """
        if not self.config.enabled:
            return {d: 0.0 for d in doc_ids}

        profile    = self.get_or_create(user_id)
        clicked    = {e.doc_id for e in profile.click_history}
        boost_map: dict[int, float] = {}

        for doc_id in doc_ids:
            if doc_id in clicked:
                boost_map[doc_id] = 0.5   # previously clicked → strong boost
            else:
                boost_map[doc_id] = 0.0

        return boost_map

    # ── Persistence ───────────────────────────────────────────────────────

    def _save(self, profile: UserProfile) -> None:
        self.db.upsert_user_profile(
            user_id              = profile.user_id,
            search_history_json  = json.dumps(profile.search_history),
            click_history_json   = json.dumps([c.__dict__ for c in profile.click_history]),
            preferences_json     = json.dumps(profile.preferences),
        )
