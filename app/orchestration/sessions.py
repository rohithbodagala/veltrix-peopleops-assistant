"""In-memory conversation sessions. A session belongs to the employee who sent its first message."""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field


class SessionOwnershipError(Exception):
    pass


@dataclass
class Proposal:
    """A write action waiting for confirmation (at most one per session)."""

    kind: str  # "apply_leave" | "cancel_leave"
    args: dict
    summary: str
    details: dict = field(default_factory=dict)


@dataclass
class Session:
    session_id: str
    owner_id: str
    history: deque = field(default_factory=lambda: deque(maxlen=6))  # (role, text)
    pending: Proposal | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str, employee_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = Session(session_id=session_id, owner_id=employee_id)
                self._sessions[session_id] = session
            elif session.owner_id != employee_id:
                raise SessionOwnershipError(session_id)
            return session
