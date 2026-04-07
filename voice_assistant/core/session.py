"""
Session and Conversation State Management.
Handles multi-user sessions with conversation history persistence.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading

from ..config import settings
from ..utils.logging import logger, debug_log


class ConversationState(Enum):
    """State of the conversation."""
    IDLE = auto()           # Waiting for user
    LISTENING = auto()      # User is speaking
    PROCESSING = auto()     # Processing user input
    RESPONDING = auto()     # Bot is responding
    INTERRUPTED = auto()    # User interrupted


@dataclass
class Message:
    """A single message in conversation."""
    role: str               # user, assistant, system
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(**data)


@dataclass
class SessionStats:
    """Statistics for a session."""
    total_messages: int = 0
    user_messages: int = 0
    assistant_messages: int = 0
    total_audio_duration_ms: float = 0
    total_latency_ms: float = 0
    interrupts: int = 0
    errors: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Session:
    """
    Represents a user session with conversation history.

    Features:
    - Unique session ID
    - Conversation history with timestamps
    - Session metadata (created, last active)
    - Statistics tracking
    - State management
    """
    id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    state: ConversationState = ConversationState.IDLE
    history: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    stats: SessionStats = field(default_factory=SessionStats)

    # Context for LLM (can store user preferences, etc.)
    context: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, **metadata) -> Message:
        """Add a message to conversation history."""
        message = Message(
            role=role,
            content=content,
            metadata=metadata,
        )
        self.history.append(message)
        self.last_active = time.time()

        # Update stats
        self.stats.total_messages += 1
        if role == "user":
            self.stats.user_messages += 1
        elif role == "assistant":
            self.stats.assistant_messages += 1

        # Trim history if too long
        max_len = int(settings.session.max_history_length)
        if len(self.history) > max_len:
            # Keep system messages and recent history
            system_msgs = [m for m in self.history if m.role == "system"]
            other_msgs = [m for m in self.history if m.role != "system"]
            self.history = system_msgs + other_msgs[-(max_len - len(system_msgs)):]

        return message

    def get_history_for_llm(self, limit: int = 20) -> List[Dict[str, str]]:
        """Get conversation history formatted for LLM API."""
        messages = []

        # Add recent messages (excluding metadata)
        for msg in self.history[-limit:]:
            messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        return messages

    def clear_history(self, keep_system: bool = True):
        """Clear conversation history."""
        if keep_system:
            self.history = [m for m in self.history if m.role == "system"]
        else:
            self.history = []

    def set_state(self, state: ConversationState):
        """Update conversation state."""
        debug_log(f"Session {self.id[:8]} state: {self.state.name} → {state.name}")
        self.state = state
        self.last_active = time.time()

    def is_expired(self, timeout_seconds: int = None) -> bool:
        """Check if session has expired."""
        timeout = timeout_seconds or int(settings.session.timeout)
        return (time.time() - self.last_active) > timeout

    def to_dict(self) -> dict:
        """Serialize session to dictionary."""
        return {
            "id": self.id,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "state": self.state.name,
            "history": [m.to_dict() for m in self.history],
            "metadata": self.metadata,
            "stats": self.stats.to_dict(),
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        """Deserialize session from dictionary."""
        session = cls(
            id=data["id"],
            created_at=data.get("created_at", time.time()),
            last_active=data.get("last_active", time.time()),
            metadata=data.get("metadata", {}),
            context=data.get("context", {}),
        )
        session.state = ConversationState[data.get("state", "IDLE")]
        session.history = [Message.from_dict(m) for m in data.get("history", [])]
        session.stats = SessionStats(**data.get("stats", {}))
        return session


class SessionManager:
    """
    Manages multiple user sessions.

    Features:
    - Create/get/delete sessions
    - Auto-cleanup expired sessions
    - Optional persistence to disk
    - Thread-safe operations
    """

    def __init__(
        self,
        timeout: int = None,
        max_sessions: int = 1000,
        persistence_enabled: bool = None,
        storage_path: str = None,
    ):
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.RLock()

        self.timeout = timeout or int(settings.session.timeout)
        self.max_sessions = max_sessions
        self.persistence_enabled = (
            persistence_enabled
            if persistence_enabled is not None
            else settings.session.persistence
        )
        self.storage_path = Path(storage_path or settings.session.storage_path)

        # Load persisted sessions
        if self.persistence_enabled:
            self._load_sessions()

        # Start cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None

    def create_session(self, session_id: str = None, **metadata) -> Session:
        """Create a new session."""
        with self._lock:
            # Generate ID if not provided
            if not session_id:
                session_id = str(uuid.uuid4())

            # Check max sessions
            if len(self._sessions) >= self.max_sessions:
                self._cleanup_expired()
                if len(self._sessions) >= self.max_sessions:
                    # Remove oldest session
                    oldest = min(self._sessions.values(), key=lambda s: s.last_active)
                    self.delete_session(oldest.id)

            # Create session
            session = Session(id=session_id, metadata=metadata)

            # Add default system message
            session.add_message(
                role="system",
                content=settings.llm.system_prompt,
            )

            self._sessions[session_id] = session

            logger.info(f"Created session: {session_id[:8]}...")
            debug_log("Session created", id=session_id, metadata=metadata)

            return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session and not session.is_expired(self.timeout):
                session.last_active = time.time()
                return session
            elif session:
                # Session expired
                self.delete_session(session_id)
            return None

    def get_or_create_session(self, session_id: str = None, **metadata) -> Session:
        """Get existing session or create new one."""
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session

        return self.create_session(session_id, **metadata)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Deleted session: {session_id[:8]}...")

                # Remove persisted file
                if self.persistence_enabled:
                    session_file = self.storage_path / f"{session_id}.json"
                    if session_file.exists():
                        session_file.unlink()

                return True
            return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions (summary only)."""
        with self._lock:
            return [
                {
                    "id": s.id,
                    "created_at": datetime.fromtimestamp(s.created_at).isoformat(),
                    "last_active": datetime.fromtimestamp(s.last_active).isoformat(),
                    "state": s.state.name,
                    "message_count": len(s.history),
                    "metadata": s.metadata,
                }
                for s in self._sessions.values()
                if not s.is_expired(self.timeout)
            ]

    def _cleanup_expired(self):
        """Remove expired sessions."""
        with self._lock:
            expired = [
                sid for sid, session in self._sessions.items()
                if session.is_expired(self.timeout)
            ]
            for sid in expired:
                self.delete_session(sid)

            if expired:
                logger.info(f"Cleaned up {len(expired)} expired sessions")

    async def start_cleanup_loop(self, interval: int = 300):
        """Start background cleanup task."""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(interval)
                self._cleanup_expired()

                # Persist sessions periodically
                if self.persistence_enabled:
                    self._save_sessions()

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    def stop_cleanup_loop(self):
        """Stop cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()

    def _save_sessions(self):
        """Persist all sessions to disk."""
        if not self.persistence_enabled:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)

        with self._lock:
            for session in self._sessions.values():
                session_file = self.storage_path / f"{session.id}.json"
                with open(session_file, "w", encoding="utf-8") as f:
                    json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

    def _load_sessions(self):
        """Load persisted sessions from disk."""
        if not self.storage_path.exists():
            return

        for session_file in self.storage_path.glob("*.json"):
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    session = Session.from_dict(data)

                    # Skip expired sessions
                    if not session.is_expired(self.timeout):
                        self._sessions[session.id] = session
                        debug_log(f"Loaded session: {session.id[:8]}")
            except Exception as e:
                logger.warning(f"Failed to load session {session_file}: {e}")

        logger.info(f"Loaded {len(self._sessions)} sessions from disk")

    def save_session(self, session_id: str):
        """Save a specific session to disk."""
        if not self.persistence_enabled:
            return

        session = self._sessions.get(session_id)
        if session:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            session_file = self.storage_path / f"{session_id}.json"
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)


# Global session manager
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get or create session manager singleton."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
