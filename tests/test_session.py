"""
Tests for session management module.
"""

import asyncio
import time
import pytest

from voice_assistant.core.session import (
    Session,
    SessionManager,
    SessionStats,
    ConversationState,
    Message,
)


class TestMessage:
    """Tests for Message."""

    def test_create_message(self):
        msg = Message(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp > 0

    def test_message_to_dict(self):
        msg = Message(role="user", content="Hello", metadata={"lang": "vi"})
        data = msg.to_dict()

        assert data["role"] == "user"
        assert data["content"] == "Hello"
        assert data["metadata"]["lang"] == "vi"

    def test_message_from_dict(self):
        data = {
            "role": "assistant",
            "content": "Hi there",
            "timestamp": 1234567890,
            "metadata": {},
        }
        msg = Message.from_dict(data)

        assert msg.role == "assistant"
        assert msg.content == "Hi there"


class TestSessionStats:
    """Tests for SessionStats."""

    def test_default_values(self):
        stats = SessionStats()

        assert stats.total_messages == 0
        assert stats.user_messages == 0
        assert stats.assistant_messages == 0
        assert stats.interrupts == 0
        assert stats.errors == 0

    def test_to_dict(self):
        stats = SessionStats(total_messages=5, user_messages=3)
        data = stats.to_dict()

        assert data["total_messages"] == 5
        assert data["user_messages"] == 3


class TestSession:
    """Tests for Session."""

    def test_create_session(self):
        session = Session(id="test-123")

        assert session.id == "test-123"
        assert session.state == ConversationState.IDLE
        assert len(session.history) == 0

    def test_add_message(self):
        session = Session(id="test-123")

        msg = session.add_message("user", "Hello")

        assert len(session.history) == 1
        assert session.stats.total_messages == 1
        assert session.stats.user_messages == 1

    def test_add_multiple_messages(self):
        session = Session(id="test-123")

        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there!")
        session.add_message("user", "How are you?")

        assert len(session.history) == 3
        assert session.stats.user_messages == 2
        assert session.stats.assistant_messages == 1

    def test_history_trimming(self):
        session = Session(id="test-123")

        # Add many messages
        for i in range(60):
            session.add_message("user", f"Message {i}")

        # Should be trimmed to max_history_length (50)
        assert len(session.history) <= 50

    def test_get_history_for_llm(self):
        session = Session(id="test-123")
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi!")

        llm_history = session.get_history_for_llm(limit=10)

        assert len(llm_history) == 2
        assert llm_history[0]["role"] == "user"
        assert llm_history[1]["role"] == "assistant"

    def test_clear_history(self):
        session = Session(id="test-123")
        session.add_message("system", "You are a helpful assistant")
        session.add_message("user", "Hello")

        session.clear_history(keep_system=True)

        assert len(session.history) == 1
        assert session.history[0].role == "system"

    def test_set_state(self):
        session = Session(id="test-123")

        session.set_state(ConversationState.LISTENING)

        assert session.state == ConversationState.LISTENING

    def test_is_expired(self):
        session = Session(id="test-123")
        session.last_active = time.time() - 3600  # 1 hour ago

        assert session.is_expired(timeout_seconds=1800)  # 30 min timeout

    def test_not_expired(self):
        session = Session(id="test-123")

        assert not session.is_expired(timeout_seconds=1800)

    def test_to_dict(self):
        session = Session(id="test-123")
        session.add_message("user", "Hello")

        data = session.to_dict()

        assert data["id"] == "test-123"
        assert data["state"] == "IDLE"
        assert len(data["history"]) == 1

    def test_from_dict(self):
        data = {
            "id": "test-456",
            "created_at": time.time(),
            "last_active": time.time(),
            "state": "LISTENING",
            "history": [{"role": "user", "content": "Hi", "timestamp": time.time(), "metadata": {}}],
            "metadata": {},
            "stats": {"total_messages": 1, "user_messages": 1, "assistant_messages": 0,
                     "total_audio_duration_ms": 0, "total_latency_ms": 0, "interrupts": 0, "errors": 0},
            "context": {},
        }

        session = Session.from_dict(data)

        assert session.id == "test-456"
        assert session.state == ConversationState.LISTENING
        assert len(session.history) == 1


class TestSessionManager:
    """Tests for SessionManager."""

    @pytest.fixture
    def manager(self):
        return SessionManager(timeout=60, persistence_enabled=False)

    def test_create_session(self, manager):
        session = manager.create_session()

        assert session is not None
        assert session.id is not None
        assert len(manager._sessions) == 1

    def test_create_session_with_id(self, manager):
        session = manager.create_session(session_id="my-session")

        assert session.id == "my-session"

    def test_get_session(self, manager):
        created = manager.create_session(session_id="test-session")
        retrieved = manager.get_session("test-session")

        assert retrieved is not None
        assert retrieved.id == created.id

    def test_get_nonexistent_session(self, manager):
        session = manager.get_session("nonexistent")

        assert session is None

    def test_get_or_create_session(self, manager):
        # First call creates
        session1 = manager.get_or_create_session(session_id="test-session")
        # Second call retrieves
        session2 = manager.get_or_create_session(session_id="test-session")

        assert session1.id == session2.id

    def test_delete_session(self, manager):
        manager.create_session(session_id="to-delete")

        result = manager.delete_session("to-delete")

        assert result == True
        assert manager.get_session("to-delete") is None

    def test_delete_nonexistent_session(self, manager):
        result = manager.delete_session("nonexistent")

        assert result == False

    def test_list_sessions(self, manager):
        manager.create_session(session_id="session-1")
        manager.create_session(session_id="session-2")

        sessions = manager.list_sessions()

        assert len(sessions) == 2

    def test_expired_session_cleanup(self, manager):
        # Create session and make it expired
        session = manager.create_session(session_id="expired")
        session.last_active = time.time() - 120  # 2 minutes ago (timeout is 60s)

        manager._cleanup_expired()

        assert manager.get_session("expired") is None

    def test_max_sessions_limit(self):
        manager = SessionManager(max_sessions=3, persistence_enabled=False)

        manager.create_session(session_id="s1")
        manager.create_session(session_id="s2")
        manager.create_session(session_id="s3")
        manager.create_session(session_id="s4")  # Should remove oldest

        assert len(manager._sessions) == 3


class TestConversationState:
    """Tests for ConversationState enum."""

    def test_states_exist(self):
        assert ConversationState.IDLE is not None
        assert ConversationState.LISTENING is not None
        assert ConversationState.PROCESSING is not None
        assert ConversationState.RESPONDING is not None
        assert ConversationState.INTERRUPTED is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
