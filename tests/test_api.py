"""
Tests for API server.
"""

import pytest
from fastapi.testclient import TestClient

from voice_assistant.api.server import app


@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as c:
        yield c


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_returns_html_or_json(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        # Should be HTML (web UI) or JSON
        content_type = response.headers.get("content-type", "")
        assert "text/html" in content_type or "application/json" in content_type

    def test_api_info(self, client):
        """Test API info endpoint."""
        response = client.get("/api")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "Vietnamese Voice Assistant"
        assert "endpoints" in data
        assert data["endpoints"]["websocket"] == "/ws"

    def test_health(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "config" in data


class TestSessionEndpoints:
    """Tests for session management endpoints."""

    def test_list_sessions_empty(self, client):
        """Test listing sessions when empty."""
        response = client.get("/sessions")
        assert response.status_code == 200
        
        data = response.json()
        assert "sessions" in data
        assert "total" in data

    def test_get_nonexistent_session(self, client):
        """Test getting a session that doesn't exist."""
        response = client.get("/sessions/nonexistent-id")
        assert response.status_code == 404

    def test_delete_nonexistent_session(self, client):
        """Test deleting a session that doesn't exist."""
        response = client.delete("/sessions/nonexistent-id")
        assert response.status_code == 404


class TestWebSocket:
    """Tests for WebSocket endpoint."""

    def test_websocket_connect(self, client):
        """Test WebSocket connection."""
        with client.websocket_connect("/ws") as ws:
            # Should receive session info
            data = ws.receive_json()
            assert data["type"] == "session"
            assert data["action"] == "connected"
            assert "session_id" in data

    def test_websocket_with_session_id(self, client):
        """Test WebSocket with existing session ID."""
        # First connection
        with client.websocket_connect("/ws") as ws:
            data = ws.receive_json()
            session_id = data["session_id"]

        # Reconnect with same session
        with client.websocket_connect(f"/ws?session_id={session_id}") as ws:
            data = ws.receive_json()
            assert data["session_id"] == session_id

    def test_websocket_text_message(self, client):
        """Test sending text message via WebSocket."""
        with client.websocket_connect("/ws") as ws:
            # Receive session info
            ws.receive_json()

            # Send text message
            ws.send_json({"type": "text", "text": "Xin chào"})

            # Note: Full response depends on LLM service
            # Just verify no error occurs

    def test_websocket_get_state(self, client):
        """Test getting state via WebSocket."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # Session info

            ws.send_json({"type": "get_state"})
            data = ws.receive_json()

            assert data["type"] == "state"
            assert "state" in data

    def test_websocket_get_history(self, client):
        """Test getting history via WebSocket."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # Session info

            ws.send_json({"type": "get_history"})
            data = ws.receive_json()

            assert data["type"] == "history"
            assert "messages" in data

    def test_websocket_reset(self, client):
        """Test reset via WebSocket."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # Session info

            ws.send_json({"type": "reset"})
            data = ws.receive_json()

            assert data["type"] == "control"
            assert data["action"] == "reset_complete"


class TestStaticFiles:
    """Tests for static file serving."""

    def test_static_index(self, client):
        """Test static index.html is served."""
        response = client.get("/static/index.html")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Vietnamese Voice Assistant" in response.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
