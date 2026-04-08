"""
FastAPI WebSocket server for Voice Assistant.
With session management and conversation state tracking.
"""

import asyncio
import json
from pathlib import Path
from typing import Optional, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from ..config import settings
from ..utils.logging import logger, debug_log
from ..core.pipeline import PipelineOrchestrator, PipelineEvent
from ..core.session import (
    SessionManager,
    Session,
    ConversationState,
    get_session_manager,
)

app = FastAPI(
    title="Vietnamese Voice Assistant",
    description="Real-time voice assistant with ASR, LLM, and TTS streaming",
    version="1.0.0",
)

# CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Session manager
session_manager: SessionManager = None


@app.on_event("startup")
async def startup():
    """Initialize on server startup."""
    global session_manager
    session_manager = get_session_manager()
    await session_manager.start_cleanup_loop()
    logger.info("Session manager initialized")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on server shutdown."""
    if session_manager:
        session_manager.stop_cleanup_loop()
        if settings.session.persistence:
            session_manager._save_sessions()


class ConnectionManager:
    """Manage WebSocket connections with sessions."""

    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.orchestrators: Dict[str, PipelineOrchestrator] = {}

    async def connect(
        self,
        websocket: WebSocket,
        client_id: str,
        session: Session,
    ) -> PipelineOrchestrator:
        await websocket.accept()
        self.connections[client_id] = websocket

        # Create orchestrator for this connection
        orchestrator = PipelineOrchestrator()

        # Set up event callback
        async def on_event(event: PipelineEvent):
            await self._handle_event(client_id, session, event)

        orchestrator.on_event = on_event

        # Restore conversation history to orchestrator
        orchestrator._conversation_history = [
            type('Message', (), {'role': m.role, 'content': m.content})()
            for m in session.history
        ]

        self.orchestrators[client_id] = orchestrator

        logger.info(f"Client connected: {client_id} (session: {session.id[:8]})")
        return orchestrator

    async def _handle_event(
        self,
        client_id: str,
        session: Session,
        event: PipelineEvent,
    ):
        """Handle pipeline event and update session."""
        try:
            if event.type == "audio":
                await self.send_bytes(client_id, event.data)

            elif event.type == "transcript":
                data = event.data
                # Update session with user message
                if data.get("is_final") and data.get("text"):
                    session.add_message("user", data["text"])
                await self.send_json(client_id, {
                    "type": "transcript",
                    **data,
                })

            elif event.type == "response":
                # Update session with assistant message
                if event.data.get("text"):
                    session.add_message("assistant", event.data["text"])
                await self.send_json(client_id, {
                    "type": "response",
                    **event.data,
                })

            elif event.type == "control":
                action = event.data.get("action")
                if action == "interrupt":
                    session.stats.interrupts += 1
                    session.set_state(ConversationState.INTERRUPTED)
                elif action == "mic_mute":
                    session.set_state(ConversationState.RESPONDING)
                elif action == "mic_unmute":
                    session.set_state(ConversationState.IDLE)

                await self.send_json(client_id, {
                    "type": "control",
                    **event.data,
                })

        except Exception as e:
            debug_log(f"Error handling event: {e}")
            session.stats.errors += 1

    def disconnect(self, client_id: str):
        if client_id in self.connections:
            del self.connections[client_id]
        if client_id in self.orchestrators:
            del self.orchestrators[client_id]
        logger.info(f"Client disconnected: {client_id}")

    async def send_bytes(self, client_id: str, data: bytes):
        if client_id in self.connections:
            await self.connections[client_id].send_bytes(data)

    async def send_json(self, client_id: str, data: dict):
        if client_id in self.connections:
            await self.connections[client_id].send_text(json.dumps(data, ensure_ascii=False))


manager = ConnectionManager()


@app.get("/")
async def root():
    """Serve web UI."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "status": "ok",
        "service": "Vietnamese Voice Assistant",
        "version": "1.0.0",
        "ui": "/static/index.html",
    }


@app.get("/api")
async def api_info():
    """API info endpoint."""
    return {
        "status": "ok",
        "service": "Vietnamese Voice Assistant",
        "version": "1.0.0",
        "endpoints": {
            "websocket": "/ws",
            "health": "/health",
            "sessions": "/sessions",
        },
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "debug": settings.debug,
        "config": {
            "asr_model": settings.asr.model_id,
            "llm_model": settings.llm.model,
            "tts_model": settings.tts.model_id,
        },
    }


# ============================================================================
# Session Management Endpoints
# ============================================================================

@app.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    return {
        "sessions": session_manager.list_sessions(),
        "total": len(session_manager._sessions),
    }


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session": session.to_dict(),
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_manager.delete_session(session_id):
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


@app.post("/sessions/{session_id}/clear")
async def clear_session_history(session_id: str):
    """Clear conversation history for a session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.clear_history(keep_system=True)
    return {"status": "cleared", "session_id": session_id}


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: Optional[str] = Query(None, description="Session ID to resume"),
):
    """
    WebSocket endpoint for voice assistant with session support.

    Query params:
    - session_id: Optional session ID to resume conversation

    Protocol:
    - Client → Server: Binary PCM S16LE audio (16kHz, mono, 100ms chunks)
    - Server → Client:
        - Binary: TTS audio chunks
        - Text JSON: {"type": "transcript|response|control|session", ...}
    """
    import uuid
    client_id = str(uuid.uuid4())[:8]

    # Get or create session
    session = session_manager.get_or_create_session(
        session_id=session_id,
        client_id=client_id,
    )

    # Connect and get orchestrator
    orchestrator = await manager.connect(websocket, client_id, session)

    # Send session info to client
    await manager.send_json(client_id, {
        "type": "session",
        "action": "connected",
        "session_id": session.id,
        "history_length": len(session.history),
        "state": session.state.name,
    })

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data:
                # Update session state
                if session.state == ConversationState.IDLE:
                    session.set_state(ConversationState.LISTENING)

                # Process audio chunk
                await orchestrator.handle_audio_chunk(data["bytes"])

            elif "text" in data:
                try:
                    message = json.loads(data["text"])
                    msg_type = message.get("type")

                    if msg_type == "text":
                        # Process text input directly
                        session.set_state(ConversationState.PROCESSING)
                        async for event in orchestrator.process_text(message["text"]):
                            pass  # Events handled by on_event callback
                        session.set_state(ConversationState.IDLE)

                    elif msg_type == "reset":
                        # Reset conversation
                        orchestrator.reset()
                        session.clear_history(keep_system=True)
                        session.set_state(ConversationState.IDLE)
                        await manager.send_json(client_id, {
                            "type": "control",
                            "action": "reset_complete",
                        })

                    elif msg_type == "get_history":
                        # Send conversation history
                        await manager.send_json(client_id, {
                            "type": "history",
                            "messages": [m.to_dict() for m in session.history],
                        })

                    elif msg_type == "get_state":
                        # Send current state
                        await manager.send_json(client_id, {
                            "type": "state",
                            "state": session.state.name,
                            "stats": session.stats.to_dict(),
                        })

                except json.JSONDecodeError:
                    debug_log("Invalid JSON received")

    except WebSocketDisconnect:
        # Save session on disconnect
        if settings.session.persistence:
            session_manager.save_session(session.id)
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        session.stats.errors += 1
        manager.disconnect(client_id)


def run_server(host: str = None, port: int = None):
    """Run the FastAPI server."""
    import uvicorn
    host = host or settings.server.host
    port = port or settings.server.port
    logger.info(f"Starting server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
