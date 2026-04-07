"""
FastAPI WebSocket server for Voice Assistant.
"""

import asyncio
import json
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from ..config import settings
from ..utils.logging import logger, debug_log
from ..core.pipeline import PipelineOrchestrator, PipelineEvent, get_orchestrator

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


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client connected: {client_id}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client disconnected: {client_id}")

    async def send_bytes(self, client_id: str, data: bytes):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_bytes(data)

    async def send_json(self, client_id: str, data: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(json.dumps(data))


manager = ConnectionManager()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Vietnamese Voice Assistant",
        "version": "1.0.0",
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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for voice assistant.

    Protocol:
    - Client → Server: Binary PCM S16LE audio (16kHz, mono, 100ms chunks)
    - Server → Client:
        - Binary: TTS audio chunks
        - Text JSON: {"type": "transcript|response|control", ...}
    """
    import uuid
    client_id = str(uuid.uuid4())[:8]

    await manager.connect(websocket, client_id)

    # Create orchestrator for this connection
    orchestrator = PipelineOrchestrator()

    # Event callback to send to client
    async def on_event(event: PipelineEvent):
        try:
            if event.type == "audio":
                # Send audio as binary
                await manager.send_bytes(client_id, event.data)
            else:
                # Send other events as JSON
                await manager.send_json(client_id, {
                    "type": event.type,
                    **event.data if isinstance(event.data, dict) else {"data": event.data},
                })
        except Exception as e:
            debug_log(f"Error sending event: {e}")

    orchestrator.on_event = on_event

    try:
        while True:
            # Receive audio chunk from client
            data = await websocket.receive()

            if "bytes" in data:
                # Process audio chunk
                await orchestrator.handle_audio_chunk(data["bytes"])

            elif "text" in data:
                # Handle text commands
                try:
                    message = json.loads(data["text"])

                    if message.get("type") == "text":
                        # Process text input directly (skip ASR)
                        async for event in orchestrator.process_text(message["text"]):
                            await on_event(event)

                    elif message.get("type") == "reset":
                        # Reset conversation
                        orchestrator.reset()
                        await manager.send_json(client_id, {
                            "type": "control",
                            "action": "reset_complete",
                        })

                except json.JSONDecodeError:
                    debug_log("Invalid JSON received")

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(client_id)


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    import uvicorn
    logger.info(f"Starting server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
