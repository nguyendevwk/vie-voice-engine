"""
Vietnamese Voice Assistant Pipeline.
End-to-end streaming pipeline for ASR, LLM, and TTS.

Quick start:
    >>> from voice_assistant import PipelineOrchestrator
    >>> pipeline = PipelineOrchestrator()
    >>> async for event in pipeline.process_text("Xin chào"):
    ...     print(event.type, event.data)

Or use individual services:
    >>> from voice_assistant import get_llm_service, get_tts_service
    >>> llm = get_llm_service()
    >>> tts = get_tts_service()
"""

__version__ = "1.0.0"
__author__ = "nguyendevwk"

# Lazy public API - only loads what you use
__all__ = [
    # Main pipeline
    "PipelineOrchestrator",
    "PipelineEvent",
    "PipelineState",
    # Services
    "get_llm_service",
    "get_tts_service",
    "get_asr_service",
    "get_vad_service",
    # Session
    "Session",
    "SessionManager",
    # Message
    "Message",
    # Config
    "Settings",
]


def __getattr__(name: str):
    """Lazy import on first access."""
    if name == "PipelineOrchestrator":
        from voice_assistant.core.pipeline import PipelineOrchestrator
        return PipelineOrchestrator
    if name == "PipelineEvent":
        from voice_assistant.core.pipeline import PipelineEvent
        return PipelineEvent
    if name == "PipelineState":
        from voice_assistant.core.pipeline import PipelineState
        return PipelineState
    if name == "get_llm_service":
        from voice_assistant.core.llm import get_llm_service
        return get_llm_service
    if name == "get_tts_service":
        from voice_assistant.core.tts import get_tts_service
        return get_tts_service
    if name == "get_asr_service":
        from voice_assistant.core.asr import get_asr_service
        return get_asr_service
    if name == "get_vad_service":
        from voice_assistant.core.vad import get_vad_service
        return get_vad_service
    if name == "Session":
        from voice_assistant.core.session import Session
        return Session
    if name == "SessionManager":
        from voice_assistant.core.session import SessionManager
        return SessionManager
    if name == "Message":
        from voice_assistant.core.llm_base import Message
        return Message
    if name == "Settings":
        from voice_assistant.config import Settings
        return Settings
    raise AttributeError(f"module 'voice_assistant' has no attribute {name!r}")
