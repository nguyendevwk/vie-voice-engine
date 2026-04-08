"""
Vietnamese Voice Assistant - End-to-End Streaming Pipeline.

A real-time voice assistant for Vietnamese with streaming ASR, LLM, and TTS.
Designed for personal use and CV demonstration with production-quality architecture.

Main Components:
    - ASR: Gipformer with ONNX/PyTorch backends
    - LLM: Groq/OpenAI with token streaming
    - TTS: Qwen-TTS with voice cloning or Edge-TTS fallback
    - VAD: Silero for speech detection
    - Pipeline: Async orchestrator with session management

Quick Start:
    >>> from voice_assistant.core.pipeline import PipelineOrchestrator
    >>> orchestrator = PipelineOrchestrator()
    >>> await orchestrator.handle_audio_chunk(audio_bytes)

See README.md for detailed documentation.

Author: Nguyen Dev (phamnguyen.devwk@gmail.com)
License: MIT
"""

__version__ = "1.0.0"
__author__ = "nguyendevwk"
__email__ = "phamnguyen.devwk@gmail.com"

from .config import settings

# Public API
try:
    from .core.pipeline import PipelineOrchestrator, PipelineEvent
    from .core.asr import ASRService
    from .core.llm import LLMService
    from .core.tts import TTSService
    from .core.vad import VADService
    from .core.session import Session, SessionManager
    
    __all__ = [
        "settings",
        "PipelineOrchestrator",
        "PipelineEvent",
        "ASRService",
        "LLMService",
        "TTSService",
        "VADService",
        "Session",
        "SessionManager",
        "__version__",
    ]
except ImportError:
    # Minimal export if core modules not yet imported
    __all__ = ["settings", "__version__"]
