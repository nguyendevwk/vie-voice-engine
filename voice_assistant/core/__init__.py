"""
Core components for the voice assistant pipeline.

This module contains the main processing components:
    - ASR: Automatic Speech Recognition
    - LLM: Large Language Model integration
    - TTS: Text-to-Speech synthesis
    - VAD: Voice Activity Detection
    - Pipeline: Orchestration and state management
    - Session: Conversation management
    - Audio: Preprocessing utilities

All components are designed for async operation with streaming support.

Usage:
    >>> from voice_assistant.core import PipelineOrchestrator
    >>> from voice_assistant.core import get_llm_service, get_tts_service
"""

# Lazy imports to avoid loading heavy dependencies at import time
# Each submodule is only loaded when actually accessed

__all__ = [
    # Services
    "ASRService",
    "LLMService",
    "TTSService",
    "VADService",
    # Singletons
    "get_asr_service",
    "get_llm_service",
    "get_tts_service",
    "get_vad_service",
    # Pipeline
    "PipelineOrchestrator",
    "PipelineEvent",
    "PipelineState",
    # Session
    "Session",
    "SessionManager",
    "ConversationState",
    # Audio
    "AudioPreprocessor",
    "AudioNormalizer",
    "NoiseReducer",
    # LLM Providers
    "FallbackProvider",
    "create_fallback_provider",
    "GroqProvider",
    "OpenAIProvider",
    "OllamaProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "create_groq_provider",
    "create_openai_provider",
    "create_ollama_provider",
    "create_anthropic_provider",
    "create_gemini_provider",
]


def __getattr__(name: str):
    """Lazy import submodule attributes on first access."""
    if name in ("ASRService", "get_asr_service"):
        from .asr import ASRService, get_asr_service
        return ASRService if name == "ASRService" else get_asr_service

    if name in ("LLMService", "get_llm_service"):
        from .llm import LLMService, get_llm_service
        return LLMService if name == "LLMService" else get_llm_service

    if name in ("TTSService", "get_tts_service"):
        from .tts import TTSService, get_tts_service
        return TTSService if name == "TTSService" else get_tts_service

    if name in ("VADService", "get_vad_service"):
        from .vad import VADService, get_vad_service
        return VADService if name == "VADService" else get_vad_service

    if name in ("PipelineOrchestrator", "PipelineEvent", "PipelineState"):
        from .pipeline import PipelineOrchestrator, PipelineEvent, PipelineState
        return {"PipelineOrchestrator": PipelineOrchestrator, "PipelineEvent": PipelineEvent, "PipelineState": PipelineState}[name]

    if name in ("Session", "SessionManager", "ConversationState"):
        from .session import Session, SessionManager, ConversationState
        return {"Session": Session, "SessionManager": SessionManager, "ConversationState": ConversationState}[name]

    if name in ("AudioPreprocessor", "AudioNormalizer", "NoiseReducer"):
        from .audio import AudioPreprocessor, AudioNormalizer, NoiseReducer
        return {"AudioPreprocessor": AudioPreprocessor, "AudioNormalizer": AudioNormalizer, "NoiseReducer": NoiseReducer}[name]

    if name in ("FallbackProvider", "create_fallback_provider"):
        from .llm_fallback import FallbackProvider, create_fallback_provider
        return FallbackProvider if name == "FallbackProvider" else create_fallback_provider

    if name in ("GroqProvider", "OpenAIProvider", "OllamaProvider", "AnthropicProvider", "GeminiProvider",
                 "create_groq_provider", "create_openai_provider", "create_ollama_provider",
                 "create_anthropic_provider", "create_gemini_provider"):
        from . import llm_providers
        return getattr(llm_providers, name)

    raise AttributeError(f"module 'voice_assistant.core' has no attribute {name!r}")
