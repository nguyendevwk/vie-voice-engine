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
"""

from .asr import ASRService, get_asr_service
from .llm import LLMService, get_llm_service
from .llm_fallback import FallbackProvider, create_fallback_provider
from .llm_providers import (
    GroqProvider, OpenAIProvider, OllamaProvider, AnthropicProvider, GeminiProvider,
    create_groq_provider, create_openai_provider, create_ollama_provider,
    create_anthropic_provider, create_gemini_provider,
)
from .tts import TTSService, get_tts_service
from .vad import VADService, get_vad_service
from .pipeline import PipelineOrchestrator, PipelineEvent, PipelineState
from .session import Session, SessionManager, ConversationState
from .audio import AudioPreprocessor, AudioNormalizer, NoiseReducer

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
