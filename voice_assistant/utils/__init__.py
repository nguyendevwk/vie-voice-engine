"""
Utility functions and helpers.

This module provides:
    - Logging: Enhanced logging with latency tracking
    - Text Utils: Text normalization for Vietnamese TTS
    - Configuration: Settings management
"""

from .logging import logger, debug_log, latency, LatencyTracker
from .text_utils import (
    normalize_for_tts,
    normalize_llm_output,
    clean_vietnamese_text,
    split_for_tts,
    remove_markdown,
    strip_thinking_blocks,
    remove_emojis,
    expand_vietnamese_abbreviations,
    TTSNormalizationConfig,
    VIENEU_V2_TURBO_CONFIG,
)

__all__ = [
    # Logging
    "logger",
    "debug_log",
    "latency",
    "LatencyTracker",
    # Text processing
    "normalize_for_tts",
    "normalize_llm_output",
    "clean_vietnamese_text",
    "split_for_tts",
    "remove_markdown",
    "strip_thinking_blocks",
    "remove_emojis",
    "expand_vietnamese_abbreviations",
    "TTSNormalizationConfig",
    "VIENEU_V2_TURBO_CONFIG",
]
