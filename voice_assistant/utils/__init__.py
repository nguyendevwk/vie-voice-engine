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
    split_into_sentences,
    remove_markdown,
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
    "split_into_sentences",
    "remove_markdown",
]
