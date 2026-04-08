"""
Logging utilities for Voice Assistant.
Provides structured logging with latency tracking.
"""

import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional

from ..config import settings


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure logging for the application."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)-15s | %(message)s",
        datefmt="%H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger("voice_assistant")
    logger.setLevel(log_level)
    logger.handlers = [handler]

    return logger


# Global logger
logger = setup_logging(settings.log_level)


@dataclass
class LatencyTracker:
    """Track latency for pipeline stages."""

    _timings: Dict[str, float] = field(default_factory=dict)
    _starts: Dict[str, float] = field(default_factory=dict)

    def start(self, stage: str) -> None:
        """Start timing a stage."""
        self._starts[stage] = time.perf_counter()

    def end(self, stage: str) -> float:
        """End timing and return duration in ms."""
        if stage not in self._starts:
            return 0.0
        duration_ms = (time.perf_counter() - self._starts[stage]) * 1000
        self._timings[stage] = duration_ms
        del self._starts[stage]
        return duration_ms

    @contextmanager
    def track(self, stage: str):
        """Context manager for timing a stage."""
        self.start(stage)
        try:
            yield
        finally:
            duration = self.end(stage)
            if settings.debug:
                logger.debug(f"[LATENCY] {stage}: {duration:.1f}ms")

    def get_summary(self) -> Dict[str, float]:
        """Get all recorded timings."""
        return dict(self._timings)

    def log_summary(self) -> None:
        """Log all timings as summary."""
        if not self._timings:
            return
        parts = [f"{k}={v:.0f}ms" for k, v in self._timings.items()]
        logger.info(f"Latency: {', '.join(parts)}")

    def reset(self) -> None:
        """Clear all timings."""
        self._timings.clear()
        self._starts.clear()


# Global latency tracker
latency = LatencyTracker()


def debug_log(message: str, **kwargs) -> None:
    """Log debug message only if DEBUG mode is enabled."""
    if settings.debug:
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.debug(f"{message} {extra}".strip())


# =============================================================================
# Pipeline Stage Logging
# =============================================================================

def log_vad_event(event: str, speech_prob: float = None, chunks: int = None) -> None:
    """Log VAD events with details."""
    if settings.debug:
        details = []
        if speech_prob is not None:
            details.append(f"prob={speech_prob:.2f}")
        if chunks is not None:
            details.append(f"chunks={chunks}")
        detail_str = f" ({', '.join(details)})" if details else ""
        logger.debug(f"[VAD] {event}{detail_str}")


def log_asr_event(event: str, text: str = None, latency_ms: float = None) -> None:
    """Log ASR events with details."""
    if settings.debug:
        details = []
        if text:
            # Truncate long text
            display_text = text[:50] + "..." if len(text) > 50 else text
            details.append(f'"{display_text}"')
        if latency_ms is not None:
            details.append(f"{latency_ms:.0f}ms")
        detail_str = f": {' '.join(details)}" if details else ""
        logger.debug(f"[ASR] {event}{detail_str}")


def log_asr_result(text: str, is_final: bool, latency_ms: float = None) -> None:
    """Log ASR result in debug mode."""
    if settings.debug:
        tag = "FINAL" if is_final else "INTERIM"
        lat_str = f" [{latency_ms:.0f}ms]" if latency_ms else ""
        logger.debug(f"[ASR {tag}]{lat_str} {text}")


def log_llm_event(event: str, tokens: int = None, latency_ms: float = None) -> None:
    """Log LLM events with details."""
    if settings.debug:
        details = []
        if tokens is not None:
            details.append(f"tokens={tokens}")
        if latency_ms is not None:
            details.append(f"{latency_ms:.0f}ms")
        detail_str = f" ({', '.join(details)})" if details else ""
        logger.debug(f"[LLM] {event}{detail_str}")


def log_llm_token(token: str) -> None:
    """Log LLM token in debug mode (prints without newline)."""
    if settings.debug:
        print(token, end="", flush=True)


def log_tts_event(event: str, text: str = None, duration_ms: float = None, latency_ms: float = None) -> None:
    """Log TTS events with details."""
    if settings.debug:
        details = []
        if text:
            display_text = text[:30] + "..." if len(text) > 30 else text
            details.append(f'"{display_text}"')
        if duration_ms is not None:
            details.append(f"audio={duration_ms:.0f}ms")
        if latency_ms is not None:
            details.append(f"latency={latency_ms:.0f}ms")
        detail_str = f": {', '.join(details)}" if details else ""
        logger.debug(f"[TTS] {event}{detail_str}")


def log_audio_chunk(stage: str, duration_ms: float, sample_rate: int = 16000) -> None:
    """Log audio chunk processing."""
    if settings.debug:
        logger.debug(f"[AUDIO] {stage}: {duration_ms:.0f}ms @ {sample_rate}Hz")


def log_pipeline_state(old_state: str, new_state: str) -> None:
    """Log pipeline state change."""
    if settings.debug:
        logger.debug(f"[PIPELINE] State: {old_state} → {new_state}")


def log_session_event(event: str, session_id: str = None, details: str = None) -> None:
    """Log session events."""
    sid = session_id[:8] if session_id else "N/A"
    detail_str = f": {details}" if details else ""
    logger.info(f"[SESSION] {event} ({sid}){detail_str}")


def log_error(stage: str, error: Exception) -> None:
    """Log error with stage context."""
    logger.error(f"[{stage}] Error: {error}", exc_info=settings.debug)
