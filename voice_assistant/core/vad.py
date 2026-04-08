"""
Voice Activity Detection using Silero VAD.
Streaming-optimized with VADIterator.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np

from ..config import settings
from ..utils.logging import debug_log, log_vad_event, logger
from .audio import AudioPreprocessor


@dataclass
class VADResult:
    """Result from VAD processing."""
    event: Optional[Literal["start", "end"]] = None
    is_speech: bool = False
    confidence: float = 0.0
    latency_ms: float = 0.0


class VADService:
    """
    Voice Activity Detection using Silero VADIterator.

    Features:
    - Real-time streaming (32ms chunks)
    - Start/end event detection
    - Low latency, low memory
    """

    def __init__(self, config=None):
        cfg = config or settings.vad
        self.threshold = cfg.threshold
        self.min_silence_duration_ms = cfg.min_silence_duration_ms
        self.speech_pad_ms = cfg.speech_pad_ms
        self.chunk_size = cfg.chunk_size  # 512 samples optimal

        self._model = None
        self._iterator = None
        self._preprocessor = AudioPreprocessor()
        self._is_speech_active = False
        self._speech_chunks = 0
        self._silence_chunks = 0

    def _ensure_loaded(self):
        """Lazy load Silero VAD model."""
        if self._model is not None:
            return

        import torch
        self._torch = torch

        log_vad_event("Loading Silero VAD model...")

        # Load Silero VAD
        self._model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )

        # Get VADIterator class
        _, _, _, VADIterator, _ = utils

        # Initialize iterator
        self._iterator = VADIterator(
            self._model,
            threshold=self.threshold,
            sampling_rate=settings.audio.sample_rate,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
        )

        log_vad_event("Model loaded", speech_prob=self.threshold)
        logger.info(f"VAD initialized: threshold={self.threshold}, silence={self.min_silence_duration_ms}ms")

    def reset(self):
        """Reset VAD state for new conversation."""
        if self._iterator is not None:
            self._iterator.reset_states()
        self._is_speech_active = False
        self._speech_chunks = 0
        self._silence_chunks = 0
        log_vad_event("Reset")

    def process_chunk(self, audio_data: bytes) -> VADResult:
        """
        Process audio chunk and detect speech events.

        Args:
            audio_data: PCM S16LE bytes (100ms chunk)

        Returns:
            VADResult with event type and speech status
        """
        start_time = time.perf_counter()
        self._ensure_loaded()

        # Preprocess audio
        audio = self._preprocessor.process(audio_data)

        # Process in 512-sample sub-chunks (Silero optimal)
        event_type = None
        last_prob = 0.0

        for i in range(0, len(audio), self.chunk_size):
            chunk = audio[i:i + self.chunk_size]
            if len(chunk) < self.chunk_size:
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))

            # Convert to tensor
            tensor = self._torch.from_numpy(chunk.astype(np.float32))

            # Run VAD
            speech_dict = self._iterator(tensor, return_seconds=True)

            # Get probability for logging
            with self._torch.no_grad():
                prob = self._model(tensor.unsqueeze(0), settings.audio.sample_rate).item()
                last_prob = prob

            # Parse events
            if speech_dict:
                if "start" in speech_dict:
                    self._is_speech_active = True
                    self._speech_chunks = 0
                    event_type = "start"
                    log_vad_event("SPEECH_START", speech_prob=prob)
                elif "end" in speech_dict:
                    self._is_speech_active = False
                    event_type = "end"
                    log_vad_event("SPEECH_END", speech_prob=prob, chunks=self._speech_chunks)

        # Track consecutive speech/silence
        if self._is_speech_active:
            self._speech_chunks += 1
            self._silence_chunks = 0
        else:
            self._silence_chunks += 1

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Periodic status logging
        if settings.debug and (self._speech_chunks % 10 == 1 or self._silence_chunks % 50 == 1):
            log_vad_event(
                f"{'SPEECH' if self._is_speech_active else 'SILENCE'}",
                speech_prob=last_prob,
                chunks=self._speech_chunks if self._is_speech_active else self._silence_chunks
            )

        return VADResult(
            event=event_type,
            is_speech=self._is_speech_active,
            confidence=last_prob,
            latency_ms=latency_ms,
        )

    async def process_chunk_async(self, audio_data: bytes) -> VADResult:
        """Async wrapper for process_chunk."""
        return await asyncio.to_thread(self.process_chunk, audio_data)

    @property
    def is_speech_active(self) -> bool:
        """Check if speech is currently active."""
        return self._is_speech_active


# Lazy singleton
_vad_service: Optional[VADService] = None


def get_vad_service() -> VADService:
    """Get or create VAD service singleton."""
    global _vad_service
    if _vad_service is None:
        _vad_service = VADService()
    return _vad_service
