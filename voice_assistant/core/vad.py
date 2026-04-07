"""
Voice Activity Detection using Silero VAD.
Streaming-optimized with VADIterator.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np

from ..config import settings
from ..utils.logging import debug_log
from .audio import AudioPreprocessor


@dataclass
class VADResult:
    """Result from VAD processing."""
    event: Optional[Literal["start", "end"]] = None
    is_speech: bool = False
    confidence: float = 0.0


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

    def _ensure_loaded(self):
        """Lazy load Silero VAD model."""
        if self._model is not None:
            return

        import torch
        self._torch = torch

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

        debug_log("VAD model loaded", threshold=self.threshold)

    def reset(self):
        """Reset VAD state for new conversation."""
        if self._iterator is not None:
            self._iterator.reset_states()
        self._is_speech_active = False

    def process_chunk(self, audio_data: bytes) -> VADResult:
        """
        Process audio chunk and detect speech events.

        Args:
            audio_data: PCM S16LE bytes (100ms chunk)

        Returns:
            VADResult with event type and speech status
        """
        self._ensure_loaded()

        # Preprocess audio
        audio = self._preprocessor.process(audio_data)

        # Process in 512-sample sub-chunks (Silero optimal)
        event_type = None

        for i in range(0, len(audio), self.chunk_size):
            chunk = audio[i:i + self.chunk_size]
            if len(chunk) < self.chunk_size:
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))

            # Convert to tensor
            tensor = self._torch.from_numpy(chunk.astype(np.float32))

            # Run VAD
            speech_dict = self._iterator(tensor, return_seconds=True)

            # Parse events
            if speech_dict:
                if "start" in speech_dict:
                    self._is_speech_active = True
                    event_type = "start"
                    debug_log("VAD speech start", time=speech_dict.get("start"))
                elif "end" in speech_dict:
                    self._is_speech_active = False
                    event_type = "end"
                    debug_log("VAD speech end", time=speech_dict.get("end"))

        return VADResult(
            event=event_type,
            is_speech=self._is_speech_active,
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
