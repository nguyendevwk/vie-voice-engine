"""
Automatic Speech Recognition service with streaming support.
Uses Gipformer model for Vietnamese ASR.
"""

import asyncio
import tempfile
import os
from dataclasses import dataclass
from typing import List, Optional, Callable, Awaitable
from pathlib import Path

import numpy as np

from ..config import settings
from ..utils.logging import debug_log, log_asr_result, latency, logger
from .audio import AudioPreprocessor


@dataclass
class TranscriptUpdate:
    """ASR transcript result."""
    text: str
    is_final: bool
    confidence: float = 1.0


class ASRService:
    """
    Vietnamese ASR using Gipformer model.

    Supports both batch and streaming modes.
    """

    def __init__(self, config=None):
        cfg = config or settings.asr
        self.model_id = cfg.model_id
        self.device = cfg.device
        self.language = cfg.language

        self._model = None
        self._preprocessor = AudioPreprocessor()

    def _ensure_loaded(self):
        """Lazy load ASR model."""
        if self._model is not None:
            return

        import torch

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading ASR model: {self.model_id} on {device}")

        # Import and setup Gipformer (from inferances_demo)
        # Note: This requires icefall setup - see inferances_demo/asr/infer_pytorch.py
        try:
            from .asr_gipformer import GipformerASR
            self._model = GipformerASR(device=device)
        except ImportError:
            # Fallback: use whisper or other model
            logger.warning("Gipformer not available, using fallback ASR")
            self._model = FallbackASR(device=device)

        logger.info("ASR model loaded")

    def transcribe_file(self, audio_path: str) -> str:
        """Transcribe audio file."""
        self._ensure_loaded()
        with latency.track("asr_file"):
            text = self._model.transcribe(audio_path)
        log_asr_result(text, is_final=True)
        return text

    def transcribe_bytes(self, audio_chunks: List[bytes]) -> str:
        """
        Transcribe audio from PCM bytes.

        Args:
            audio_chunks: List of PCM S16LE bytes

        Returns:
            Transcribed text
        """
        self._ensure_loaded()

        # Concatenate and process audio
        raw_audio = b"".join(audio_chunks)
        audio = self._preprocessor.decode_pcm16(raw_audio)

        # Save to temp file (model expects file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            import soundfile as sf
            sf.write(f.name, audio, settings.audio.sample_rate)
            temp_path = f.name

        try:
            with latency.track("asr_transcribe"):
                text = self._model.transcribe(temp_path)
            return text.strip()
        finally:
            os.unlink(temp_path)

    async def transcribe_bytes_async(self, audio_chunks: List[bytes]) -> str:
        """Async transcribe."""
        return await asyncio.to_thread(self.transcribe_bytes, audio_chunks)


class StreamingASRService:
    """
    Streaming ASR with interim results.

    Emits partial transcripts while user is speaking,
    then final transcript when speech ends.
    """

    def __init__(
        self,
        asr_service: ASRService,
        on_transcript_update: Optional[Callable[[TranscriptUpdate], Awaitable[None]]] = None,
        config=None,
    ):
        cfg = config or settings.asr
        self.asr = asr_service
        self.on_transcript_update = on_transcript_update
        self.interim_interval_ms = cfg.interim_interval_ms
        self.min_audio_for_interim_ms = cfg.min_audio_for_interim_ms

        self._audio_buffer: List[bytes] = []
        self._is_active = False
        self._interim_task: Optional[asyncio.Task] = None
        self._last_interim_text = ""

    async def start_utterance(self):
        """Called when VAD detects speech start."""
        self._audio_buffer.clear()
        self._is_active = True
        self._last_interim_text = ""

        # Start background interim loop
        self._interim_task = asyncio.create_task(self._interim_loop())
        debug_log("StreamingASR started")

    def add_audio(self, chunk: bytes):
        """Add audio chunk to buffer."""
        if self._is_active:
            self._audio_buffer.append(chunk)

    async def _interim_loop(self):
        """Background task for interim transcription."""
        try:
            while self._is_active:
                await asyncio.sleep(self.interim_interval_ms / 1000)

                # Check minimum audio requirement
                duration_ms = len(self._audio_buffer) * settings.audio.chunk_duration_ms
                if duration_ms < self.min_audio_for_interim_ms:
                    continue

                # Run interim ASR
                try:
                    interim_text = await self.asr.transcribe_bytes_async(
                        list(self._audio_buffer)
                    )

                    # Emit if changed
                    if interim_text and interim_text != self._last_interim_text:
                        self._last_interim_text = interim_text
                        log_asr_result(interim_text, is_final=False)

                        if self.on_transcript_update:
                            await self.on_transcript_update(TranscriptUpdate(
                                text=interim_text,
                                is_final=False,
                                confidence=0.8,
                            ))
                except Exception as e:
                    debug_log(f"Interim ASR error: {e}")

        except asyncio.CancelledError:
            pass

    async def end_utterance(self) -> Optional[TranscriptUpdate]:
        """Called when VAD detects speech end. Returns final transcript."""
        self._is_active = False

        # Cancel interim task
        if self._interim_task:
            self._interim_task.cancel()
            try:
                await self._interim_task
            except asyncio.CancelledError:
                pass

        # No audio buffered
        if not self._audio_buffer:
            return None

        # Final transcription
        try:
            final_text = await self.asr.transcribe_bytes_async(self._audio_buffer)
            log_asr_result(final_text, is_final=True)

            result = TranscriptUpdate(
                text=final_text.strip(),
                is_final=True,
                confidence=1.0,
            )

            if self.on_transcript_update:
                await self.on_transcript_update(result)

            return result
        except Exception as e:
            logger.error(f"Final ASR error: {e}")
            return None

    async def cancel_utterance(self):
        """Cancel current utterance without processing."""
        self._is_active = False
        if self._interim_task:
            self._interim_task.cancel()
        self._audio_buffer.clear()

    def get_buffer_duration_ms(self) -> float:
        """Get current buffer duration in ms."""
        return len(self._audio_buffer) * settings.audio.chunk_duration_ms

    @property
    def is_active(self) -> bool:
        return self._is_active


class FallbackASR:
    """Fallback ASR using Whisper (if Gipformer unavailable)."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return

        try:
            import whisper
            self._model = whisper.load_model("base", device=self.device)
            logger.info("Loaded Whisper fallback ASR")
        except ImportError:
            logger.error("No ASR model available. Install whisper: pip install openai-whisper")
            raise

    def transcribe(self, audio_path: str) -> str:
        self._ensure_loaded()
        result = self._model.transcribe(audio_path, language="vi")
        return result["text"]


# Lazy singleton
_asr_service: Optional[ASRService] = None


def get_asr_service() -> ASRService:
    """Get or create ASR service singleton."""
    global _asr_service
    if _asr_service is None:
        _asr_service = ASRService()
    return _asr_service
