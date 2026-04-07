"""
Gwen-TTS Wrapper for Vietnamese Text-to-Speech.
Supports streaming chunk output.
"""

import asyncio
import json
from pathlib import Path
from typing import Optional, Tuple, AsyncIterator, List
import numpy as np
from scipy import signal

from ..config import settings
from ..utils.logging import logger, debug_log, latency


class GwenTTS:
    """
    Gwen-TTS wrapper for Vietnamese voice synthesis.

    Based on inferances_demo/tts/inference.py
    Supports voice cloning with reference audio.
    """

    # Recommended generation config
    GENERATION_CONFIG = dict(
        temperature=0.3,
        top_k=20,
        top_p=0.9,
        max_new_tokens=4096,
        repetition_penalty=2.0,
        subtalker_do_sample=True,
        subtalker_temperature=0.1,
        subtalker_top_k=20,
        subtalker_top_p=1.0,
    )

    def __init__(
        self,
        model_id: str = "g-group-ai-lab/gwen-tts-0.6B",
        device: str = "cuda:0",
    ):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._speaker_info = None

    def _ensure_loaded(self):
        """Lazy load TTS model."""
        if self._model is not None:
            return

        import torch
        from qwen_tts import Qwen3TTSModel

        logger.info(f"Loading Gwen-TTS model: {self.model_id}")

        # Check for flash attention
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
            debug_log("Using SDPA attention")

        self._model = Qwen3TTSModel.from_pretrained(
            self.model_id,
            device_map=self.device,
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )

        logger.info("Gwen-TTS model loaded")

    def _load_speaker_info(self) -> dict:
        """Load reference speaker information."""
        if self._speaker_info is not None:
            return self._speaker_info

        # Search paths for ref_info.json
        ref_paths = [
            Path(__file__).parent.parent.parent / "inferances_demo" / "tts" / "data" / "ref_info.json",
            Path.home() / ".cache" / "gwen-tts" / "ref_info.json",
        ]

        for path in ref_paths:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    self._speaker_info = json.load(f)
                debug_log(f"Loaded speaker info from {path}")
                return self._speaker_info

        self._speaker_info = {}
        return self._speaker_info

    def synthesize(
        self,
        text: str,
        speaker: str = None,
        ref_audio: str = None,
        ref_text: str = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            speaker: Speaker key from ref_info.json
            ref_audio: Custom reference audio path
            ref_text: Transcript of reference audio

        Returns:
            (audio_array, sample_rate)
        """
        self._ensure_loaded()

        if not text.strip():
            return np.array([]), 24000

        # Get reference audio
        if ref_audio and ref_text:
            # Custom reference
            pass
        elif speaker:
            # Built-in speaker
            speaker_info = self._load_speaker_info()
            if speaker in speaker_info:
                ref_audio = speaker_info[speaker].get("audio_path")
                ref_text = speaker_info[speaker].get("text")
            else:
                logger.warning(f"Speaker '{speaker}' not found, using default synthesis")
                speaker = None

        # Generate
        if ref_audio and ref_text:
            wavs, sr = self._model.generate_voice_clone(
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                **self.GENERATION_CONFIG,
            )
        else:
            # Default synthesis without voice cloning
            wavs, sr = self._model.generate(
                text=text,
                **self.GENERATION_CONFIG,
            )

        return wavs[0], sr

    async def synthesize_async(
        self,
        text: str,
        speaker: str = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Async wrapper for synthesize."""
        return await asyncio.to_thread(
            self.synthesize, text, speaker, **kwargs
        )


class StreamingTTSService:
    """
    Streaming TTS service that processes text in chunks.

    Features:
    - Sentence-level chunking for low latency
    - Automatic resampling
    - Async streaming interface
    """

    def __init__(
        self,
        tts: GwenTTS = None,
        target_sample_rate: int = 16000,
    ):
        self.tts = tts or GwenTTS()
        self.target_sample_rate = target_sample_rate
        self.output_sample_rate = 24000  # Gwen-TTS default

    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
        speaker: str = None,
    ) -> AsyncIterator[bytes]:
        """
        Stream TTS synthesis from text chunks.

        Args:
            text_stream: Async iterator of text chunks
            speaker: Speaker key

        Yields:
            PCM S16LE audio bytes at target sample rate
        """
        async for text_chunk in text_stream:
            if not text_chunk.strip():
                continue

            # Synthesize chunk
            with latency.track("tts_chunk"):
                audio, sr = await self.tts.synthesize_async(text_chunk, speaker)

            if len(audio) == 0:
                continue

            # Resample if needed
            if sr != self.target_sample_rate:
                audio = self._resample(audio, sr, self.target_sample_rate)

            # Convert to PCM S16LE bytes
            audio_bytes = self._to_pcm16(audio)

            yield audio_bytes

    async def synthesize_text(
        self,
        text: str,
        speaker: str = None,
    ) -> bytes:
        """Synthesize complete text to audio bytes."""
        with latency.track("tts_full"):
            audio, sr = await self.tts.synthesize_async(text, speaker)

        if len(audio) == 0:
            return b""

        # Resample
        if sr != self.target_sample_rate:
            audio = self._resample(audio, sr, self.target_sample_rate)

        return self._to_pcm16(audio)

    def _resample(
        self,
        audio: np.ndarray,
        src_rate: int,
        dst_rate: int,
    ) -> np.ndarray:
        """Resample audio."""
        if src_rate == dst_rate:
            return audio
        num_samples = int(len(audio) * dst_rate / src_rate)
        return signal.resample(audio, num_samples)

    def _to_pcm16(self, audio: np.ndarray) -> bytes:
        """Convert float audio to PCM S16LE bytes."""
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        return pcm.tobytes()

    @staticmethod
    def get_duration_ms(audio_bytes: bytes, sample_rate: int = 16000) -> float:
        """Calculate audio duration from bytes."""
        return len(audio_bytes) / 2 / sample_rate * 1000


class SentenceChunker:
    """
    Chunks text stream into complete sentences for TTS.

    Yields sentences when delimiters are encountered,
    optimizing for low latency while maintaining natural speech.
    """

    SENTENCE_DELIMITERS = {'.', '!', '?', ';', ':', '\n'}
    CLAUSE_DELIMITERS = {','}

    def __init__(self, min_chars: int = 10, max_chars: int = 200):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buffer = ""

    def add_token(self, token: str) -> Optional[str]:
        """
        Add token to buffer and return sentence if ready.

        Returns:
            Complete sentence or None if still buffering
        """
        self._buffer += token

        # Check for sentence delimiter
        for delim in self.SENTENCE_DELIMITERS:
            if delim in token:
                sentence = self._buffer.strip()
                self._buffer = ""
                if len(sentence) >= self.min_chars:
                    return sentence

        # Check for clause delimiter with enough content
        if len(self._buffer) >= self.max_chars:
            for delim in self.CLAUSE_DELIMITERS:
                if delim in self._buffer:
                    last_delim = self._buffer.rfind(delim)
                    if last_delim > self.min_chars:
                        sentence = self._buffer[:last_delim + 1].strip()
                        self._buffer = self._buffer[last_delim + 1:].strip()
                        return sentence

        return None

    def flush(self) -> Optional[str]:
        """Flush remaining buffer."""
        if self._buffer.strip():
            sentence = self._buffer.strip()
            self._buffer = ""
            return sentence
        return None

    async def chunk_stream(
        self,
        token_stream: AsyncIterator[str],
    ) -> AsyncIterator[str]:
        """
        Transform token stream into sentence stream.

        Args:
            token_stream: Async iterator of tokens

        Yields:
            Complete sentences
        """
        async for token in token_stream:
            sentence = self.add_token(token)
            if sentence:
                yield sentence

        # Flush remaining
        final = self.flush()
        if final:
            yield final
