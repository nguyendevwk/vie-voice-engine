"""
Text-to-Speech service using Gwen-TTS.
Supports streaming chunk output.
"""

import asyncio
from typing import Optional, Tuple
import numpy as np
from scipy import signal

from ..config import settings
from ..utils.logging import debug_log, latency, logger


class TTSService:
    """
    Vietnamese TTS using Gwen-TTS model.

    Features:
    - Voice cloning with reference audio
    - Streaming chunk output
    - Automatic resampling to pipeline sample rate
    """

    def __init__(self, config=None):
        cfg = config or settings.tts
        self.model_id = cfg.model_id
        self.device = cfg.device
        self.output_sample_rate = cfg.output_sample_rate
        self.target_sample_rate = cfg.target_sample_rate
        self.default_speaker = cfg.default_speaker

        self._model = None
        self._speaker_info = None

    def _ensure_loaded(self):
        """Lazy load TTS model."""
        if self._model is not None:
            return

        logger.info(f"Loading TTS model: {self.model_id}")

        try:
            from qwen_tts import Qwen3TTSModel
            import torch

            # Check for flash attention
            try:
                import flash_attn
                attn_impl = "flash_attention_2"
            except ImportError:
                attn_impl = "sdpa"
                debug_log("Using SDPA attention (flash-attn not available)")

            self._model = Qwen3TTSModel.from_pretrained(
                self.model_id,
                device_map=self.device,
                dtype=torch.bfloat16,
                attn_implementation=attn_impl,
            )

            logger.info("TTS model loaded")

        except ImportError as e:
            logger.error(f"TTS model import error: {e}")
            logger.warning("Using fallback TTS (edge-tts)")
            self._model = FallbackTTS()

    def _load_speaker_info(self):
        """Load reference speaker information."""
        if self._speaker_info is not None:
            return

        import json
        from pathlib import Path

        # Try to find ref_info.json
        ref_paths = [
            Path(__file__).parent.parent.parent / "inferances_demo" / "tts" / "data" / "ref_info.json",
            Path.home() / ".cache" / "gwen-tts" / "ref_info.json",
        ]

        for path in ref_paths:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    self._speaker_info = json.load(f)
                debug_log(f"Loaded speaker info from {path}")
                return

        self._speaker_info = {}

    async def synthesize(self, text: str, speaker: str = None) -> bytes:
        """
        Synthesize text to speech.

        Args:
            text: Text to synthesize
            speaker: Speaker key (optional)

        Returns:
            PCM S16LE audio bytes at target sample rate
        """
        return await asyncio.to_thread(self._synthesize_sync, text, speaker)

    def _synthesize_sync(self, text: str, speaker: str = None) -> bytes:
        """Synchronous TTS synthesis."""
        self._ensure_loaded()

        if not text.strip():
            return b""

        speaker = speaker or self.default_speaker

        with latency.track("tts_synthesize"):
            if isinstance(self._model, FallbackTTS):
                audio, sr = self._model.synthesize(text)
            else:
                audio, sr = self._synthesize_gwen(text, speaker)

        # Resample if needed
        if sr != self.target_sample_rate:
            audio = self._resample(audio, sr, self.target_sample_rate)

        # Convert to PCM S16LE
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

        return pcm.tobytes()

    def _synthesize_gwen(self, text: str, speaker: str) -> Tuple[np.ndarray, int]:
        """Synthesize using Gwen-TTS."""
        self._load_speaker_info()

        # Generation config
        gen_config = dict(
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

        # Get reference audio if available
        if speaker in self._speaker_info:
            ref_audio = self._speaker_info[speaker].get("audio_path")
            ref_text = self._speaker_info[speaker].get("text")

            wavs, sr = self._model.generate_voice_clone(
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                **gen_config,
            )
        else:
            # Default synthesis without voice cloning
            wavs, sr = self._model.generate(text=text, **gen_config)

        return wavs[0], sr

    def _resample(
        self,
        audio: np.ndarray,
        src_rate: int,
        dst_rate: int,
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        if src_rate == dst_rate:
            return audio

        num_samples = int(len(audio) * dst_rate / src_rate)
        return signal.resample(audio, num_samples)

    def get_audio_duration_ms(self, audio_bytes: bytes) -> float:
        """Calculate audio duration from PCM bytes."""
        # bytes / 2 (16-bit) / sample_rate * 1000
        return len(audio_bytes) / 2 / self.target_sample_rate * 1000


class FallbackTTS:
    """Fallback TTS using edge-tts (Microsoft Azure TTS)."""

    def __init__(self):
        self.voice = "vi-VN-HoaiMyNeural"
        self._edge_tts = None

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize using edge-tts."""
        import io
        import asyncio
        import edge_tts
        import soundfile as sf

        async def _synthesize():
            communicate = edge_tts.Communicate(text, self.voice)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            return audio_data

        # Run async in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        audio_bytes = loop.run_until_complete(_synthesize())

        # Decode audio
        audio, sr = sf.read(io.BytesIO(audio_bytes))

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        return audio.astype(np.float32), sr


# Lazy singleton
_tts_service: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    """Get or create TTS service singleton."""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service
