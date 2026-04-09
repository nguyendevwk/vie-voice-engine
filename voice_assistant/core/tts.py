"""
Vietnamese Text-to-Speech Service.

Supports multiple backends with automatic fallback:
1. VieNeu-TTS (Turbo) - Fast, CPU-friendly, offline
2. Qwen-TTS - High quality, GPU, voice cloning
3. Edge-TTS - Microsoft Azure, online fallback

Usage:
    >>> tts = get_tts_service()
    >>> audio = await tts.synthesize("Xin chào Việt Nam")
"""

import asyncio
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, AsyncIterator, List, Dict, Any
import numpy as np
from scipy import signal

from ..config import settings
from ..utils.logging import debug_log, latency, logger


# =============================================================================
# Text Preprocessing
# =============================================================================

def prepare_text_for_tts(text: str) -> str:
    """
    Clean and prepare text for TTS synthesis.
    Handles markdown, special chars, minimum length.
    
    Optimized for VieNeu-TTS-v2 Turbo.
    """
    from ..utils.text_utils import normalize_for_tts
    return normalize_for_tts(text)


def split_sentences(text: str, max_length: int = 150) -> List[str]:
    """Split text into sentences for streaming TTS."""
    if not text:
        return []

    # Split on sentence endings
    parts = re.split(r'([.!?]+\s*)', text)

    sentences = []
    current = ""

    for part in parts:
        current += part
        if re.search(r'[.!?]\s*$', current) and len(current.strip()) > 10:
            sentences.append(current.strip())
            current = ""

    if current.strip():
        sentences.append(current.strip())

    # Merge short sentences
    merged = []
    buffer = ""
    for s in sentences:
        if len(buffer) + len(s) < 20:
            buffer = (buffer + " " + s).strip() if buffer else s
        else:
            if buffer:
                merged.append(buffer)
            buffer = s
    if buffer:
        merged.append(buffer)

    return merged


# =============================================================================
# Base TTS Provider
# =============================================================================

@dataclass
class TTSResult:
    """Result from TTS synthesis."""
    audio: np.ndarray
    sample_rate: int
    duration_ms: float = 0

    def to_pcm16(self) -> bytes:
        """Convert to PCM S16LE bytes."""
        pcm = (np.clip(self.audio, -1.0, 1.0) * 32767).astype(np.int16)
        return pcm.tobytes()


class BaseTTSProvider(ABC):
    """Abstract base class for TTS providers."""

    name: str = "base"
    supports_cloning: bool = False
    requires_gpu: bool = False
    is_online: bool = False

    @abstractmethod
    def synthesize(self, text: str, **kwargs) -> TTSResult:
        """Synthesize text to audio."""
        pass

    def is_available(self) -> bool:
        """Check if provider is available."""
        return True

    def list_voices(self) -> List[str]:
        """List available voices."""
        return ["default"]


# =============================================================================
# VieNeu-TTS Provider (CPU-friendly, offline)
# =============================================================================

class VieNeuTTSProvider(BaseTTSProvider):
    """
    VieNeu-TTS provider for fast, CPU-friendly synthesis.

    🚀 Supports TWO modes:
    1. LOCAL (turbo): VieNeu-TTS-v2 Turbo - runs locally on CPU
    2. REMOTE: Lightweight client connecting to VieNeu server

    LOCAL mode (vieneu):
    - Runs locally, no network needed
    - CPU-friendly, ~300MB memory
    - Latency: 100-500ms

    REMOTE mode (vieneu_remote):
    - Lightweight client (~50MB), server does heavy lifting
    - Optimal for Docker/low-end hardware
    - Latency: depends on network (typically 200-800ms)
    - Requires: VIENEU_REMOTE_API_BASE, VIENEU_REMOTE_MODEL_ID

    Install: pip install vieneu
    """

    name = "vieneu"
    supports_cloning = True
    requires_gpu = False
    is_online = False

    def __init__(
        self,
        voice: str = None,
        mode: str = None,  # turbo, standard, fast, turbo_gpu, remote
    ):
        self.voice = voice  # Preset voice ID
        self.mode = mode or settings.tts.vieneu_mode
        self._tts = None
        self._voice_data = None
        self._load_error: Optional[str] = None  # Store load errors

    def _ensure_loaded(self):
        """Lazy load VieNeu-TTS."""
        if self._tts is not None:
            return

        try:
            from vieneu import Vieneu

            # Check if using remote mode
            if self.mode == "remote" or settings.tts.backend == "vieneu_remote":
                # Remote mode - lightweight client
                api_base = settings.tts.vieneu_remote_api_base
                model_id = settings.tts.vieneu_remote_model_id
                logger.info(f"🚀 Loading VieNeu-TTS Remote mode: {api_base}")
                self._tts = Vieneu(
                    mode='remote',
                    api_base=api_base,
                    model_name=model_id
                )
                self.is_online = True
                logger.info(f"✅ VieNeu-TTS Remote loaded (model={model_id})")
            else:
                # Local mode - VieNeu-TTS-v2 Turbo
                logger.info(f"🚀 Loading VieNeu-TTS Local (mode={self.mode})...")
                logger.info("🚀 Using VieNeu-TTS-v2 Turbo - Optimized for edge devices!")
                
                # Load with appropriate mode
                if self.mode == "turbo":
                    # VieNeu-TTS-v2 Turbo with custom model paths
                    self._tts = Vieneu(
                        mode=self.mode,
                        backbone_repo=settings.tts.vieneu_model_backbone,
                        decoder_repo=settings.tts.vieneu_model_decoder,
                        encoder_repo=settings.tts.vieneu_model_encoder,
                    )
                else:
                    # Other modes use default models
                    self._tts = Vieneu(mode=self.mode)

                logger.info(f"✅ VieNeu-TTS-v2 Turbo loaded successfully (mode={self.mode})")

            # Get preset voice if specified
            if self.voice:
                try:
                    self._voice_data = self._tts.get_preset_voice(self.voice)
                    logger.info(f"Using VieNeu voice: {self.voice}")
                except Exception as e:
                    logger.warning(f"Voice '{self.voice}' not found: {e}")

        except ImportError as e:
            error_msg = str(e)
            self._load_error = error_msg
            
            # Provide helpful error messages
            if "remote" in self.mode.lower() or "api" in error_msg.lower():
                logger.error(
                    f"VieNeu-TTS Remote mode failed: {error_msg}. "
                    f"Check VIENEU_REMOTE_API_BASE is accessible"
                )
            elif "lmdeploy" in error_msg.lower():
                logger.error(
                    f"VieNeu-TTS '{self.mode}' mode requires lmdeploy. "
                    f"Install with: pip install vieneu[gpu] or use 'turbo' mode instead"
                )
            elif "torch" in error_msg.lower():
                logger.error(
                    f"VieNeu-TTS '{self.mode}' mode requires PyTorch. "
                    f"Install with: pip install torch or use 'turbo' mode instead"
                )
            else:
                logger.error(f"Failed to load VieNeu-TTS: {error_msg}")
            raise

        except Exception as e:
            self._load_error = str(e)
            logger.error(f"Failed to load VieNeu-TTS (mode={self.mode}): {e}")
            raise

    def is_available(self) -> bool:
        """Check if provider is available."""
        try:
            import vieneu
            # Try to load to check availability
            if self._tts is None and self._load_error is None:
                # Haven't tried loading yet, attempt it
                try:
                    self._ensure_loaded()
                    return True
                except Exception:
                    return False
            return self._tts is not None
        except ImportError:
            return False

    def synthesize(self, text: str, voice: str = None, **kwargs) -> TTSResult:
        """Synthesize using VieNeu-TTS."""
        self._ensure_loaded()

        text = prepare_text_for_tts(text)
        if not text:
            return TTSResult(np.array([], dtype=np.float32), 24000)

        # Get voice data
        voice_data = self._voice_data
        if voice and voice != self.voice:
            try:
                voice_data = self._tts.get_preset_voice(voice)
            except:
                pass

        # Support reference audio for zero-shot voice cloning
        ref_audio = kwargs.get("ref_audio")
        ref_text = kwargs.get("ref_text")

        with latency.track("tts_vieneu"):
            if ref_audio and ref_text:
                # Zero-shot voice cloning
                audio = self._tts.infer(
                    text=text,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )
            elif voice_data:
                # Use preset voice
                audio = self._tts.infer(text=text, voice=voice_data)
            else:
                # Default voice
                audio = self._tts.infer(text=text)

        # VieNeu outputs at 24kHz
        return TTSResult(
            audio=np.array(audio, dtype=np.float32),
            sample_rate=24000,
            duration_ms=len(audio) / 24000 * 1000
        )

    def clone_voice(self, ref_audio: str) -> Any:
        """Clone voice from reference audio."""
        self._ensure_loaded()
        return self._tts.encode_voice(ref_audio)

    def list_voices(self) -> List[str]:
        """List preset voices."""
        self._ensure_loaded()
        try:
            voices = self._tts.list_preset_voices()
            return [v[1] for v in voices]  # Return voice IDs
        except:
            return ["default"]


# =============================================================================
# Qwen-TTS Provider (GPU, high quality)
# =============================================================================

class QwenTTSProvider(BaseTTSProvider):
    """
    Qwen-TTS provider for high-quality voice cloning.

    Requires:
    - CUDA GPU with 4-6GB+ VRAM
    - qwen_tts package

    Features:
    - High quality synthesis
    - Voice cloning with reference audio
    """

    name = "qwen"
    supports_cloning = True
    requires_gpu = True
    is_online = False

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
        """Lazy load Qwen-TTS."""
        if self._model is not None:
            return

        import torch
        from qwen_tts import Qwen3TTSModel

        logger.info(f"Loading Qwen-TTS: {self.model_id}")

        # Check attention implementation
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"

        self._model = Qwen3TTSModel.from_pretrained(
            self.model_id,
            device_map=self.device,
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )

        self._load_speaker_info()
        logger.info("Qwen-TTS loaded successfully")

    def _load_speaker_info(self):
        """Load reference speaker information."""
        import json
        from pathlib import Path

        data_dir = Path(__file__).parent.parent / "data"
        ref_path = data_dir / "ref_info.json"

        if ref_path.exists():
            with open(ref_path, "r", encoding="utf-8") as f:
                self._speaker_info = json.load(f)

            # Resolve relative audio paths
            for key, data in self._speaker_info.items():
                if isinstance(data, dict) and "audio_path" in data:
                    audio_path = data.get("audio_path")
                    if audio_path and not os.path.isabs(audio_path):
                        resolved = data_dir.parent / audio_path
                        if resolved.exists():
                            data["audio_path"] = str(resolved)

            valid = [k for k, v in self._speaker_info.items()
                    if isinstance(v, dict) and v.get("audio_path") and os.path.exists(v.get("audio_path", ""))]
            logger.info(f"Available Qwen speakers: {', '.join(valid)}")
        else:
            self._speaker_info = {}

    def is_available(self) -> bool:
        try:
            import torch
            import qwen_tts
            return torch.cuda.is_available()
        except ImportError:
            return False

    def synthesize(self, text: str, speaker: str = None, **kwargs) -> TTSResult:
        """Synthesize using Qwen-TTS with voice cloning."""
        self._ensure_loaded()

        text = prepare_text_for_tts(text)
        if not text:
            return TTSResult(np.array([], dtype=np.float32), 24000)

        # Get reference audio for voice cloning
        ref_audio = kwargs.get("ref_audio")
        ref_text = kwargs.get("ref_text")

        if not ref_audio and speaker and self._speaker_info:
            if speaker in self._speaker_info:
                data = self._speaker_info[speaker]
                ref_audio = data.get("audio_path")
                ref_text = data.get("text")

        # Synthesize
        with latency.track("tts_qwen"):
            if ref_audio and ref_text and os.path.exists(ref_audio):
                wavs, sr = self._model.generate_voice_clone(
                    text=text,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    **self.GENERATION_CONFIG,
                )
            else:
                raise ValueError("Qwen-TTS requires reference audio for voice cloning")

        return TTSResult(
            audio=wavs[0].astype(np.float32),
            sample_rate=sr,
            duration_ms=len(wavs[0]) / sr * 1000
        )

    def list_voices(self) -> List[str]:
        """List available speakers."""
        self._ensure_loaded()
        return [k for k in self._speaker_info.keys() if not k.startswith("_")]


# =============================================================================
# Edge-TTS Provider (Online fallback)
# =============================================================================

class EdgeTTSProvider(BaseTTSProvider):
    """
    Edge-TTS provider using Microsoft Azure TTS.

    Features:
    - Free, no API key needed
    - Good quality Vietnamese voices
    - Online only

    Install: pip install edge-tts
    """

    name = "edge"
    supports_cloning = False
    requires_gpu = False
    is_online = True

    # Vietnamese voices
    VOICES = {
        "male": "vi-VN-NamMinhNeural",
        "female": "vi-VN-HoaiMyNeural",
    }

    def __init__(self, voice: str = "male", speech_rate: float = 1.0):
        self.voice = self.VOICES.get(voice, voice)
        self.speech_rate = max(1.0, min(speech_rate, 1.25))  # Limit for stability

    def is_available(self) -> bool:
        try:
            import edge_tts
            return True
        except ImportError:
            return False

    def synthesize(self, text: str, **kwargs) -> TTSResult:
        """Synthesize using edge-tts."""
        import io
        import asyncio
        import edge_tts
        import soundfile as sf

        text = prepare_text_for_tts(text)
        if not text or len(text) < 10:
            return TTSResult(np.array([], dtype=np.float32), 16000)

        # Calculate rate
        rate_percent = int((self.speech_rate - 1.0) * 100)
        rate_str = f"+{rate_percent}%" if rate_percent > 0 else "+0%"

        async def _synthesize():
            try:
                communicate = edge_tts.Communicate(
                    text=text,
                    voice=kwargs.get("voice", self.voice),
                    rate=rate_str,
                )
                audio_data = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data += chunk["data"]
                return audio_data
            except Exception as e:
                logger.error(f"Edge-TTS error: {e}")
                return b""

        # Run async
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        with latency.track("tts_edge"):
            audio_bytes = loop.run_until_complete(_synthesize())

        if not audio_bytes:
            return TTSResult(np.array([], dtype=np.float32), 16000)

        # Decode
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        return TTSResult(
            audio=audio.astype(np.float32),
            sample_rate=sr,
            duration_ms=len(audio) / sr * 1000
        )

    def list_voices(self) -> List[str]:
        return list(self.VOICES.keys())


# =============================================================================
# Main TTS Service (with fallback chain)
# =============================================================================

class TTSService:
    """
    Main TTS service with automatic provider selection and fallback.

    Provider priority:
    1. VieNeu-TTS (if installed, CPU-friendly)
    2. Qwen-TTS (if GPU available)
    3. Edge-TTS (online fallback)

    Usage:
        >>> tts = TTSService()
        >>> audio = await tts.synthesize("Xin chào")
        >>> # Or with specific provider
        >>> tts = TTSService(backend="vieneu")
    """

    def __init__(
        self,
        backend: str = "auto",  # auto, vieneu, qwen, edge
        default_speaker: str = None,
        speech_rate: float = 1.0,
        target_sample_rate: int = 16000,
    ):
        self.backend = backend
        self.default_speaker = default_speaker or settings.tts.default_speaker
        self.speech_rate = speech_rate
        self.target_sample_rate = target_sample_rate

        self._provider: Optional[BaseTTSProvider] = None
        self._fallback: Optional[BaseTTSProvider] = None

    def _get_provider(self) -> BaseTTSProvider:
        """Get or create TTS provider."""
        if self._provider is not None:
            return self._provider

        backend = self.backend if self.backend != "auto" else settings.tts.backend

        # Try requested backend
        if backend == "vieneu":
            try:
                provider = VieNeuTTSProvider()
                if provider.is_available():
                    self._provider = provider
                    logger.info(f"Using VieNeu-TTS Local (mode={settings.tts.vieneu_mode})")
                    return self._provider
            except Exception as e:
                logger.warning(f"VieNeu-TTS Local not available: {e}")

        elif backend == "vieneu_remote":
            try:
                provider = VieNeuTTSProvider(mode="remote")
                if provider.is_available():
                    self._provider = provider
                    self._fallback = EdgeTTSProvider(speech_rate=self.speech_rate)
                    logger.info(f"🚀 Using VieNeu-TTS Remote: {settings.tts.vieneu_remote_api_base}")
                    return self._provider
            except Exception as e:
                logger.warning(f"VieNeu-TTS Remote not available: {e}")
                logger.info("Falling back to Edge-TTS")

        elif backend == "qwen":
            try:
                provider = QwenTTSProvider()
                if provider.is_available():
                    self._provider = provider
                    logger.info("Using Qwen-TTS")
                    return self._provider
            except Exception as e:
                logger.warning(f"Qwen-TTS not available: {e}")

        elif backend == "edge":
            self._provider = EdgeTTSProvider(speech_rate=self.speech_rate)
            logger.info("Using Edge-TTS")
            return self._provider

        # Auto mode: try providers in order for optimal latency
        if backend == "auto":
            # 1. VieNeu-TTS Remote (lightest client, optimal for Docker)
            try:
                vieneu_remote = VieNeuTTSProvider(mode="remote")
                if vieneu_remote.is_available():
                    self._provider = vieneu_remote
                    self._fallback = EdgeTTSProvider(speech_rate=self.speech_rate)
                    logger.info(f"🚀 Using VieNeu-TTS Remote (auto): {settings.tts.vieneu_remote_api_base}")
                    return self._provider
            except Exception:
                pass

            # 2. VieNeu-TTS Local v2 Turbo (CPU-friendly)
            try:
                vieneu = VieNeuTTSProvider()
                if vieneu.is_available():
                    self._provider = vieneu
                    self._fallback = EdgeTTSProvider(speech_rate=self.speech_rate)
                    logger.info(f"Using VieNeu-TTS Local v2 Turbo (auto)")
                    return self._provider
            except Exception:
                pass

            # 3. Qwen-TTS (GPU)
            try:
                qwen = QwenTTSProvider()
                if qwen.is_available():
                    self._provider = qwen
                    self._fallback = EdgeTTSProvider(speech_rate=self.speech_rate)
                    logger.info("Using Qwen-TTS (auto)")
                    return self._provider
            except Exception:
                pass

        # 4. Edge-TTS fallback
        self._provider = EdgeTTSProvider(speech_rate=self.speech_rate)
        logger.info("Using Edge-TTS (fallback)")
        return self._provider

    async def synthesize(self, text: str, speaker: str = None) -> bytes:
        """
        Synthesize text to PCM audio bytes.

        Args:
            text: Text to synthesize
            speaker: Speaker/voice ID (optional)

        Returns:
            PCM S16LE audio bytes at target sample rate
        """
        from ..utils.text_utils import normalize_for_tts

        # Normalize text before synthesis
        text = normalize_for_tts(text)
        if not text:
            return b""

        provider = self._get_provider()
        speaker = speaker or self.default_speaker

        try:
            result = await asyncio.to_thread(
                provider.synthesize, text, speaker=speaker
            )
        except Exception as e:
            logger.warning(f"{provider.name} failed: {e}")

            # Try fallback
            if self._fallback:
                logger.info(f"Falling back to {self._fallback.name}")
                result = await asyncio.to_thread(
                    self._fallback.synthesize, text
                )
            else:
                return b""

        if result.audio.size == 0:
            return b""

        # Resample if needed
        audio = result.audio
        if result.sample_rate != self.target_sample_rate:
            audio = self._resample(audio, result.sample_rate, self.target_sample_rate)

        # Convert to PCM
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        return pcm.tobytes()

    async def synthesize_stream(self, text: str, speaker: str = None) -> AsyncIterator[bytes]:
        """Stream TTS synthesis sentence by sentence."""
        sentences = split_sentences(text)

        for sentence in sentences:
            audio = await self.synthesize(sentence, speaker)
            if audio:
                yield audio

    def _resample(self, audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """Resample audio."""
        if src_rate == dst_rate:
            return audio
        num_samples = int(len(audio) * dst_rate / src_rate)
        return signal.resample(audio, num_samples)

    def get_audio_duration_ms(self, audio_bytes: bytes) -> float:
        """Calculate audio duration from PCM bytes."""
        return len(audio_bytes) / 2 / self.target_sample_rate * 1000

    def list_voices(self) -> List[str]:
        """List available voices for current provider."""
        return self._get_provider().list_voices()


# =============================================================================
# Singleton
# =============================================================================

_tts_service: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    """Get or create TTS service singleton."""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService(
            backend=settings.tts.backend,
            default_speaker=settings.tts.default_speaker,
            speech_rate=settings.tts.speech_rate,
            target_sample_rate=settings.tts.target_sample_rate,
        )
    return _tts_service
