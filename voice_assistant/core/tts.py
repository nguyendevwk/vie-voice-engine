"""
Text-to-Speech service with Qwen-TTS and edge-tts fallback.
Supports streaming chunk output for low latency.
"""

import asyncio
from typing import Optional, Tuple, AsyncIterator
import numpy as np
from scipy import signal
import os

from ..config import settings
from ..utils.logging import debug_log, latency, logger
from ..utils.text_utils import normalize_for_tts, clean_vietnamese_text, split_into_sentences


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
        self.use_onnx = cfg.use_onnx
        self.speech_rate = cfg.speech_rate
        self.streaming = cfg.streaming
        self.backend = cfg.backend

        self._model = None
        self._speaker_info = None
        
    async def synthesize_stream(self, text: str, speaker: str = None) -> AsyncIterator[bytes]:
        """
        Streaming TTS synthesis - yields audio chunks as they're generated.
        
        Args:
            text: Text to synthesize
            speaker: Speaker key (optional)
            
        Yields:
            Audio chunks (PCM S16LE bytes)
        """
        # Normalize text
        text = normalize_for_tts(text)
        text = clean_vietnamese_text(text)
        
        if not text:
            return
        
        # Split into sentences for streaming
        sentences = split_into_sentences(text, max_length=150)
        
        # Merge short sentences to avoid edge-tts failure on short texts
        merged_sentences = []
        buffer = ""
        MIN_TTS_LENGTH = 15  # Minimum characters for edge-tts
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(buffer) + len(sentence) < MIN_TTS_LENGTH:
                # Merge short segments
                buffer = (buffer + " " + sentence).strip() if buffer else sentence
            else:
                if buffer:
                    merged_sentences.append(buffer)
                buffer = sentence
        
        # Don't forget remaining buffer
        if buffer:
            merged_sentences.append(buffer)
        
        for sentence in merged_sentences:
            if sentence.strip():
                audio_bytes = await self.synthesize(sentence, speaker)
                if audio_bytes:
                    yield audio_bytes

    def _ensure_loaded(self):
        """Lazy load TTS model."""
        if self._model is not None:
            return

        logger.info(f"Loading TTS model: {self.model_id}")

        # Check backend preference
        cfg = settings.tts
        if cfg.backend == "edge-tts":
            logger.info("Using edge-tts (TTS_BACKEND=edge-tts)")
            self._model = FallbackTTS(speech_rate=self.speech_rate)
            return

        # Try Qwen-TTS (gwen-tts)
        try:
            logger.info("Attempting to load Qwen-TTS...")
            from qwen_tts import Qwen3TTSModel
            import torch
            
            # Setup HF token
            hf_token = os.getenv("HF_HUB_TOKEN")
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token

            # Check for flash attention
            try:
                import flash_attn
                attn_impl = "flash_attention_2"
                debug_log("Using Flash Attention 2")
            except ImportError:
                attn_impl = "sdpa"
                debug_log("Using SDPA attention (flash-attn not available)")

            self._model = Qwen3TTSModel.from_pretrained(
                self.model_id,
                device_map=self.device,
                dtype=torch.bfloat16,
                attn_implementation=attn_impl,
            )

            # Load speaker info
            self._load_speaker_info()

            logger.info("Qwen-TTS PyTorch model loaded successfully")
            return

        except Exception as e:
            logger.warning(f"Qwen-TTS failed to load: {e}")
            logger.warning("Falling back to edge-tts")
            self._model = FallbackTTS(speech_rate=self.speech_rate)

    def _load_speaker_info(self):
        """Load reference speaker information."""
        if self._speaker_info is not None:
            return

        import json
        from pathlib import Path

        # Try to find ref_info.json
        data_dir = Path(__file__).parent.parent / "data"
        ref_paths = [
            data_dir / "ref_info.json",
            Path(__file__).parent.parent.parent / "inferances_demo" / "tts" / "data" / "ref_info.json",
            Path.home() / ".cache" / "gwen-tts" / "ref_info.json",
        ]
        
        ref_info_path = None
        for path in ref_paths:
            if path.exists():
                ref_info_path = path
                with open(path, "r", encoding="utf-8") as f:
                    self._speaker_info = json.load(f)
                debug_log(f"Loaded speaker info from {path}")
                break

        if not self._speaker_info:
            # Create default speaker info if not found
            logger.warning("No speaker info found, creating default")
            self._speaker_info = {
                "default": {
                    "name": "Default Voice",
                    "text": "Xin chào, tôi là trợ lý ảo thông minh.",
                    "audio_path": None,
                }
            }
            return
        
        # Resolve relative audio paths
        base_dir = ref_info_path.parent.parent if ref_info_path else data_dir.parent
        
        for speaker_key, speaker_data in self._speaker_info.items():
            if isinstance(speaker_data, dict) and "audio_path" in speaker_data:
                audio_path = speaker_data.get("audio_path")
                if audio_path and not os.path.isabs(audio_path):
                    # Resolve relative path from voice_assistant directory
                    resolved_path = base_dir / audio_path
                    if resolved_path.exists():
                        speaker_data["audio_path"] = str(resolved_path)
                        debug_log(f"Resolved audio path for {speaker_key}: {resolved_path}")
                    else:
                        # Try from data directory directly
                        alt_path = data_dir / Path(audio_path).name
                        if not alt_path.exists():
                            # Try ref_audio subdirectory
                            alt_path = data_dir / "ref_audio" / Path(audio_path).name
                        if alt_path.exists():
                            speaker_data["audio_path"] = str(alt_path)
                            debug_log(f"Resolved audio path for {speaker_key}: {alt_path}")
                        else:
                            logger.warning(f"Audio file not found for {speaker_key}: {audio_path}")
        
        valid_speakers = [k for k, v in self._speaker_info.items() 
                         if isinstance(v, dict) and v.get("audio_path") and os.path.exists(v.get("audio_path", ""))]
        logger.info(f"Available speakers: {', '.join(valid_speakers)}")

    async def synthesize(self, text: str, speaker: str = None) -> bytes:
        """
        Synthesize text to speech.

        Args:
            text: Text to synthesize
            speaker: Speaker key (optional)

        Returns:
            PCM S16LE audio bytes at target sample rate
        """
        # Normalize text before synthesis
        text = normalize_for_tts(text)
        text = clean_vietnamese_text(text)
        
        if not text:
            return b""
        
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
                try:
                    audio, sr = self._synthesize_gwen(text, speaker)
                except (ValueError, FileNotFoundError) as e:
                    # Fall back to edge-tts if Gwen-TTS fails (missing ref audio, etc.)
                    logger.warning(f"Gwen-TTS failed ({e}), falling back to edge-tts")
                    if not isinstance(self._model, FallbackTTS):
                        self._model = FallbackTTS()
                    audio, sr = self._model.synthesize(text)

        # Resample if needed
        if sr != self.target_sample_rate:
            audio = self._resample(audio, sr, self.target_sample_rate)

        # Convert to PCM S16LE
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

        return pcm.tobytes()

    def _synthesize_gwen(self, text: str, speaker: str) -> Tuple[np.ndarray, int]:
        """Synthesize using Gwen-TTS (Qwen3TTS)."""
        self._load_speaker_info()

        # Generation config (optimized for Vietnamese)
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

        # Qwen3TTSModel requires reference audio for voice cloning
        ref_audio = None
        ref_text = None
        
        if speaker in self._speaker_info:
            speaker_data = self._speaker_info[speaker]
            ref_audio = speaker_data.get("audio_path")
            ref_text = speaker_data.get("text")
        
        # If no reference audio, use default or first available
        if not ref_audio:
            if self._speaker_info:
                # Use first available speaker with valid audio
                for key, speaker_data in self._speaker_info.items():
                    if key.startswith("_"):  # Skip metadata entries
                        continue
                    audio_path = speaker_data.get("audio_path")
                    if audio_path and os.path.exists(audio_path):
                        ref_audio = audio_path
                        ref_text = speaker_data.get("text")
                        logger.info(f"Using speaker: {key}")
                        break
            
        # If still no valid reference, raise error to trigger fallback
        if not ref_audio or not ref_text:
            raise ValueError("No reference audio available - Qwen-TTS requires reference audio for voice cloning")
        
        # Check if reference audio file exists
        if not os.path.exists(ref_audio):
            raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

        # Voice cloning with reference
        wavs, sr = self._model.generate_voice_clone(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            **gen_config,
        )

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

    def __init__(self, speech_rate: float = 1.5):
        # Try male voice for better stability
        self.voice = "vi-VN-NamMinhNeural"  # Male voice, more stable
        self.speech_rate = max(1.0, min(speech_rate, 1.5))  # Limit to 1.0-1.5x (edge-tts can be unstable at 2x)
        self._edge_tts = None

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize using edge-tts."""
        import io
        import asyncio
        import edge_tts
        import soundfile as sf

        if not text or not text.strip():
            return np.array([], dtype=np.float32), 16000

        # Text is already normalized by TTSService.synthesize()
        text = text.strip()
        
        # Edge-TTS fails with short texts (< 10 chars), need minimum length
        if len(text) < 10:
            logger.debug(f"Text too short for TTS: '{text}' ({len(text)} chars)")
            return np.array([], dtype=np.float32), 16000
        
        # Ensure text ends with punctuation (edge-tts works better)
        if text and text[-1] not in '.!?,;:。':
            text += '.'
        
        # Length limit
        if len(text) > 1000:
            text = text[:1000]
        
        logger.debug(f"Edge-TTS synthesizing: '{text[:50]}...' ({len(text)} chars)")

        # Calculate rate parameter for edge-tts
        # Limit to +25% for stability (HoaiMyNeural fails at +50%)
        rate_percent = int((min(self.speech_rate, 1.25) - 1.0) * 100)
        rate_str = f"+{rate_percent}%" if rate_percent > 0 else "+0%"

        async def _synthesize():
            try:
                communicate = edge_tts.Communicate(
                    text=text,
                    voice=self.voice,
                    rate=rate_str,
                )
                audio_data = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data += chunk["data"]
                
                # Retry with slower rate if failed
                if not audio_data and rate_percent > 0:
                    logger.warning(f"Retrying with slower rate (+0%)")
                    communicate = edge_tts.Communicate(
                        text=text,
                        voice=self.voice,
                        rate="+0%",
                    )
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            audio_data += chunk["data"]
                
                if not audio_data:
                    logger.warning(f"Edge-TTS returned empty audio for: '{text[:50]}'")
                
                return audio_data
            except Exception as e:
                logger.error(f"Edge-TTS synthesis error for text '{text[:50]}': {e}")
                # Try one more time with default parameters
                try:
                    logger.info("Retrying with default parameters...")
                    communicate = edge_tts.Communicate(text=text, voice=self.voice)
                    audio_data = b""
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            audio_data += chunk["data"]
                    return audio_data
                except:
                    return b""

        # Run async in sync context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context, create new loop
                import nest_asyncio
                nest_asyncio.apply()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        audio_bytes = loop.run_until_complete(_synthesize())
        
        if not audio_bytes:
            logger.warning("edge-tts returned empty audio")
            return np.array([], dtype=np.float32), 16000

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
