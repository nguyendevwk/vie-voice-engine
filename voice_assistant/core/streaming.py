"""
End-to-End Streaming Pipeline for Voice Assistant.
Optimized for real-time ASR → LLM → TTS with low latency.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncIterator, Callable, List, Optional, Awaitable, Tuple
import numpy as np

from ..config import settings
from ..utils.logging import (
    debug_log, latency, logger, log_error, log_asr_result,
    log_asr_event, log_llm_event, log_tts_event, log_vad_event,
    log_pipeline_state, log_audio_chunk,
)
from .audio import AudioPreprocessor


class StreamState(Enum):
    """Streaming pipeline state."""
    IDLE = auto()
    LISTENING = auto()      # User speaking
    TRANSCRIBING = auto()   # ASR processing
    GENERATING = auto()     # LLM generating
    SYNTHESIZING = auto()   # TTS synthesizing
    INTERRUPTED = auto()


@dataclass
class StreamMetrics:
    """Real-time metrics for pipeline."""
    utterance_start: float = 0
    first_interim: float = 0
    final_transcript: float = 0
    first_llm_token: float = 0
    first_audio_chunk: float = 0
    total_audio_ms: float = 0

    def log_summary(self):
        """Log latency summary."""
        if self.utterance_start == 0:
            return

        now = time.perf_counter()
        metrics = {
            "interim_latency": f"{(self.first_interim - self.utterance_start) * 1000:.0f}ms" if self.first_interim else "N/A",
            "asr_latency": f"{(self.final_transcript - self.utterance_start) * 1000:.0f}ms" if self.final_transcript else "N/A",
            "llm_first_token": f"{(self.first_llm_token - self.final_transcript) * 1000:.0f}ms" if self.first_llm_token and self.final_transcript else "N/A",
            "first_audio": f"{(self.first_audio_chunk - self.utterance_start) * 1000:.0f}ms" if self.first_audio_chunk else "N/A",
            "total_audio": f"{self.total_audio_ms:.0f}ms",
        }
        logger.info(f"Pipeline metrics: {metrics}")


class RealtimeASRProcessor:
    """
    Real-time ASR with interim results.

    Processes audio chunks and emits:
    - Interim transcripts (while speaking)
    - Final transcript (on speech end)
    """

    def __init__(self, model_type: str = "auto"):
        self.model_type = model_type
        self._model = None
        self._preprocessor = AudioPreprocessor()
        self._audio_buffer: List[np.ndarray] = []
        self._last_interim_text = ""

    def _ensure_loaded(self):
        """Lazy load ASR model."""
        if self._model is not None:
            return

        try:
            from .asr_gipformer import GipformerASR
            self._model = GipformerASR(device=settings.asr.device)
            logger.info("Using Gipformer ASR")
        except Exception as e:
            debug_log(f"Gipformer not available: {e}")
            # Fallback
            from .asr import FallbackASR
            self._model = FallbackASR(device=settings.asr.device)
            logger.info("Using fallback ASR (Whisper)")

    def reset(self):
        """Reset for new utterance."""
        self._audio_buffer.clear()
        self._last_interim_text = ""

    def add_audio(self, audio_chunk: bytes) -> None:
        """Add audio chunk to buffer."""
        audio = self._preprocessor.process(audio_chunk)
        self._audio_buffer.append(audio)

    async def get_interim(self) -> Optional[str]:
        """Get interim transcript of current buffer."""
        if len(self._audio_buffer) < 6:  # Minimum ~600ms
            return None

        self._ensure_loaded()

        # Concatenate audio
        audio = np.concatenate(self._audio_buffer)

        # Transcribe
        text = await asyncio.to_thread(
            self._model.transcribe_array, audio, settings.audio.sample_rate
        )

        if text and text != self._last_interim_text:
            self._last_interim_text = text
            log_asr_result(text, is_final=False)
            return text

        return None

    async def get_final(self) -> str:
        """Get final transcript and clear buffer."""
        if not self._audio_buffer:
            return ""

        self._ensure_loaded()

        # Concatenate all audio
        audio = np.concatenate(self._audio_buffer)

        # Final transcription
        text = await asyncio.to_thread(
            self._model.transcribe_array, audio, settings.audio.sample_rate
        )

        log_asr_result(text, is_final=True)
        self.reset()

        return text.strip()

    @property
    def buffer_duration_ms(self) -> float:
        """Get current buffer duration."""
        if not self._audio_buffer:
            return 0
        total_samples = sum(len(a) for a in self._audio_buffer)
        return total_samples / settings.audio.sample_rate * 1000


class RealtimeTTSProcessor:
    """
    Real-time TTS with streaming output.

    Synthesizes text chunks and streams audio.
    """

    def __init__(self):
        self._tts = None
        self._target_sr = settings.audio.sample_rate

    def _ensure_loaded(self):
        """Lazy load TTS model."""
        if self._tts is not None:
            return

        try:
            from .tts_gwen import GwenTTS
            self._tts = GwenTTS(
                model_id=settings.tts.model_id,
                device=settings.tts.device,
            )
            logger.info("Using Gwen-TTS")
        except Exception as e:
            debug_log(f"Gwen-TTS not available: {e}")
            from .tts import FallbackTTS
            self._tts = FallbackTTS()
            logger.info("Using fallback TTS (edge-tts)")

    async def synthesize(self, text: str, speaker: str = None) -> bytes:
        """Synthesize text to audio bytes."""
        if not text.strip():
            return b""

        self._ensure_loaded()

        # Synthesize
        audio, sr = await asyncio.to_thread(
            self._tts.synthesize, text, speaker
        )

        if len(audio) == 0:
            return b""

        # Resample if needed
        if sr != self._target_sr:
            from scipy import signal
            num_samples = int(len(audio) * self._target_sr / sr)
            audio = signal.resample(audio, num_samples)

        # Convert to PCM S16LE
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        return pcm.tobytes()

    @staticmethod
    def get_duration_ms(audio_bytes: bytes, sample_rate: int = 16000) -> float:
        """Calculate audio duration."""
        return len(audio_bytes) / 2 / sample_rate * 1000


class StreamingPipeline:
    """
    End-to-End Streaming Pipeline.

    Flow:
    Audio → VAD → ASR (streaming) → LLM (streaming) → TTS (chunked) → Audio

    Features:
    - Real-time interim transcripts
    - Token-by-token LLM streaming
    - Sentence-level TTS chunking
    - Interrupt handling
    - Latency tracking
    """

    def __init__(
        self,
        on_interim_transcript: Callable[[str], Awaitable[None]] = None,
        on_final_transcript: Callable[[str], Awaitable[None]] = None,
        on_llm_token: Callable[[str], Awaitable[None]] = None,
        on_llm_sentence: Callable[[str], Awaitable[None]] = None,
        on_audio_chunk: Callable[[bytes], Awaitable[None]] = None,
        on_state_change: Callable[[StreamState], Awaitable[None]] = None,
        enable_tts: bool = True,
    ):
        # Components
        self._vad = None
        self._asr = RealtimeASRProcessor()
        self._llm = None
        self._tts = RealtimeTTSProcessor() if enable_tts else None
        self._enable_tts = enable_tts

        # Callbacks
        self.on_interim_transcript = on_interim_transcript
        self.on_final_transcript = on_final_transcript
        self.on_llm_token = on_llm_token
        self.on_llm_sentence = on_llm_sentence
        self.on_audio_chunk = on_audio_chunk
        self.on_state_change = on_state_change

        # State
        self._state = StreamState.IDLE
        self._should_interrupt = False
        self._metrics = StreamMetrics()

        # Conversation
        self._history: List[dict] = []

        # Tasks
        self._interim_task: Optional[asyncio.Task] = None
        self._pipeline_task: Optional[asyncio.Task] = None

    def _ensure_vad(self):
        """Lazy load VAD."""
        if self._vad is None:
            from .vad import VADService
            self._vad = VADService()

    def _ensure_llm(self):
        """Lazy load LLM."""
        if self._llm is None:
            from .llm import LLMService
            self._llm = LLMService()

    async def _set_state(self, state: StreamState):
        """Update state and notify."""
        if state != self._state:
            self._state = state
            debug_log(f"Pipeline state: {state.name}")
            if self.on_state_change:
                await self.on_state_change(state)

    async def process_audio(self, audio_chunk: bytes) -> Optional[str]:
        """
        Process incoming audio chunk.

        Returns final transcript when speech ends, None otherwise.
        """
        self._ensure_vad()

        # Log audio chunk
        chunk_duration_ms = len(audio_chunk) / 2 / settings.audio.sample_rate * 1000
        log_audio_chunk("input", chunk_duration_ms)

        # Run VAD
        vad_result = await self._vad.process_chunk_async(audio_chunk)

        # Log VAD result
        if vad_result.event:
            log_vad_event(f"Event: {vad_result.event}", speech_prob=vad_result.confidence)

        # Speech start
        if vad_result.event == "start":
            await self._on_speech_start()

        # During speech
        if self._state == StreamState.LISTENING and vad_result.is_speech:
            self._asr.add_audio(audio_chunk)
            log_asr_event("Buffer add", latency_ms=self._asr.buffer_duration_ms())

        # Speech end
        if vad_result.event == "end" and self._state == StreamState.LISTENING:
            return await self._on_speech_end()

        # Check for interrupt during generation
        if self._state in (StreamState.GENERATING, StreamState.SYNTHESIZING):
            if vad_result.is_speech:
                await self._handle_interrupt()

        return None

    async def _on_speech_start(self):
        """Handle speech start."""
        if self._state != StreamState.IDLE:
            return

        self._metrics = StreamMetrics()
        self._metrics.utterance_start = time.perf_counter()

        self._asr.reset()
        await self._set_state(StreamState.LISTENING)
        log_pipeline_state("IDLE", "LISTENING")
        logger.info("[PIPELINE] Speech detected, start listening...")

        # Start interim ASR loop
        self._interim_task = asyncio.create_task(self._interim_loop())

    async def _interim_loop(self):
        """Background task for interim transcripts."""
        try:
            while self._state == StreamState.LISTENING:
                await asyncio.sleep(0.8)  # Every 800ms

                interim = await self._asr.get_interim()
                if interim:
                    if self._metrics.first_interim == 0:
                        self._metrics.first_interim = time.perf_counter()
                        latency_ms = (self._metrics.first_interim - self._metrics.utterance_start) * 1000
                        log_asr_event("First interim", text=interim, latency_ms=latency_ms)

                    log_asr_result(interim, is_final=False)

                    if self.on_interim_transcript:
                        await self.on_interim_transcript(interim)

        except asyncio.CancelledError:
            pass

    async def _on_speech_end(self) -> Optional[str]:
        """Handle speech end."""
        log_pipeline_state("LISTENING", "TRANSCRIBING")
        logger.info("[PIPELINE] Speech ended, transcribing...")

        # Cancel interim task
        if self._interim_task:
            self._interim_task.cancel()
            try:
                await self._interim_task
            except asyncio.CancelledError:
                pass

        # Check minimum duration
        buffer_ms = self._asr.buffer_duration_ms
        if buffer_ms < settings.pipeline.min_utterance_duration_ms:
            log_asr_event("Utterance too short", latency_ms=buffer_ms)
            await self._set_state(StreamState.IDLE)
            return None

        # Get final transcript
        await self._set_state(StreamState.TRANSCRIBING)
        transcribe_start = time.perf_counter()
        final_text = await self._asr.get_final()
        transcribe_latency = (time.perf_counter() - transcribe_start) * 1000
        self._metrics.final_transcript = time.perf_counter()

        if not final_text:
            log_asr_event("No transcript", latency_ms=transcribe_latency)
            await self._set_state(StreamState.IDLE)
            return None

        total_latency = (self._metrics.final_transcript - self._metrics.utterance_start) * 1000
        log_asr_result(final_text, is_final=True, latency_ms=total_latency)
        logger.info(f"[ASR] Final: \"{final_text}\" ({total_latency:.0f}ms)")

        if self.on_final_transcript:
            await self.on_final_transcript(final_text)

        # Start LLM → TTS pipeline
        self._pipeline_task = asyncio.create_task(
            self._run_generation(final_text)
        )

        return final_text

    async def _run_generation(self, user_text: str):
        """Run LLM → TTS generation pipeline."""
        self._ensure_llm()
        self._should_interrupt = False

        try:
            await self._set_state(StreamState.GENERATING)
            log_pipeline_state("TRANSCRIBING", "GENERATING")
            log_llm_event("Start generation")

            # Add to history
            self._history.append({"role": "user", "content": user_text})

            # Stream LLM → TTS
            full_response = ""
            sentence_buffer = ""
            first_token = True
            token_count = 0
            llm_start = time.perf_counter()

            async for token in self._llm.generate_tokens(
                user_text,
                history=[type('M', (), h)() for h in self._history[-10:]],
            ):
                if self._should_interrupt:
                    log_llm_event("Interrupted", tokens=token_count)
                    break

                token_count += 1

                if first_token:
                    self._metrics.first_llm_token = time.perf_counter()
                    first_token_latency = (self._metrics.first_llm_token - self._metrics.final_transcript) * 1000
                    log_llm_event("First token", latency_ms=first_token_latency)
                    first_token = False

                # Emit token
                if self.on_llm_token:
                    await self.on_llm_token(token)

                full_response += token
                sentence_buffer += token

                # Check for sentence boundary
                if any(d in token for d in {'.', '!', '?', '\n'}):
                    sentence = sentence_buffer.strip()
                    sentence_buffer = ""

                    if sentence and len(sentence) >= 5:
                        # Emit sentence
                        if self.on_llm_sentence:
                            await self.on_llm_sentence(sentence)

                        # Synthesize TTS
                        await self._set_state(StreamState.SYNTHESIZING)
                        log_tts_event("Synthesizing", text=sentence)
                        tts_start = time.perf_counter()

                        if self._enable_tts and self._tts:
                            audio = await self._tts.synthesize(sentence)
                        else:
                            audio = None

                        if audio:
                            tts_latency = (time.perf_counter() - tts_start) * 1000
                            audio_duration = self._tts.get_duration_ms(audio)
                            log_tts_event("Complete", duration_ms=audio_duration, latency_ms=tts_latency)

                            if self._metrics.first_audio_chunk == 0:
                                self._metrics.first_audio_chunk = time.perf_counter()
                                first_audio_latency = (self._metrics.first_audio_chunk - self._metrics.utterance_start) * 1000
                                logger.info(f"[TTS] First audio chunk: {first_audio_latency:.0f}ms from utterance start")

                            self._metrics.total_audio_ms += audio_duration

                            if self.on_audio_chunk:
                                await self.on_audio_chunk(audio)

                        await self._set_state(StreamState.GENERATING)

            # Log LLM completion
            llm_duration = (time.perf_counter() - llm_start) * 1000
            log_llm_event("Complete", tokens=token_count, latency_ms=llm_duration)

            # Flush remaining text
            if sentence_buffer.strip() and not self._should_interrupt:
                sentence = sentence_buffer.strip()
                if self.on_llm_sentence:
                    await self.on_llm_sentence(sentence)

                await self._set_state(StreamState.SYNTHESIZING)
                if self._enable_tts and self._tts:
                    audio = await self._tts.synthesize(sentence)
                else:
                    audio = None
                if audio:
                    self._metrics.total_audio_ms += self._tts.get_duration_ms(audio)
                    if self.on_audio_chunk:
                        await self.on_audio_chunk(audio)

            # Save response
            if full_response.strip():
                self._history.append({
                    "role": "assistant",
                    "content": full_response.strip(),
                })

        except Exception as e:
            log_error("pipeline", e)
        finally:
            await self._set_state(StreamState.IDLE)
            log_pipeline_state("*", "IDLE")
            self._metrics.log_summary()
            logger.info("[PIPELINE] Turn complete")

    async def _handle_interrupt(self):
        """Handle user interrupt."""
        if self._should_interrupt:
            return

        logger.info("[PIPELINE] Interrupt detected!")
        self._should_interrupt = True

        if self._pipeline_task:
            self._pipeline_task.cancel()

        await self._set_state(StreamState.INTERRUPTED)
        log_pipeline_state("*", "INTERRUPTED")

        # Reset for new speech
        self._asr.reset()
        await self._set_state(StreamState.LISTENING)
        self._interim_task = asyncio.create_task(self._interim_loop())

    def reset(self):
        """Reset pipeline state."""
        self._state = StreamState.IDLE
        self._asr.reset()
        self._history.clear()
        if self._vad:
            self._vad.reset()

    async def process_text(self, text: str) -> AsyncIterator[Tuple[str, bytes]]:
        """
        Process text input directly (skip ASR).

        Yields (sentence, audio_bytes) tuples.
        """
        self._ensure_llm()

        self._history.append({"role": "user", "content": text})

        sentence_buffer = ""
        full_response = ""

        async for token in self._llm.generate_tokens(
            text,
            history=[type('M', (), h)() for h in self._history[-10:]],
        ):
            full_response += token
            sentence_buffer += token

            if any(d in token for d in {'.', '!', '?', '\n'}):
                sentence = sentence_buffer.strip()
                sentence_buffer = ""

                if sentence:
                    audio = b""
                    if self._enable_tts and self._tts:
                        audio = await self._tts.synthesize(sentence)
                    yield sentence, audio

        # Flush
        if sentence_buffer.strip():
            sentence = sentence_buffer.strip()
            audio = b""
            if self._enable_tts and self._tts:
                audio = await self._tts.synthesize(sentence)
            yield sentence, audio

        self._history.append({"role": "assistant", "content": full_response.strip()})

    @property
    def state(self) -> StreamState:
        return self._state

    @property
    def history(self) -> List[dict]:
        return self._history
