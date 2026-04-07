"""
Pipeline Orchestrator - Core coordinator for voice assistant.
Manages VAD → ASR → LLM → TTS flow with interrupt support.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncIterator, Callable, List, Optional, Awaitable

from ..config import settings
from ..utils.logging import debug_log, latency, logger, log_error
from .vad import VADService, get_vad_service
from .asr import ASRService, StreamingASRService, TranscriptUpdate, get_asr_service
from .llm import LLMService, Message, get_llm_service
from .tts import TTSService, get_tts_service


class PipelineState(Enum):
    """Pipeline state machine."""
    IDLE = auto()           # Waiting for speech
    LISTENING = auto()      # VAD detected speech, buffering
    PROCESSING = auto()     # Running ASR → LLM → TTS
    SPEAKING = auto()       # TTS audio playing
    INTERRUPTED = auto()    # User interrupted bot


@dataclass
class PipelineEvent:
    """Event emitted by pipeline."""
    type: str  # audio, transcript, response, control
    data: any = None


class AudioBuffer:
    """Thread-safe async audio buffer."""

    def __init__(self, max_duration_ms: int = 15000):
        self.max_chunks = max_duration_ms // settings.audio.chunk_duration_ms
        self._chunks: List[bytes] = []
        self._lock = asyncio.Lock()

    async def add(self, chunk: bytes):
        async with self._lock:
            self._chunks.append(chunk)
            # Trim if exceeds max
            if len(self._chunks) > self.max_chunks:
                self._chunks = self._chunks[-self.max_chunks:]

    async def get_all(self) -> List[bytes]:
        async with self._lock:
            return list(self._chunks)

    async def clear(self):
        async with self._lock:
            self._chunks.clear()

    @property
    def duration_ms(self) -> float:
        return len(self._chunks) * settings.audio.chunk_duration_ms

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)


class PipelineOrchestrator:
    """
    Main pipeline coordinator.

    Flow:
    1. Audio Input → VAD (detect speech)
    2. Speech detected → Buffer audio + Streaming ASR
    3. Speech ended → Final ASR → LLM (streaming)
    4. LLM chunks → TTS (per sentence)
    5. TTS audio → Output

    Features:
    - Async streaming throughout
    - Interrupt detection
    - State management
    - Latency tracking
    """

    def __init__(
        self,
        vad: VADService = None,
        asr: ASRService = None,
        llm: LLMService = None,
        tts: TTSService = None,
        on_event: Optional[Callable[[PipelineEvent], Awaitable[None]]] = None,
    ):
        self.vad = vad or get_vad_service()
        self.asr = asr or get_asr_service()
        self.llm = llm or get_llm_service()
        self.tts = tts or get_tts_service()

        self.on_event = on_event

        # State
        self._state = PipelineState.IDLE
        self._audio_buffer = AudioBuffer()
        self._streaming_asr: Optional[StreamingASRService] = None
        self._conversation_history: List[Message] = []

        # Pipeline control
        self._pipeline_task: Optional[asyncio.Task] = None
        self._should_interrupt = False
        self._mic_muted = False

        # Metrics
        self._verified_speech_count = 0
        self._interrupt_speech_count = 0

        # Config
        cfg = settings.pipeline
        self._min_utterance_duration_ms = cfg.min_utterance_duration_ms
        self._min_verified_chunks = cfg.min_verified_chunks
        self._interrupt_threshold = cfg.interrupt_threshold_chunks
        self._post_tts_buffer_ms = cfg.post_tts_buffer_ms

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def is_listening(self) -> bool:
        return self._state == PipelineState.LISTENING

    @property
    def is_speaking(self) -> bool:
        return self._state == PipelineState.SPEAKING

    async def _emit(self, event_type: str, data=None):
        """Emit event to callback."""
        if self.on_event:
            await self.on_event(PipelineEvent(type=event_type, data=data))

    async def handle_audio_chunk(self, audio_chunk: bytes) -> Optional[PipelineEvent]:
        """
        Process incoming audio chunk.

        This is the main entry point for audio data.
        """
        # Skip if mic is muted (during TTS playback)
        if self._mic_muted:
            return None

        # Run VAD
        vad_result = await self.vad.process_chunk_async(audio_chunk)

        # Handle VAD events
        if vad_result.event == "start":
            await self._on_speech_start()

        # If listening, buffer audio and run streaming ASR
        if self._state == PipelineState.LISTENING:
            await self._audio_buffer.add(audio_chunk)

            if self._streaming_asr:
                self._streaming_asr.add_audio(audio_chunk)

            if vad_result.is_speech:
                self._verified_speech_count += 1

        # Check for interrupt during bot speech
        if self._state == PipelineState.SPEAKING and vad_result.is_speech:
            self._interrupt_speech_count += 1

            if self._interrupt_speech_count >= self._interrupt_threshold:
                await self._handle_interrupt()

        # Handle speech end
        if vad_result.event == "end":
            await self._on_speech_end()

        return None

    async def _on_speech_start(self):
        """Handle VAD speech start event."""
        if self._state != PipelineState.IDLE:
            return

        debug_log("Speech started")
        self._state = PipelineState.LISTENING
        self._verified_speech_count = 0

        await self._audio_buffer.clear()

        # Initialize streaming ASR
        self._streaming_asr = StreamingASRService(
            asr_service=self.asr,
            on_transcript_update=self._on_transcript_update,
        )
        await self._streaming_asr.start_utterance()

    async def _on_transcript_update(self, update: TranscriptUpdate):
        """Handle ASR transcript updates."""
        await self._emit("transcript", {
            "text": update.text,
            "is_final": update.is_final,
        })

    async def _on_speech_end(self):
        """Handle VAD speech end event."""
        if self._state != PipelineState.LISTENING:
            return

        debug_log("Speech ended", buffer_ms=self._audio_buffer.duration_ms)

        # Guards before processing
        if not await self._should_process():
            await self._reset_listening()
            return

        # Get final transcript
        if self._streaming_asr:
            final_result = await self._streaming_asr.end_utterance()

            if final_result and final_result.text.strip():
                # Run pipeline
                self._state = PipelineState.PROCESSING
                self._pipeline_task = asyncio.create_task(
                    self._run_pipeline(final_result.text)
                )
            else:
                await self._reset_listening()

    async def _should_process(self) -> bool:
        """Check guard conditions before processing."""
        # Guard 1: Pipeline already running
        if self._pipeline_task and not self._pipeline_task.done():
            debug_log("Skip: Pipeline already running")
            return False

        # Guard 2: Too short
        if self._audio_buffer.duration_ms < self._min_utterance_duration_ms:
            debug_log("Skip: Utterance too short",
                     duration=self._audio_buffer.duration_ms)
            return False

        # Guard 3: Not enough verified speech
        if self._verified_speech_count < self._min_verified_chunks:
            debug_log("Skip: Not enough verified speech",
                     verified=self._verified_speech_count)
            return False

        return True

    async def _run_pipeline(self, user_text: str):
        """Run LLM → TTS pipeline."""
        latency.reset()
        latency.start("pipeline_total")

        try:
            # Mute mic during response
            self._mic_muted = True
            await self._emit("control", {"action": "mic_mute"})

            self._state = PipelineState.SPEAKING
            self._should_interrupt = False
            self._interrupt_speech_count = 0

            debug_log("Running pipeline", query=user_text[:50])

            # Add to history
            self._conversation_history.append(Message(role="user", content=user_text))

            # Stream LLM → TTS
            full_response = ""
            tts_duration_ms = 0

            async for sentence in self.llm.generate_response_stream(
                user_text,
                history=self._conversation_history[-10:],  # Keep last 10 messages
            ):
                if self._should_interrupt:
                    debug_log("Pipeline interrupted")
                    break

                # Emit text response
                await self._emit("response", {"text": sentence})
                full_response += sentence + " "

                # Synthesize TTS
                audio_bytes = await self.tts.synthesize(sentence)

                if audio_bytes:
                    # Track duration
                    tts_duration_ms += self.tts.get_audio_duration_ms(audio_bytes)

                    # Emit audio
                    await self._emit("audio", audio_bytes)

            # Add assistant response to history
            if full_response.strip():
                self._conversation_history.append(
                    Message(role="assistant", content=full_response.strip())
                )

            # Wait for TTS playback before unmute
            playback_wait = (tts_duration_ms / 1000) + (self._post_tts_buffer_ms / 1000)
            await asyncio.sleep(playback_wait)

        except Exception as e:
            log_error("pipeline", e)
        finally:
            # Unmute and reset
            self._mic_muted = False
            await self._emit("control", {"action": "mic_unmute"})
            self._state = PipelineState.IDLE

            latency.end("pipeline_total")
            latency.log_summary()

    async def _handle_interrupt(self):
        """Handle user interrupt during bot speech."""
        debug_log("Interrupt detected")

        self._should_interrupt = True
        self._state = PipelineState.INTERRUPTED

        # Cancel pipeline
        if self._pipeline_task:
            self._pipeline_task.cancel()

        await self._emit("control", {"action": "interrupt"})

        # Reset for new speech
        self._interrupt_speech_count = 0
        self._state = PipelineState.LISTENING
        self._mic_muted = False

        # Start new utterance
        await self._audio_buffer.clear()
        self._streaming_asr = StreamingASRService(
            asr_service=self.asr,
            on_transcript_update=self._on_transcript_update,
        )
        await self._streaming_asr.start_utterance()

    async def _reset_listening(self):
        """Reset to idle state."""
        if self._streaming_asr:
            await self._streaming_asr.cancel_utterance()
        await self._audio_buffer.clear()
        self._state = PipelineState.IDLE
        self._verified_speech_count = 0

    def reset(self):
        """Reset orchestrator state."""
        self.vad.reset()
        self._state = PipelineState.IDLE
        self._conversation_history.clear()
        self._mic_muted = False

    async def process_text(self, text: str) -> AsyncIterator[PipelineEvent]:
        """
        Process text input directly (skip ASR).
        Useful for testing or text-only mode.
        """
        self._state = PipelineState.PROCESSING

        async for sentence in self.llm.generate_response_stream(text):
            yield PipelineEvent(type="response", data={"text": sentence})

            audio = await self.tts.synthesize(sentence)
            if audio:
                yield PipelineEvent(type="audio", data=audio)

        self._state = PipelineState.IDLE


# Lazy singleton
_orchestrator: Optional[PipelineOrchestrator] = None


def get_orchestrator() -> PipelineOrchestrator:
    """Get or create pipeline orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = PipelineOrchestrator()
    return _orchestrator
