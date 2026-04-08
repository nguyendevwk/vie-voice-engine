"""
CLI interface for Voice Assistant.
Interactive voice assistant with real-time streaming.
"""

import argparse
import asyncio
import sys
from typing import Optional

from ..config import settings
from ..utils.logging import logger, setup_logging
from ..core.streaming import StreamingPipeline, StreamState


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Vietnamese Voice Assistant - CLI Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m voice_assistant.cli.main
    python -m voice_assistant.cli.main --text-only
    python -m voice_assistant.cli.main --debug
    python -m voice_assistant.cli.main --streaming
        """,
    )

    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Text input mode (no microphone)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Use streaming pipeline (default)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for models (cpu/cuda/cuda:0)",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable TTS output",
    )

    return parser.parse_args()


class CLIAssistant:
    """CLI-based voice assistant with real-time streaming."""

    def __init__(self, text_only: bool = False, no_tts: bool = False):
        self.text_only = text_only
        self.no_tts = no_tts
        self.pipeline: Optional[StreamingPipeline] = None
        self._audio_player = None
        self._running = False
        self._current_response = ""

    async def setup(self):
        """Initialize components."""
        logger.info("Initializing Voice Assistant (Streaming Mode)...")

        # Create streaming pipeline with callbacks
        self.pipeline = StreamingPipeline(
            on_interim_transcript=self._on_interim,
            on_final_transcript=self._on_final_transcript,
            on_llm_token=self._on_llm_token,
            on_llm_sentence=self._on_llm_sentence,
            on_audio_chunk=self._on_audio_chunk,
            on_state_change=self._on_state_change,
            enable_tts=not self.no_tts,
        )

        if not self.no_tts:
            try:
                import sounddevice as sd
                self._audio_player = AudioPlayer()
                logger.info("Audio output initialized")
            except ImportError:
                logger.warning("sounddevice not installed, audio playback disabled")
                self.no_tts = True

        logger.info("Voice Assistant ready!")

    # Streaming callbacks
    async def _on_interim(self, text: str):
        """Handle interim transcript."""
        print(f"\r🎤 {text}...", end="", flush=True)

    async def _on_final_transcript(self, text: str):
        """Handle final transcript."""
        print(f"\n🎤 You: {text}")
        self._current_response = ""
        print("🤖 ", end="", flush=True)

    async def _on_llm_token(self, token: str):
        """Handle LLM token (for display)."""
        if settings.debug:
            print(token, end="", flush=True)

    async def _on_llm_sentence(self, sentence: str):
        """Handle complete sentence."""
        self._current_response += sentence + " "
        print(sentence, end=" ", flush=True)

    async def _on_audio_chunk(self, audio: bytes):
        """Handle TTS audio chunk."""
        if self._audio_player and not self.no_tts:
            await self._audio_player.play(audio)

    async def _on_state_change(self, state: StreamState):
        """Handle state change."""
        if state == StreamState.IDLE and self._current_response:
            print()  # Newline after response
        elif state == StreamState.INTERRUPTED:
            print("\n⚡ [Interrupted]")

    async def run_text_mode(self):
        """Run in text input mode."""
        print("\n" + "="*50)
        print("🇻🇳 Vietnamese Voice Assistant - Text Mode")
        print("="*50)
        print("Nhập câu hỏi của bạn (hoặc 'exit' để thoát)")
        print()

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit", "q"):
                    print("Goodbye! 👋")
                    break

                print("Bot: ", end="", flush=True)

                async for sentence, audio in self.pipeline.process_text(user_input):
                    print(sentence, end=" ", flush=True)
                    if self._audio_player and audio:
                        await self._audio_player.play(audio)

                print()

            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break
            except EOFError:
                break

    async def run_voice_mode(self):
        """Run in voice input mode with real-time streaming."""
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            logger.error("sounddevice not installed. Run: pip install sounddevice")
            return

        print("\n" + "="*50)
        print("🇻🇳 Vietnamese Voice Assistant - Streaming Mode")
        print("="*50)
        print("Nói vào microphone (Ctrl+C để thoát)")
        print("Pipeline: Audio → VAD → ASR → LLM → TTS")
        print()

        self._running = True
        audio_queue = asyncio.Queue()

        # Audio processing task
        async def process_audio():
            while self._running:
                try:
                    audio_bytes = await asyncio.wait_for(
                        audio_queue.get(), timeout=0.1
                    )
                    await self.pipeline.process_audio(audio_bytes)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Audio processing error: {e}")

        # Start processing task
        process_task = asyncio.create_task(process_audio())

        try:
            # Audio input callback (runs in separate thread)
            def audio_callback(indata, frames, time_info, status):
                if status:
                    logger.warning(f"Audio status: {status}")
                if self._running:
                    # Convert to PCM S16LE bytes
                    audio_bytes = (indata[:, 0] * 32767).astype(np.int16).tobytes()
                    try:
                        audio_queue.put_nowait(audio_bytes)
                    except asyncio.QueueFull:
                        pass  # Drop if queue full

            # Start audio stream
            stream = sd.InputStream(
                samplerate=settings.audio.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=settings.audio.chunk_samples,
                callback=audio_callback,
            )

            stream.start()
            logger.info("🎙️ Listening... (Ctrl+C to stop)")
            print()

            # Keep running
            while self._running:
                await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            self._running = False
            process_task.cancel()
            try:
                await process_task
            except asyncio.CancelledError:
                pass
            if stream:
                stream.stop()
                stream.close()

    async def run(self):
        """Main entry point."""
        await self.setup()

        if self.text_only:
            await self.run_text_mode()
        else:
            await self.run_voice_mode()


class AudioPlayer:
    """Simple audio player using sounddevice."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._queue = asyncio.Queue()
        self._playing = False

    async def play(self, audio_bytes: bytes):
        """Play audio bytes."""
        import numpy as np
        import sounddevice as sd

        # Convert bytes to float32
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Play (non-blocking)
        sd.play(audio, self.sample_rate)


def main():
    """CLI entry point."""
    args = parse_args()

    # Configure logging
    if args.debug:
        settings.debug = True
        setup_logging("DEBUG")

    # Configure device
    if args.device != "auto":
        settings.asr.device = args.device
        settings.tts.device = args.device

    # Run assistant
    assistant = CLIAssistant(
        text_only=args.text_only,
        no_tts=args.no_tts,
    )

    try:
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
