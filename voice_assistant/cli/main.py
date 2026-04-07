"""
CLI interface for Voice Assistant.
Interactive voice assistant with microphone input.
"""

import argparse
import asyncio
import sys
from typing import Optional

from ..config import settings
from ..utils.logging import logger, setup_logging
from ..core.pipeline import PipelineOrchestrator, PipelineEvent


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
        """,
    )

    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Text input mode (no microphone)",
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
    """CLI-based voice assistant."""

    def __init__(self, text_only: bool = False, no_tts: bool = False):
        self.text_only = text_only
        self.no_tts = no_tts
        self.orchestrator: Optional[PipelineOrchestrator] = None
        self._audio_player = None
        self._running = False

    async def setup(self):
        """Initialize components."""
        logger.info("Initializing Voice Assistant...")

        self.orchestrator = PipelineOrchestrator(
            on_event=self._handle_event,
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

    async def _handle_event(self, event: PipelineEvent):
        """Handle pipeline events."""
        if event.type == "transcript":
            data = event.data
            if data.get("is_final"):
                print(f"\n🎤 You: {data['text']}")
            else:
                print(f"\r🎤 {data['text']}...", end="", flush=True)

        elif event.type == "response":
            print(f"🤖 {event.data['text']}", end=" ", flush=True)

        elif event.type == "audio":
            if self._audio_player:
                await self._audio_player.play(event.data)

        elif event.type == "control":
            action = event.data.get("action")
            if action == "interrupt":
                print("\n⚡ [Interrupted]")
            elif action == "mic_mute":
                pass  # Silent
            elif action == "mic_unmute":
                print()  # Newline after response

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

                async for event in self.orchestrator.process_text(user_input):
                    if event.type == "response":
                        print(event.data["text"], end=" ", flush=True)
                    elif event.type == "audio" and self._audio_player:
                        await self._audio_player.play(event.data)

                print()

            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break
            except EOFError:
                break

    async def run_voice_mode(self):
        """Run in voice input mode."""
        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice not installed. Run: pip install sounddevice")
            return

        print("\n" + "="*50)
        print("🇻🇳 Vietnamese Voice Assistant - Voice Mode")
        print("="*50)
        print("Nói vào microphone (Ctrl+C để thoát)")
        print()

        self._running = True
        stream = None

        try:
            # Audio input callback
            def audio_callback(indata, frames, time, status):
                if status:
                    logger.warning(f"Audio status: {status}")
                if self._running:
                    # Convert to bytes and queue
                    audio_bytes = (indata[:, 0] * 32767).astype("int16").tobytes()
                    asyncio.create_task(
                        self.orchestrator.handle_audio_chunk(audio_bytes)
                    )

            # Start audio stream
            stream = sd.InputStream(
                samplerate=settings.audio.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=settings.audio.chunk_samples,
                callback=audio_callback,
            )

            stream.start()
            logger.info("Listening... (Ctrl+C to stop)")

            # Keep running
            while self._running:
                await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            self._running = False
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
