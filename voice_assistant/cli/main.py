"""
CLI interface for Voice Assistant.
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
    uv run python -m voice_assistant.cli.main
    uv run python -m voice_assistant.cli.main --text-only
    uv run python -m voice_assistant.cli.main --no-tts
        """,
    )

    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Text input mode (no microphone)",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable TTS output",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


class CLIAssistant:
    """CLI-based voice assistant."""

    def __init__(self, text_only: bool = False, no_tts: bool = False):
        self.text_only = text_only
        self.no_tts = no_tts
        self.orchestrator: Optional[PipelineOrchestrator] = None
        self._current_response = ""

    async def setup(self):
        """Initialize pipeline."""
        logger.info("Initializing Voice Assistant...")
        
        self.orchestrator = PipelineOrchestrator(
            on_event=self._on_event
        )
        
        logger.info("Voice Assistant ready!")

    async def _on_event(self, event: PipelineEvent):
        """Handle pipeline events."""
        if event.type == "transcript":
            text = event.data.get("text", "")
            is_final = event.data.get("is_final", False)
            
            if is_final:
                print(f"\n🎤 You: {text}")
                self._current_response = ""
                print("🤖 ", end="", flush=True)
            else:
                print(f"\r🎤 {text}...", end="", flush=True)
        
        elif event.type == "response":
            text = event.data.get("text", "")
            is_final = event.data.get("is_final", False)
            
            self._current_response += text + " "
            print(text, end=" " if not is_final else "\n", flush=True)
        
        elif event.type == "audio":
            # Play audio if needed
            if not self.no_tts:
                # Audio playback would go here
                pass

    async def run_text_mode(self):
        """Run in text input mode."""
        print("\n" + "="*50)
        print("🇻🇳 Vietnamese Voice Assistant - Text Mode")
        print("="*50)
        print("Type your question (or 'exit' to quit)")
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
                
                # Process text through pipeline
                async for event in self.orchestrator.process_text(user_input):
                    # Events are handled by callback
                    pass
                
                print()  # Newline after response

            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\nError: {e}")

    async def run_voice_mode(self):
        """Run in voice input mode."""
        print("\n" + "="*50)
        print("🇻🇳 Vietnamese Voice Assistant - Voice Mode")
        print("="*50)
        print("Speak into your microphone (Ctrl+C to quit)")
        print()

        try:
            import sounddevice as sd
            import numpy as np
        except (ImportError, OSError) as e:
            print("\n❌ Error: Audio system not available")
            print()
            if "PortAudio" in str(e):
                print("PortAudio library not found. Please install:")
                print()
                print("  Ubuntu/Debian:")
                print("    sudo apt-get install portaudio19-dev")
                print()
                print("  Fedora/RHEL:")
                print("    sudo dnf install portaudio-devel")
                print()
                print("  macOS:")
                print("    brew install portaudio")
                print()
                print("Then reinstall sounddevice:")
                print("    pip install --force-reinstall sounddevice")
            else:
                print("sounddevice not installed")
                print("Install with: pip install sounddevice")
            print()
            print("💡 Try text mode instead: --text-only")
            return

        # Audio stream parameters
        sample_rate = settings.audio.sample_rate
        chunk_size = settings.audio.chunk_samples

        async def audio_callback(indata, frames, time_info, status):
            """Process audio chunks."""
            if status:
                logger.warning(f"Audio status: {status}")
            
            # Convert to bytes
            audio_bytes = indata.tobytes()
            await self.orchestrator.handle_audio_chunk(audio_bytes)

        # Start audio stream
        try:
            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype='int16',
                blocksize=chunk_size,
                callback=lambda *args: asyncio.create_task(audio_callback(*args))
            ):
                print("🎤 Listening...")
                while True:
                    await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")

    async def run(self):
        """Run assistant."""
        await self.setup()
        
        if self.text_only:
            await self.run_text_mode()
        else:
            await self.run_voice_mode()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else settings.log_level
    setup_logging(level=log_level)
    
    # Update settings
    if args.debug:
        settings.debug = True
    
    # Run assistant
    assistant = CLIAssistant(
        text_only=args.text_only,
        no_tts=args.no_tts,
    )
    
    try:
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
