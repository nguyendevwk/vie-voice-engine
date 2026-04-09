"""
Example: Using VieNeu-TTS Remote Mode

This example demonstrates how to use VieNeu-TTS in remote mode,
which is lightweight and doesn't require local GPU/CPU inference.

Prerequisites:
    - Access to a VieNeu server (http://your-server-ip:23333/v1)
    - vieneu package installed: pip install vieneu

Usage:
    uv run python examples/vieneu_remote_example.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vieneu import Vieneu


def example_remote_tts():
    """Example: Basic remote TTS usage."""
    print("="*70)
    print("Example 1: Basic Remote TTS")
    print("="*70)

    # Configuration
    REMOTE_API_BASE = 'http://localhost:23333/v1'  # Or your server
    REMOTE_MODEL_ID = "pnnbao-ump/VieNeu-TTS-0.3B"

    # Initialization (LIGHTWEIGHT - only loads small codec locally)
    tts = Vieneu(
        mode='remote',
        api_base=REMOTE_API_BASE,
        model_name=REMOTE_MODEL_ID
    )

    os.makedirs("outputs", exist_ok=True)

    # List remote voices
    print("\n📋 Available voices:")
    available_voices = tts.list_preset_voices()
    for desc, name in available_voices:
        print(f"   - {desc} (ID: {name})")

    # Use specific voice
    if available_voices:
        _, my_voice_id = available_voices[1]
        print(f"\n🎤 Using voice: {my_voice_id}")
        voice_data = tts.get_preset_voice(my_voice_id)
        audio_spec = tts.infer(
            text="Chào bạn, tôi đang nói bằng giọng của bác sĩ Tuyên.",
            voice=voice_data
        )
        tts.save(audio_spec, f"outputs/remote_{my_voice_id}.wav")
        print(f"💾 Saved synthesis to: outputs/remote_{my_voice_id}.wav")

    # Standard synthesis with default voice
    print("\n🎤 Standard synthesis:")
    text_input = "Chế độ remote giúp tích hợp VieNeu vào ứng dụng Web hoặc App cực nhanh mà không cần GPU tại máy khách."
    audio = tts.infer(text=text_input)
    tts.save(audio, "outputs/remote_output.wav")
    print("💾 Saved remote synthesis to: outputs/remote_output.wav")


def example_zero_shot_cloning():
    """Example: Zero-shot voice cloning via remote server."""
    print("\n" + "="*70)
    print("Example 2: Zero-Shot Voice Cloning (Remote)")
    print("="*70)

    REMOTE_API_BASE = 'http://localhost:23333/v1'
    REMOTE_MODEL_ID = "pnnbao-ump/VieNeu-TTS-0.3B"

    tts = Vieneu(
        mode='remote',
        api_base=REMOTE_API_BASE,
        model_name=REMOTE_MODEL_ID
    )

    os.makedirs("outputs", exist_ok=True)

    # Example reference audio path
    ref_audio_path = "examples/audio_ref/example_ngoc_huyen.wav"
    ref_text = "Tác phẩm dự thi bảo đảm tính khoa học, tính đảng, tính chiến đấu, tính định hướng."

    if os.path.exists(ref_audio_path):
        print(f"\n🎤 Using reference audio: {ref_audio_path}")
        cloned_audio = tts.infer(
            text="Đây là giọng nói được clone và xử lý thông qua VieNeu Server.",
            ref_audio=ref_audio_path,
            ref_text=ref_text
        )
        tts.save(cloned_audio, "outputs/remote_cloned_output.wav")
        print("💾 Saved remote cloned voice to: outputs/remote_cloned_output.wav")
    else:
        print(f"⚠ Reference audio not found: {ref_audio_path}")
        print("Skipping voice cloning example")


def example_with_voice_assistant():
    """Example: Using VieNeu remote mode with the voice assistant pipeline."""
    print("\n" + "="*70)
    print("Example 3: Integration with Voice Assistant Pipeline")
    print("="*70)

    print("\n📝 To use VieNeu remote mode in the voice assistant:")
    print("\nOption 1: Environment variables")
    print("  export TTS_BACKEND=vieneu_remote")
    print("  export VIENEU_REMOTE_API_BASE=http://your-server:23333/v1")
    print("  export VIENEU_REMOTE_MODEL_ID=pnnbao-ump/VieNeu-TTS-0.3B")

    print("\nOption 2: .env file")
    print("  TTS_BACKEND=vieneu_remote")
    print("  VIENEU_REMOTE_API_BASE=http://your-server:23333/v1")
    print("  VIENEU_REMOTE_MODEL_ID=pnnbao-ump/VieNeu-TTS-0.3B")

    print("\nOption 3: Programmatic usage")
    print("""
    from voice_assistant.core.tts import get_tts_service, TTSService

    # Create TTS service with remote mode
    tts_service = TTSService(backend="vieneu_remote")
    audio_bytes = await tts_service.synthesize("Xin chào Việt Nam")
    """)


def example_comparison():
    """Show comparison of local vs remote mode."""
    print("\n" + "="*70)
    print("Comparison: Local vs Remote VieNeu-TTS")
    print("="*70)

    print("""
┌─────────────────────┬──────────────────────┬──────────────────────┐
│ Feature             │ Local Mode           │ Remote Mode          │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ CPU/GPU Required    │ Yes (CPU)            │ No (lightweight)     │
│ Network Required    │ No (offline)         │ Yes (API calls)      │
│ Memory Usage        │ ~500MB-1GB           │ ~50MB                │
│ First Load Time     │ 5-10 seconds         │ <1 second            │
│ Inference Speed     │ Fast (local)         │ Depends on server    │
│ Voice Cloning       │ Yes (local)          │ Yes (server-side)    │
│ Best For            │ Offline, privacy     │ Web apps, low-end HW │
└─────────────────────┴──────────────────────┴──────────────────────┘
    """)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("VieNeu-TTS Remote Mode Examples")
    print("="*70)

    try:
        # Run examples
        example_remote_tts()
        example_zero_shot_cloning()
        example_with_voice_assistant()
        example_comparison()

        print("\n" + "="*70)
        print("✅ All examples completed!")
        print("="*70)

    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\nInstall vieneu package:")
        print("  pip install vieneu")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
