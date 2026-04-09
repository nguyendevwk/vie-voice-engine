"""
Example: Using VieNeu-TTS 0.3B Local Model

This example demonstrates how to use VieNeu-TTS 0.3B model locally
for optimal performance with Python SDK.

VieNeu-TTS 0.3B provides the best speed/quality balance for production use.

Usage:
    uv run python examples/vieneu_03b_example.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vieneu import Vieneu


def example_basic_usage():
    """Example 1: Basic VieNeu-TTS 0.3B usage."""
    print("="*70)
    print("Example 1: Basic VieNeu-TTS 0.3B Usage")
    print("="*70)

    # Initialize VieNeu-TTS with turbo mode (recommended)
    # The 0.3B model is loaded automatically
    print("\n📦 Initializing VieNeu-TTS (turbo mode)...")
    tts = Vieneu(mode='turbo')
    print("✅ VieNeu-TTS loaded successfully")

    os.makedirs("outputs", exist_ok=True)

    # List available voices
    print("\n🎤 Available voices:")
    available_voices = tts.list_preset_voices()
    for desc, name in available_voices:
        print(f"   - {desc} (ID: {name})")

    # Synthesize with default voice
    print("\n🔊 Synthesizing speech...")
    text = "Xin chào! Tôi là trợ lý ảo tiếng Việt."
    audio = tts.infer(text=text)
    tts.save(audio, "outputs/vieneu_03b_output.wav")
    print(f"💾 Saved to: outputs/vieneu_03b_output.wav")


def example_specific_voice():
    """Example 2: Using specific voice."""
    print("\n" + "="*70)
    print("Example 2: Using Specific Voice")
    print("="*70)

    tts = Vieneu(mode='turbo')

    # Get available voices
    available_voices = tts.list_preset_voices()
    if available_voices:
        # Use first voice
        desc, voice_id = available_voices[0]
        print(f"\n🎤 Using voice: {desc} ({voice_id})")
        
        voice_data = tts.get_preset_voice(voice_id)
        
        text = f"Chào bạn, tôi đang nói bằng giọng {desc}."
        audio = tts.infer(text=text, voice=voice_data)
        tts.save(audio, f"outputs/vieneu_voice_{voice_id}.wav")
        print(f"💾 Saved to: outputs/vieneu_voice_{voice_id}.wav")


def example_voice_cloning():
    """Example 3: Zero-shot voice cloning."""
    print("\n" + "="*70)
    print("Example 3: Zero-Shot Voice Cloning")
    print("="*70)

    tts = Vieneu(mode='turbo')
    
    os.makedirs("outputs", exist_ok=True)

    # Example reference audio
    ref_audio_path = "examples/audio_ref/example_ngoc_huyen.wav"
    ref_text = "Tác phẩm dự thi bảo đảm tính khoa học, tính đảng, tính chiến đấu, tính định hướng."

    if os.path.exists(ref_audio_path):
        print(f"\n🎤 Cloning voice from: {ref_audio_path}")
        cloned_audio = tts.infer(
            text="Đây là giọng nói được clone từ audio tham chiếu.",
            ref_audio=ref_audio_path,
            ref_text=ref_text
        )
        tts.save(cloned_audio, "outputs/vieneu_cloned_voice.wav")
        print("💾 Saved cloned voice to: outputs/vieneu_cloned_voice.wav")
    else:
        print(f"⚠ Reference audio not found: {ref_audio_path}")
        print("Skipping voice cloning example")


def example_different_modes():
    """Example 4: Comparing different modes."""
    print("\n" + "="*70)
    print("Example 4: VieNeu-TTS Modes Comparison")
    print("="*70)

    modes = [
        ("turbo", "Default optimized (recommended)"),
        ("standard", "CPU/GPU GGUF"),
        ("fast", "GPU LMDeploy"),
        ("turbo_gpu", "GPU optimized"),
    ]

    text = "Xin chào Việt Nam!"

    for mode, description in modes:
        try:
            print(f"\n🔧 Testing mode: {mode} - {description}")
            tts = Vieneu(mode=mode)
            audio = tts.infer(text=text)
            duration = len(audio) / 24000 * 1000  # 24kHz sample rate
            print(f"   ✅ Success - Generated {duration:.0f}ms of audio")
        except Exception as e:
            print(f"   ⚠ Not available: {e}")


def example_with_voice_assistant():
    """Example 5: Integration with voice assistant pipeline."""
    print("\n" + "="*70)
    print("Example 5: Integration with Voice Assistant")
    print("="*70)

    print("\n📝 To use VieNeu-TTS 0.3B in the voice assistant:")
    print("\nOption 1: Environment variables")
    print("  export TTS_BACKEND=vieneu")
    print("  export VIENEU_MODE=turbo")
    print("  export VIENEU_MODEL_ID=pnnbao-ump/VieNeu-TTS-0.3B")

    print("\nOption 2: .env file")
    print("  TTS_BACKEND=vieneu")
    print("  VIENEU_MODE=turbo")
    print("  VIENEU_MODEL_ID=pnnbao-ump/VieNeu-TTS-0.3B")

    print("\nOption 3: Programmatic usage")
    print("""
    from voice_assistant.core.tts import get_tts_service, TTSService
    
    # TTS service automatically uses VieNeu 0.3B
    tts_service = TTSService(backend="vieneu")
    audio_bytes = await tts_service.synthesize("Xin chào Việt Nam")
    """)


def show_comparison():
    """Show comparison table."""
    print("\n" + "="*70)
    print("VieNeu-TTS 0.3B vs Other TTS Options")
    print("="*70)

    print("""
┌──────────────────┬──────────┬─────────┬──────────┬─────────────┐
│ Feature          │ 0.3B     │ 0.6B    │ Remote   │ Edge-TTS    │
├──────────────────┼──────────┼─────────┼──────────┼─────────────┤
│ Speed            │ ⚡⚡⚡   │ ⚡⚡    │ ⚡⚡     │ ⚡⚡        │
│ Quality          │ ⭐⭐⭐   │ ⭐⭐⭐⭐ │ ⭐⭐⭐   │ ⭐⭐⭐      │
│ CPU Usage        │ Low      │ Medium  │ None     │ None        │
│ GPU Required     │ ❌       │ ❌/✅   │ ❌       │ ❌          │
│ Offline          │ ✅       │ ✅      │ ❌       │ ❌          │
│ Memory           │ ~300MB   │ ~600MB  │ ~50MB    │ Minimal     │
│ Best For         │ Production│Quality │ Web apps │ Fallback    │
└──────────────────┴──────────┴─────────┴──────────┴─────────────┘

✅ Recommendation: Use 0.3B with turbo mode for best performance!
    """)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("VieNeu-TTS 0.3B Local Model Examples")
    print("="*70)

    try:
        # Run examples
        example_basic_usage()
        example_specific_voice()
        example_voice_cloning()
        example_different_modes()
        example_with_voice_assistant()
        show_comparison()

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
