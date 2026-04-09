# VieNeu-TTS Remote Mode

## Overview

VieNeu-TTS now supports **remote mode**, which allows you to use a VieNeu server for TTS inference instead of running models locally. This is perfect for:

- **Web applications** - Lightweight client, no local GPU needed
- **Low-end hardware** - Minimal CPU/RAM requirements
- **Fast deployment** - Quick setup, no model downloads
- **Voice cloning** - Server-side voice cloning with reference audio

## Architecture

```
┌─────────────────────┐         ┌──────────────────────┐
│   Your Application │  HTTP   │   VieNeu Server      │
│   (Lightweight)    │ ──────> │   (GPU-powered)      │
│   - Vieneu client  │ <────── │   - TTS Models       │
│   - Codec only     │  Audio  │   - Voice Library    │
└─────────────────────┘         └──────────────────────┘
```

## Quick Start

### 1. Using with Voice Assistant

#### Option A: Environment Variables

```bash
# Set environment variables
export TTS_BACKEND=vieneu_remote
export VIENEU_REMOTE_API_BASE=http://your-server-ip:23333/v1
export VIENEU_REMOTE_MODEL_ID=pnnbao-ump/VieNeu-TTS-0.3B

# Start the server
uv run python -m voice_assistant.api.server
```

#### Option B: .env File

Create or update your `.env` file:

```env
# TTS Backend
TTS_BACKEND=vieneu_remote

# VieNeu Remote Configuration
VIENEU_REMOTE_API_BASE=http://your-server-ip:23333/v1
VIENEU_REMOTE_MODEL_ID=pnnbao-ump/VieNeu-TTS-0.3B
```

Then start the server normally.

#### Option C: Programmatic Usage

```python
from voice_assistant.core.tts import TTSService

# Create TTS service with remote mode
tts_service = TTSService(backend="vieneu_remote")

# Synthesize speech
audio_bytes = await tts_service.synthesize("Xin chào Việt Nam")
```

### 2. Standalone Usage

```python
from vieneu import Vieneu
import os

# Configuration
REMOTE_API_BASE = 'http://your-server-ip:23333/v1'
REMOTE_MODEL_ID = "pnnbao-ump/VieNeu-TTS-0.3B"

# Initialization (LIGHTWEIGHT - only loads small codec locally)
tts = Vieneu(
    mode='remote',
    api_base=REMOTE_API_BASE,
    model_name=REMOTE_MODEL_ID
)

os.makedirs("outputs", exist_ok=True)

# List remote voices
available_voices = tts.list_preset_voices()
for desc, name in available_voices:
    print(f"   - {desc} (ID: {name})")

# Use specific voice
if available_voices:
    _, my_voice_id = available_voices[1]
    voice_data = tts.get_preset_voice(my_voice_id)
    audio_spec = tts.infer(
        text="Chào bạn, tôi đang nói bằng giọng của bác sĩ Tuyên.",
        voice=voice_data
    )
    tts.save(audio_spec, f"outputs/remote_{my_voice_id}.wav")
    print(f"💾 Saved to: outputs/remote_{my_voice_id}.wav")

# Standard synthesis
text_input = "Chế độ remote giúp tích hợp VieNeu vào ứng dụng Web hoặc App cực nhanh."
audio = tts.infer(text=text_input)
tts.save(audio, "outputs/remote_output.wav")
print("💾 Saved to: outputs/remote_output.wav")
```

### 3. Zero-Shot Voice Cloning

```python
from vieneu import Vieneu

tts = Vieneu(
    mode='remote',
    api_base='http://your-server-ip:23333/v1',
    model_name="pnnbao-ump/VieNeu-TTS-0.3B"
)

# Clone voice from reference audio
cloned_audio = tts.infer(
    text="Đây là giọng nói được clone thông qua VieNeu Server.",
    ref_audio="path/to/reference_audio.wav",
    ref_text="Text content of the reference audio"
)

tts.save(cloned_audio, "outputs/cloned_voice.wav")
```

## Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TTS_BACKEND` | TTS backend selector (`auto`, `vieneu`, `vieneu_remote`, `qwen`, `edge`) | `auto` |
| `VIENEU_REMOTE_API_BASE` | VieNeu server API base URL | `http://localhost:23333/v1` |
| `VIENEU_REMOTE_MODEL_ID` | Remote model identifier | `pnnbao-ump/VieNeu-TTS-0.3B` |

### Python Configuration

```python
from voice_assistant.config import settings

# Access configuration
print(settings.tts.vieneu_remote_api_base)
print(settings.tts.vieneu_remote_model_id)
```

## Comparison: Local vs Remote

| Feature | Local Mode | Remote Mode |
|---------|-----------|-------------|
| **CPU/GPU Required** | Yes (CPU) | No (lightweight client) |
| **Network Required** | No (offline) | Yes (API calls) |
| **Memory Usage** | ~500MB-1GB | ~50MB |
| **First Load Time** | 5-10 seconds | <1 second |
| **Inference Speed** | Fast (local) | Depends on server |
| **Voice Cloning** | Yes (local) | Yes (server-side) |
| **Best For** | Offline, privacy | Web apps, low-end HW |

## TTS Backend Priority

When `TTS_BACKEND=auto`, the system tries backends in this order:

1. **VieNeu-TTS Local** - If installed and available
2. **VieNeu-TTS Remote** - If configured with server URL
3. **Qwen-TTS** - If GPU available
4. **Edge-TTS** - Online fallback

## Examples

See `examples/vieneu_remote_example.py` for complete working examples.

Run it with:
```bash
uv run python examples/vieneu_remote_example.py
```

## Troubleshooting

### Connection Issues

**Problem:** Can't connect to remote server

```
Error: Connection refused to http://localhost:23333/v1
```

**Solution:**
- Verify the server is running and accessible
- Check the `VIENEU_REMOTE_API_BASE` URL
- Test connectivity: `curl http://your-server:23333/v1`

### Voice Not Found

**Problem:** Voice ID not found

```
Voice 'xyz' not found
```

**Solution:**
- List available voices first: `tts.list_preset_voices()`
- Verify voice ID is correct
- Check server voice library

### Import Error

**Problem:** Cannot import Vieneu

```
ImportError: No module named 'vieneu'
```

**Solution:**
```bash
pip install vieneu
# or with uv
uv add vieneu
```

## Performance Tips

1. **Use connection pooling** - Reuse Vieneu instance for multiple requests
2. **Cache voice data** - Don't fetch voice data repeatedly
3. **Batch requests** - Group short texts when possible
4. **Monitor latency** - Track API response times

## API Reference

### VieNeuTTSProvider (Remote Mode)

```python
from voice_assistant.core.tts import VieNeuTTSProvider

provider = VieNeuTTSProvider(
    voice="yen_nhi",  # Optional: preset voice
    mode="remote",
    api_base="http://your-server:23333/v1",
    model_id="pnnbao-ump/VieNeu-TTS-0.3B"
)

# Check availability
if provider.is_available():
    # Synthesize
    result = provider.synthesize("Xin chào")
    print(f"Audio: {result.audio.shape}, Sample rate: {result.sample_rate}")
```

### TTSService (Remote Mode)

```python
from voice_assistant.core.tts import TTSService

tts = TTSService(backend="vieneu_remote")

# Synthesize
audio_bytes = await tts.synthesize("Xin chào Việt Nam")

# List voices
voices = tts.list_voices()
print(f"Available voices: {voices}")
```

## Resources

- [VieNeu-TTS Documentation](https://github.com/pnnbao97/vieneu-tts)
- [Voice Assistant README](../README.md)
- [API Documentation](../API.md)
- [Example Script](../examples/vieneu_remote_example.py)
