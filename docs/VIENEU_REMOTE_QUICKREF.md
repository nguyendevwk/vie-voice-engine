# Quick Reference: VieNeu Remote Mode

## TL;DR

```bash
# 1. Set environment variables
export TTS_BACKEND=vieneu_remote
export VIENEU_REMOTE_API_BASE=http://your-server:23333/v1
export VIENEU_REMOTE_MODEL_ID=pnnbao-ump/VieNeu-TTS-0.3B

# 2. Start server
uv run python -m voice_assistant.api.server
```

## Configuration Options

### Via `.env` file
```env
TTS_BACKEND=vieneu_remote
VIENEU_REMOTE_API_BASE=http://your-server:23333/v1
VIENEU_REMOTE_MODEL_ID=pnnbao-ump/VieNeu-TTS-0.3B
```

### Via Python code
```python
from voice_assistant.core.tts import TTSService

# Create service
tts = TTSService(backend="vieneu_remote")

# Synthesize
audio = await tts.synthesize("Xin chào Việt Nam")
```

### Via command line
```bash
TTS_BACKEND=vieneu_remote uv run python -m voice_assistant.api.server
```

## Available TTS Backends

| Backend | Command | Offline | GPU Required |
|---------|---------|---------|--------------|
| VieNeu Local | `TTS_BACKEND=vieneu` | ✅ | ❌ |
| **VieNeu Remote** | `TTS_BACKEND=vieneu_remote` | ❌ | ❌ |
| Qwen-TTS | `TTS_BACKEND=qwen` | ✅ | ✅ |
| Edge-TTS | `TTS_BACKEND=edge` | ❌ | ❌ |
| Auto | `TTS_BACKEND=auto` | - | - |

## Quick Examples

### Basic Usage
```python
from vieneu import Vieneu

tts = Vieneu(mode='remote', api_base='http://server:23333/v1')
audio = tts.infer(text="Xin chào")
tts.save(audio, "output.wav")
```

### Voice Cloning
```python
audio = tts.infer(
    text="Đây là giọng clone",
    ref_audio="reference.wav",
    ref_text="Reference text"
)
```

### List Voices
```python
voices = tts.list_preset_voices()
for desc, name in voices:
    print(f"{desc}: {name}")
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Connection refused | Check server URL and port |
| Voice not found | List available voices first |
| Import error | Run `pip install vieneu` |

## More Info

- Full docs: `docs/VIENEU_REMOTE_MODE.md`
- Examples: `examples/vieneu_remote_example.py`
