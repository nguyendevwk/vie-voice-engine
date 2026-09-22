# VieNeu-TTS Configuration

## Overview

VieNeu-TTS provides fast, offline Vietnamese text-to-speech with multiple modes:

| Mode | Model | Hardware | Speed | Quality | Recommended |
|------|-------|----------|-------|---------|-------------|
| `turbo` | VieNeu-TTS-v2 Turbo | CPU | 50x+ RT | Good | **Yes** |
| `standard` | VieNeu-TTS Standard | CPU/GPU | 20x RT | Better | Compatibility |
| `fast` | LMDeploy GPU | GPU | Very fast | Better | Max GPU speed |
| `turbo_gpu` | GPU Turbo | GPU | Fast | Good | GPU available |
| `remote` | VieNeu Server | Network | Network | Good | Web apps |

## Quick Start

```bash
# Set in .env file
TTS_BACKEND=vieneu
VIENEU_MODE=turbo

# Start server
uv run python -m voice_assistant.api.server
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TTS_BACKEND` | TTS backend (`auto`, `vieneu`, `vieneu_remote`, `qwen`, `edge`) | `auto` |
| `VIENEU_MODE` | Inference mode (`turbo`, `standard`, `fast`, `turbo_gpu`, `remote`) | `turbo` |
| `VIENEU_MODEL_BACKBONE` | Main TTS model repo | `pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF` |
| `VIENEU_MODEL_DECODER` | Audio decoder repo | `pnnbao-ump/VieNeu-Codec` |
| `VIENEU_MODEL_ENCODER` | Reference encoder repo | `pnnbao-ump/VieNeu-Codec` |
| `VIENEU_REMOTE_API_BASE` | Remote server URL | `http://localhost:23333/v1` |
| `VIENEU_REMOTE_MODEL_ID` | Remote model ID | `pnnbao-ump/VieNeu-TTS-0.3B` |

### Python Usage

```python
from voice_assistant.core.tts import TTSService

# Auto-select best available backend
tts = TTSService(backend="auto")

# Or specific backend
tts = TTSService(backend="vieneu")
audio = await tts.synthesize("Xin chào Việt Nam")
```

### Direct SDK Usage

```python
from vieneu import Vieneu

# Local turbo mode (recommended)
tts = Vieneu(mode='turbo')
audio = tts.infer(text="Xin chào!")
tts.save(audio, "output.wav")

# Remote mode
tts = Vieneu(mode='remote', api_base='http://server:23333/v1')
audio = tts.infer(text="Xin chào!")
```

## Voice Cloning

```python
from vieneu import Vieneu

tts = Vieneu(mode='turbo')

# Clone from reference audio
audio = tts.infer(
    text="Đây là giọng clone",
    ref_audio="reference.wav",
    ref_text="Content of reference audio"
)
tts.save(audio, "cloned.wav")
```

## Performance

### Benchmarks (CPU - Intel i7)

| Metric | v2 Turbo | Standard |
|--------|----------|----------|
| Speed | 50x+ RT | 20x RT |
| Latency (10 chars) | <100ms | <300ms |
| Latency (100 chars) | <500ms | <1s |
| Memory | ~300MB | ~500MB |

### Best Practices

1. Use `turbo` mode for best speed on CPU
2. Cache voice data - don't fetch repeatedly
3. Batch short texts when possible
4. Warm up model on startup

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ImportError: No module named 'vieneu'` | `pip install vieneu` |
| `ImportError: Failed to import lmdeploy` | Use `turbo` mode or `pip install vieneu[gpu]` |
| Slow first load | Normal - first load downloads models (~200MB) |
| Quality issues with short text | Use text > 5 words, or try `standard` mode |
| Out of memory | Use `turbo` mode (lightest) |

## Resources

- [VieNeu-TTS GitHub](https://github.com/pnnbao97/vieneu-tts)
- [Voice Assistant README](../README.md)
