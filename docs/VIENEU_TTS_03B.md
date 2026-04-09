# VieNeu-TTS 0.3B Configuration

## Overview

VieNeu-TTS 0.3B is the **recommended model** for production use, providing the best balance between:
- ⚡ **Speed** - Fast inference on CPU
- 🎯 **Quality** - Natural Vietnamese speech synthesis
- 💾 **Size** - Small footprint (~300MB)
- 🔒 **Offline** - No network required

## Quick Start

### 1. Basic Usage

```bash
# Set environment variables
export TTS_BACKEND=vieneu
export VIENEU_MODE=turbo

# Start server
uv run python -m voice_assistant.api.server
```

### 2. Via .env File

```env
TTS_BACKEND=vieneu
VIENEU_MODE=turbo
VIENEU_MODEL_ID=pnnbao-ump/VieNeu-TTS-0.3B
```

### 3. Python SDK Direct Usage

```python
from vieneu import Vieneu

# Initialize with turbo mode (recommended)
tts = Vieneu(mode='turbo')

# Synthesize
audio = tts.infer(text="Xin chào Việt Nam")
tts.save(audio, "output.wav")
```

## Configuration Options

### Environment Variables

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `TTS_BACKEND` | TTS backend selector | `auto` | `auto`, `vieneu`, `qwen`, `edge` |
| `VIENEU_MODE` | VieNeu inference mode | `turbo` | `turbo`, `standard`, `fast`, `turbo_gpu` |
| `VIENEU_MODEL_ID` | Model identifier | `pnnbao-ump/VieNeu-TTS-0.3B` | - |

### VieNeu Modes Explained

| Mode | Description | Hardware | Dependencies | Speed | Recommended |
|------|-------------|----------|--------------|-------|-------------|
| `turbo` | Default optimized | CPU | None | ⚡⚡⚡ | ✅ **Yes** |
| `standard` | CPU/GPU GGUF | CPU/GPU | PyTorch | ⚡⚡ | For compatibility |
| `fast` | GPU LMDeploy | GPU | lmdeploy | ⚡⚡⚡⚡ | Max GPU speed |
| `turbo_gpu` | GPU optimized | GPU | PyTorch | ⚡⚡⚡ | GPU available |

**Quick recommendation**: Use `turbo` mode - works out of the box, no extra dependencies!

## Python Examples

### Basic Synthesis

```python
from vieneu import Vieneu

tts = Vieneu(mode='turbo')
audio = tts.infer(text="Xin chào, tôi là trợ lý ảo tiếng Việt.")
tts.save(audio, "output.wav")
```

### Using Specific Voice

```python
tts = Vieneu(mode='turbo')

# List voices
voices = tts.list_preset_voices()
for desc, name in voices:
    print(f"{desc}: {name}")

# Use specific voice
voice_data = tts.get_preset_voice("yen_nhi")
audio = tts.infer(text="Xin chào!", voice=voice_data)
tts.save(audio, "output.wav")
```

### Zero-Shot Voice Cloning

```python
tts = Vieneu(mode='turbo')

# Clone voice from reference audio
audio = tts.infer(
    text="Đây là giọng clone",
    ref_audio="reference.wav",
    ref_text="Content of reference audio"
)
tts.save(audio, "cloned.wav")
```

### Integration with Voice Assistant

```python
from voice_assistant.core.tts import TTSService

# Create service (automatically uses VieNeu 0.3B)
tts_service = TTSService(backend="vieneu")

# Synthesize
audio_bytes = await tts_service.synthesize("Xin chào Việt Nam")
```

## Performance

### Benchmarks (CPU - Intel i7)

| Metric | Value |
|--------|-------|
| Model Size | ~300MB |
| Memory Usage | ~500MB |
| First Load Time | 3-5 seconds |
| Inference Speed | ~50x real-time |
| Latency (10 chars) | <200ms |
| Latency (100 chars) | <1s |

### Optimization Tips

1. **Use turbo mode** - Best performance out of the box
2. **Cache voice data** - Don't fetch voice repeatedly
3. **Batch short texts** - Group when possible
4. **Warm up on startup** - Load model before first request

## Troubleshooting

### Import Error

```
ImportError: No module named 'vieneu'
```

**Solution:**
```bash
pip install vieneu
# or
uv add vieneu
```

### Fast Mode Error

```
ImportError: Failed to import `lmdeploy`. Install with: pip install vieneu[gpu]
```

**Solution:**
```bash
# Option 1: Install GPU dependencies
pip install vieneu[gpu]

# Option 2: Use turbo mode instead (recommended)
export VIENEU_MODE=turbo
```

### Standard Mode Error

```
ImportError: Codec requires PyTorch. Install torch via: pip install vieneu[gpu]
```

**Solution:**
```bash
# Option 1: Install PyTorch
pip install torch torchaudio

# Option 2: Use turbo mode instead (recommended)
export VIENEU_MODE=turbo
```

### Out of Memory

```
RuntimeError: Not enough memory
```

**Solution:**
- Use `turbo` mode (lightest)
- Close other applications
- Consider using GPU modes if available

### Slow Inference

**Solution:**
- Verify using 0.3B model (not larger)
- Use `turbo` or `turbo_gpu` mode
- Check CPU/GPU utilization
- Don't use `fast` mode without proper GPU setup

## Comparison with Other Models

| Feature | 0.3B (Recommended) | 0.6B | Remote |
|---------|-------------------|------|--------|
| **Speed** | ⚡⚡⚡ Fast | ⚡⚡ Medium | ⚡⚡ Network |
| **Quality** | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Best | ⭐⭐⭐ Good |
| **Offline** | ✅ Yes | ✅ Yes | ❌ No |
| **Memory** | ~300MB | ~600MB | ~50MB |
| **Setup** | Easy | Easy | Needs server |
| **Best For** | **Production** | Quality focus | Web apps |

## Resources

- [VieNeu-TTS GitHub](https://github.com/pnnbao97/vieneu-tts)
- [Voice Assistant README](../README.md)
- [Example Code](../examples/vieneu_03b_example.py)
