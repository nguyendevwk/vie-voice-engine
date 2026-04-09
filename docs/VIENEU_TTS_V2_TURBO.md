# 🚀 VieNeu-TTS-v2 Turbo

## Overview

**VieNeu-TTS-v2 Turbo** is the **fastest** TTS option available, specifically optimized for:
- ⚡ **Extremely fast inference** on CPU
- 📱 **Edge devices** and low-end hardware
- 💻 **CPU-only** systems without GPU
- 🚀 **Real-time** applications requiring low latency

### ⚠️ Important Notes

- **Quality**: Lower than Standard VieNeu-TTS (trade-off for speed)
- **Short segments**: May struggle with very short text (< 5 words)
- **Coming soon**: VieNeu-TTS-v2 (Non-Turbo) for better quality

## Quick Start

### 1. Environment Setup

```bash
# Set in .env file
TTS_BACKEND=vieneu
VIENEU_MODE=turbo
VIENEU_MODEL_BACKBONE=pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF
VIENEU_MODEL_DECODER=pnnbao-ump/VieNeu-Codec
VIENEU_MODEL_ENCODER=pnnbao-ump/VieNeu-Codec
```

### 2. Start Server

```bash
uv run python -m voice_assistant.api.server
```

That's it! The server will automatically use VieNeu-TTS-v2 Turbo.

## Architecture

```
VieNeu-TTS-v2 Turbo Components:
┌─────────────────────────────────────────┐
│  Backbone: VieNeu-TTS-v2-Turbo-GGUF    │ ← Main TTS model (quantized)
│  Decoder:  VieNeu-Codec                │ ← Audio decoder (ONNX)
│  Encoder:  VieNeu-Codec                │ ← Reference encoder (ONNX)
└─────────────────────────────────────────┘
         ↓
    Audio Output (24kHz)
```

## Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TTS_BACKEND` | TTS backend selector | `auto` |
| `VIENEU_MODE` | VieNeu inference mode | `turbo` |
| `VIENEU_MODEL_BACKBONE` | Main TTS model repo | `pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF` |
| `VIENEU_MODEL_DECODER` | Audio decoder repo | `pnnbao-ump/VieNeu-Codec` |
| `VIENEU_MODEL_ENCODER` | Reference encoder repo | `pnnbao-ump/VieNeu-Codec` |

### Python Usage

```python
from voice_assistant.core.tts import TTSService

# Create service (automatically uses v2 Turbo)
tts = TTSService(backend="vieneu")

# Synthesize
audio_bytes = await tts.synthesize("Xin chào Việt Nam")
```

### Direct SDK Usage

```python
from vieneu import Vieneu

# Initialize VieNeu-TTS-v2 Turbo
tts = Vieneu(
    mode='turbo',
    backbone_repo='pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF',
    decoder_repo='pnnbao-ump/VieNeu-Codec',
    encoder_repo='pnnbao-ump/VieNeu-Codec'
)

# Synthesize
audio = tts.infer(text="Xin chào!")
tts.save(audio, "output.wav")
```

## Performance

### Benchmarks (CPU - Intel i7)

| Metric | VieNeu-TTS-v2 Turbo | Standard VieNeu |
|--------|---------------------|-----------------|
| **Speed** | ⚡⚡⚡⚡⚡ (50x+ RT) | ⚡⚡⚡ (20x RT) |
| **Quality** | ⭐⭐⭐ (Good) | ⭐⭐⭐⭐ (Better) |
| **Latency (10 chars)** | <100ms | <300ms |
| **Latency (100 chars)** | <500ms | <1s |
| **Memory** | ~300MB | ~500MB |
| **CPU Usage** | Low | Medium |

### Best Use Cases

✅ **Perfect for:**
- Real-time voice assistants
- Live captioning and dubbing
- Interactive applications
- Low-end hardware deployment
- CPU-only environments

⚠️ **Not ideal for:**
- Very short text (< 5 words)
- Highest quality requirements
- Professional audio production

## Available Modes

| Mode | Model | Hardware | Speed | Quality |
|------|-------|----------|-------|---------|
| `turbo` 🚀 | **VieNeu-TTS-v2-Turbo** | CPU | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ |
| `standard` | VieNeu-TTS Standard | CPU/GPU | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| `fast` | LMDeploy GPU | GPU | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ |
| `turbo_gpu` | GPU Turbo | GPU | ⚡⚡⚡⚡ | ⭐⭐⭐ |

**Recommendation**: Use `turbo` mode for best speed on CPU!

## Examples

### Basic Synthesis

```python
from voice_assistant.core.tts import get_tts_service

tts = get_tts_service()
audio = await tts.synthesize("Xin chào! Tôi là trợ lý ảo tiếng Việt.")
```

### Voice Cloning

```python
from vieneu import Vieneu

tts = Vieneu(mode='turbo')

# Clone voice from reference
audio = tts.infer(
    text="Đây là giọng nói được clone.",
    ref_audio="reference.wav",
    ref_text="Content of reference audio"
)
tts.save(audio, "cloned.wav")
```

### Batch Processing

```python
texts = [
    "Xin chào!",
    "Tôi là trợ lý ảo.",
    "Rất vui được gặp bạn!"
]

for text in texts:
    audio = tts.infer(text=text)
    tts.save(audio, f"output_{text[:10]}.wav")
```

## Troubleshooting

### Import Error

```
ImportError: No module named 'vieneu'
```

**Solution:**
```bash
pip install vieneu
```

### Slow First Load

**Normal behavior** - First load downloads models (~200MB):
```
Downloading VieNeu-TTS-v2-Turbo-GGUF... (202MB)
```

**Solution**: Wait for download, subsequent loads are fast.

### Quality Issues

**Problem**: Audio quality not as expected

**Solutions:**
1. Use longer text (> 5 words)
2. Try `standard` mode for better quality
3. Ensure proper punctuation

### Very Short Text

**Problem**: Struggles with text < 5 words

**Workaround:**
```python
# Pad short texts if needed
if len(text.split()) < 5:
    text = text + "..."  # Add padding
```

## Migration from v1/0.3B

If you were using the previous 0.3B model:

### Old Config (0.3B)
```env
VIENEU_MODE=turbo
VIENEU_MODEL_ID=pnnbao-ump/VieNeu-TTS-0.3B
```

### New Config (v2 Turbo)
```env
VIENEU_MODE=turbo
VIENEU_MODEL_BACKBONE=pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF
VIENEU_MODEL_DECODER=pnnbao-ump/VieNeu-Codec
VIENEU_MODEL_ENCODER=pnnbao-ump/VieNeu-Codec
```

### Benefits of v2 Turbo
- ⚡ **2-3x faster** inference
- 💾 **Smaller memory** footprint
- 📱 **Better edge device** support
- ⚠️ Trade-off: Slightly lower quality

## Resources

- [VieNeu-TTS GitHub](https://github.com/pnnbao97/vieneu-tts)
- [Voice Assistant README](../README.md)
- [Fast Mode Fix Documentation](FAST_MODE_FIX.md)
