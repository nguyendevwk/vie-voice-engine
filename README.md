# Vietnamese Voice Assistant

End-to-end streaming pipeline for Vietnamese speech recognition and synthesis.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Real-time ASR** - Vietnamese speech recognition with ONNX/PyTorch backends
- **LLM Integration** - Groq/OpenAI with streaming response
- **Multiple TTS** - VieNeu-TTS (CPU), Qwen-TTS (GPU), Edge-TTS (fallback)
- **Voice Activity Detection** - Silero VAD for speech detection
- **Session Management** - Conversation history and state tracking
- **Web UI** - Modern interface with WebSocket streaming
- **CLI** - Command-line interface for testing

## Architecture

```
Audio Input → VAD → ASR → LLM → TTS → Audio Output
     ↓         ↓      ↓      ↓      ↓
  16kHz    Silero  ONNX   Groq  VieNeu/
  PCM16    Model   Model  API   Edge-TTS
```

## Installation

### Prerequisites

```bash
# Ubuntu/Debian - Install PortAudio for microphone support
sudo apt-get install portaudio19-dev

# Fedora/RHEL
sudo dnf install portaudio-devel

# macOS
brew install portaudio
```

### With uv (Recommended)

```bash
# Clone
git clone https://github.com/nguyendevwk/end2end_asr_tts_vie.git
cd end2end_asr_tts_vie

# Install with uv
uv sync

# Configure
cp .env.example .env
# Edit .env with your GROQ_API_KEY
```

### With pip

```bash
# Clone
git clone https://github.com/nguyendevwk/end2end_asr_tts_vie.git
cd end2end_asr_tts_vie

# Setup environment
python -m venv .venv
source .venv/bin/activate

# Install
pip install -r requirements.txt
pip install -e .

# Configure
cp .env.example .env
# Edit .env with your GROQ_API_KEY
```

## Configuration

Create `.env` file:

```bash
# Required
GROQ_API_KEY=your_key_here

# ASR
ASR_USE_ONNX=true
ASR_DEVICE=auto

# TTS: auto, vieneu, qwen, edge
TTS_BACKEND=auto

# VieNeu-TTS v2 Turbo Configuration (recommended for speed)
# 🚀 Optimized for edge devices and extremely fast inference!
VIENEU_MODE=turbo
VIENEU_MODEL_BACKBONE=pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF
VIENEU_MODEL_DECODER=pnnbao-ump/VieNeu-Codec
VIENEU_MODEL_ENCODER=pnnbao-ump/VieNeu-Codec
TTS_SPEECH_RATE=1.25

# Server
SERVER_PORT=8000
DEBUG=false
```

## Usage

### Docker (Recommended for Production)

```bash
# Copy environment file
cp .env.docker.example .env
# Edit .env with your GROQ_API_KEY

# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

Access at `http://localhost:8000`

### Local Development

#### Web UI

```bash
# With uv
uv run python -m voice_assistant.api.server

# With pip/venv
python -m voice_assistant.api.server

# Open http://localhost:8000
```

#### CLI

```bash
# Voice mode
uv run python -m voice_assistant.cli.main

# Text mode
uv run python -m voice_assistant.cli.main --text-only --no-tts
```

### Python API

```python
from voice_assistant.core.pipeline import PipelineOrchestrator

async def main():
    orchestrator = PipelineOrchestrator()

    # Process audio
    await orchestrator.handle_audio_chunk(audio_bytes)

    # Or text
    async for event in orchestrator.process_text("Xin chào"):
        print(event.type, event.data)
```

## Project Structure

```
voice_assistant/
├── api/           # FastAPI server
├── cli/           # CLI interface
├── core/          # Core components
│   ├── vad.py     # Voice Activity Detection
│   ├── asr.py     # Speech Recognition
│   ├── llm.py     # Language Model
│   ├── tts.py     # Text-to-Speech
│   └── pipeline.py
├── utils/         # Utilities
└── config.py      # Configuration
```

## TTS Backends

| Backend | Hardware | Quality | Speed | Offline |
|---------|----------|---------|-------|---------|
| **VieNeu-TTS v2 Turbo** 🚀 | CPU | Good | ⚡⚡⚡⚡⚡ | Yes |
| VieNeu-TTS Standard | CPU/GPU | Better | ⚡⚡⚡ | Yes |
| Qwen-TTS | GPU (4-6GB) | Best | ⚡⚡ | Yes |
| Edge-TTS | Any | Good | ⚡⚡ | No |

**🚀 VieNeu-TTS-v2 Turbo:**
- Optimized for edge devices and extremely fast inference
- Perfect for CPU-only and low-end devices
- ⚠️ Note: May struggle with very short segments (< 5 words)
- Version VieNeu-TTS-v2 (Non-Turbo) coming soon!

**Quick setup:**
```bash
# Use VieNeu-TTS v2 Turbo (fastest, recommended for speed)
TTS_BACKEND=vieneu
VIENEU_MODE=turbo
```

## Performance

- VAD: ~10ms per chunk
- ASR: <1s for short utterances
- LLM first token: <1s
- TTS: <2s per sentence
- Total: <3s first response

## Testing

```bash
# With uv
uv run pytest tests/ -v

# With pip/venv
pytest tests/ -v
```

## Documentation

- [Architecture](ARCHITECTURE.md)
- [API Reference](API.md)
- [Contributing](CONTRIBUTING.md)
- [LLM Integration](docs/LLM_INTEGRATION.md)

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

- Gipformer ASR - G-Group AI Lab
- Qwen-TTS - G-Group AI Lab
- Silero VAD - Silero Team
- VieNeu-TTS - pnnbao97
