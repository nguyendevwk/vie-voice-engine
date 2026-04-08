# 🎙️ Vietnamese Voice Assistant - End-to-End Streaming Pipeline

> Real-time Vietnamese voice assistant with ASR, LLM, and TTS streaming for personal use and CV demo

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## ✨ Features

### Core Capabilities

- 🎤 **Real-time Speech Recognition** - Gipformer ASR with streaming support
- 🧠 **LLM Integration** - Groq/OpenAI with token streaming
- 🗣️ **Voice Synthesis** - Qwen-TTS with voice cloning or Edge-TTS fallback
- 👂 **Voice Activity Detection** - Silero VAD for speech start/end
- ⚡ **Low Latency** - <3s first response, async throughout
- 🔄 **Interrupt Support** - Can interrupt assistant while speaking
- 💬 **Session Management** - Conversation history with persistence

### Technical Highlights

- 🔧 **Modular Architecture** - Easy to extend and customize
- 🚀 **Multiple Backends** - ONNX/PyTorch for ASR, Qwen/Edge-TTS for TTS
- 📊 **Audio Preprocessing** - Comprehensive normalization pipeline
- 🌐 **Web UI** - Modern dark theme with real-time visualizer
- 🖥️ **CLI Interface** - Command-line mode for testing
- 📝 **Text Normalization** - Markdown/special char cleaning for TTS

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Voice Assistant Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Audio Input → VAD → ASR → LLM → TTS → Audio Output            │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   VAD    │→ │   ASR    │→ │   LLM    │→ │   TTS    │       │
│  │ Silero   │  │Gipformer │  │  Groq    │  │ Qwen-TTS │       │
│  │  Model   │  │ ONNX/PT  │  │ Stream   │  │ +Edge-TTS│       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│       ↓            ↓             ↓              ↓               │
│  Speech Start  Vietnamese   AI Response   Audio Chunks         │
│  Detection     Transcription  Streaming    Synthesis           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- CUDA 11.8+ (optional, for GPU acceleration)
- Groq API key (for LLM)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/nguyendevwk/end2end_asr_tts_vie.git
cd end2end_asr_tts_vie

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install package in development mode
pip install -e .

# 5. Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Configuration

Create `.env` file with your settings:

```bash
# Required: LLM API Key
GROQ_API_KEY=your_groq_api_key_here

# Optional: ASR Backend (default: ONNX)
ASR_USE_ONNX=true
ASR_USE_PYTORCH_CUDA=false
ASR_DEVICE=auto

# Optional: TTS Backend (default: auto with edge-tts fallback)
TTS_BACKEND=auto
TTS_SPEECH_RATE=1.25
TTS_DEVICE=cuda

# Optional: Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
DEBUG=false
```

### Running

#### Web UI (Recommended)

```bash
python -m voice_assistant.api.server
# Open http://localhost:8000 in browser
```

#### CLI Mode

```bash
# Text-only mode (for testing LLM)
python -m voice_assistant.cli.main --text-only --no-tts

# Voice mode (requires microphone)
python -m voice_assistant.cli.main
```

## 📖 Usage

### Web Interface

1. **Start Server**

   ```bash
   python -m voice_assistant.api.server
   ```

2. **Open Browser**
   - Navigate to `http://localhost:8000`
   - Click "Connect" to establish WebSocket connection
   - Click microphone icon to start recording
   - Speak in Vietnamese
   - Receive AI response with voice

3. **Features**
   - 🎤 Click mic to record
   - ✋ Click stop to end recording
   - 💬 Type text for testing
   - 📊 Real-time audio visualizer
   - 📝 Conversation history
   - 🔧 Debug panel (toggle with button)

### CLI Interface

```bash
# Interactive voice mode
python -m voice_assistant.cli.main

# Text-only mode (no microphone needed)
python -m voice_assistant.cli.main --text-only

# Disable TTS (text response only)
python -m voice_assistant.cli.main --no-tts

# Custom LLM model
python -m voice_assistant.cli.main --model llama-3.3-70b-versatile
```

### Python API

```python
import asyncio
from voice_assistant.core.pipeline import PipelineOrchestrator

async def main():
    # Create pipeline
    orchestrator = PipelineOrchestrator()
    
    # Process audio chunks
    audio_chunk = b'\x00' * 3200  # 100ms PCM16
    await orchestrator.handle_audio_chunk(audio_chunk)
    
    # Or process text directly
    async for event in orchestrator.process_text("Xin chào"):
        print(event.type, event.data)

asyncio.run(main())
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| **LLM** | | |
| `GROQ_API_KEY` | - | Groq API key (required) |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | LLM model name |
| `LLM_PROVIDER` | `groq` | LLM provider (groq/openai) |
| **ASR** | | |
| `ASR_DEVICE` | `auto` | Device (cpu/cuda/auto) |
| `ASR_USE_ONNX` | `true` | Use ONNX backend |
| `ASR_USE_PYTORCH_CUDA` | `false` | Use PyTorch CUDA |
| **TTS** | | |
| `TTS_BACKEND` | `auto` | Backend (auto/qwen/edge-tts) |
| `TTS_SPEECH_RATE` | `1.25` | Speech speed (1.0-2.0) |
| `TTS_DEVICE` | `cuda` | TTS device |
| `TTS_EDGE_VOICE` | `vi-VN-NamMinhNeural` | Edge-TTS voice |
| **Pipeline** | | |
| `PIPELINE_ASR_TIMEOUT` | `10` | ASR timeout (seconds) |
| `PIPELINE_LLM_TIMEOUT` | `30` | LLM timeout (seconds) |
| `PIPELINE_TTS_TIMEOUT` | `15` | TTS timeout (seconds) |
| **Server** | | |
| `SERVER_HOST` | `0.0.0.0` | Server host |
| `SERVER_PORT` | `8000` | Server port |
| `DEBUG` | `false` | Enable debug mode |

### Backend Options

#### ASR Backends

1. **ONNX** (Recommended) - Fast, simple, no dependencies
2. **PyTorch CUDA** - Best quality, requires k2/icefall
3. **Whisper** - Universal fallback

#### TTS Backends

1. **Qwen-TTS** - Best quality, voice cloning, GPU required
2. **Edge-TTS** - Reliable fallback, Microsoft Azure

## 📚 API Documentation

### Core Components

#### PipelineOrchestrator

Main pipeline coordinator for audio processing.

```python
from voice_assistant.core.pipeline import PipelineOrchestrator

orchestrator = PipelineOrchestrator(
    on_event=lambda event: print(event)
)

# Process audio
await orchestrator.handle_audio_chunk(audio_bytes)

# Process text
async for event in orchestrator.process_text("Hello"):
    # Handle events
    pass
```

#### ASRService

Speech recognition service with multiple backends.

```python
from voice_assistant.core.asr import ASRService

asr = ASRService(use_onnx=True)

# Transcribe file
text = asr.transcribe_file("audio.wav")

# Transcribe audio
text = asr.transcribe(audio_samples, sample_rate=16000)
```

#### LLMService

Language model service with streaming support.

```python
from voice_assistant.core.llm import LLMService

llm = LLMService()

# Stream responses
async for sentence in llm.generate_response_stream("Hello"):
    print(sentence)
```

#### TTSService

Text-to-speech service with multiple backends.

```python
from voice_assistant.core.tts import TTSService

tts = TTSService()

# Synthesize
audio_bytes = await tts.synthesize("Xin chào")

# Stream synthesis
async for chunk in tts.synthesize_stream("Long text..."):
    # Play chunk
    pass
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_audio.py -v

# Run with coverage
pytest tests/ --cov=voice_assistant --cov-report=html

# Run integration tests
pytest tests/test_api.py -v
```

## 📊 Performance

### Latency Benchmarks

- **VAD**: ~10ms per 100ms chunk
- **ASR (ONNX)**: <1s for short utterances
- **LLM First Token**: <1s (Groq)
- **TTS**: <2s per sentence
- **End-to-End**: <3s first response

### Resource Usage

- **RAM**: 2-4GB (ONNX), 4-8GB (PyTorch)
- **VRAM**: 2-4GB (TTS), 1-2GB (ASR)
- **CPU**: 2-4 cores recommended
- **Network**: <1MB/s (LLM streaming)

## 🎨 Project Structure

```
end2end_asr_tts_vie/
├── voice_assistant/          # Main package
│   ├── api/                  # Web API & UI
│   │   ├── server.py        # FastAPI server
│   │   └── static/
│   │       └── index.html   # Web UI
│   ├── cli/                  # CLI interface
│   │   └── main.py
│   ├── core/                 # Core components
│   │   ├── vad.py           # Voice Activity Detection
│   │   ├── asr*.py          # ASR implementations
│   │   ├── llm.py           # LLM service
│   │   ├── tts*.py          # TTS implementations
│   │   ├── audio.py         # Audio preprocessing
│   │   ├── pipeline.py      # Pipeline orchestrator
│   │   └── session.py       # Session management
│   ├── utils/                # Utilities
│   │   ├── logging.py       # Enhanced logging
│   │   └── text_utils.py    # Text normalization
│   ├── data/                 # Data files
│   └── config.py            # Configuration
├── tests/                    # Test suite
├── docs/                     # Documentation
├── .env.example             # Environment template
├── requirements.txt         # Dependencies
├── pyproject.toml          # Package metadata
├── LICENSE                 # MIT License
└── README.md              # This file
```

## 🛠️ Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

### Key Points

- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Use type hints
- Write descriptive commit messages

## 🐛 Troubleshooting

### Common Issues

**1. Module not found errors**

```bash
# Install package in development mode
pip install -e .
```

**2. ASR k2 dependency errors**

```bash
# Use ONNX backend instead
export ASR_USE_ONNX=true
export ASR_USE_PYTORCH_CUDA=false
```

**3. TTS errors**

```bash
# Use Edge-TTS fallback
export TTS_BACKEND=edge-tts
```

**4. CUDA out of memory**

```bash
# Use CPU for TTS
export TTS_DEVICE=cpu
```

**5. WebSocket connection fails**

```bash
# Check server is running
python -m voice_assistant.api.server

# Check firewall settings
```

### Debug Mode

Enable debug logging:

```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
python -m voice_assistant.api.server
```

## 📝 CV/Portfolio Highlights

### For Your CV

- ✅ Real-time streaming pipeline with <3s latency
- ✅ Multiple backend support (ONNX/PyTorch/Edge-TTS)
- ✅ Production-ready async architecture
- ✅ Comprehensive audio preprocessing
- ✅ Modern web UI with WebSocket
- ✅ Session management with persistence
- ✅ Text normalization for Vietnamese
- ✅ Error handling and timeouts
- ✅ 90+ unit tests

### Demo Video Script

1. Open web UI at localhost:8000
2. Click "Connect" and "Record"
3. Ask: "Việt Nam có bao nhiêu tỉnh thành?"
4. Show real-time transcription
5. Show LLM response streaming
6. Show audio visualizer
7. Demonstrate interrupt feature
8. Show conversation history

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Nguyen Dev**

- GitHub: [@nguyendevwk](https://github.com/nguyendevwk)
- Email: <phamnguyen.devwk@gmail.com>

## 🙏 Acknowledgments

- Gipformer ASR model by G-Group AI Lab
- Qwen-TTS model by G-Group AI Lab
- Silero VAD by Silero Team
- Groq for fast LLM inference
- Microsoft Azure for Edge-TTS

## 🔗 Links

- [Architecture Documentation](ARCHITECTURE.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Gipformer Model](https://huggingface.co/g-group-ai-lab/gipformer-65M-rnnt)
- [Qwen-TTS Model](https://huggingface.co/g-group-ai-lab/gwen-tts-0.6B)
- [Groq API](https://groq.com/)

---

**Note**: This is a personal project suitable for CV/portfolio demonstration. For production use, consider additional security measures, monitoring, and scalability improvements.
