# 🎙️ Vietnamese Voice Assistant Pipeline

> Real-time voice assistant với ASR, LLM và TTS streaming - Phiên bản personal/demo

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## 🎯 Features

- **Real-time Speech Recognition** - Streaming ASR với interim results
- **LLM Integration** - Token streaming cho phản hồi nhanh
- **Voice Synthesis** - TTS chunked streaming
- **Voice Activity Detection** - Silero VAD cho start/end detection
- **Low Latency** - Async pipeline design
- **Interrupt Support** - Có thể ngắt lời bot đang nói

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         VOICE ASSISTANT PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌───────┐ │
│  │  Audio   │   │   VAD    │   │   ASR    │   │   LLM    │   │  TTS  │ │
│  │  Input   │──►│ (Silero) │──►│(Gipformer│──►│ (Groq)   │──►│(Gwen) │ │
│  │          │   │          │   │ stream)  │   │ stream   │   │stream │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └───┬───┘ │
│       │              │              │              │              │     │
│       └──────────────┴──────────────┴──────────────┴──────────────┘     │
│                           Async Queues & Buffers                         │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│  Pipeline Orchestrator: State Management, Interrupt Handling, Timing    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Streaming Flow

```
User speaks      VAD detects      ASR processes       LLM generates      TTS synthesizes
    │            speech start     incrementally       token by token     sentence chunks
    │                 │                │                    │                  │
    ▼                 ▼                ▼                    ▼                  ▼
┌───────┐        ┌────────┐      ┌──────────┐        ┌──────────┐       ┌──────────┐
│ Audio │───────►│ Start  │─────►│ Interim  │───────►│ "Xin"    │──────►│ Chunk 1  │
│ chunk │        │ event  │      │ "xin"    │        │ "chào"   │       │ audio    │
│ 100ms │        └────────┘      │ "xin chào│        │ "bạn"    │       └──────────┘
└───────┘                        │ "xin chào│        │ "..."    │       ┌──────────┐
    │                            │  tôi"    │        └──────────┘──────►│ Chunk 2  │
    ▼                            └──────────┘                           │ audio    │
┌───────┐        ┌────────┐      ┌──────────┐                           └──────────┘
│ More  │───────►│  End   │─────►│ Final:   │
│ audio │        │ event  │      │"xin chào │
└───────┘        │(silence│      │ tôi cần  │
                 │ 500ms) │      │ hỗ trợ"  │
                 └────────┘      └──────────┘
```

## 📁 Project Structure

```
voice_assistant/
├── __init__.py
├── config.py              # Configuration management
├── core/
│   ├── __init__.py
│   ├── audio.py           # Audio preprocessing
│   ├── vad.py             # Voice Activity Detection
│   ├── asr.py             # Speech Recognition
│   ├── llm.py             # Language Model
│   ├── tts.py             # Text-to-Speech
│   └── pipeline.py        # Orchestrator
├── api/
│   ├── __init__.py
│   └── server.py          # FastAPI WebSocket server
├── cli/
│   ├── __init__.py
│   └── main.py            # CLI interface
└── utils/
    ├── __init__.py
    └── logging.py         # Logging utilities
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/nguyendevwk/voice-assistant-vie.git
cd voice-assistant-vie

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Set API keys
export GROQ_API_KEY="your-groq-api-key"

# Optional: Debug mode
export DEBUG=true
```

### Run CLI Mode

```bash
# Interactive voice assistant
python -m voice_assistant.cli.main

# With custom settings
python -m voice_assistant.cli.main --device cuda:0
```

### Run API Server

```bash
# Start WebSocket server
python -m voice_assistant.api.server

# Server runs at ws://localhost:8000/ws
```

## 🔧 Configuration

Edit `voice_assistant/config.py` or use environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key for LLM | Required |
| `ASR_DEVICE` | ASR device (cpu/cuda) | auto |
| `TTS_DEVICE` | TTS device | cuda:0 |
| `DEBUG` | Enable debug logging | false |

## 📊 Latency Breakdown

| Stage | Typical Latency | Notes |
|-------|-----------------|-------|
| VAD | ~10ms | Per 100ms chunk |
| ASR Interim | ~200-400ms | Every 800ms |
| LLM First Token | ~200-500ms | Groq API |
| TTS First Chunk | ~300-600ms | Per sentence |
| **End-to-end** | **~1-2s** | User speaks → Bot responds |

## 💡 CV Highlights

**Technical Skills Demonstrated:**

- **Real-time Audio Processing**: VAD, streaming ASR
- **Async Programming**: Python asyncio, concurrent pipelines
- **API Integration**: LLM streaming (Groq/OpenAI compatible)
- **WebSocket Protocol**: Bidirectional real-time communication
- **System Design**: Modular architecture, buffer management
- **ML Integration**: Speech models (ASR, TTS, VAD)

**Key Achievements:**

- Sub-2s end-to-end latency
- Interrupt handling for natural conversation
- Production-ready code structure
- Comprehensive error handling & logging

## 📝 API Reference

### WebSocket Protocol

**Endpoint:** `ws://localhost:8000/ws`

**Client → Server:**

- Binary: PCM S16LE audio chunks (16kHz, mono, 100ms)

**Server → Client:**

- Binary: TTS audio chunks
- Text JSON:

  ```json
  {"type": "transcript", "text": "...", "is_final": false}
  {"type": "response", "text": "..."}
  {"type": "control", "action": "interrupt"}
  ```

## 🛠️ Development

```bash
# Run tests
pytest tests/

# Format code
black voice_assistant/
isort voice_assistant/

# Type check
mypy voice_assistant/
```

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 👤 Author

**nguyendevwk**

- GitHub: [@nguyendevwk](https://github.com/nguyendevwk)
- Email: <phamnguyen.devwk@gmail.com>

---

> Built with ❤️ for Vietnamese voice AI
