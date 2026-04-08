# Vietnamese Voice Assistant - Architecture

## 🎯 Overview

End-to-end real-time Vietnamese voice assistant with streaming ASR, LLM, and TTS pipeline.

```
┌─────────────┐
│   Client    │ (Web UI / CLI)
│  (Browser)  │
└──────┬──────┘
       │ WebSocket
       ↓
┌─────────────────────────────────────────────────┐
│            FastAPI Server                        │
│  ┌───────────────────────────────────────────┐ │
│  │      Session Manager                       │ │
│  │  • Session state tracking                  │ │
│  │  • Conversation history                    │ │
│  │  • Auto cleanup                           │ │
│  └───────────────────────────────────────────┘ │
└───────────────┬─────────────────────────────────┘
                │
                ↓
┌─────────────────────────────────────────────────┐
│         Pipeline Orchestrator                    │
│                                                  │
│  Audio Input → VAD → ASR → LLM → TTS → Output   │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │   VAD    │→ │   ASR    │→ │   LLM    │      │
│  │ (Silero) │  │(Gipformer)│  │ (Groq)  │      │
│  └──────────┘  └──────────┘  └────┬─────┘      │
│                                    │             │
│                               ┌────↓─────┐      │
│                               │   TTS    │      │
│                               │(Qwen-TTS)│      │
│                               └──────────┘      │
└─────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
voice_assistant/
├── api/                    # Web API & UI
│   ├── server.py          # FastAPI server with WebSocket
│   └── static/
│       └── index.html     # Modern web UI
│
├── cli/                    # Command-line interface
│   └── main.py            # CLI entry point
│
├── core/                   # Core components
│   ├── vad.py             # Voice Activity Detection
│   ├── asr.py             # ASR service (multi-backend)
│   ├── asr_onnx.py        # ONNX ASR implementation
│   ├── asr_pytorch.py     # PyTorch CUDA ASR
│   ├── audio.py           # Audio preprocessing
│   ├── llm.py             # LLM streaming service
│   ├── tts.py             # TTS service (multi-backend)
│   ├── pipeline.py        # Pipeline orchestrator
│   ├── session.py         # Session management
│   └── streaming.py       # Streaming pipeline (alternative)
│
├── utils/                  # Utilities
│   ├── logging.py         # Enhanced logging
│   └── text_utils.py      # Text normalization
│
├── data/                   # Data files
│   └── ref_info.json      # TTS speaker references
│
└── config.py              # Configuration management
```

## 🔄 Data Flow

### 1. Audio Input Processing
```
Raw Audio (PCM16)
    ↓
Audio Preprocessing
    ├── DC offset removal
    ├── Noise reduction (spectral gating)
    ├── Band-pass filter (80Hz-7600Hz)
    ├── RMS normalization (-20dB)
    └── Pre-emphasis (α=0.97)
    ↓
Preprocessed Audio → VAD
```

### 2. Voice Activity Detection (VAD)
```
Audio Chunks (100ms)
    ↓
Silero VAD Model
    ├── Speech probability
    ├── Start/End detection
    └── Silence tracking (500ms threshold)
    ↓
Speech Segments → ASR
```

### 3. Speech Recognition (ASR)
```
Speech Audio
    ↓
Feature Extraction (Fbank 80-dim)
    ↓
Gipformer Model (ONNX/PyTorch)
    ├── Encoder: Conformer
    ├── Decoder: RNNT
    └── Beam search decoding
    ↓
Vietnamese Text → LLM
```

### 4. Language Model (LLM)
```
User Query
    ↓
Context + History
    ↓
LLM API (Groq/OpenAI)
    ├── Token streaming
    └── Sentence chunking
    ↓
Response Sentences → TTS
```

### 5. Text-to-Speech (TTS)
```
Text Sentences
    ↓
Text Normalization
    ├── Remove markdown
    ├── Clean special chars
    └── Vietnamese abbreviations
    ↓
Qwen-TTS / Edge-TTS
    ├── Voice cloning (Qwen)
    └── Speed adjustment
    ↓
Audio Output (PCM16, 16kHz)
```

## 🔌 Backend Options

### ASR Backends (Priority Order)
1. **ONNX** (Default) - Fast, simple, no k2 dependency
2. **PyTorch CUDA** - Best quality, requires k2/icefall
3. **Whisper** - Universal fallback

### TTS Backends (Priority Order)
1. **Qwen-TTS** - Best quality, voice cloning, needs GPU
2. **Edge-TTS** - Reliable fallback, Microsoft Azure

## ⚙️ Key Features

### Async Streaming
- Non-blocking I/O throughout
- Progressive audio output
- Low latency (<3s first response)

### Session Management
- UUID-based sessions
- Conversation history (max 50 messages)
- Auto-cleanup (30min timeout)
- Optional disk persistence

### Error Handling
- Timeouts: ASR 10s, LLM 30s, TTS 15s
- Graceful degradation
- Backend fallback
- Retry logic

### Performance Optimizations
- Lazy model loading
- Audio preprocessing pipeline
- Text normalization
- Streaming sentence-by-sentence

## �� Configuration

### Environment Variables
```bash
# LLM API
GROQ_API_KEY=your_key
LLM_MODEL=llama-3.3-70b-versatile

# ASR Backend
ASR_DEVICE=auto
ASR_USE_ONNX=true
ASR_USE_PYTORCH_CUDA=false

# TTS Backend
TTS_BACKEND=auto
TTS_SPEECH_RATE=1.25
TTS_DEVICE=cuda

# Pipeline Timeouts
PIPELINE_ASR_TIMEOUT=10
PIPELINE_LLM_TIMEOUT=30
PIPELINE_TTS_TIMEOUT=15

# Session
SESSION_TIMEOUT=1800
SESSION_PERSISTENCE=false

# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
DEBUG=false
```

## 📊 Performance Metrics

### Latency Targets
- VAD: ~10ms per chunk
- ASR (interim): ~800ms
- ASR (final): <3s
- LLM first token: <1s
- TTS per sentence: <2s
- Total pipeline: <5s

### Resource Usage
- RAM: 2-4GB (ONNX), 4-8GB (PyTorch)
- VRAM: 2-4GB (TTS), 1-2GB (ASR)
- CPU: 2-4 cores recommended
- GPU: CUDA 11.8+ for optimal performance

## 🔐 Security Considerations

### API Keys
- Stored in .env (not committed)
- Accessed via environment variables
- No hardcoded credentials

### WebSocket
- Session-based authentication
- Client connection tracking
- Auto-disconnect on errors

### Data Privacy
- No conversation logging by default
- Optional session persistence
- Clear session data on cleanup

## 🚀 Deployment Options

### Local Development
```bash
python -m voice_assistant.api.server
```

### Production (Uvicorn)
```bash
uvicorn voice_assistant.api.server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

### Docker (Future)
```dockerfile
FROM python:3.12
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "-m", "voice_assistant.api.server"]
```

## 📈 Scalability

### Current: Monolith
- Single process
- In-memory sessions
- Local model loading

### Future: Microservices
- Separate ASR/LLM/TTS services
- Redis for session store
- Model serving (Triton)
- Load balancing

## 🔍 Monitoring & Logging

### Logging Levels
- DEBUG: Detailed traces
- INFO: Component events
- WARNING: Fallbacks, retries
- ERROR: Failures

### Metrics Tracked
- Latency per stage
- Token counts
- Error rates
- Session statistics

## 📚 References

### Models
- Gipformer: https://huggingface.co/g-group-ai-lab/gipformer-65M-rnnt
- Qwen-TTS: https://huggingface.co/g-group-ai-lab/gwen-tts-0.6B
- Silero VAD: https://github.com/snakers4/silero-vad

### APIs
- Groq: https://groq.com/
- OpenAI: https://openai.com/
- Edge-TTS: https://github.com/rany2/edge-tts
