# Architecture

## Overview

```
┌─────────────────────────────────────────────────┐
│              Pipeline Orchestrator               │
│                                                  │
│  Audio → VAD → ASR → LLM → TTS → Audio Output   │
│    ↓       ↓      ↓      ↓      ↓                │
│  PCM16  Silero  ONNX   Groq  VieNeu/Edge        │
└─────────────────────────────────────────────────┘
```

## Components

### Voice Activity Detection (VAD)
- Model: Silero VAD
- Input: 100ms PCM16 chunks @ 16kHz
- Output: speech start/end events
- Threshold: 0.75, silence: 500ms

### Speech Recognition (ASR)
- Primary: ONNX (sherpa-onnx)
- Fallback: Whisper
- Model: Gipformer-65M-RNNT
- Output: Vietnamese text

### Language Model (LLM)
- Provider: Groq (default), Anthropic, Gemini, FallbackProvider
- Model: llama-3.3-70b-versatile
- Streaming: token-by-token
- Chunking: sentence-level for TTS
- Features: Provider fallback, health checks, metrics

### Text-to-Speech (TTS)
Priority:
1. **VieNeu-TTS** - CPU, fast, offline
2. **Qwen-TTS** - GPU, voice cloning
3. **Edge-TTS** - Online fallback

## Data Flow

```
1. Audio Input (16kHz PCM16)
   ↓
2. VAD → detect speech start/end
   ↓
3. ASR → transcribe to Vietnamese text
   ↓
4. LLM → generate response (streaming)
   ↓
5. TTS → synthesize audio (per sentence)
   ↓
6. Audio Output (16kHz PCM16)
```

## Project Structure

```
voice_assistant/
├── __init__.py            # Lazy public API
├── config.py              # Settings (pydantic-settings)
├── core/
│   ├── pipeline.py        # PipelineOrchestrator (main entry point)
│   ├── vad.py             # Voice Activity Detection
│   ├── asr.py             # ASR service
│   ├── asr_onnx.py        # ONNX implementation
│   ├── asr_pytorch.py     # PyTorch CUDA
│   ├── llm.py             # LLM service
│   ├── llm_base.py        # Message, BaseLLMProvider
│   ├── llm_providers.py   # Anthropic, Gemini, etc.
│   ├── llm_fallback.py    # FallbackProvider with metrics
│   ├── llm_extended.py    # Extended LLM capabilities
│   ├── llm_tasks.py       # LLM task utilities
│   ├── tts.py             # TTS service (unified)
│   ├── audio.py           # Audio preprocessing
│   ├── session.py         # Session management (dirty flag)
│   └── warmup.py          # Model warmup utilities
├── api/
│   ├── server.py          # FastAPI + WebSocket (lifespan)
│   └── static/
│       └── index.html     # Test UI
├── cli/
│   └── main.py            # CLI interface
└── utils/
    ├── logging.py         # Logging utilities
    └── text_utils.py      # Text normalization
```

## Public API

### Quick Start

```python
from voice_assistant import PipelineOrchestrator

# Full pipeline
pipeline = PipelineOrchestrator()
async for event in pipeline.process_text("Xin chào"):
    print(event.type, event.data)
```

### Individual Services

```python
from voice_assistant import get_llm_service, get_tts_service

llm = get_llm_service()
tts = get_tts_service()

# Use individually
response = await llm.generate_response("Hello")
audio = await tts.synthesize("Xin chào")
```

### Message Class

```python
from voice_assistant import Message

msg = Message(content="Hello", role="user", metadata={"source": "test"})
```

## Configuration

```bash
# ASR
ASR_USE_ONNX=true
ASR_DEVICE=auto

# TTS
TTS_BACKEND=auto      # auto/vieneu/vieneu_remote/qwen/edge
TTS_SPEECH_RATE=1.25

# LLM
LLM_PROVIDER=groq     # groq/anthropic/gemini
GROQ_API_KEY=your_key

# Timeouts
PIPELINE_ASR_TIMEOUT=10
PIPELINE_LLM_TIMEOUT=30
PIPELINE_TTS_TIMEOUT=15
```

## Performance

| Stage | Latency |
|-------|---------|
| VAD | ~10ms |
| ASR | <1s |
| LLM first token | <1s |
| TTS | <2s |
| **Total** | **<3s** |

## Resource Usage

- RAM: 2-4GB (ONNX), 4-8GB (PyTorch)
- VRAM: 2-4GB (TTS GPU only)
- CPU: 2-4 cores

## Module Features

- Lazy imports (~0.1s import time)
- Thread-safe singleton initialization
- Public API methods (cancel_pipeline, set_history, get_history)
- Dirty flag for session persistence
- Provider fallback with health checks
