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
- Provider: Groq (default), OpenAI
- Model: llama-3.3-70b-versatile
- Streaming: token-by-token
- Chunking: sentence-level for TTS

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
├── core/
│   ├── vad.py          # Voice Activity Detection
│   ├── asr.py          # ASR service
│   ├── asr_onnx.py     # ONNX implementation
│   ├── asr_pytorch.py  # PyTorch CUDA
│   ├── llm.py          # LLM service
│   ├── llm_base.py     # LLM extension framework
│   ├── tts.py          # TTS service (unified)
│   ├── audio.py        # Audio preprocessing
│   ├── pipeline.py     # Pipeline orchestrator
│   └── session.py      # Session management
├── api/
│   └── server.py       # FastAPI + WebSocket
├── cli/
│   └── main.py         # CLI interface
└── utils/
    ├── logging.py      # Logging utilities
    └── text_utils.py   # Text normalization
```

## Configuration

```bash
# ASR
ASR_USE_ONNX=true
ASR_DEVICE=auto

# TTS
TTS_BACKEND=auto      # auto/vieneu/qwen/edge
TTS_SPEECH_RATE=1.25

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
