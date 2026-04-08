# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-08

### Added
- Initial release of Vietnamese Voice Assistant
- Real-time ASR with Gipformer (ONNX/PyTorch backends)
- LLM integration with Groq/OpenAI (token streaming)
- TTS with Qwen-TTS and Edge-TTS fallback
- Voice Activity Detection with Silero VAD
- Modern web UI with audio visualizer
- CLI interface for testing
- Session management with conversation history
- Comprehensive audio preprocessing pipeline
- Text normalization for Vietnamese TTS
- Multi-backend support (ONNX priority, PyTorch fallback)
- Timeout protection (ASR 10s, LLM 30s, TTS 15s)
- Streaming pipeline for low latency (<3s first response)
- 90+ unit and integration tests
- Complete API documentation
- Architecture documentation
- Contributing guidelines

### Features
- **ASR**: Gipformer 65M RNNT model
  - ONNX backend (fast, no k2 dependency)
  - PyTorch CUDA backend (best quality)
  - Whisper fallback
  - Real-time streaming with interim results

- **LLM**: Groq/OpenAI integration
  - Token streaming
  - Sentence chunking for TTS
  - Conversation history (max 50 messages)
  - Configurable models

- **TTS**: Multi-backend support
  - Qwen-TTS with voice cloning (1.25x speed)
  - Edge-TTS fallback (Microsoft Azure)
  - Text normalization and cleaning
  - Markdown/special char removal
  - Vietnamese abbreviation expansion

- **VAD**: Silero model
  - Speech start/end detection
  - 500ms silence threshold
  - Consecutive chunk tracking
  - Real-time probability reporting

- **Pipeline**: Async orchestration
  - State machine (IDLE → LISTENING → PROCESSING → SPEAKING)
  - Interrupt support
  - Error handling with graceful degradation
  - Latency tracking per stage
  - Session persistence (optional)

- **Web UI**: Modern interface
  - Dark theme
  - Real-time audio visualizer (32 bars)
  - Conversation display
  - Debug panel
  - WebSocket streaming
  - Mic recording (WebAudio API)

- **Audio Processing**: Comprehensive pipeline
  - DC offset removal
  - Spectral gating noise reduction
  - Band-pass filter (80Hz-7600Hz)
  - RMS normalization (-20dB)
  - Pre-emphasis filter (α=0.97)
  - Dynamic range compression

### Technical
- Python 3.12+ with type hints
- Async/await throughout
- FastAPI WebSocket server
- Lazy model loading
- Singleton services
- Environment-based configuration
- Comprehensive logging
- pytest test suite

### Performance
- VAD: ~10ms per 100ms chunk
- ASR (ONNX): <1s for short utterances
- LLM first token: <1s (Groq)
- TTS: <2s per sentence
- End-to-end: <3s first response

### Known Limitations
- Qwen-TTS requires reference audio for voice cloning
- PyTorch ASR requires k2/icefall setup
- Edge-TTS rate limited to 1.25x for stability
- Session persistence to disk is optional (off by default)

## [Unreleased]

### Planned
- Docker containerization
- Redis session storage
- Prometheus metrics
- Model serving with Triton
- Microservices architecture
- Load balancing
- WebRTC support
- Mobile app (React Native)

---

[1.0.0]: https://github.com/nguyendevwk/end2end_asr_tts_vie/releases/tag/v1.0.0
