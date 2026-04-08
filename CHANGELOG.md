# Changelog

## [1.0.0] - 2026-04-08

### Added
- Vietnamese ASR with Gipformer (ONNX/PyTorch)
- LLM integration (Groq/OpenAI) with streaming
- TTS backends: VieNeu-TTS, Qwen-TTS, Edge-TTS
- Voice Activity Detection (Silero VAD)
- Web UI with WebSocket streaming
- CLI interface
- Session management
- Audio preprocessing pipeline
- Text normalization for Vietnamese

### Performance
- End-to-end latency: <3s
- ASR: <1s
- LLM first token: <1s
- TTS: <2s per sentence
