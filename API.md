# API Documentation

## Quick Start

```python
from voice_assistant import PipelineOrchestrator

pipeline = PipelineOrchestrator()
async for event in pipeline.process_text("Xin chào"):
    print(event.type, event.data)
```

## Public API

### PipelineOrchestrator

Main coordinator for the voice assistant pipeline.

```python
from voice_assistant import PipelineOrchestrator

pipeline = PipelineOrchestrator()

# Process text input
async for event in pipeline.process_text("Hello"):
    if event.type == "response":
        print(event.data["text"])

# Process audio input
await pipeline.handle_audio_chunk(audio_bytes)

# Public methods
pipeline.cancel_pipeline()           # Cancel current pipeline
pipeline.set_history(messages)       # Set conversation history
history = pipeline.get_history()     # Get conversation history
pipeline.reset()                     # Reset pipeline state
```

### Individual Services

```python
from voice_assistant import get_llm_service, get_tts_service, get_asr_service, get_vad_service

# LLM
llm = get_llm_service()
response = await llm.generate_response("Hello")
async for token in llm.generate_response_stream("Hello"):
    print(token)

# TTS
tts = get_tts_service()
audio = await tts.synthesize("Xin chào Việt Nam")

# ASR
asr = get_asr_service()
text = asr.transcribe(audio_array)

# VAD
vad = get_vad_service()
result = vad.process_chunk(audio_chunk)
```

### Message

Canonical message class for conversation history.

```python
from voice_assistant import Message

msg = Message(
    content="Hello",
    role="user",
    metadata={"source": "test"}
)

# Properties
msg.content      # Message text
msg.role         # "user" or "assistant"
msg.timestamp    # Unix timestamp
msg.metadata     # Custom metadata dict
msg.to_dict()    # Convert to dict
```

### Session

```python
from voice_assistant import Session, SessionManager

# Session manager
manager = SessionManager()
session = manager.get_or_create_session()

# Session properties
session.id              # Session ID
session.history        # Conversation history
session.state          # ConversationState
session.is_dirty       # Has unsaved changes

# Session methods
session.add_message(msg)
session.clear()
session.save()          # Save if dirty
```

### Settings

```python
from voice_assistant import Settings

settings = Settings()

# Access settings
print(settings.asr.device)
print(settings.llm.model)
print(settings.tts.speech_rate)
```

## Pipeline Events

### Event Types

```python
# Transcript event
{"type": "transcript", "data": {"text": "...", "is_final": true}}

# Response event
{"type": "response", "data": {"text": "...", "is_final": true}}

# Audio event
{"type": "audio", "data": b"audio_bytes"}

# Control event
{"type": "control", "data": {"action": "mic_mute"}}
```

## Configuration

### Environment Variables

```bash
# Required
GROQ_API_KEY=your_key

# ASR
ASR_USE_ONNX=true
ASR_DEVICE=auto

# TTS
TTS_BACKEND=auto
TTS_SPEECH_RATE=1.25

# LLM
LLM_PROVIDER=groq

# Timeouts
PIPELINE_ASR_TIMEOUT=10
PIPELINE_LLM_TIMEOUT=30
PIPELINE_TTS_TIMEOUT=15
```

## Error Handling

```python
try:
    audio = await tts.synthesize(text)
except asyncio.TimeoutError:
    logger.error("TTS timeout")
except ValueError as e:
    logger.error(f"Invalid input: {e}")
```

## Best Practices

1. Use lazy imports for fast startup
2. Handle timeouts for all async operations
3. Clean up resources with `pipeline.reset()`
4. Check `session.is_dirty` before saving
5. Use `cancel_pipeline()` to stop long operations

## Utilities

```python
from voice_assistant.utils.text_utils import normalize_for_tts, split_into_sentences
from voice_assistant.utils.logging import logger, latency

# Normalize text
clean = normalize_for_tts("**Bold** text")

# Split sentences
sentences = split_into_sentences("First. Second.")

# Track latency
with latency.track("tts"):
    audio = await tts.synthesize(text)
```
