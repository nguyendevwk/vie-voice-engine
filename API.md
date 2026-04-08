# API Documentation

## PipelineOrchestrator

Main coordinator for the voice assistant pipeline.

### Constructor

```python
PipelineOrchestrator(
    on_event: Optional[Callable[[PipelineEvent], Awaitable[None]]] = None
)
```

**Parameters:**
- `on_event`: Async callback for pipeline events

**Example:**
```python
async def handle_event(event: PipelineEvent):
    if event.type == "transcript":
        print(f"User said: {event.data['text']}")
    elif event.type == "response":
        print(f"Assistant: {event.data['text']}")

orchestrator = PipelineOrchestrator(on_event=handle_event)
```

### Methods

#### `handle_audio_chunk(audio_bytes: bytes)`

Process raw audio chunk.

**Parameters:**
- `audio_bytes`: PCM S16LE audio data (100ms recommended)

**Returns:**
- None (events sent via callback)

**Example:**
```python
# 100ms chunk at 16kHz, mono, 16-bit
audio_chunk = b'\x00' * 3200
await orchestrator.handle_audio_chunk(audio_chunk)
```

#### `process_text(text: str) -> AsyncIterator[PipelineEvent]`

Process text input directly (skip ASR).

**Parameters:**
- `text`: User query text

**Yields:**
- `PipelineEvent`: Events for response and audio

**Example:**
```python
async for event in orchestrator.process_text("Xin chào"):
    if event.type == "response":
        print(event.data["text"])
```

#### `reset()`

Reset pipeline state and clear buffers.

**Example:**
```python
orchestrator.reset()
```

---

## ASRService

Automatic Speech Recognition service.

### Constructor

```python
ASRService(
    config: Optional[ASRConfig] = None,
    use_onnx: bool = True,
    use_pytorch_cuda: bool = False
)
```

**Parameters:**
- `config`: ASR configuration
- `use_onnx`: Use ONNX backend (faster)
- `use_pytorch_cuda`: Use PyTorch with CUDA

### Methods

#### `transcribe_file(audio_path: str) -> str`

Transcribe audio file to text.

**Parameters:**
- `audio_path`: Path to audio file (WAV, MP3, etc.)

**Returns:**
- `str`: Transcribed Vietnamese text

**Example:**
```python
asr = ASRService()
text = asr.transcribe_file("audio.wav")
print(text)
```

#### `transcribe(audio: np.ndarray, sample_rate: int = 16000) -> str`

Transcribe audio samples to text.

**Parameters:**
- `audio`: Audio samples (float32, mono, -1.0 to 1.0)
- `sample_rate`: Sample rate in Hz

**Returns:**
- `str`: Transcribed text

**Example:**
```python
import numpy as np

audio = np.random.randn(16000).astype(np.float32)
text = asr.transcribe(audio, sample_rate=16000)
```

---

## LLMService

Large Language Model service with streaming.

### Constructor

```python
LLMService(config: Optional[LLMConfig] = None)
```

### Methods

#### `generate_response(prompt: str, history: List[Message] = None) -> str`

Generate complete response (non-streaming).

**Parameters:**
- `prompt`: User query
- `history`: Conversation history

**Returns:**
- `str`: Complete response

**Example:**
```python
llm = LLMService()
response = await llm.generate_response("Việt Nam ở đâu?")
print(response)
```

#### `generate_response_stream(prompt: str, history: List[Message] = None) -> AsyncIterator[str]`

Stream response sentence by sentence.

**Parameters:**
- `prompt`: User query
- `history`: Conversation history

**Yields:**
- `str`: Response sentences

**Example:**
```python
async for sentence in llm.generate_response_stream("Tell me a story"):
    print(sentence)
    # Process sentence immediately for TTS
```

---

## TTSService

Text-to-Speech service with multiple backends.

### Constructor

```python
TTSService(config: Optional[TTSConfig] = None)
```

### Methods

#### `synthesize(text: str, speaker: str = None) -> bytes`

Synthesize text to speech.

**Parameters:**
- `text`: Text to synthesize
- `speaker`: Speaker key (optional)

**Returns:**
- `bytes`: PCM S16LE audio at target sample rate

**Example:**
```python
tts = TTSService()
audio_bytes = await tts.synthesize("Xin chào Việt Nam")

# Save to file
with open("output.raw", "wb") as f:
    f.write(audio_bytes)
```

#### `synthesize_stream(text: str, speaker: str = None) -> AsyncIterator[bytes]`

Stream synthesis sentence by sentence.

**Parameters:**
- `text`: Long text to synthesize
- `speaker`: Speaker key (optional)

**Yields:**
- `bytes`: Audio chunks

**Example:**
```python
long_text = "First sentence. Second sentence. Third sentence."
async for audio_chunk in tts.synthesize_stream(long_text):
    # Play chunk immediately
    play_audio(audio_chunk)
```

---

## VADService

Voice Activity Detection service.

### Constructor

```python
VADService(config: Optional[VADConfig] = None)
```

### Methods

#### `process_chunk(audio_chunk: bytes) -> VADResult`

Process audio chunk for speech detection.

**Parameters:**
- `audio_chunk`: PCM S16LE audio (100ms recommended)

**Returns:**
- `VADResult`: Detection result with event, confidence, latency

**Example:**
```python
vad = VADService()
result = vad.process_chunk(audio_chunk)

if result.event == "speech_start":
    print("Speech started!")
elif result.event == "speech_end":
    print("Speech ended!")
```

---

## SessionManager

Manage conversation sessions.

### Constructor

```python
SessionManager(
    cleanup_interval_s: int = 60,
    enable_persistence: bool = False
)
```

### Methods

#### `get_or_create_session(session_id: str = None, client_id: str = None) -> Session`

Get existing session or create new one.

**Parameters:**
- `session_id`: Optional session ID to resume
- `client_id`: Client identifier

**Returns:**
- `Session`: Session object

**Example:**
```python
manager = SessionManager()
session = manager.get_or_create_session()
print(f"Session ID: {session.id}")
```

#### `delete_session(session_id: str) -> bool`

Delete a session.

**Parameters:**
- `session_id`: Session ID

**Returns:**
- `bool`: True if deleted

---

## Configuration

### Settings

Global settings object loaded from environment variables.

```python
from voice_assistant.config import settings

# Access settings
print(settings.asr.device)
print(settings.llm.model)
print(settings.tts.speech_rate)

# Update at runtime
settings.debug = True
```

### Environment Variables

See `.env.example` for all available options.

**Example .env:**
```bash
GROQ_API_KEY=your_key
ASR_USE_ONNX=true
TTS_BACKEND=auto
TTS_SPEECH_RATE=1.25
DEBUG=false
```

---

## Events

### PipelineEvent

Event emitted by pipeline.

**Attributes:**
- `type`: Event type ("audio", "transcript", "response", "control")
- `data`: Event data (varies by type)

**Event Types:**

1. **transcript**
   ```python
   {
       "text": "Transcribed text",
       "is_final": true,
       "latency_ms": 500
   }
   ```

2. **response**
   ```python
   {
       "text": "AI response",
       "is_final": true,
       "latency_ms": 1000
   }
   ```

3. **audio**
   ```python
   audio_bytes  # PCM S16LE
   ```

4. **control**
   ```python
   {
       "action": "mic_mute",  # or "mic_unmute", "interrupt"
   }
   ```

---

## Utilities

### Text Normalization

```python
from voice_assistant.utils.text_utils import (
    normalize_for_tts,
    clean_vietnamese_text,
    split_into_sentences
)

# Normalize for TTS
text = "**Bold** text with *markdown*"
clean = normalize_for_tts(text)  # "Bold text with markdown"

# Clean Vietnamese
text = "TP.HCM là thành phố lớn"
clean = clean_vietnamese_text(text)  # "thành phố Hồ Chí Minh là thành phố lớn"

# Split sentences
text = "First. Second! Third?"
sentences = split_into_sentences(text)  # ["First.", "Second!", "Third?"]
```

### Logging

```python
from voice_assistant.utils.logging import logger, latency

# Log messages
logger.info("Starting process")
logger.error("Error occurred")

# Track latency
latency.start("process_time")
# ... do work ...
latency.end("process_time")

# Track with context manager
with latency.track("tts_synthesis"):
    audio = await tts.synthesize(text)

# Get tracked times
times = latency.get_all()
print(times)  # {"process_time": 123, "tts_synthesis": 456}
```

---

## Error Handling

All async methods may raise:

- `asyncio.TimeoutError`: Operation timed out
- `ValueError`: Invalid parameters
- `RuntimeError`: Service initialization failed
- `ImportError`: Required package not installed

**Example:**
```python
try:
    audio = await tts.synthesize(text)
except asyncio.TimeoutError:
    logger.error("TTS timeout")
except ValueError as e:
    logger.error(f"Invalid input: {e}")
```

---

## Best Practices

1. **Use Context Managers**
   ```python
   with latency.track("operation"):
       result = await operation()
   ```

2. **Handle Timeouts**
   ```python
   try:
       result = await asyncio.wait_for(operation(), timeout=10)
   except asyncio.TimeoutError:
       # Handle timeout
       pass
   ```

3. **Clean Up Resources**
   ```python
   orchestrator.reset()
   session_manager.cleanup_expired()
   ```

4. **Check Session State**
   ```python
   if session.state == ConversationState.IDLE:
       # Process new input
       pass
   ```

5. **Stream for Long Responses**
   ```python
   async for chunk in tts.synthesize_stream(long_text):
       await play_audio(chunk)
   ```
