# Bug Fixes & Feature Enhancements

## Latest Update: VieNeu-TTS 0.3B Local Mode

**Changed from remote mode to local 0.3B model for optimal performance.**

### What Changed:
- ✅ **Removed remote mode** - No longer using external VieNeu server
- ✅ **Added local 0.3B model** - `pnnbao-ump/VieNeu-TTS-0.3B` for best speed/quality
- ✅ **Multiple mode support** - `turbo`, `standard`, `fast`, `turbo_gpu`
- ✅ **Python SDK direct usage** - Using Vieneu factory function locally

### Configuration:
```bash
# Recommended setup
TTS_BACKEND=vieneu
VIENEU_MODE=turbo
VIENEU_MODEL_ID=pnnbao-ump/VieNeu-TTS-0.3B
```

### Benefits:
- 🚀 **Faster** - No network latency, local inference
- 🔒 **Private** - All processing happens locally
- 💰 **Free** - No server costs
- 📦 **Simple** - Just `pip install vieneu`

### Files Updated:
- `voice_assistant/config.py` - Changed to local mode config
- `voice_assistant/core/tts.py` - Simplified VieNeuTTSProvider
- `.env.example` - Updated environment variables
- `README.md` - Updated TTS backend table
- `docs/VIENEU_TTS_03B.md` - New comprehensive guide
- `examples/vieneu_03b_example.py` - New working examples

---

## Previous Changes

### 1. **IndentationError in llm.py** (CRITICAL)
**Location:** `voice_assistant/core/llm.py:338`

**Issue:** 
Duplicate lines at the end of the file caused an `IndentationError`, preventing the module from loading.

```python
# Before (BROKEN):
    return _llm_service
          _llm_service = LLMService()
    return _llm_service

# After (FIXED):
    return _llm_service
```

**Impact:** Server couldn't start at all - complete failure.

**Fix:** Removed duplicate lines (lines 338-339).

---

### 2. **Missing Final Response Event in Text Mode** (MAJOR)
**Location:** `voice_assistant/core/pipeline.py:499-514`

**Issue:**
The `process_text()` method didn't:
- Add user messages to conversation history
- Emit a final response event with `is_final=True`
- Track the full assistant response

This caused:
- Session history to be incomplete (missing user messages)
- Client couldn't determine when response was complete
- Conversation context lost between requests

**Fix:**
```python
# Added user message to history
self._conversation_history.append(Message(role="user", content=text))

# Track full response
full_response = ""
async for sentence in self.llm.generate_response_stream(text):
    yield PipelineEvent(type="response", data={"text": sentence, "is_final": False})
    full_response += sentence + " "

# Added assistant response and final event
if full_response.strip():
    self._conversation_history.append(
        Message(role="assistant", content=full_response.strip())
    )
    yield PipelineEvent(
        type="response",
        data={"text": "", "full_text": full_response.strip(), "is_final": True}
    )
```

---

### 3. **User Messages Not Saved to Session in Text Mode** (MAJOR)
**Location:** `voice_assistant/api/server.py:420-425`

**Issue:**
When processing text input via WebSocket, the server didn't add user messages to the session object, only assistant responses were saved.

**Impact:** Session history showed only assistant responses, missing user questions.

**Fix:**
```python
if msg_type == "text":
    user_text = message.get("text", "").strip()
    if user_text:
        # Add user message to session history
        session.add_message("user", user_text)
    
    session.set_state(ConversationState.PROCESSING)
    async for event in orchestrator.process_text(user_text):
        await manager._handle_event(client_id, session, event)
    session.set_state(ConversationState.IDLE)
```

---

### 4. **Wrong Default Value for min_tts_chunk_chars** (MINOR)
**Location:** `voice_assistant/config.py:120`

**Issue:**
Config had `min_tts_chunk_chars = 5` but test expected `20`. The pipeline code used `max(8, settings.pipeline.min_tts_chunk_chars)`, meaning 5 was too small and would only use 8 characters minimum.

**Impact:** TTS could fail on short texts (edge-tts especially fails on very short inputs).

**Fix:** Changed default from `5` to `20` to match test expectations and improve TTS stability.

---

## Test Results

### Before Fixes:
- ❌ Server couldn't start (IndentationError)
- ❌ Session history incomplete
- ❌ No final response events in text mode
- ❌ 1 test failure (config value mismatch)

### After Fixes:
- ✅ Server starts successfully
- ✅ All 76 unit tests pass
- ✅ All 4 comprehensive integration tests pass
- ✅ Full conversation flow works correctly
- ✅ Session history properly tracks all messages
- ✅ Audio processing and VAD work as expected
- ✅ Error handling works properly

---

## Files Modified

1. `voice_assistant/core/llm.py` - Removed duplicate lines
2. `voice_assistant/core/pipeline.py` - Added proper history tracking and final events in text mode
3. `voice_assistant/api/server.py` - Added user message saving in text mode
4. `voice_assistant/config.py` - Fixed min_tts_chunk_chars default value

---

## Verification

All fixes verified with:
1. Unit tests: `uv run pytest tests/ -v` → 76 passed
2. Integration tests: `uv run python test_comprehensive.py` → 4/4 passed
3. Manual WebSocket testing
4. Server startup and health checks
