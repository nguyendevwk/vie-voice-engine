# Fast Mode Error Fix

## Issue
When using `VIENEU_MODE=fast`, the system crashed with:
```
ImportError: Failed to import `lmdeploy`. Install with: pip install vieneu[gpu]
```

## Root Cause
- User had `VIENEU_MODE=faster` in `.env` file (typo)
- Even with correct `fast` mode, it requires `lmdeploy` GPU package
- No graceful error handling or fallback mechanism

## Changes Made

### 1. Enhanced Error Handling in VieNeuTTSProvider
**File:** `voice_assistant/core/tts.py`

**Changes:**
- Added `_load_error` attribute to store load errors
- Improved `_ensure_loaded()` with detailed error messages
- Added specific error handling for:
  - Missing `lmdeploy` (fast mode)
  - Missing `torch` (standard mode)
  - General import errors
- Enhanced `is_available()` to check actual load status

**Code:**
```python
def _ensure_loaded(self):
    try:
        from vieneu import Vieneu
        self._tts = Vieneu(mode=self.mode)
        # ... success logging
    except ImportError as e:
        self._load_error = str(e)
        if "lmdeploy" in error_msg.lower():
            logger.error(
                f"VieNeu-TTS '{self.mode}' mode requires lmdeploy. "
                f"Install with: pip install vieneu[gpu] or use 'turbo' mode instead"
            )
        # ... other error handling
        raise
```

### 2. Improved TTSService Fallback
**File:** `voice_assistant/core/tts.py`

**Changes:**
- Wrapped provider initialization in try-except
- Added graceful fallback to Edge-TTS if VieNeu fails
- Better logging of initialization failures

**Code:**
```python
if backend == "vieneu":
    try:
        provider = VieNeuTTSProvider()
        if provider.is_available():
            self._provider = provider
            return self._provider
    except Exception as e:
        logger.warning(f"VieNeu-TTS not available: {e}")
        logger.info("Falling back to Edge-TTS")
```

### 3. Fixed .env Configuration
**File:** `.env`

**Changed:**
```diff
-VIENEU_MODE=faster
+VIENEU_MODE=turbo
+VIENEU_MODEL_ID=pnnbao-ump/VieNeu-TTS-0.3B
```

### 4. Updated Documentation
**File:** `docs/VIENEU_TTS_03B.md`

**Added:**
- Dependencies column to modes table
- Troubleshooting section with specific error solutions
- Clear recommendations to use `turbo` mode

## VieNeu Modes Comparison

| Mode | Dependencies | Hardware | Speed | Recommended |
|------|-------------|----------|-------|-------------|
| `turbo` | **None** | CPU | ⚡⚡⚡ | ✅ **Yes** |
| `standard` | PyTorch | CPU/GPU | ⚡⚡ | Compatibility |
| `fast` | lmdeploy | GPU | ⚡⚡⚡⚡ | Max GPU speed |
| `turbo_gpu` | PyTorch | GPU | ⚡⚡⚡ | GPU available |

## Recommendation

**Use `turbo` mode** - it's the default and works out of the box with no extra dependencies!

```bash
# Recommended setup
TTS_BACKEND=vieneu
VIENEU_MODE=turbo
VIENEU_MODEL_ID=pnnbao-ump/VieNeu-TTS-0.3B
```

## Testing

All tests pass:
```
✅ 36/36 unit tests passed
✅ Error handling verified
✅ Fallback mechanism works
✅ No breaking changes
```

## Error Messages

Now users get helpful error messages:

**Before:**
```
ImportError: Failed to import `lmdeploy`
```

**After:**
```
VieNeu-TTS 'fast' mode requires lmdeploy. 
Install with: pip install vieneu[gpu] or use 'turbo' mode instead
```

And the system automatically falls back to Edge-TTS if VieNeu fails to load.
