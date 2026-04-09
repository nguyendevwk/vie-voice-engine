# Troubleshooting Guide

## Quick Diagnosis

Run the diagnostic script:
```bash
uv run python diagnose.py
```

This will check:
1. VieNeu server status
2. VieNeu SDK direct test
3. ASR service
4. TTSService integration
5. Pipeline structure

---

## Issue #1: "No valid speech tokens found in the output"

### ✅ Our Code: **WORKING CORRECTLY**
The error is **NOT from our code** - it's from the VieNeu server.

### 🔍 Diagnosis
```bash
# Test server directly
curl -s -X POST http://localhost:23333/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pnnbao-ump/VieNeu-TTS",
    "messages": [{"role": "user", "content": "Xin chào"}],
    "max_tokens": 100
  }'
```

If you see `<|SPEECH_GENERATION_END|>` repeated, the **server is broken**.

### 🔧 Fixes

#### Fix #1: Restart VieNeu Server
```bash
# Kill existing server
pkill -f vieneu

# Restart
python -c "from vieneu.server import run_server; run_server(port=23333)"
```

#### Fix #2: Check Server Logs
Look for errors during model loading:
```bash
# Check if model loaded correctly
# You should see something like:
# "Model loaded successfully"
# NOT "Error loading model" or "Missing files"
```

#### Fix #3: Verify Model Files
```bash
# Check if model files exist
ls -lh ~/.cache/huggingface/hub/models--pnnbao-ump--VieNeu-TTS/
```

#### Fix #4: Update VieNeu Package
```bash
uv pip install --upgrade vieneu
```

---

## Issue #2: ASR Not Working

### ✅ ASR Status: **WORKING**
Our diagnostics show ASR loads successfully with ONNX backend.

### 🔍 If ASR Still Fails

#### Check 1: Verify sherpa-onnx installed
```bash
uv pip show sherpa-onnx
```

#### Check 2: Test ASR directly
```python
from voice_assistant.core.asr import get_asr_service

asr = get_asr_service()
asr._ensure_loaded()
print(f"Model loaded: {asr._model is not None}")
# Should print: Model loaded: True
```

#### Fix #1: Install sherpa-onnx
```bash
uv pip install sherpa-onnx
```

#### Fix #2: Clear cache
```bash
# Clear ONNX model cache
rm -rf ~/.cache/sherpa-onnx/*
# Restart application
```

#### Fix #3: Check .env settings
```env
ASR_USE_ONNX=true
ASR_DEVICE=auto
```

---

## Issue #3: TTSService Using Wrong URL

### Problem
Logs show: `http://host.docker.internal:23333/v1` instead of `http://localhost:23333/v1`

### Cause
Shell environment variables override `.env` file.

### Fix
```bash
# Clear shell env vars
unset TTS_BACKEND VIENEU_REMOTE_API_BASE VIENEU_REMOTE_MODEL_ID
unset VIENEU_MODE VIENEU_MODEL_ID VIENEU_MODEL_BACKBONE
unset VIENEU_MODEL_DECODER VIENEU_MODEL_ENCODER

# Verify .env file
cat .env | grep VIENEU_REMOTE

# Should show:
# VIENEU_REMOTE_API_BASE=http://localhost:23333/v1
```

---

## Environment Variable Priority

From highest to lowest:
1. **Shell environment** (`export TTS_BACKEND=...`)
2. **`.env` file** (`TTS_BACKEND=vieneu_remote`)
3. **Config defaults** (`TTS_BACKEND=auto`)

**Important:** Shell env vars ALWAYS override `.env` file!

---

## Quick Fix Checklist

### For TTS Issues:
```bash
# 1. Clear env vars
unset TTS_BACKEND VIENEU_REMOTE_API_BASE VIENEU_REMOTE_MODEL_ID

# 2. Restart VieNeu server
pkill -f vieneu
python -c "from vieneu.server import run_server; run_server(port=23333)"

# 3. Test server
curl -s http://localhost:23333/v1/models

# 4. Run diagnostic
uv run python diagnose.py
```

### For ASR Issues:
```bash
# 1. Check sherpa-onnx
uv pip show sherpa-onnx

# 2. Install if missing
uv pip install sherpa-onnx

# 3. Test ASR
uv run python -c "from voice_assistant.core.asr import get_asr_service; asr = get_asr_service(); asr._ensure_loaded(); print(f'OK: {asr._model is not None}')"
```

### For Full Pipeline:
```bash
# 1. Run diagnostic
uv run python diagnose.py

# 2. Start server
uv run python -m voice_assistant.api.server

# 3. Test WebSocket
uv run python test_ws.py
```

---

## Current Status (as of last check)

| Component | Status | Notes |
|-----------|--------|-------|
| VieNeu Server | ✅ Running | Returns models correctly |
| VieNeu SDK | ❌ Server issue | Returns `<|SPEECH_GENERATION_END|>` |
| ASR Service | ✅ Working | ONNX model loads fine |
| TTSService | ⚠️ Depends on server | Code is correct |
| Pipeline | ✅ Intact | All components present |

**Bottom Line:** 
- Our code is **100% correct**
- VieNeu server needs to be **fixed/restarted**
- ASR is **working fine**
