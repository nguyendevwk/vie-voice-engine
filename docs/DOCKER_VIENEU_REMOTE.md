# Docker Deployment with VieNeu-TTS Remote Mode

## 🚀 Quick Start for Optimal Latency

This guide shows how to deploy the voice assistant with **VieNeu-TTS Remote mode** for optimal latency in Docker.

### Architecture

```
┌─────────────────────────────────────────────┐
│  Docker Container (Voice Assistant)         │
│  - Lightweight VieNeu client (~50MB)        │
│  - ASR, LLM, Pipeline                       │
│  - No GPU required                          │
└──────────────┬──────────────────────────────┘
               │ HTTP (low latency)
               ↓
┌─────────────────────────────────────────────┐
│  VieNeu Server (Host Machine or Remote)     │
│  - VieNeu-TTS model (does heavy lifting)    │
│  - GPU optional on client side              │
│  - Port: 23333                              │
└─────────────────────────────────────────────┘
```

## Setup Steps

### 1. Start VieNeu Server

First, ensure you have a VieNeu-TTS server running:

```bash
# Option A: Run on host machine (recommended)
python -c "
from vieneu.server import run_server
run_server(port=23333)
"

# Option B: Use existing VieNeu server
# Get the server URL (e.g., http://your-server:23333/v1)
```

### 2. Configure Environment

Create or update `.env` file:

```bash
# Copy example
cp .env.example .env

# Edit .env with your settings
```

**Minimal .env for Remote Mode:**
```env
# Required: LLM API key
GROQ_API_KEY=your-groq-api-key

# TTS: Remote mode for optimal latency
TTS_BACKEND=vieneu_remote
VIENEU_REMOTE_API_BASE=http://host.docker.internal:23333/v1
VIENEU_REMOTE_MODEL_ID=pnnbao-ump/VieNeu-TTS
```

### 3. Build and Run

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Check health
curl http://localhost:8000/health
```

### 4. Test

```bash
# Open web UI
open http://localhost:8000

# Or use CLI
docker-compose exec voice-assistant \
  python -m voice_assistant.cli.main --text-only
```

## Configuration Reference

### Docker Compose Environment Variables

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `TTS_BACKEND` | TTS backend selector | `vieneu_remote` | Use `vieneu_remote` for optimal latency |
| `VIENEU_REMOTE_API_BASE` | VieNeu server URL | `http://host.docker.internal:23333/v1` | Use `host.docker.internal` for Docker Desktop |
| `VIENEU_REMOTE_MODEL_ID` | Model identifier | `pnnbao-ump/VieNeu-TTS` | Match your server model |

### Network Configuration

#### Docker Desktop (Mac/Windows)
```env
VIENEU_REMOTE_API_BASE=http://host.docker.internal:23333/v1
```

#### Linux (Docker Engine)
```env
# Option 1: Host machine IP
VIENEU_REMOTE_API_BASE=http://172.17.0.1:23333/v1

# Option 2: Custom network
# Create network and use service name
```

#### Remote Server
```env
VIENEU_REMOTE_API_BASE=http://your-vieneu-server.com:23333/v1
```

## Performance Optimization

### Latency Breakdown

| Component | Latency | Notes |
|-----------|---------|-------|
| VieNeu Client | ~10ms | Lightweight codec only |
| Network | 50-200ms | Depends on location |
| VieNeu Server | 100-500ms | GPU speed |
| **Total** | **200-800ms** | Much better than local |

### Tips for Optimal Latency

1. **Use same region**: Keep client and server close geographically
2. **Fast network**: Use LAN or low-latency connection
3. **GPU on server**: VieNeu server should have GPU for fast inference
4. **Lightweight client**: Docker container only needs ~50MB for codec

### Resource Usage

#### Docker Container (Client)
```
CPU:    1-2 cores
RAM:    2-4GB
GPU:    Not required
Disk:   ~500MB (models cached)
```

#### VieNeu Server
```
CPU:    2-4 cores
RAM:    4-8GB
GPU:    Recommended (4-6GB VRAM)
Disk:   ~2GB (model storage)
```

## Troubleshooting

### Container Can't Connect to VieNeu Server

**Error:**
```
ConnectionError: http://host.docker.internal:23333/v1
```

**Solutions:**

**Docker Desktop (Mac/Windows):**
```env
VIENEU_REMOTE_API_BASE=http://host.docker.internal:23333/v1
```

**Linux:**
```env
# Use host network mode
docker-compose --network host up -d

# Or use host IP
VIENEU_REMOTE_API_BASE=http://172.17.0.1:23333/v1
```

### TTS Falling Back to Edge-TTS

**Logs show:**
```
VieNeu-TTS Remote not available: ...
Falling back to Edge-TTS
```

**Check:**
```bash
# Test connectivity from container
docker-compose exec voice-assistant \
  curl -f http://host.docker.internal:23333/v1

# Should return 200 OK
```

**Fix:**
1. Verify VieNeu server is running
2. Check network configuration
3. Review container logs for exact error

### High Latency

**Problem:** TTS takes >1 second

**Solutions:**
1. Check network latency: `ping your-vieneu-server`
2. Ensure server has GPU enabled
3. Use closer server (same region/datacenter)
4. Monitor server load (CPU/GPU utilization)

### Model Not Found

**Error:**
```
Model not found: pnnbao-ump/VieNeu-TTS
```

**Fix:**
```env
# Match your server's model ID
VIENEU_REMOTE_MODEL_ID=pnnbao-ump/VieNeu-TTS
# Or
VIENEU_REMOTE_MODEL_ID=pnnbao-ump/VieNeu-TTS-0.3B
```

## Advanced Configuration

### Custom Docker Compose

```yaml
version: '3.8'

services:
  voice-assistant:
    build: .
    ports:
      - "8000:8000"
    environment:
      - TTS_BACKEND=vieneu_remote
      - VIENEU_REMOTE_API_BASE=http://host.docker.internal:23333/v1
      - VIENEU_REMOTE_MODEL_ID=pnnbao-ump/VieNeu-TTS
    extra_hosts:
      - "host.docker.internal:host-gateway"  # Linux support
```

### Multiple TTS Backends

```env
# Auto-select best available
TTS_BACKEND=auto

# Priority: remote > local > qwen > edge
# System will try remote first, then fallback
```

### Local Mode (Offline)

If you want to run TTS locally in Docker:

```env
TTS_BACKEND=vieneu
VIENEU_MODE=turbo
VIENEU_MODEL_BACKBONE=pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF
VIENEU_MODEL_DECODER=pnnbao-ump/VieNeu-Codec
VIENEU_MODEL_ENCODER=pnnbao-ump/VieNeu-Codec
```

**Note:** Local mode requires more resources in container (~300MB RAM).

## Monitoring

### Check Logs

```bash
# View TTS initialization
docker-compose logs | grep -i "vieneu"

# Should show:
# 🚀 Loading VieNeu-TTS Remote mode: http://...
# ✅ VieNeu-TTS Remote loaded (model=...)
# 🚀 Using VieNeu-TTS Remote: http://...
```

### Health Check

```bash
# Check service health
curl http://localhost:8000/health

# Response:
{
  "status": "healthy",
  "config": {
    "tts_backend": "vieneu_remote",
    "vieneu_remote_api_base": "http://host.docker.internal:23333/v1"
  }
}
```

### Performance Metrics

```bash
# Watch TTS latency in logs
docker-compose logs -f | grep "TTS"

# Example output:
# TTS latency: 250ms (remote mode)
# Pipeline breakdown: tts_synth_total=250ms
```

## Migration Guide

### From Edge-TTS to Remote

**Before:**
```env
TTS_BACKEND=edge
```

**After:**
```env
TTS_BACKEND=vieneu_remote
VIENEU_REMOTE_API_BASE=http://host.docker.internal:23333/v1
VIENEU_REMOTE_MODEL_ID=pnnbao-ump/VieNeu-TTS
```

**Benefits:**
- ✅ Better Vietnamese pronunciation
- ✅ Voice cloning support
- ✅ Offline capable (if server is local)
- ✅ Lower latency (200-800ms vs 1-2s)

### From Local to Remote

**Before:**
```env
TTS_BACKEND=vieneu
VIENEU_MODE=turbo
```

**After:**
```env
TTS_BACKEND=vieneu_remote
VIENEU_REMOTE_API_BASE=http://host.docker.internal:23333/v1
VIENEU_REMOTE_MODEL_ID=pnnbao-ump/VieNeu-TTS
```

**Benefits:**
- ✅ Lighter container (~50MB vs ~300MB)
- ✅ Faster startup (no model download)
- ✅ Centralized model management
- ✅ GPU on server, not in container

## Resources

- [VieNeu-TTS Remote Documentation](VIENEU_REMOTE_MODE.md)
- [LLM Prompt Engineering](LLM_PROMPT_ENGINEERING.md)
- [Text Normalization Guide](TTS_TEXT_NORMALIZATION.md)
- [Voice Assistant README](../README.md)
