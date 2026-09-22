# Docker Deployment Guide

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/nguyendevwk/vie-voice-engine.git
cd vie-voice-engine

# 2. Configure environment
cp .env.docker.example .env
nano .env  # Add your GROQ_API_KEY

# 3. Build and run
docker-compose up -d

# 4. Access
# Open http://localhost:8000
```

## Configuration

### Environment Variables

Edit `.env` file:

```bash
# Required
GROQ_API_KEY=your_key_here

# ASR Backend (cpu recommended for Docker)
ASR_USE_ONNX=true
ASR_DEVICE=cpu

# TTS Backend
TTS_BACKEND=edge  # or vieneu_remote for better latency
TTS_SPEECH_RATE=1.25

# Server
SERVER_PORT=8000
DEBUG=false
```

### TTS Backends

| Backend | Command | GPU Required | Latency |
|---------|---------|--------------|---------|
| Edge-TTS | `TTS_BACKEND=edge` | No | 1-2s |
| VieNeu Remote | `TTS_BACKEND=vieneu_remote` | No | 200-800ms |
| VieNeu Local | `TTS_BACKEND=vieneu` | No | 100-500ms |
| Qwen-TTS | `TTS_BACKEND=qwen` | Yes | 100-300ms |

### VieNeu Remote Mode (Recommended for Docker)

For optimal latency, use VieNeu-TTS Remote mode:

```bash
# 1. Start VieNeu server on host
python -c "from vieneu.server import run_server; run_server(port=23333)"

# 2. Configure .env
TTS_BACKEND=vieneu_remote
VIENEU_REMOTE_API_BASE=http://host.docker.internal:23333/v1
VIENEU_REMOTE_MODEL_ID=pnnbao-ump/VieNeu-TTS

# 3. Restart container
docker-compose restart
```

**Network Configuration:**

| Platform | API Base URL |
|----------|--------------|
| Docker Desktop (Mac/Windows) | `http://host.docker.internal:23333/v1` |
| Linux | `http://172.17.0.1:23333/v1` |
| Remote Server | `http://your-server:23333/v1` |

### Resource Limits

Default limits in `docker-compose.yml`:

```yaml
limits:
  cpus: '4'
  memory: 8G
reservations:
  cpus: '2'
  memory: 4G
```

## Management

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Restart
docker-compose restart

# View logs
docker-compose logs -f voice-assistant

# Execute commands in container
docker-compose exec voice-assistant bash

# Remove volumes (clear cache)
docker-compose down -v
```

## GPU Support (Optional)

For TTS with Qwen-TTS on GPU:

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Update `docker-compose.yml`:

```yaml
services:
  voice-assistant:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

    environment:
      - TTS_BACKEND=qwen
      - TTS_DEVICE=cuda:0
```

3. Rebuild and run:

```bash
docker-compose up -d --build
```

## Persistence

### Model Cache

Models are cached in Docker volume:

```bash
# View volume
docker volume inspect vie-voice-engine_model-cache

# Backup cache
docker run --rm -v vie-voice-engine_model-cache:/data -v $(pwd):/backup alpine tar czf /backup/model-cache.tar.gz /data
```

### Session Storage

Sessions are stored in `./sessions` directory (mounted volume).

## Production Deployment

### Behind Reverse Proxy

Example nginx configuration:

```nginx
server {
    listen 80;
    server_name voice-assistant.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket timeout
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}
```

### SSL/TLS

Use Let's Encrypt with certbot:

```bash
sudo certbot --nginx -d voice-assistant.example.com
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Container won't start | Check `docker-compose logs` for errors |
| Health check failing | Verify `GROQ_API_KEY` is set in `.env` |
| Out of memory | Reduce `memory` limit in `docker-compose.yml` |
| Models not downloading | Check internet connection from container |
| TTS falling back to Edge | Verify VieNeu server is running and accessible |
| High TTS latency | Use same region for client and server |

## Resources

- [VieNeu-TTS Documentation](VIENEU_TTS.md)
- [VieNeu Remote Mode](VIENEU_REMOTE_MODE.md)
- [Voice Assistant README](../README.md)
