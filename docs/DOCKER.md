# Docker Deployment Guide

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/nguyendevwk/end2end_asr_tts_vie.git
cd end2end_asr_tts_vie

# 2. Configure environment
cp .env.docker.example .env
nano .env  # Add your GROQ_API_KEY

# 3. Build and run
docker-compose up -d

# 4. Check logs
docker-compose logs -f

# 5. Access
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
TTS_BACKEND=edge  # edge-tts works without GPU
TTS_SPEECH_RATE=1.25

# Server
SERVER_PORT=8000
DEBUG=false
```

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

Adjust based on your server specs.

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
docker volume inspect end2end_asr_tts_vie_model-cache

# Backup cache
docker run --rm -v end2end_asr_tts_vie_model-cache:/data -v $(pwd):/backup alpine tar czf /backup/model-cache.tar.gz /data
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

### Monitoring

Add to `docker-compose.yml`:

```yaml
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs voice-assistant

# Common issues:
# 1. Missing GROQ_API_KEY in .env
# 2. Port 8000 already in use
# 3. Insufficient memory
```

### Health check failing

```bash
# Check health status
docker-compose ps

# Manual health check
curl http://localhost:8000/health
```

### Out of memory

Reduce resource usage:

```bash
# Edit docker-compose.yml
deploy:
  resources:
    limits:
      memory: 4G  # Reduce from 8G
```

### Models not downloading

```bash
# Check internet connection from container
docker-compose exec voice-assistant ping -c 3 google.com

# Check disk space
docker system df
```

## Building Custom Image

```bash
# Build with custom tag
docker build -t voice-assistant:custom .

# Push to registry
docker tag voice-assistant:custom registry.example.com/voice-assistant:latest
docker push registry.example.com/voice-assistant:latest
```

## Multi-stage Build Details

The Dockerfile uses multi-stage build to minimize image size:

- **Builder stage**: Compiles dependencies (~2GB)
- **Runtime stage**: Only includes runtime deps (~1GB)

Final image size: ~1.5GB (includes models cache)
