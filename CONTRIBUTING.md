# Contributing

## Setup

```bash
git clone https://github.com/nguyendevwk/vie-voice-engine.git
cd vie-voice-engine

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"

cp .env.example .env
```

## Development

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=voice_assistant

# Start server
python -m voice_assistant.api.server
```

## Code Style

- Follow PEP 8
- Use type hints
- Google-style docstrings
- Max line length: 88 (black default)

```python
def process_audio(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Process audio with normalization.

    Args:
        audio: Input audio samples
        sample_rate: Sample rate in Hz

    Returns:
        Processed audio samples
    """
    pass
```

## Commits

```
type(scope): description

- feat: new feature
- fix: bug fix
- docs: documentation
- refactor: code refactoring
- test: add tests
- chore: maintenance
```

## Pull Requests

1. Create feature branch
2. Write tests
3. Update docs
4. Open PR

## License

By contributing, you agree to MIT License.
