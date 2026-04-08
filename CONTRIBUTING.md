# Contributing Guide

## 🎯 Development Setup

### Prerequisites
- Python 3.12+
- CUDA 11.8+ (optional, for GPU)
- Git

### Initial Setup
```bash
# Clone repository
git clone https://github.com/nguyendevwk/end2end_asr_tts_vie.git
cd end2end_asr_tts_vie

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_audio.py -v

# Run with coverage
pytest tests/ --cov=voice_assistant --cov-report=html
```

## 📝 Code Style

### Python Style Guide
- Follow PEP 8
- Use type hints
- Max line length: 100 characters
- Use docstrings (Google style)

### Example
```python
from typing import Optional, List

def process_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    normalize: bool = True
) -> np.ndarray:
    """
    Process audio with normalization.
    
    Args:
        audio: Input audio samples
        sample_rate: Sample rate in Hz
        normalize: Whether to normalize audio
        
    Returns:
        Processed audio samples
        
    Raises:
        ValueError: If sample_rate is invalid
    """
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample_rate: {sample_rate}")
    
    # Processing logic
    return audio
```

### Imports Order
1. Standard library
2. Third-party packages
3. Local modules

```python
import os
import sys
from typing import List

import numpy as np
import torch

from ..config import settings
from ..utils.logging import logger
```

## 🔧 Adding New Features

### 1. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Implement Feature
- Write code with tests
- Add docstrings
- Update documentation

### 3. Run Tests
```bash
pytest tests/ -v
```

### 4. Commit Changes
```bash
git add .
git commit -m "feat: Add your feature description"
```

### 5. Create Pull Request
- Push to your fork
- Open PR with description
- Link related issues

## 🧪 Testing Guidelines

### Unit Tests
```python
import pytest
from voice_assistant.core.audio import AudioPreprocessor

def test_audio_normalization():
    """Test audio normalization."""
    preprocessor = AudioPreprocessor()
    
    # Test data
    audio = np.random.randn(16000).astype(np.float32)
    
    # Process
    processed = preprocessor.normalize(audio)
    
    # Assertions
    assert processed.dtype == np.float32
    assert len(processed) == len(audio)
    assert -1.0 <= processed.max() <= 1.0
```

### Integration Tests
```python
@pytest.mark.asyncio
async def test_full_pipeline():
    """Test complete pipeline flow."""
    from voice_assistant.core.pipeline import PipelineOrchestrator
    
    orchestrator = PipelineOrchestrator()
    
    # Test audio input
    audio_chunk = b'\x00' * 3200
    await orchestrator.handle_audio_chunk(audio_chunk)
    
    # Verify state
    assert orchestrator._state != PipelineState.IDLE
```

## 📚 Documentation

### Docstring Format (Google Style)
```python
def function_name(param1: str, param2: int) -> bool:
    """
    Short description of function.
    
    Longer description if needed. Explain complex logic,
    edge cases, and usage examples.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param2 is negative
        RuntimeError: When operation fails
        
    Example:
        >>> result = function_name("test", 42)
        >>> assert result is True
    """
    pass
```

### README Updates
- Keep README.md in sync with features
- Add usage examples for new features
- Update configuration section

## 🐛 Bug Reports

### Bug Report Template
```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Run command '...'
2. Click on '...'
3. See error

**Expected behavior**
What you expected to happen.

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.12.0]
- CUDA version: [e.g., 11.8]
- Package versions: [pip list]

**Logs**
```
Paste relevant logs here
```

**Additional context**
Any other relevant information.
```

## 🎨 UI/Frontend

### Web UI Development
```bash
# Edit static files
vim voice_assistant/api/static/index.html

# Test locally
python -m voice_assistant.api.server
# Open http://localhost:8000
```

### UI Guidelines
- Mobile-responsive design
- Dark theme by default
- Accessibility (ARIA labels)
- WebSocket error handling

## 🔄 Release Process

### Version Bumping
```bash
# Update version in pyproject.toml
# Update CHANGELOG.md
# Commit changes
git commit -m "chore: Bump version to X.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

### Changelog Format
```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New feature description

### Changed
- Changed feature description

### Fixed
- Bug fix description

### Removed
- Removed feature description
```

## 📞 Getting Help

### Communication Channels
- GitHub Issues: Bug reports, feature requests
- GitHub Discussions: Questions, ideas
- Email: phamnguyen.devwk@gmail.com

### Code Review
- All PRs require review
- Address feedback promptly
- Keep PRs focused and small

## 🙏 Code of Conduct

### Our Standards
- Be respectful and inclusive
- Accept constructive criticism
- Focus on what's best for the project
- Show empathy towards others

### Unacceptable Behavior
- Harassment or discrimination
- Trolling or insulting comments
- Personal or political attacks
- Publishing others' private information

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.
