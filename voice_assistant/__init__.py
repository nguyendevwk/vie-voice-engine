"""
Vietnamese Voice Assistant Pipeline.

Real-time voice assistant with ASR, LLM, and TTS streaming.
"""

__version__ = "1.0.0"
__author__ = "nguyendevwk"
__email__ = "phamnguyen.devwk@gmail.com"

from .config import settings

__all__ = ["settings", "__version__"]
