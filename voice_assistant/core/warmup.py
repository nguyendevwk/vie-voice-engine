"""
Model warmup module.

Preloads models and runs initial inference to avoid first-request latency.
Used by server startup and can be called independently.
"""

import time

import numpy as np

from ..config import settings
from ..utils.logging import logger


async def warmup_all():
    """
    Run warmup inference on all models to avoid first-request delay.

    This preloads weights into memory and runs a small inference
    to trigger any JIT compilation or lazy initialization.
    """
    if not settings.server.warmup:
        logger.info("Model warmup disabled")
        return

    start = time.time()
    logger.info("Starting model warmup...")

    # 1. VAD warmup
    try:
        from .vad import get_vad_service
        vad = get_vad_service()
        dummy_pcm = np.zeros(1600, dtype=np.int16).tobytes()
        vad.process_chunk(dummy_pcm)
        vad.reset()
        logger.info("✓ VAD warmed up")
    except Exception as e:
        logger.warning(f"VAD warmup failed: {e}")

    # 2. ASR warmup
    try:
        from .asr import get_asr_service
        asr = get_asr_service()
        asr._ensure_loaded()
        dummy_pcm = np.zeros(16000, dtype=np.int16).tobytes()
        asr.transcribe_bytes([dummy_pcm])
        logger.info("✓ ASR warmed up")
    except Exception as e:
        logger.warning(f"ASR warmup failed: {e}")

    # 3. LLM warmup (just init client, no actual inference)
    try:
        from .llm import get_llm_service
        llm = get_llm_service()
        if hasattr(llm, '_ensure_client'):
            llm._ensure_client()
        logger.info("✓ LLM client initialized")
    except Exception as e:
        logger.warning(f"LLM warmup failed: {e}")

    # 4. TTS warmup
    try:
        from .tts import get_tts_service
        tts = get_tts_service()
        provider = tts._get_provider()
        audio = await tts.synthesize("Xin chào.")
        if audio:
            logger.info(f"✓ TTS warmed up ({provider.name})")
        else:
            logger.info(f"✓ TTS provider loaded ({provider.name})")
    except Exception as e:
        logger.warning(f"TTS warmup failed: {e}")

    elapsed = time.time() - start
    logger.info(f"Model warmup complete in {elapsed:.1f}s")
