"""
Configuration for Voice Assistant Pipeline.
Simplified from production for personal/demo use.
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class AudioConfig:
    """Audio processing settings."""
    sample_rate: int = 16000
    channels: int = 1
    bit_depth: int = 16
    chunk_duration_ms: int = 100  # 100ms chunks = 1600 samples

    # Preprocessing
    high_pass_hz: int = 80
    low_pass_hz: int = 7600
    noise_gate_db: float = -40.0

    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_duration_ms / 1000)

    @property
    def bytes_per_chunk(self) -> int:
        return self.chunk_samples * (self.bit_depth // 8)


@dataclass
class VADConfig:
    """Voice Activity Detection settings."""
    threshold: float = 0.5
    min_silence_duration_ms: int = 500
    speech_pad_ms: int = 30
    chunk_size: int = 512  # Silero optimal


@dataclass
class ASRConfig:
    """Automatic Speech Recognition settings."""
    model_id: str = "g-group-ai-lab/gipformer-65M-rnnt"
    device: str = "auto"
    language: str = "Vietnamese"
    # Streaming
    interim_interval_ms: int = 800
    min_audio_for_interim_ms: int = 600


@dataclass
class LLMConfig:
    """Language Model settings."""
    provider: str = "groq"  # groq, openai, local
    model: str = "llama-3.3-70b-versatile"
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    base_url: str = "https://api.groq.com/openai/v1"
    max_tokens: int = 256
    temperature: float = 0.7
    system_prompt: str = "Bạn là trợ lý ảo thông minh, trả lời ngắn gọn và hữu ích bằng tiếng Việt."


@dataclass
class TTSConfig:
    """Text-to-Speech settings."""
    model_id: str = "g-group-ai-lab/gwen-tts-0.6B"
    device: str = "cuda:0"
    output_sample_rate: int = 24000
    target_sample_rate: int = 16000
    default_speaker: str = "yen_nhi"


@dataclass
class PipelineConfig:
    """Pipeline orchestration settings."""
    min_utterance_duration_ms: int = 500
    min_verified_chunks: int = 3  # Relaxed for personal use
    max_buffer_duration_ms: int = 15000
    interrupt_threshold_chunks: int = 6
    post_tts_buffer_ms: int = 500


@dataclass
class Settings:
    """Main configuration container."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    # Debug options
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "").lower() == "true")
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings with environment variable overrides."""
        settings = cls()
        if os.getenv("ASR_DEVICE"):
            settings.asr.device = os.getenv("ASR_DEVICE")
        if os.getenv("TTS_DEVICE"):
            settings.tts.device = os.getenv("TTS_DEVICE")
        if os.getenv("LLM_PROVIDER"):
            settings.llm.provider = os.getenv("LLM_PROVIDER")
        if os.getenv("DEBUG"):
            settings.debug = os.getenv("DEBUG", "").lower() == "true"
        return settings


# Global settings instance
settings = Settings.from_env()
