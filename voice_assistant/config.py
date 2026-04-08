"""
Configuration for Voice Assistant Pipeline.
Simplified from production for personal/demo use.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os

# Load .env file if exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed


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
    threshold: float = 0.75
    min_silence_duration_ms: int = 500
    speech_pad_ms: int = 30
    chunk_size: int = 512  # Silero optimal


@dataclass
class ASRConfig:
    """Automatic Speech Recognition settings."""
    model_id: str = "g-group-ai-lab/gipformer-65M-rnnt"
    device: str = "auto"
    language: str = "Vietnamese"
    use_onnx: bool = True  # Prefer ONNX (faster, simpler)
    use_pytorch_cuda: bool = False  # Use PyTorch with CUDA if ONNX fails
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
    backend: str = "auto"  # auto, vieneu, qwen, edge
    default_speaker: str = "yen_nhi"
    target_sample_rate: int = 16000
    speech_rate: float = 1.25  # Speed multiplier (1.0 = normal, max 1.25 for edge-tts)
    # Qwen-TTS specific (requires GPU)
    qwen_model_id: str = "g-group-ai-lab/gwen-tts-0.6B"
    qwen_device: str = "cuda:0"
    # Legacy compatibility
    model_id: str = "g-group-ai-lab/gwen-tts-0.6B"
    device: str = "cuda:0"
    output_sample_rate: int = 24000


@dataclass
class PipelineConfig:
    """Pipeline orchestration settings."""
    min_utterance_duration_ms: int = 500
    min_verified_chunks: int = 3  # Relaxed for personal use
    max_buffer_duration_ms: int = 15000
    interrupt_threshold_chunks: int = 6
    post_tts_buffer_ms: int = 500
    # Timeouts (prevent hanging)
    asr_timeout_s: int = 10  # ASR should complete within 10s
    llm_timeout_s: int = 30  # LLM should start streaming within 30s
    tts_timeout_s: int = 15  # TTS should complete within 15s
    pipeline_timeout_s: int = 60  # Total pipeline timeout


@dataclass
class SessionConfig:
    """Session management settings."""
    timeout: int = 1800  # 30 minutes
    max_history_length: int = 50
    persistence: bool = False
    storage_path: str = "./sessions"


@dataclass
class ServerConfig:
    """Server settings."""
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class Settings:
    """Main configuration container."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    # Debug options
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings with environment variable overrides."""
        settings = cls()

        # Device settings
        if os.getenv("ASR_DEVICE"):
            settings.asr.device = os.getenv("ASR_DEVICE")
        if os.getenv("TTS_DEVICE"):
            settings.tts.device = os.getenv("TTS_DEVICE")
        if os.getenv("TTS_BACKEND"):
            settings.tts.backend = os.getenv("TTS_BACKEND")
        if os.getenv("TTS_SPEECH_RATE"):
            settings.tts.speech_rate = float(os.getenv("TTS_SPEECH_RATE"))
        if os.getenv("TTS_DEFAULT_SPEAKER"):
            settings.tts.default_speaker = os.getenv("TTS_DEFAULT_SPEAKER")

        # LLM settings
        if os.getenv("LLM_PROVIDER"):
            settings.llm.provider = os.getenv("LLM_PROVIDER")
        if os.getenv("GROQ_API_KEY"):
            settings.llm.api_key = os.getenv("GROQ_API_KEY")
        if os.getenv("OPENAI_API_KEY") and settings.llm.provider == "openai":
            settings.llm.api_key = os.getenv("OPENAI_API_KEY")
            settings.llm.base_url = "https://api.openai.com/v1"

        # Debug settings
        if os.getenv("DEBUG"):
            settings.debug = os.getenv("DEBUG", "").lower() == "true"
        if os.getenv("LOG_LEVEL"):
            settings.log_level = os.getenv("LOG_LEVEL")

        # Server settings
        if os.getenv("SERVER_HOST"):
            settings.server.host = os.getenv("SERVER_HOST")
        if os.getenv("SERVER_PORT"):
            settings.server.port = int(os.getenv("SERVER_PORT"))

        # Session settings
        if os.getenv("SESSION_TIMEOUT"):
            settings.session.timeout = int(os.getenv("SESSION_TIMEOUT"))
        if os.getenv("MAX_HISTORY_LENGTH"):
            settings.session.max_history_length = int(os.getenv("MAX_HISTORY_LENGTH"))
        if os.getenv("SESSION_PERSISTENCE"):
            settings.session.persistence = os.getenv("SESSION_PERSISTENCE", "").lower() == "true"
        if os.getenv("SESSION_STORAGE_PATH"):
            settings.session.storage_path = os.getenv("SESSION_STORAGE_PATH")

        return settings


# Global settings instance
settings = Settings.from_env()
