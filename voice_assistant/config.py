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
    model: str = "qwen/qwen3-32b"
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    base_url: str = "https://api.groq.com/openai/v1"
    max_tokens: int = 256
    temperature: float = 0.7  # Lower for more predictable TTS-friendly output
    use_extended: bool = True
    auto_register_task_handlers: bool = True
    system_prompt: str = """Bạn là Nam, trợ lý ảo tiếng Việt.

## QUAN TRỌNG NHẤT
Câu trả lời của bạn sẽ được đọc TO thành tiếng bằng TTS. Hãy viết như đang NÓI CHUYỆN.

## QUY TẮC BẮT BUỘC

1. ĐỊNH DẠNG VĂN NÓI:
   - Viết như đang nói, không viết như đang viết văn bản
   - Câu ngắn, rõ ràng, dễ đọc thành tiếng
   - Ưu tiên 1-3 câu cho mỗi phản hồi
   - Mỗi câu không quá 30 từ

2. TUYỆT ĐỐI KHÔNG DÙNG:
   - Markdown: **bold**, *italic*, # header, v.v.
   - Danh sách: gạch đầu dòng, số thứ tự, bullet points
   - Code blocks: ```code``` hoặc `inline code`
   - Bảng biểu, ký tự đặc biệt, emoji
   - URL, email, link
   - Thẻ HTML/XML: <tag>, <think>, v.v.
   - Viết tắt không rõ nghĩa

3. LUÔN LUÔN DÙNG:
   - Câu hoàn chỉnh có chủ ngữ, vị ngữ
   - Kết thúc câu bằng dấu chấm, chấm hỏi hoặc chấm than
   - Dấu phẩy để ngắt câu dài thành các đoạn ngắn
   - Từ nối tự nhiên: "và", "nhưng", "vì", "nên", "thì", "do đó"

4. XỬ LÝ NỘI DUNG ĐẶC BIỆT:
   - Số: viết rõ ràng "mười phần trăm" thay vì "10%"
   - Ngày tháng: viết đầy đủ "ngày mùng một tháng một" 
   - Tiền tệ: viết rõ "một triệu đồng" thay vì "1.000.000đ"
   - Đơn vị: viết đầy đủ "kilômet", "kiôgam"
   - Tên riêng: giữ nguyên, không viết tắt

5. PHONG CÁCH:
   - Thân thiện, tự nhiên như đang nói chuyện
   - Tiếng Việt đời thường, không quá trang trọng
   - Nếu không biết, nói thẳng "Mình không rõ về việc này"
   - Không giải thích dài dòng về quy trình suy nghĩ

## VÍ DỤ PHẢN HỒI TỐT

✅ "Chào bạn! Mình có thể giúp gì cho bạn nào?"
✅ "Hôm nay trời nắng đẹp, nhiệt độ khoảng ba mươi độ."
✅ "Thành phố Hồ Chí Minh nằm ở miền Nam Việt Nam."

## VÍ DỤ PHẢN HỒI KHÔNG TỐT

❌ "**Thành phố Hồ Chí Minh** (TP.HCM) là..."
❌ "- Dân số: 9 triệu\n- Diện tích: 2000 km²"
❌ "Xem thêm tại https://example.com"
❌ "10% người dùng thích điều này."

## BẮT ĐẦU PHẢN HỒI
Hãy trả lời ngắn gọn, tự nhiên, dễ đọc thành tiếng."""


@dataclass
class TTSConfig:
    """Text-to-Speech settings."""
    backend: str = "auto"  # auto, vieneu, vieneu_remote, qwen, edge
    default_speaker: str = "yen_nhi"
    target_sample_rate: int = 16000
    speech_rate: float = 1.25  # Speed multiplier (1.0 = normal, max 1.25 for edge-tts)
    # Qwen-TTS specific (requires GPU)
    qwen_model_id: str = "g-group-ai-lab/gwen-tts-0.6B"
    qwen_device: str = "cuda:0"
    # VieNeu-TTS v2 Turbo (local mode)
    vieneu_mode: str = "turbo"  # turbo (v2), standard, fast, turbo_gpu
    vieneu_model_backbone: str = "pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF"
    vieneu_model_decoder: str = "pnnbao-ump/VieNeu-Codec"
    vieneu_model_encoder: str = "pnnbao-ump/VieNeu-Codec"
    # VieNeu-TTS Remote mode (for optimal latency with remote server)
    vieneu_remote_api_base: str = "http://localhost:23333/v1"  # Local server, no tunnel
    vieneu_remote_model_id: str = "pnnbao-ump/VieNeu-TTS"  # Remote model ID
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
    llm_history_window: int = 8
    min_tts_chunk_chars: int = 20  # Minimum chars before TTS synthesis
    tts_overlap_enabled: bool = True
    max_playback_wait_s: float = 0.2
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
    warmup: bool = True  # Run warmup inference on startup


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
        if os.getenv("VIENEU_MODE"):
            settings.tts.vieneu_mode = os.getenv("VIENEU_MODE")
        if os.getenv("VIENEU_MODEL_BACKBONE"):
            settings.tts.vieneu_model_backbone = os.getenv("VIENEU_MODEL_BACKBONE")
        if os.getenv("VIENEU_MODEL_DECODER"):
            settings.tts.vieneu_model_decoder = os.getenv("VIENEU_MODEL_DECODER")
        if os.getenv("VIENEU_MODEL_ENCODER"):
            settings.tts.vieneu_model_encoder = os.getenv("VIENEU_MODEL_ENCODER")
        if os.getenv("VIENEU_REMOTE_API_BASE"):
            settings.tts.vieneu_remote_api_base = os.getenv("VIENEU_REMOTE_API_BASE")
        if os.getenv("VIENEU_REMOTE_MODEL_ID"):
            settings.tts.vieneu_remote_model_id = os.getenv("VIENEU_REMOTE_MODEL_ID")

        # LLM settings
        if os.getenv("LLM_PROVIDER"):
            settings.llm.provider = os.getenv("LLM_PROVIDER")
        if os.getenv("LLM_MODEL"):
            settings.llm.model = os.getenv("LLM_MODEL")
        if os.getenv("LLM_TEMPERATURE"):
            settings.llm.temperature = float(os.getenv("LLM_TEMPERATURE"))
        if os.getenv("LLM_USE_EXTENDED"):
            settings.llm.use_extended = os.getenv("LLM_USE_EXTENDED", "").lower() == "true"
        if os.getenv("LLM_AUTO_REGISTER_TASK_HANDLERS"):
            settings.llm.auto_register_task_handlers = (
                os.getenv("LLM_AUTO_REGISTER_TASK_HANDLERS", "").lower() == "true"
            )
        if os.getenv("LLM_SYSTEM_PROMPT"):
            settings.llm.system_prompt = os.getenv("LLM_SYSTEM_PROMPT")
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
        if os.getenv("SERVER_WARMUP"):
            settings.server.warmup = os.getenv("SERVER_WARMUP", "").lower() == "true"

        # Session settings
        if os.getenv("SESSION_TIMEOUT"):
            settings.session.timeout = int(os.getenv("SESSION_TIMEOUT"))
        if os.getenv("MAX_HISTORY_LENGTH"):
            settings.session.max_history_length = int(os.getenv("MAX_HISTORY_LENGTH"))
        if os.getenv("SESSION_PERSISTENCE"):
            settings.session.persistence = os.getenv("SESSION_PERSISTENCE", "").lower() == "true"
        if os.getenv("SESSION_STORAGE_PATH"):
            settings.session.storage_path = os.getenv("SESSION_STORAGE_PATH")

        # Pipeline settings
        if os.getenv("PIPELINE_MIN_UTTERANCE_MS"):
            settings.pipeline.min_utterance_duration_ms = int(os.getenv("PIPELINE_MIN_UTTERANCE_MS"))
        if os.getenv("PIPELINE_MIN_VERIFIED_CHUNKS"):
            settings.pipeline.min_verified_chunks = int(os.getenv("PIPELINE_MIN_VERIFIED_CHUNKS"))
        if os.getenv("PIPELINE_INTERRUPT_THRESHOLD_CHUNKS"):
            settings.pipeline.interrupt_threshold_chunks = int(
                os.getenv("PIPELINE_INTERRUPT_THRESHOLD_CHUNKS")
            )
        if os.getenv("PIPELINE_LLM_HISTORY_WINDOW"):
            settings.pipeline.llm_history_window = int(os.getenv("PIPELINE_LLM_HISTORY_WINDOW"))
        if os.getenv("PIPELINE_MIN_TTS_CHUNK_CHARS"):
            settings.pipeline.min_tts_chunk_chars = int(os.getenv("PIPELINE_MIN_TTS_CHUNK_CHARS"))
        if os.getenv("PIPELINE_TTS_OVERLAP_ENABLED"):
            settings.pipeline.tts_overlap_enabled = (
                os.getenv("PIPELINE_TTS_OVERLAP_ENABLED", "").lower() == "true"
            )
        if os.getenv("PIPELINE_MAX_PLAYBACK_WAIT_S"):
            settings.pipeline.max_playback_wait_s = float(os.getenv("PIPELINE_MAX_PLAYBACK_WAIT_S"))
        if os.getenv("PIPELINE_ASR_TIMEOUT"):
            settings.pipeline.asr_timeout_s = int(os.getenv("PIPELINE_ASR_TIMEOUT"))
        if os.getenv("PIPELINE_LLM_TIMEOUT"):
            settings.pipeline.llm_timeout_s = int(os.getenv("PIPELINE_LLM_TIMEOUT"))
        if os.getenv("PIPELINE_TTS_TIMEOUT"):
            settings.pipeline.tts_timeout_s = int(os.getenv("PIPELINE_TTS_TIMEOUT"))
        if os.getenv("PIPELINE_TOTAL_TIMEOUT"):
            settings.pipeline.pipeline_timeout_s = int(os.getenv("PIPELINE_TOTAL_TIMEOUT"))

        return settings


# Global settings instance
settings = Settings.from_env()
