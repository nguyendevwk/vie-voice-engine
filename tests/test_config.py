"""
Tests for configuration module.
"""

import os
import pytest

from voice_assistant.config import (
    Settings,
    AudioConfig,
    VADConfig,
    ASRConfig,
    LLMConfig,
    TTSConfig,
    PipelineConfig,
    SessionConfig,
    ServerConfig,
)


class TestAudioConfig:
    """Tests for AudioConfig."""

    def test_default_values(self):
        config = AudioConfig()

        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.bit_depth == 16
        assert config.chunk_duration_ms == 100

    def test_chunk_samples(self):
        config = AudioConfig()

        # 100ms at 16kHz = 1600 samples
        assert config.chunk_samples == 1600

    def test_bytes_per_chunk(self):
        config = AudioConfig()

        # 1600 samples * 2 bytes (16-bit) = 3200 bytes
        assert config.bytes_per_chunk == 3200


class TestVADConfig:
    """Tests for VADConfig."""

    def test_default_values(self):
        config = VADConfig()

        assert config.threshold == 0.75
        assert config.min_silence_duration_ms == 500
        assert config.speech_pad_ms == 30
        assert config.chunk_size == 512


class TestASRConfig:
    """Tests for ASRConfig."""

    def test_default_values(self):
        config = ASRConfig()

        assert config.model_id == "g-group-ai-lab/gipformer-65M-rnnt"
        assert config.device == "auto"
        assert config.language == "Vietnamese"


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_default_values(self):
        config = LLMConfig()

        assert config.provider == "groq"
        assert config.model == "qwen/qwen3-32b"
        assert config.max_tokens == 256
        assert config.temperature == 0.7  # Lower for more predictable TTS-friendly output
        assert config.use_extended is True
        assert config.auto_register_task_handlers is True


class TestTTSConfig:
    """Tests for TTSConfig."""

    def test_default_values(self):
        config = TTSConfig()

        assert config.model_id == "g-group-ai-lab/gwen-tts-0.6B"
        assert config.output_sample_rate == 24000
        assert config.target_sample_rate == 16000


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_values(self):
        config = PipelineConfig()

        assert config.min_utterance_duration_ms == 500
        assert config.min_verified_chunks == 3
        assert config.max_buffer_duration_ms == 15000
        assert config.interrupt_threshold_chunks == 6
        assert config.llm_history_window == 8
        assert config.min_tts_chunk_chars == 20
        assert config.tts_overlap_enabled is True


class TestSessionConfig:
    """Tests for SessionConfig."""

    def test_default_values(self):
        config = SessionConfig()

        assert config.timeout == 1800  # 30 minutes
        assert config.max_history_length == 50
        assert config.persistence == False


class TestServerConfig:
    """Tests for ServerConfig."""

    def test_default_values(self):
        config = ServerConfig()

        assert config.host == "0.0.0.0"
        assert config.port == 8000


class TestSettings:
    """Tests for Settings."""

    def test_default_settings(self):
        settings = Settings()

        assert isinstance(settings.audio, AudioConfig)
        assert isinstance(settings.vad, VADConfig)
        assert isinstance(settings.asr, ASRConfig)
        assert isinstance(settings.llm, LLMConfig)
        assert isinstance(settings.tts, TTSConfig)
        assert isinstance(settings.pipeline, PipelineConfig)
        assert isinstance(settings.session, SessionConfig)
        assert isinstance(settings.server, ServerConfig)

    def test_from_env(self, monkeypatch):
        """Test loading settings from environment."""
        monkeypatch.setenv("ASR_DEVICE", "cuda:1")
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.setenv("SERVER_PORT", "9000")
        monkeypatch.setenv("PIPELINE_LLM_HISTORY_WINDOW", "6")
        monkeypatch.setenv("PIPELINE_MIN_TTS_CHUNK_CHARS", "16")
        monkeypatch.setenv("PIPELINE_TTS_OVERLAP_ENABLED", "false")

        settings = Settings.from_env()

        assert settings.asr.device == "cuda:1"
        assert settings.debug == True
        assert settings.server.port == 9000
        assert settings.pipeline.llm_history_window == 6
        assert settings.pipeline.min_tts_chunk_chars == 16
        assert settings.pipeline.tts_overlap_enabled is False

    def test_groq_api_key(self, monkeypatch):
        """Test GROQ_API_KEY loading."""
        monkeypatch.setenv("GROQ_API_KEY", "test-key-123")

        settings = Settings.from_env()

        assert settings.llm.api_key == "test-key-123"

    def test_llm_extended_flags(self, monkeypatch):
        """Test extended LLM toggle flags from environment."""
        monkeypatch.setenv("LLM_USE_EXTENDED", "false")
        monkeypatch.setenv("LLM_AUTO_REGISTER_TASK_HANDLERS", "false")

        settings = Settings.from_env()

        assert settings.llm.use_extended is False
        assert settings.llm.auto_register_task_handlers is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
