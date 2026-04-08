"""
Tests for streaming pipeline module.
"""

import asyncio
import numpy as np
import pytest

from voice_assistant.core.streaming import (
    StreamingPipeline,
    StreamState,
    StreamMetrics,
    RealtimeASRProcessor,
    RealtimeTTSProcessor,
)
from voice_assistant.core.audio import AudioPreprocessor


class TestStreamState:
    """Tests for StreamState enum."""

    def test_states_exist(self):
        assert StreamState.IDLE is not None
        assert StreamState.LISTENING is not None
        assert StreamState.TRANSCRIBING is not None
        assert StreamState.GENERATING is not None
        assert StreamState.SYNTHESIZING is not None
        assert StreamState.INTERRUPTED is not None


class TestStreamMetrics:
    """Tests for StreamMetrics."""

    def test_default_values(self):
        metrics = StreamMetrics()

        assert metrics.utterance_start == 0
        assert metrics.first_interim == 0
        assert metrics.total_audio_ms == 0

    def test_log_summary(self):
        """Test that log_summary doesn't crash."""
        metrics = StreamMetrics()
        metrics.utterance_start = 1.0
        metrics.first_interim = 1.5
        metrics.final_transcript = 2.0

        # Should not raise
        metrics.log_summary()


class TestRealtimeASRProcessor:
    """Tests for RealtimeASRProcessor."""

    @pytest.fixture
    def processor(self):
        return RealtimeASRProcessor()

    @pytest.fixture
    def sample_chunk(self):
        """Generate 100ms audio chunk."""
        t = np.linspace(0, 0.1, 1600, endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t) * 0.5
        return (audio * 32767).astype(np.int16).tobytes()

    def test_reset(self, processor):
        processor._audio_buffer.append(np.zeros(1600))
        processor._last_interim_text = "test"

        processor.reset()

        assert len(processor._audio_buffer) == 0
        assert processor._last_interim_text == ""

    def test_add_audio(self, processor, sample_chunk):
        processor.add_audio(sample_chunk)

        assert len(processor._audio_buffer) == 1

    def test_buffer_duration_ms(self, processor, sample_chunk):
        # Add 10 chunks (100ms each)
        for _ in range(10):
            processor.add_audio(sample_chunk)

        # Should be ~1000ms
        duration = processor.buffer_duration_ms
        assert 900 < duration < 1100


class TestRealtimeTTSProcessor:
    """Tests for RealtimeTTSProcessor."""

    @pytest.fixture
    def processor(self):
        return RealtimeTTSProcessor()

    def test_get_duration_ms(self, processor):
        # 16000 samples = 1 second = 1000ms
        # PCM S16LE: 16000 samples * 2 bytes = 32000 bytes
        audio_bytes = np.zeros(16000, dtype=np.int16).tobytes()

        duration = processor.get_duration_ms(audio_bytes)

        assert abs(duration - 1000.0) < 1.0


class TestStreamingPipeline:
    """Tests for StreamingPipeline."""

    @pytest.fixture
    def pipeline(self):
        return StreamingPipeline()

    def test_initial_state(self, pipeline):
        assert pipeline.state == StreamState.IDLE

    def test_reset(self, pipeline):
        pipeline._state = StreamState.LISTENING
        pipeline._history.append({"role": "user", "content": "test"})

        pipeline.reset()

        assert pipeline.state == StreamState.IDLE
        assert len(pipeline.history) == 0

    @pytest.mark.asyncio
    async def test_state_change_callback(self, pipeline):
        states_received = []

        async def on_state(state):
            states_received.append(state)

        pipeline.on_state_change = on_state

        await pipeline._set_state(StreamState.LISTENING)
        await pipeline._set_state(StreamState.TRANSCRIBING)

        assert StreamState.LISTENING in states_received
        assert StreamState.TRANSCRIBING in states_received


class TestIntegration:
    """Integration tests for streaming pipeline."""

    @pytest.mark.asyncio
    async def test_audio_chunk_processing(self):
        """Test processing audio chunks through pipeline."""
        # Skip if torch is not properly installed (needed for VAD)
        try:
            import torch
        except (ImportError, OSError):
            pytest.skip("torch not properly installed")

        pipeline = StreamingPipeline()

        # Generate test audio chunk
        t = np.linspace(0, 0.1, 1600, endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t) * 0.5
        chunk = (audio * 32767).astype(np.int16).tobytes()

        # Process should not raise
        result = await pipeline.process_audio(chunk)

        # With just one silent chunk, should stay in IDLE
        assert pipeline.state == StreamState.IDLE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
