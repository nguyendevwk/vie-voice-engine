"""
Tests for audio preprocessing module.
"""

import numpy as np
import pytest

from voice_assistant.core.audio import (
    AudioPreprocessor,
    AudioNormalizer,
    NoiseReducer,
)


class TestAudioPreprocessor:
    """Tests for AudioPreprocessor."""

    @pytest.fixture
    def preprocessor(self):
        return AudioPreprocessor()

    @pytest.fixture
    def sample_audio_bytes(self):
        """Generate sample PCM S16LE audio (1 second of 440Hz sine wave)."""
        sample_rate = 16000
        duration = 1.0
        frequency = 440

        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = np.sin(2 * np.pi * frequency * t) * 0.5  # 50% amplitude
        pcm = (audio * 32767).astype(np.int16)
        return pcm.tobytes()

    @pytest.fixture
    def silence_bytes(self):
        """Generate silence."""
        return np.zeros(1600, dtype=np.int16).tobytes()

    @pytest.fixture
    def noisy_audio_bytes(self):
        """Generate audio with noise."""
        sample_rate = 16000
        duration = 1.0
        frequency = 440

        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        signal = np.sin(2 * np.pi * frequency * t) * 0.5
        noise = np.random.normal(0, 0.1, len(signal))
        audio = signal + noise
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        return pcm.tobytes()

    def test_decode_pcm16(self, preprocessor, sample_audio_bytes):
        """Test PCM16 decoding."""
        audio = preprocessor.decode_pcm16(sample_audio_bytes)

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert np.max(np.abs(audio)) <= 1.0
        assert len(audio) == 16000

    def test_encode_pcm16(self, preprocessor):
        """Test PCM16 encoding."""
        audio = np.sin(np.linspace(0, 2 * np.pi, 1000)).astype(np.float32)
        pcm_bytes = preprocessor.encode_pcm16(audio)

        assert isinstance(pcm_bytes, bytes)
        assert len(pcm_bytes) == len(audio) * 2  # 16-bit = 2 bytes

    def test_remove_dc_offset(self, preprocessor):
        """Test DC offset removal."""
        # Audio with DC offset
        audio = np.ones(1000, dtype=np.float32) * 0.5 + np.sin(np.linspace(0, 2 * np.pi, 1000)) * 0.1
        processed = preprocessor.remove_dc_offset(audio)

        assert np.abs(np.mean(processed)) < 0.01

    def test_noise_gate_silence(self, preprocessor, silence_bytes):
        """Test noise gate with silence."""
        audio = preprocessor.decode_pcm16(silence_bytes)
        gated = preprocessor.apply_noise_gate(audio)

        assert np.all(gated == 0)

    def test_noise_gate_signal(self, preprocessor, sample_audio_bytes):
        """Test noise gate passes signal."""
        audio = preprocessor.decode_pcm16(sample_audio_bytes)
        gated = preprocessor.apply_noise_gate(audio)

        # Signal should pass through
        assert not np.all(gated == 0)

    def test_peak_normalize(self, preprocessor):
        """Test peak normalization."""
        audio = np.sin(np.linspace(0, 2 * np.pi, 1000)).astype(np.float32) * 0.3
        normalized = preprocessor.peak_normalize(audio)

        assert np.abs(np.max(np.abs(normalized)) - 1.0) < 0.01

    def test_full_pipeline(self, preprocessor, sample_audio_bytes):
        """Test full preprocessing pipeline."""
        processed = preprocessor.process(sample_audio_bytes)

        assert isinstance(processed, np.ndarray)
        assert processed.dtype == np.float32
        assert np.max(np.abs(processed)) <= 1.0
        assert len(processed) == 16000

    def test_process_to_bytes(self, preprocessor, sample_audio_bytes):
        """Test processing to bytes."""
        output = preprocessor.process_to_bytes(sample_audio_bytes)

        assert isinstance(output, bytes)
        assert len(output) == len(sample_audio_bytes)

    def test_get_rms_db(self, preprocessor, sample_audio_bytes):
        """Test RMS calculation."""
        audio = preprocessor.decode_pcm16(sample_audio_bytes)
        rms_db = preprocessor.get_rms_db(audio)

        assert isinstance(rms_db, float)
        assert rms_db < 0  # Should be negative dB for normalized audio
        assert rms_db > -60  # Should not be too quiet

    def test_get_duration_ms(self, preprocessor, sample_audio_bytes):
        """Test duration calculation."""
        audio = preprocessor.decode_pcm16(sample_audio_bytes)
        duration = preprocessor.get_duration_ms(audio)

        assert abs(duration - 1000.0) < 1.0  # ~1000ms

    def test_snr_estimate(self, preprocessor, sample_audio_bytes, noisy_audio_bytes):
        """Test SNR estimation returns valid ratio."""
        clean_audio = preprocessor.decode_pcm16(sample_audio_bytes)
        noisy_audio = preprocessor.decode_pcm16(noisy_audio_bytes)

        clean_snr = preprocessor.get_snr_estimate(clean_audio)
        noisy_snr = preprocessor.get_snr_estimate(noisy_audio)

        # Both should return non-negative values
        assert clean_snr >= 0
        assert noisy_snr >= 0


class TestAudioNormalizer:
    """Tests for AudioNormalizer."""

    @pytest.fixture
    def normalizer(self):
        return AudioNormalizer()

    @pytest.fixture
    def quiet_audio(self):
        """Generate quiet audio (-40dB)."""
        t = np.linspace(0, 1, 16000, endpoint=False)
        return np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.01

    @pytest.fixture
    def loud_audio(self):
        """Generate loud audio (near clipping)."""
        t = np.linspace(0, 1, 16000, endpoint=False)
        return np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.99

    def test_rms_normalize_quiet(self, normalizer, quiet_audio):
        """Test RMS normalization of quiet audio."""
        normalized = normalizer.rms_normalize(quiet_audio, target_db=-20.0)

        # Should be louder
        original_rms = np.sqrt(np.mean(quiet_audio ** 2))
        normalized_rms = np.sqrt(np.mean(normalized ** 2))

        assert normalized_rms > original_rms

    def test_rms_normalize_loud(self, normalizer, loud_audio):
        """Test RMS normalization of loud audio."""
        normalized = normalizer.rms_normalize(loud_audio, target_db=-20.0)

        # Should be quieter
        original_rms = np.sqrt(np.mean(loud_audio ** 2))
        normalized_rms = np.sqrt(np.mean(normalized ** 2))

        assert normalized_rms < original_rms

    def test_pre_emphasis(self, normalizer):
        """Test pre-emphasis filter."""
        t = np.linspace(0, 1, 16000, endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        emphasized = normalizer.apply_pre_emphasis(audio)

        assert len(emphasized) == len(audio)
        # Pre-emphasis should change the audio
        assert not np.allclose(emphasized, audio)

    def test_detect_clipping(self, normalizer):
        """Test clipping detection."""
        clean = np.sin(np.linspace(0, 2 * np.pi, 1000)).astype(np.float32) * 0.5
        clipped = np.sin(np.linspace(0, 2 * np.pi, 1000)).astype(np.float32) * 1.5

        assert not normalizer.detect_clipping(clean)
        assert normalizer.detect_clipping(np.clip(clipped, -1, 1))

    def test_repair_clipping(self, normalizer):
        """Test clipping repair."""
        clipped = np.array([0.5, 0.8, 1.0, 1.0, 1.0, 0.8, 0.5], dtype=np.float32)
        repaired = normalizer.repair_clipping(clipped)

        # Should not have values at exactly 1.0
        assert np.all(np.abs(repaired) < 1.0)

    def test_compression(self, normalizer):
        """Test dynamic range compression."""
        # Audio with varying amplitude
        t = np.linspace(0, 1, 16000, endpoint=False)
        envelope = np.linspace(0.1, 0.9, 16000)
        audio = (np.sin(2 * np.pi * 440 * t) * envelope).astype(np.float32)

        compressed = normalizer.apply_compression(audio, threshold_db=-20.0, ratio=4.0)

        # Dynamic range should be reduced
        original_range = np.max(np.abs(audio)) - np.min(np.abs(audio[audio != 0]))
        compressed_range = np.max(np.abs(compressed)) - np.min(np.abs(compressed[compressed != 0]))

        # Compressed should have smaller dynamic range
        assert compressed_range <= original_range


class TestNoiseReducer:
    """Tests for NoiseReducer."""

    @pytest.fixture
    def reducer(self):
        return NoiseReducer()

    @pytest.fixture
    def noisy_signal(self):
        """Generate signal with noise."""
        t = np.linspace(0, 1, 16000, endpoint=False)
        signal = np.sin(2 * np.pi * 440 * t) * 0.5
        noise = np.random.normal(0, 0.1, len(signal))
        return (signal + noise).astype(np.float32)

    def test_spectral_gate(self, reducer, noisy_signal):
        """Test spectral gating."""
        denoised = reducer.spectral_gate(noisy_signal, threshold_db=-40.0)

        assert len(denoised) == len(noisy_signal)
        # Should reduce noise
        # Note: This is a weak test, spectral gating doesn't always reduce overall energy

    def test_wiener_filter(self, reducer, noisy_signal):
        """Test Wiener filter."""
        filtered = reducer.wiener_filter(noisy_signal)

        assert len(filtered) == len(noisy_signal)

    def test_impulse_removal(self, reducer):
        """Test impulse noise removal."""
        # Clean signal with impulse noise
        signal = np.sin(np.linspace(0, 10 * np.pi, 1000)).astype(np.float32) * 0.5
        signal[500] = 1.0  # Impulse

        cleaned = reducer.remove_impulse_noise(signal)

        # Impulse should be reduced
        assert cleaned[500] < 1.0


class TestIntegration:
    """Integration tests for audio preprocessing."""

    def test_full_asr_pipeline(self):
        """Test complete preprocessing pipeline for ASR."""
        preprocessor = AudioPreprocessor()

        # Generate test audio with various issues
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        # Signal components
        speech_freq = 200  # Typical fundamental frequency
        speech = np.sin(2 * np.pi * speech_freq * t) * 0.3

        # Add harmonics (makes it more speech-like)
        for harmonic in [2, 3, 4, 5]:
            speech += np.sin(2 * np.pi * speech_freq * harmonic * t) * (0.3 / harmonic)

        # Add noise
        noise = np.random.normal(0, 0.05, len(speech))

        # Add DC offset
        dc_offset = 0.1

        # Combine
        audio = speech + noise + dc_offset

        # Convert to PCM bytes
        pcm_bytes = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()

        # Process
        processed = preprocessor.process(pcm_bytes, for_asr=True)

        # Verify output
        assert isinstance(processed, np.ndarray)
        assert processed.dtype == np.float32
        assert np.max(np.abs(processed)) <= 1.0

        # Should have removed DC offset
        assert np.abs(np.mean(processed)) < 0.05

        # Should have reasonable RMS level
        rms_db = preprocessor.get_rms_db(processed)
        assert -35 < rms_db < -10  # Normalized range

    def test_realtime_processing(self):
        """Test real-time chunk processing."""
        preprocessor = AudioPreprocessor()

        # Simulate real-time 100ms chunks
        chunk_size = 1600  # 100ms at 16kHz
        total_chunks = 10

        results = []

        for i in range(total_chunks):
            # Generate chunk
            t = np.linspace(i * 0.1, (i + 1) * 0.1, chunk_size, endpoint=False)
            audio = np.sin(2 * np.pi * 440 * t) * 0.5
            pcm_bytes = (audio * 32767).astype(np.int16).tobytes()

            # Process
            processed = preprocessor.process_simple(pcm_bytes)
            results.append(processed)

        # Verify all chunks processed
        assert len(results) == total_chunks
        for chunk in results:
            assert len(chunk) == chunk_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
