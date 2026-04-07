"""
Audio preprocessing module.
Simplified pipeline: decode → filter → normalize.
"""

import numpy as np
from scipy.signal import butter, sosfilt
from typing import Optional

from ..config import settings


class AudioPreprocessor:
    """
    Lightweight audio preprocessing for voice assistant.

    Pipeline:
    1. Decode PCM16 → float32
    2. Remove DC offset
    3. High-pass filter (remove low freq noise)
    4. Low-pass filter (remove high freq noise)
    5. Noise gate (suppress silence)
    6. Peak normalize
    """

    def __init__(self, config=None):
        cfg = config or settings.audio
        self.sample_rate = cfg.sample_rate

        # Pre-compute Butterworth filters
        self._hp_sos = butter(
            N=5,
            Wn=cfg.high_pass_hz,
            btype="highpass",
            fs=self.sample_rate,
            output="sos",
        )
        self._lp_sos = butter(
            N=5,
            Wn=cfg.low_pass_hz,
            btype="lowpass",
            fs=self.sample_rate,
            output="sos",
        )

        # Noise gate threshold: dB → linear
        self._noise_gate_linear = 10.0 ** (cfg.noise_gate_db / 20.0)

    def decode_pcm16(self, raw: bytes) -> np.ndarray:
        """Convert PCM S16LE bytes to float32 [-1.0, 1.0]."""
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        return audio / 32768.0

    def encode_pcm16(self, audio: np.ndarray) -> bytes:
        """Convert float32 [-1.0, 1.0] to PCM S16LE bytes."""
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        return pcm.tobytes()

    def remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """Remove DC offset (center waveform at zero)."""
        return audio - np.mean(audio)

    def apply_highpass(self, audio: np.ndarray) -> np.ndarray:
        """Apply high-pass filter to remove low frequency noise."""
        if len(audio) < 13:  # Filter needs minimum samples
            return audio
        return sosfilt(self._hp_sos, audio)

    def apply_lowpass(self, audio: np.ndarray) -> np.ndarray:
        """Apply low-pass filter to remove high frequency noise."""
        if len(audio) < 13:
            return audio
        return sosfilt(self._lp_sos, audio)

    def apply_noise_gate(self, audio: np.ndarray) -> np.ndarray:
        """Zero out audio if RMS is below threshold."""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < self._noise_gate_linear:
            return np.zeros_like(audio)
        return audio

    def peak_normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize to peak amplitude."""
        peak = np.max(np.abs(audio))
        if peak > 0:
            return audio / peak
        return audio

    def process(self, raw: bytes) -> np.ndarray:
        """
        Full preprocessing pipeline.

        Args:
            raw: PCM S16LE bytes

        Returns:
            Processed float32 audio array
        """
        audio = self.decode_pcm16(raw)
        audio = self.remove_dc_offset(audio)
        audio = self.apply_highpass(audio)
        audio = self.apply_lowpass(audio)
        audio = self.apply_noise_gate(audio)
        audio = self.peak_normalize(audio)
        return audio

    def process_to_bytes(self, raw: bytes) -> bytes:
        """Process and return as PCM S16LE bytes."""
        audio = self.process(raw)
        return self.encode_pcm16(audio)

    @staticmethod
    def get_rms_db(audio: np.ndarray) -> float:
        """Get RMS level in dB."""
        rms = np.sqrt(np.mean(audio ** 2))
        return 20 * np.log10(max(rms, 1e-10))

    @staticmethod
    def get_duration_ms(audio: np.ndarray, sample_rate: int = 16000) -> float:
        """Get audio duration in milliseconds."""
        return len(audio) / sample_rate * 1000


# Convenience instance
preprocessor = AudioPreprocessor()
