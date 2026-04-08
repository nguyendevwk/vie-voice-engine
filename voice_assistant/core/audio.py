"""
Audio preprocessing module for ASR.
Advanced normalization pipeline for optimal ASR accuracy.
"""

import numpy as np
from scipy.signal import butter, sosfilt, wiener, medfilt
from scipy.ndimage import uniform_filter1d
from typing import Optional, Tuple

from ..config import settings


class AudioNormalizer:
    """
    Advanced audio normalization for ASR.

    Techniques:
    - RMS normalization (consistent volume)
    - Pre-emphasis (boost high frequencies for speech)
    - Spectral subtraction (noise reduction)
    - Dynamic range compression
    - Clipping detection and repair
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        target_rms_db: float = -20.0,
        pre_emphasis: float = 0.97,
    ):
        self.sample_rate = sample_rate
        self.target_rms_linear = 10.0 ** (target_rms_db / 20.0)
        self.pre_emphasis = pre_emphasis

    def rms_normalize(
        self,
        audio: np.ndarray,
        target_db: float = -20.0,
    ) -> np.ndarray:
        """
        Normalize audio to target RMS level.

        This ensures consistent volume regardless of input level.
        """
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-10:
            return audio

        target_linear = 10.0 ** (target_db / 20.0)
        gain = target_linear / rms

        # Limit gain to prevent noise amplification
        gain = min(gain, 10.0)  # Max 20dB gain

        return audio * gain

    def apply_pre_emphasis(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply pre-emphasis filter to boost high frequencies.

        Speech has more energy in lower frequencies. Pre-emphasis
        helps balance the spectrum for better ASR performance.

        y[n] = x[n] - α * x[n-1]
        """
        return np.append(audio[0], audio[1:] - self.pre_emphasis * audio[:-1])

    def remove_pre_emphasis(self, audio: np.ndarray) -> np.ndarray:
        """Remove pre-emphasis (de-emphasis) for playback."""
        return np.append(audio[0], audio[1:] + self.pre_emphasis * audio[:-1])

    def detect_clipping(self, audio: np.ndarray, threshold: float = 0.99) -> bool:
        """Detect if audio has clipping."""
        return np.any(np.abs(audio) > threshold)

    def repair_clipping(self, audio: np.ndarray) -> np.ndarray:
        """
        Attempt to repair clipped audio using cubic interpolation.

        Note: This is a best-effort repair, severely clipped audio
        cannot be fully recovered.
        """
        # Find clipped regions
        clipped = np.abs(audio) > 0.99

        if not np.any(clipped):
            return audio

        # Simple soft-clip using tanh with scaling to stay strictly under 1.0
        repaired = np.tanh(audio * 0.8) / np.tanh(0.8)
        return repaired * 0.999

    def apply_compression(
        self,
        audio: np.ndarray,
        threshold_db: float = -20.0,
        ratio: float = 4.0,
    ) -> np.ndarray:
        """
        Apply dynamic range compression.

        Reduces the difference between loud and quiet parts,
        making speech more consistent.
        """
        threshold_linear = 10.0 ** (threshold_db / 20.0)

        # Compute envelope
        envelope = np.abs(audio)
        envelope = uniform_filter1d(envelope, size=int(self.sample_rate * 0.01))

        # Apply compression
        gain = np.ones_like(audio)
        above_threshold = envelope > threshold_linear

        if np.any(above_threshold):
            # Compression: output = threshold + (input - threshold) / ratio
            excess = envelope[above_threshold] - threshold_linear
            compressed_excess = excess / ratio
            gain[above_threshold] = (threshold_linear + compressed_excess) / envelope[above_threshold]

        return audio * gain

    def normalize_full(self, audio: np.ndarray) -> np.ndarray:
        """
        Full normalization pipeline for ASR.

        Steps:
        1. Repair clipping if detected
        2. RMS normalize to -20dB
        3. Apply pre-emphasis
        4. Apply light compression
        """
        # 1. Repair clipping
        if self.detect_clipping(audio):
            audio = self.repair_clipping(audio)

        # 2. RMS normalize
        audio = self.rms_normalize(audio, target_db=-20.0)

        # 3. Pre-emphasis
        audio = self.apply_pre_emphasis(audio)

        # 4. Light compression
        audio = self.apply_compression(audio, threshold_db=-15.0, ratio=2.0)

        # Final clip to [-1, 1]
        return np.clip(audio, -1.0, 1.0)


class NoiseReducer:
    """
    Noise reduction for speech enhancement.

    Methods:
    - Spectral gating (simple, fast)
    - Wiener filter (adaptive)
    - Median filter (impulse noise)
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def spectral_gate(
        self,
        audio: np.ndarray,
        threshold_db: float = -40.0,
        frame_size: int = 512,
    ) -> np.ndarray:
        """
        Apply spectral gating to reduce background noise.

        Zeros out frequency bins below threshold.
        """
        hop_size = frame_size // 2
        n_frames = (len(audio) - frame_size) // hop_size + 1

        if n_frames < 1:
            return audio

        # Window
        window = np.hanning(frame_size)
        threshold_linear = 10.0 ** (threshold_db / 20.0)

        output = np.zeros_like(audio)
        window_sum = np.zeros_like(audio)

        for i in range(n_frames):
            start = i * hop_size
            end = start + frame_size

            frame = audio[start:end] * window
            spectrum = np.fft.rfft(frame)
            magnitude = np.abs(spectrum)

            # Gate
            mask = magnitude > threshold_linear
            spectrum = spectrum * mask

            # Inverse FFT
            frame_out = np.fft.irfft(spectrum, n=frame_size)
            output[start:end] += frame_out * window
            window_sum[start:end] += window ** 2

        # Normalize by window overlap
        window_sum = np.maximum(window_sum, 1e-10)
        return output / window_sum

    def wiener_filter(self, audio: np.ndarray, noise_power: float = None) -> np.ndarray:
        """
        Apply Wiener filter for adaptive noise reduction.

        If noise_power is not provided, estimates from signal.
        """
        return wiener(audio)

    def remove_impulse_noise(self, audio: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Remove impulse noise (clicks, pops) using median filter.
        """
        return medfilt(audio, kernel_size=kernel_size)


class AudioPreprocessor:
    """
    Complete audio preprocessing pipeline for ASR.

    Pipeline:
    1. Decode PCM16 → float32
    2. Remove DC offset
    3. Noise reduction (optional)
    4. Band-pass filter (80Hz - 7600Hz)
    5. Noise gate
    6. Normalization (RMS + pre-emphasis)

    This produces clean, normalized audio optimal for ASR.
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

        # Advanced processors
        self._normalizer = AudioNormalizer(sample_rate=self.sample_rate)
        self._noise_reducer = NoiseReducer(sample_rate=self.sample_rate)

        # Processing options
        self.enable_noise_reduction = True
        self.enable_pre_emphasis = True
        self.enable_compression = False  # Optional, can increase latency

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

    def apply_bandpass(self, audio: np.ndarray) -> np.ndarray:
        """Apply band-pass filter (combines HP and LP)."""
        audio = self.apply_highpass(audio)
        audio = self.apply_lowpass(audio)
        return audio

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

    def process(self, raw: bytes, for_asr: bool = True) -> np.ndarray:
        """
        Full preprocessing pipeline.

        Args:
            raw: PCM S16LE bytes
            for_asr: If True, applies ASR-optimized processing

        Returns:
            Processed float32 audio array
        """
        audio = self.decode_pcm16(raw)

        # 1. Remove DC offset
        audio = self.remove_dc_offset(audio)

        # 2. Noise reduction (light)
        if self.enable_noise_reduction and len(audio) >= 512:
            audio = self._noise_reducer.spectral_gate(audio, threshold_db=-45.0)

        # 3. Band-pass filter
        audio = self.apply_bandpass(audio)

        # 4. Noise gate
        audio = self.apply_noise_gate(audio)

        if for_asr:
            # 5. RMS normalize
            audio = self._normalizer.rms_normalize(audio, target_db=-20.0)

            # 6. Pre-emphasis (important for ASR)
            if self.enable_pre_emphasis:
                audio = self._normalizer.apply_pre_emphasis(audio)

            # 7. Optional compression
            if self.enable_compression:
                audio = self._normalizer.apply_compression(audio)

        # Final clip and ensure float32
        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    def process_simple(self, raw: bytes) -> np.ndarray:
        """
        Simplified processing (faster, less aggressive).

        Good for real-time with low latency requirement.
        """
        audio = self.decode_pcm16(raw)
        audio = self.remove_dc_offset(audio)
        audio = self.apply_bandpass(audio)
        audio = self.apply_noise_gate(audio)
        audio = self.peak_normalize(audio)
        return audio

    def process_to_bytes(self, raw: bytes, for_asr: bool = True) -> bytes:
        """Process and return as PCM S16LE bytes."""
        audio = self.process(raw, for_asr=for_asr)
        return self.encode_pcm16(audio)

    @staticmethod
    def get_rms_db(audio: np.ndarray) -> float:
        """Get RMS level in dB."""
        rms = np.sqrt(np.mean(audio ** 2))
        return float(20 * np.log10(max(rms, 1e-10)))

    @staticmethod
    def get_duration_ms(audio: np.ndarray, sample_rate: int = 16000) -> float:
        """Get audio duration in milliseconds."""
        return len(audio) / sample_rate * 1000

    @staticmethod
    def get_snr_estimate(audio: np.ndarray, frame_size: int = 512) -> float:
        """
        Estimate Signal-to-Noise Ratio.

        Uses the ratio of maximum to minimum frame energy.
        """
        n_frames = len(audio) // frame_size
        if n_frames < 2:
            return 0.0

        frame_energies = []
        for i in range(n_frames):
            frame = audio[i * frame_size:(i + 1) * frame_size]
            energy = np.sqrt(np.mean(frame ** 2))
            if energy > 1e-10:
                frame_energies.append(energy)

        if len(frame_energies) < 2:
            return 0.0

        # SNR estimate: max energy / min energy (in dB)
        max_energy = np.max(frame_energies)
        min_energy = np.percentile(frame_energies, 10)  # 10th percentile as noise floor

        if min_energy < 1e-10:
            return 60.0  # Very clean

        return 20 * np.log10(max_energy / min_energy)


# Convenience instances
preprocessor = AudioPreprocessor()
normalizer = AudioNormalizer()
noise_reducer = NoiseReducer()
