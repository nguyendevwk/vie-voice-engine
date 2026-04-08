"""
Gipformer ONNX ASR Wrapper for Vietnamese Speech Recognition.
Uses sherpa-onnx for fast, cross-platform inference.

Based on inferances_demo/asr/infer_onnx.py
"""

import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from ..config import settings
from ..utils.logging import logger, log_asr_event


class GipformerONNXASR:
    """
    Gipformer ASR using ONNX runtime via sherpa-onnx.

    Advantages over PyTorch version:
    - No k2/icefall dependencies
    - Faster inference
    - Cross-platform (CPU/GPU)
    - Supports streaming
    """

    REPO_ID = "g-group-ai-lab/gipformer-65M-rnnt"
    SAMPLE_RATE = 16000
    FEATURE_DIM = 80

    ONNX_FILES = {
        "fp32": {
            "encoder": "encoder-epoch-35-avg-6.onnx",
            "decoder": "decoder-epoch-35-avg-6.onnx",
            "joiner": "joiner-epoch-35-avg-6.onnx",
        },
        "int8": {
            "encoder": "encoder-epoch-35-avg-6.int8.onnx",
            "decoder": "decoder-epoch-35-avg-6.int8.onnx",
            "joiner": "joiner-epoch-35-avg-6.int8.onnx",
        },
    }

    def __init__(
        self,
        quantize: str = "int8",
        num_threads: int = 4,
        decoding_method: str = "modified_beam_search",
    ):
        """
        Initialize ONNX ASR.

        Args:
            quantize: "fp32" for full precision, "int8" for quantized (faster)
            num_threads: Number of threads for inference
            decoding_method: "greedy_search" or "modified_beam_search"
        """
        self.quantize = quantize
        self.num_threads = num_threads
        self.decoding_method = decoding_method

        self._recognizer = None
        self._online_recognizer = None  # For streaming
        self._model_paths = None

    def _download_model(self) -> dict:
        """Download ONNX model files from HuggingFace."""
        from huggingface_hub import hf_hub_download

        files = self.ONNX_FILES[self.quantize]
        logger.info(f"Downloading {self.quantize} model from {self.REPO_ID}...")

        paths = {}
        for key, filename in files.items():
            log_asr_event(f"Downloading {key}...")
            paths[key] = hf_hub_download(repo_id=self.REPO_ID, filename=filename)

        paths["tokens"] = hf_hub_download(repo_id=self.REPO_ID, filename="tokens.txt")

        logger.info("ONNX model downloaded successfully.")
        return paths

    def _ensure_loaded(self):
        """Lazy load recognizer."""
        if self._recognizer is not None:
            return

        try:
            import sherpa_onnx
        except ImportError:
            raise ImportError(
                "sherpa-onnx is required for ONNX inference. "
                "Install it with: pip install sherpa-onnx"
            )

        # Download model
        self._model_paths = self._download_model()

        # Create offline recognizer
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=self._model_paths["encoder"],
            decoder=self._model_paths["decoder"],
            joiner=self._model_paths["joiner"],
            tokens=self._model_paths["tokens"],
            num_threads=self.num_threads,
            sample_rate=self.SAMPLE_RATE,
            feature_dim=self.FEATURE_DIM,
            decoding_method=self.decoding_method,
        )

        logger.info(f"ONNX ASR loaded: quantize={self.quantize}, threads={self.num_threads}")

    def _ensure_streaming_loaded(self):
        """Lazy load streaming recognizer."""
        if self._online_recognizer is not None:
            return

        try:
            import sherpa_onnx
        except ImportError:
            raise ImportError(
                "sherpa-onnx is required for ONNX inference. "
                "Install it with: pip install sherpa-onnx"
            )

        # Download model if needed
        if self._model_paths is None:
            self._model_paths = self._download_model()

        # Create online recognizer for streaming
        self._online_recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=self._model_paths["encoder"],
            decoder=self._model_paths["decoder"],
            joiner=self._model_paths["joiner"],
            tokens=self._model_paths["tokens"],
            num_threads=self.num_threads,
            sample_rate=self.SAMPLE_RATE,
            feature_dim=self.FEATURE_DIM,
            decoding_method=self.decoding_method,
            enable_endpoint_detection=True,
        )

        logger.info("Streaming ONNX ASR loaded")

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file (WAV, FLAC, etc.)

        Returns:
            Transcribed text
        """
        self._ensure_loaded()

        import soundfile as sf

        start_time = time.perf_counter()

        # Read audio
        samples, sample_rate = sf.read(audio_path, dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        # Create stream and process
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self._recognizer.decode_streams([stream])

        text = stream.result.text.strip()

        latency_ms = (time.perf_counter() - start_time) * 1000
        log_asr_event("Transcribe complete", text=text, latency_ms=latency_ms)

        return text

    def transcribe_array(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio array.

        Args:
            audio: Audio samples (float32 or int16)
            sample_rate: Sample rate

        Returns:
            Transcribed text
        """
        self._ensure_loaded()

        start_time = time.perf_counter()

        # Ensure float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Create stream and process
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        self._recognizer.decode_streams([stream])

        text = stream.result.text.strip()

        latency_ms = (time.perf_counter() - start_time) * 1000
        log_asr_event("Transcribe array", text=text, latency_ms=latency_ms)

        return text

    def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """
        Transcribe PCM16 audio bytes.

        Args:
            audio_bytes: PCM S16LE bytes
            sample_rate: Sample rate

        Returns:
            Transcribed text
        """
        # Convert to float32 array
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return self.transcribe_array(audio, sample_rate)


class StreamingONNXASR:
    """
    Streaming ASR using sherpa-onnx OnlineRecognizer.

    Provides real-time transcription with endpoint detection.
    """

    def __init__(self, model: GipformerONNXASR = None):
        """
        Initialize streaming ASR.

        Args:
            model: Optional pre-initialized GipformerONNXASR
        """
        self._model = model or GipformerONNXASR()
        self._stream = None
        self._accumulated_text = ""

    def start_stream(self):
        """Start a new streaming session."""
        self._model._ensure_streaming_loaded()
        self._stream = self._model._online_recognizer.create_stream()
        self._accumulated_text = ""
        log_asr_event("Stream started")

    def add_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> Tuple[str, bool]:
        """
        Add audio chunk to stream.

        Args:
            audio: Audio samples (float32)
            sample_rate: Sample rate

        Returns:
            (partial_text, is_endpoint)
        """
        if self._stream is None:
            self.start_stream()

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Add to stream
        self._stream.accept_waveform(sample_rate, audio)

        # Process
        while self._model._online_recognizer.is_ready(self._stream):
            self._model._online_recognizer.decode_stream(self._stream)

        # Get result
        result = self._model._online_recognizer.get_result(self._stream)
        text = result.text.strip()
        is_endpoint = self._model._online_recognizer.is_endpoint(self._stream)

        if is_endpoint:
            log_asr_event("Endpoint detected", text=text)
            self._model._online_recognizer.reset(self._stream)

        return text, is_endpoint

    def add_audio_bytes(self, audio_bytes: bytes, sample_rate: int = 16000) -> Tuple[str, bool]:
        """
        Add PCM16 audio bytes to stream.

        Args:
            audio_bytes: PCM S16LE bytes
            sample_rate: Sample rate

        Returns:
            (partial_text, is_endpoint)
        """
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return self.add_audio(audio, sample_rate)

    def get_final_result(self) -> str:
        """Get final transcription and close stream."""
        if self._stream is None:
            return ""

        # Finalize
        self._stream.input_finished()

        while self._model._online_recognizer.is_ready(self._stream):
            self._model._online_recognizer.decode_stream(self._stream)

        result = self._model._online_recognizer.get_result(self._stream)
        text = result.text.strip()

        log_asr_event("Stream finalized", text=text)
        self._stream = None

        return text

    def reset(self):
        """Reset stream without getting result."""
        self._stream = None
        self._accumulated_text = ""
