"""
ONNX-based TTS for faster inference.
Supports Gwen-TTS ONNX models.
"""

import numpy as np
from typing import Tuple, Optional
from pathlib import Path

from ..utils.logging import logger, debug_log


class GwenONNXTTS:
    """
    Gwen-TTS using ONNX Runtime for faster inference.
    
    Uses ONNX runtime with optimizations:
    - CUDA/TensorRT execution provider for GPU
    - Quantization support (fp16, int8)
    - Faster than PyTorch
    """

    def __init__(
        self,
        model_id: str = "g-group-ai-lab/gwen-tts-0.6B",
        device: str = "cuda",
        speech_rate: float = 1.5,
        quantization: str = "fp16",  # fp32, fp16, int8
    ):
        self.model_id = model_id
        self.device = device
        self.speech_rate = speech_rate
        self.quantization = quantization
        
        self._session = None
        self._tokenizer = None
        
    def _ensure_loaded(self):
        """Lazy load ONNX model."""
        if self._session is not None:
            return
            
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            import os
            
            # Set HF token if available
            hf_token = os.getenv("HF_HUB_TOKEN")
            
            # Download ONNX model
            model_filename = f"model_{self.quantization}.onnx"
            model_path = hf_hub_download(
                repo_id=self.model_id,
                filename=model_filename,
                token=hf_token,
            )
            
            # Setup execution providers
            providers = []
            if self.device.startswith("cuda"):
                providers.append(("CUDAExecutionProvider", {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                }))
                # Try TensorRT if available
                try:
                    providers.insert(0, ("TensorrtExecutionProvider", {
                        "device_id": 0,
                        "trt_max_workspace_size": 2147483648,
                        "trt_fp16_enable": self.quantization == "fp16",
                    }))
                except:
                    pass
            providers.append("CPUExecutionProvider")
            
            # Create session
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self._session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers,
            )
            
            debug_log(f"ONNX TTS loaded with providers: {self._session.get_providers()}")
            
        except Exception as e:
            raise ImportError(f"Failed to load ONNX TTS: {e}")
    
    def generate_voice_clone(
        self,
        text: str,
        ref_audio: str,
        ref_text: str,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """Generate speech with voice cloning."""
        self._ensure_loaded()
        
        # ONNX inference
        # Note: Actual implementation depends on ONNX model structure
        # This is a placeholder - you'll need to adapt based on exported model
        
        raise NotImplementedError(
            "ONNX TTS requires proper model export. "
            "Use PyTorch version or edge-tts for now."
        )


class StreamingONNXTTS:
    """Streaming ONNX TTS for chunk-by-chunk synthesis."""
    
    def __init__(self, base_model: GwenONNXTTS):
        self.model = base_model
        
    async def synthesize_stream(self, text: str, **kwargs):
        """Yield audio chunks as they're generated."""
        # Split text into sentences for streaming
        sentences = self._split_sentences(text)
        
        for sentence in sentences:
            if sentence.strip():
                audio, sr = self.model.generate_voice_clone(sentence, **kwargs)
                yield audio, sr
    
    def _split_sentences(self, text: str) -> list:
        """Split text into sentences for streaming."""
        import re
        # Vietnamese sentence endings
        sentences = re.split(r'([.!?]+\s+)', text)
        result = []
        for i in range(0, len(sentences) - 1, 2):
            result.append(sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else ''))
        if len(sentences) % 2 == 1:
            result.append(sentences[-1])
        return [s for s in result if s.strip()]
