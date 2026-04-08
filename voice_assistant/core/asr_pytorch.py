"""
PyTorch CUDA ASR implementation for Gipformer.
Rewritten from inferances_demo for production use.
"""

import torch
import torchaudio
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path
import sys
import types

from ..utils.logging import logger, debug_log


class GipformerPyTorchASR:
    """
    Gipformer ASR using PyTorch with CUDA acceleration.
    
    Requires:
    - k2 (compiled with CUDA)
    - kaldifeat
    - icefall (auto-setup)
    - sentencepiece
    """

    def __init__(
        self,
        model_id: str = "g-group-ai-lab/gipformer-65M-rnnt",
        device: str = "cuda",
        decoding_method: str = "modified_beam_search",
        beam_size: int = 4,
    ):
        self.model_id = model_id
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.decoding_method = decoding_method
        self.beam_size = beam_size
        self.sample_rate = 16000
        
        self._model = None
        self._sp_model = None
        self._token_table = None
        self._feature_extractor = None
        
        # Setup icefall (for k2/icefall imports)
        self._icefall_dir = Path.home() / ".cache" / "gipformer" / "icefall"
        
    def _ensure_loaded(self):
        """Lazy load model and dependencies."""
        if self._model is not None:
            return
            
        try:
            # Setup icefall in sys.path
            self._setup_icefall()
            
            # Mock lhotse (not needed for inference)
            self._mock_lhotse()
            
            # Import dependencies
            import k2
            import kaldifeat
            import sentencepiece as spm
            from huggingface_hub import hf_hub_download
            
            logger.info(f"Loading PyTorch ASR: {self.model_id}")
            
            # Download model files
            import os
            hf_token = os.getenv("HF_HUB_TOKEN")
            
            checkpoint_path = hf_hub_download(
                repo_id=self.model_id,
                filename="epoch-35-avg-6.pt",
                token=hf_token,
            )
            bpe_path = hf_hub_download(
                repo_id=self.model_id,
                filename="bpe.model",
                token=hf_token,
            )
            tokens_path = hf_hub_download(
                repo_id=self.model_id,
                filename="tokens.txt",
                token=hf_token,
            )
            
            # Load tokenizer
            self._sp_model = spm.SentencePieceProcessor()
            self._sp_model.load(bpe_path)
            
            # Load token table
            self._token_table = k2.SymbolTable.from_file(tokens_path)
            blank_id = self._token_table["<blk>"]
            
            # Count vocabulary
            num_tokens = sum(1 for s in self._token_table.symbols if not s.startswith("#"))
            if blank_id == 0:
                num_tokens -= 1
            vocab_size = num_tokens + 1
            
            # Load model (import train module from icefall)
            sys.path.insert(0, str(self._icefall_dir / "egs" / "vietnamese" / "ASR" / "pruned_transducer_stateless7_streaming"))
            from train import get_model
            
            # Build model with params
            params = types.SimpleNamespace(
                vocab_size=vocab_size,
                blank_id=blank_id,
                context_size=2,
                encoder_dim=512,
                decoder_dim=512,
                joiner_dim=512,
                num_encoder_layers=12,
                use_transducer_loss=True,
            )
            
            self._model = get_model(params)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self._model.load_state_dict(checkpoint["model"], strict=False)
            self._model.to(self.device)
            self._model.eval()
            
            # Feature extractor
            self._feature_extractor = kaldifeat.Fbank(kaldifeat.FbankOptions(
                frame_opts=kaldifeat.FrameExtractionOptions(samp_freq=16000),
                mel_opts=kaldifeat.MelBanksOptions(num_bins=80),
            ))
            
            logger.info("PyTorch ASR loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch ASR: {e}")
            raise ImportError(f"PyTorch ASR setup failed: {e}")
    
    def _setup_icefall(self):
        """Setup icefall in sys.path."""
        marker = self._icefall_dir / "icefall" / "__init__.py"
        if not marker.exists():
            logger.info("Setting up icefall (one-time download)...")
            self._icefall_dir.parent.mkdir(parents=True, exist_ok=True)
            
            import subprocess
            subprocess.run([
                "git", "clone", "--depth", "1", "--filter=blob:none",
                "--sparse", "https://github.com/k2-fsa/icefall.git",
                str(self._icefall_dir)
            ], check=True, capture_output=True)
            
            subprocess.run([
                "git", "-C", str(self._icefall_dir),
                "sparse-checkout", "set",
                "egs/vietnamese/ASR/pruned_transducer_stateless7_streaming",
                "icefall"
            ], check=True, capture_output=True)
            
        # Add to sys.path
        if str(self._icefall_dir) not in sys.path:
            sys.path.insert(0, str(self._icefall_dir))
    
    def _mock_lhotse(self):
        """Mock lhotse module (not needed for inference)."""
        if "lhotse" in sys.modules:
            return
            
        class _MockModule(types.ModuleType):
            class _Dummy:
                def __init__(self, *a, **kw): pass
                def __call__(self, *a, **kw): return self
                def __getattr__(self, name): return type(self)()
            def __getattr__(self, name): return self._Dummy
        
        sys.modules["lhotse"] = _MockModule("lhotse")
    
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio samples (float32, mono)
            sample_rate: Sample rate (will resample if needed)
        
        Returns:
            Transcribed text
        """
        self._ensure_loaded()
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            audio = self._resample(audio, sample_rate, self.sample_rate)
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Extract features
        features = self._feature_extractor(audio_tensor)
        features = features.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Decode
        with torch.no_grad():
            if self.decoding_method == "greedy_search":
                from beam_search import greedy_search_batch
                result = greedy_search_batch(
                    model=self._model,
                    encoder_out=self._model.encoder(features),
                    processed_lens=torch.tensor([features.size(1)]).to(self.device),
                )
            else:
                from beam_search import modified_beam_search
                result = modified_beam_search(
                    model=self._model,
                    encoder_out=self._model.encoder(features),
                    processed_lens=torch.tensor([features.size(1)]).to(self.device),
                    beam=self.beam_size,
                )
        
        # Decode tokens
        token_ids = result[0]
        tokens = [self._token_table[i] for i in token_ids if i != self._token_table["<blk>"]]
        text = self._sp_model.decode(tokens)
        
        return text
    
    def _resample(self, audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """Resample audio."""
        from scipy import signal
        num_samples = int(len(audio) * dst_rate / src_rate)
        return signal.resample(audio, num_samples).astype(np.float32)
