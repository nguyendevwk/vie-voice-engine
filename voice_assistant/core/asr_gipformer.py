"""
Gipformer ASR Wrapper for Vietnamese Speech Recognition.
Integrates with icefall/k2 for streaming inference.
"""

import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

from ..config import settings
from ..utils.logging import logger, debug_log, latency


class GipformerASR:
    """
    Gipformer ASR wrapper for Vietnamese speech recognition.

    Based on inferances_demo/asr/infer_pytorch.py
    Supports both batch and streaming modes.
    """

    REPO_ID = "g-group-ai-lab/gipformer-65M-rnnt"
    ICEFALL_REPO = "https://github.com/k2-fsa/icefall.git"
    SAMPLE_RATE = 16000

    PT_FILES = {
        "checkpoint": "epoch-35-avg-6.pt",
        "bpe_model": "bpe.model",
        "tokens": "tokens.txt",
    }

    def __init__(self, device: str = "auto"):
        self.device = device
        self._model = None
        self._sp = None  # SentencePiece
        self._fbank = None
        self._params = None

        # Paths
        self._icefall_dir = Path.home() / ".cache" / "gipformer" / "icefall"
        self._model_paths = None

    def _setup_icefall(self):
        """Setup icefall library (one-time)."""
        marker = self._icefall_dir / "icefall" / "__init__.py"
        if not marker.exists():
            logger.info("Setting up icefall (one-time download)...")
            self._icefall_dir.parent.mkdir(parents=True, exist_ok=True)

            if self._icefall_dir.exists():
                import shutil
                shutil.rmtree(self._icefall_dir)

            import subprocess
            subprocess.run([
                "git", "clone", "--depth", "1",
                "--filter=blob:none", "--sparse",
                self.ICEFALL_REPO, str(self._icefall_dir),
            ], check=True)

            subprocess.run(
                ["git", "sparse-checkout", "set", "icefall", "egs/librispeech/ASR"],
                cwd=str(self._icefall_dir),
                check=True,
            )
            logger.info("Icefall setup complete.")

        # Mock lhotse (not needed for inference)
        self._mock_lhotse()

        # Add to sys.path
        for p in [
            str(self._icefall_dir),
            str(self._icefall_dir / "egs" / "librispeech" / "ASR"),
            str(self._icefall_dir / "egs" / "librispeech" / "ASR" / "zipformer"),
        ]:
            if p not in sys.path:
                sys.path.insert(0, p)

    def _mock_lhotse(self):
        """Mock lhotse module (only used for training)."""
        class _MockModule(types.ModuleType):
            class _Dummy:
                def __init__(self, *a, **kw): pass
                def __call__(self, *a, **kw): return self
                def __getattr__(self, name): return type(self)()
            def __getattr__(self, name): return self._Dummy

        class _LhotseFinder:
            def find_module(self, fullname, path=None):
                if fullname == "lhotse" or fullname.startswith("lhotse."):
                    return self
                return None

            def load_module(self, fullname):
                if fullname in sys.modules:
                    return sys.modules[fullname]
                mod = _MockModule(fullname)
                mod.__path__ = []
                mod.__loader__ = self
                mod.__file__ = "<mocked>"
                mod.__version__ = "0.0.0"
                sys.modules[fullname] = mod
                return mod

        sys.meta_path.insert(0, _LhotseFinder())

    def _download_model(self) -> dict:
        """Download model files from HuggingFace."""
        from huggingface_hub import hf_hub_download

        logger.info(f"Downloading model from {self.REPO_ID}...")
        paths = {}
        for key, filename in self.PT_FILES.items():
            paths[key] = hf_hub_download(repo_id=self.REPO_ID, filename=filename)
        logger.info("Model downloaded successfully.")
        return paths

    def _ensure_loaded(self):
        """Lazy load model."""
        if self._model is not None:
            return

        import torch

        # Setup icefall
        self._setup_icefall()

        # Import after icefall setup
        import k2
        import kaldifeat
        import sentencepiece as spm
        from train import add_model_arguments, get_model, get_params
        import argparse

        # Download model
        self._model_paths = self._download_model()

        # Device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(self.device)

        # Build params
        old_git_dir = os.environ.get("GIT_DIR")
        os.environ["GIT_DIR"] = str(self._icefall_dir / ".git")

        parser = argparse.ArgumentParser()
        add_model_arguments(parser)
        args, _ = parser.parse_known_args([])

        self._params = get_params()
        self._params.update(vars(args))
        self._params.decoding_method = "modified_beam_search"
        self._params.beam_size = 4
        self._params.context_size = 2

        if old_git_dir is None:
            os.environ.pop("GIT_DIR", None)
        else:
            os.environ["GIT_DIR"] = old_git_dir

        # Token table
        token_table = k2.SymbolTable.from_file(self._model_paths["tokens"])
        self._params.blank_id = token_table["<blk>"]
        self._params.unk_id = token_table["<unk>"]

        # Count tokens
        num_tokens = sum(1 for s in token_table.symbols if not s.startswith("#"))
        if token_table["<blk>"] == 0:
            num_tokens -= 1
        self._params.vocab_size = num_tokens + 1

        # Build model
        self._model = get_model(self._params)

        checkpoint = torch.load(
            self._model_paths["checkpoint"],
            map_location="cpu",
            weights_only=False,
        )
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            self._model.load_state_dict(checkpoint["model"], strict=False)
        else:
            self._model.load_state_dict(checkpoint, strict=False)

        self._model.to(device)
        self._model.eval()
        self._device = device

        logger.info(f"Model parameters: {sum(p.numel() for p in self._model.parameters()):,}")

        # SentencePiece
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(self._model_paths["bpe_model"])

        # Feature extractor
        opts = kaldifeat.FbankOptions()
        opts.device = device
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = self.SAMPLE_RATE
        opts.mel_opts.num_bins = 80
        opts.mel_opts.high_freq = -400
        self._fbank = kaldifeat.Fbank(opts)

        logger.info("Gipformer ASR model loaded")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file."""
        self._ensure_loaded()

        import torch
        import torchaudio
        import math
        from torch.nn.utils.rnn import pad_sequence
        from beam_search import modified_beam_search

        with torch.no_grad():
            # Load audio
            wave, sr = torchaudio.load(audio_path)
            if sr != self.SAMPLE_RATE:
                wave = torchaudio.functional.resample(wave, sr, self.SAMPLE_RATE)
            wave = wave[0].contiguous().to(self._device)

            # Extract features
            features = self._fbank([wave])
            feature_lengths = torch.tensor([features[0].size(0)], device=self._device)
            features = pad_sequence(
                features, batch_first=True, padding_value=math.log(1e-10)
            )

            # Encode
            encoder_out, encoder_out_lens = self._model.forward_encoder(
                features, feature_lengths
            )

            # Decode
            hyp_tokens = modified_beam_search(
                model=self._model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                beam=self._params.beam_size,
            )

            # Convert to text
            text = self._sp.decode(hyp_tokens[0])

        return text

    def transcribe_array(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio array."""
        self._ensure_loaded()

        import torch
        import torchaudio
        import math
        from torch.nn.utils.rnn import pad_sequence
        from beam_search import modified_beam_search

        with torch.no_grad():
            # Convert to tensor
            wave = torch.from_numpy(audio.astype(np.float32))
            if sample_rate != self.SAMPLE_RATE:
                wave = torchaudio.functional.resample(wave, sample_rate, self.SAMPLE_RATE)
            wave = wave.contiguous().to(self._device)

            # Extract features
            features = self._fbank([wave])
            feature_lengths = torch.tensor([features[0].size(0)], device=self._device)
            features = pad_sequence(
                features, batch_first=True, padding_value=math.log(1e-10)
            )

            # Encode
            encoder_out, encoder_out_lens = self._model.forward_encoder(
                features, feature_lengths
            )

            # Decode
            hyp_tokens = modified_beam_search(
                model=self._model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                beam=self._params.beam_size,
            )

            # Convert to text
            text = self._sp.decode(hyp_tokens[0])

        return text


class StreamingGipformerASR(GipformerASR):
    """
    Streaming version of Gipformer ASR.

    Uses chunked processing for lower latency interim results.
    """

    def __init__(self, device: str = "auto", chunk_size_ms: int = 1000):
        super().__init__(device)
        self.chunk_size_ms = chunk_size_ms
        self._encoder_states = None

    def reset_stream(self):
        """Reset streaming state."""
        self._encoder_states = None

    def transcribe_chunk(self, audio_chunk: np.ndarray) -> Tuple[str, bool]:
        """
        Transcribe a chunk of audio in streaming mode.

        Args:
            audio_chunk: Audio samples (float32, 16kHz)

        Returns:
            (text, is_endpoint) - Partial transcription and whether endpoint detected
        """
        self._ensure_loaded()

        import torch
        import math
        from torch.nn.utils.rnn import pad_sequence
        from beam_search import modified_beam_search

        with torch.no_grad():
            wave = torch.from_numpy(audio_chunk.astype(np.float32)).to(self._device)

            # Extract features for this chunk
            features = self._fbank([wave])
            feature_lengths = torch.tensor([features[0].size(0)], device=self._device)
            features = pad_sequence(
                features, batch_first=True, padding_value=math.log(1e-10)
            )

            # Encode
            encoder_out, encoder_out_lens = self._model.forward_encoder(
                features, feature_lengths
            )

            # Decode current chunk
            hyp_tokens = modified_beam_search(
                model=self._model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                beam=self._params.beam_size,
            )

            text = self._sp.decode(hyp_tokens[0])

        return text, False  # Endpoint detection not implemented yet
