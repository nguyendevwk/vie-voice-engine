#!/usr/bin/env python3
"""
Diagnostic script for VieNeu Remote TTS and ASR issues.
Run: uv run python diagnose.py
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

# Clear conflicting env vars
for key in ['TTS_BACKEND', 'VIENEU_MODE', 'VIENEU_MODEL_ID', 'VIENEU_MODEL_BACKBONE',
            'VIENEU_MODEL_DECODER', 'VIENEU_MODEL_ENCODER']:
    os.environ.pop(key, None)

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def check_vieneu_server():
    """Check if VieNeu server is running and responding correctly."""
    print_section("1. VieNeu Server Check")
    
    try:
        import requests
        base = "http://localhost:23333"
        
        # Check models endpoint
        resp = requests.get(f"{base}/v1/models", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", [])
            print(f"✅ Server running at {base}")
            print(f"📋 Available models: {len(models)}")
            for m in models:
                print(f"   - {m['id']}")
            return True
        else:
            print(f"❌ Server returned {resp.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot reach server: {e}")
        print()
        print("Solution:")
        print("  1. Start VieNeu server:")
        print("     python -c \"from vieneu.server import run_server; run_server(port=23333)\"")
        print()
        print("  2. Or check if it's already running:")
        print("     ps aux | grep vieneu")
        return False

def test_vieneu_direct():
    """Test VieNeu SDK directly."""
    print_section("2. VieNeu SDK Direct Test")
    
    try:
        from vieneu import Vieneu
        
        print("Loading VieNeu Remote SDK...")
        tts = Vieneu(
            mode='remote',
            api_base='http://localhost:23333/v1',
            model_name='pnnbao-ump/VieNeu-TTS'
        )
        print("✅ SDK loaded")
        
        # List voices
        voices = tts.list_preset_voices()
        print(f"📋 Voices: {len(voices)}")
        for desc, name in voices[:3]:
            print(f"   - {desc} ({name})")
        
        # Test synthesis
        print("\n🎤 Testing synthesis...")
        audio = tts.infer(text="Xin chào Việt Nam")
        
        if len(audio) > 0:
            print(f"✅ Success! Audio: {len(audio)} samples")
            return True
        else:
            print("❌ No audio returned")
            print()
            print("DIAGNOSIS: Server is returning <|SPEECH_GENERATION_END|> tokens")
            print("instead of actual speech tokens. This is a SERVER issue.")
            print()
            print("Possible causes:")
            print("  1. Model not properly loaded on server")
            print("  2. Server needs restart")
            print("  3. Wrong model version")
            print()
            print("Solutions:")
            print("  1. Restart VieNeu server")
            print("  2. Check server logs for errors")
            print("  3. Verify model files are complete")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_asr():
    """Test ASR service."""
    print_section("3. ASR Service Check")
    
    try:
        from voice_assistant.core.asr import ASRService, get_asr_service
        
        print("Initializing ASR...")
        asr = get_asr_service()
        print(f"✅ ASR service: {type(asr).__name__}")
        print(f"   use_onnx: {asr.use_onnx}")
        print(f"   device: {asr.device}")
        
        # Test model loading
        print("\nLoading ASR model (this may take a moment)...")
        asr._ensure_loaded()
        
        if asr._model:
            print(f"✅ Model loaded: {type(asr._model).__name__}")
            return True
        else:
            print("❌ Model failed to load")
            return False
    except Exception as e:
        print(f"❌ ASR Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tts_service():
    """Test TTSService integration."""
    print_section("4. TTSService Integration Check")
    
    try:
        from voice_assistant.config import Settings
        from voice_assistant.core.tts import TTSService
        import asyncio
        
        # Load fresh settings
        settings = Settings.from_env()
        
        print("Configuration:")
        print(f"  TTS Backend: {settings.tts.backend}")
        print(f"  Remote API: {settings.tts.vieneu_remote_api_base}")
        print(f"  Remote Model: {settings.tts.vieneu_remote_model_id}")
        
        # Test with remote mode
        print("\nCreating TTSService (vieneu_remote)...")
        tts = TTSService(backend='vieneu_remote')
        print(f"✅ Service created")
        
        # Test synthesis
        print("\n🎤 Testing synthesis via TTSService...")
        async def test():
            audio = await tts.synthesize("Xin chào")
            return len(audio) if audio else 0
        
        result = asyncio.run(test())
        
        if result > 0:
            print(f"✅ Success! Audio: {result} bytes")
            return True
        else:
            print("⚠️ No audio (server issue - see test #2)")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline():
    """Test full pipeline structure."""
    print_section("5. Pipeline Structure Check")
    
    try:
        from voice_assistant.core.pipeline import get_orchestrator
        
        orch = get_orchestrator()
        
        print("Pipeline components:")
        print(f"  ✅ VAD: {orch.vad is not None}")
        print(f"  ✅ ASR: {orch.asr is not None}")
        print(f"  ✅ LLM: {orch.llm is not None}")
        print(f"  ✅ TTS: {orch.tts is not None}")
        print()
        print("✅ Pipeline structure intact")
        return True
    except Exception as e:
        print(f"❌ Pipeline Error: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("  VieNeu Remote TTS & ASR Diagnostic Tool")
    print("="*70)
    
    results = {
        "VieNeu Server": check_vieneu_server(),
        "VieNeu SDK Direct": test_vieneu_direct(),
        "ASR Service": test_asr(),
        "TTSService Integration": test_tts_service(),
        "Pipeline Structure": test_pipeline(),
    }
    
    print_section("Summary")
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {name}")
    
    print()
    
    # Provide specific fixes
    if not results["VieNeu Server"]:
        print("🔧 Fix #1: Start VieNeu Server")
        print("   python -c \"from vieneu.server import run_server; run_server(port=23333)\"")
        print()
    
    if not results["VieNeu SDK Direct"]:
        print("🔧 Fix #2: Fix VieNeu Server")
        print("   The server is returning <|SPEECH_GENERATION_END|> instead of speech tokens")
        print("   - Restart the server")
        print("   - Check model files are complete")
        print("   - Verify server logs")
        print()
    
    if not results["ASR Service"]:
        print("🔧 Fix #3: Fix ASR")
        print("   - Install sherpa-onnx: pip install sherpa-onnx")
        print("   - Or enable Whisper fallback")
        print()
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("="*70)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
