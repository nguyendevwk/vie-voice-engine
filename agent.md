# 🚀 High Performance ASR Pipeline - Technical Documentation

> **Tài liệu kỹ thuật chi tiết về version đạt performance cao nhất**
>
> Version: 1.0.0 | Date: 2026-04-07 | Branch: agent-copilot

---

## 📑 Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Audio Preprocessing Pipeline](#2-audio-preprocessing-pipeline)
3. [VAD Service - Silero VADIterator](#3-vad-service---silero-vaditerator)
4. [Voice Verification Layer](#4-voice-verification-layer)
5. [Streaming ASR với Interim Results](#5-streaming-asr-với-interim-results)
6. [Orchestrator - Pipeline Coordinator](#6-orchestrator---pipeline-coordinator)
7. [Interrupt Detection & Validation](#7-interrupt-detection--validation)
8. [Echo Cancellation & Mic Mute](#8-echo-cancellation--mic-mute)
9. [LLM & TTS Integration](#9-llm--tts-integration)
10. [WebSocket Protocol](#10-websocket-protocol)
11. [Performance Tuning Guide](#11-performance-tuning-guide)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Tổng quan kiến trúc

### 1.1 Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              AUDIO INPUT PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   WebSocket    Audio         VAD           Voice         Streaming              │
│   ─────────► Preprocessor ─────────► (Silero) ─────────► Verifier ─────────►   │
│   100ms PCM   HP/LP/Gate    VADIterator    Energy/ZCR    Buffer Audio          │
│   chunks      Normalize     start/end      Spectral                             │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                              PROCESSING PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Audio          Streaming        LLM            TTS           WebSocket        │
│   Buffer ─────► ASR ─────────► (Groq) ─────► (VieNeu) ─────► Audio Out         │
│   (Guards)     Qwen3-0.6B      Streaming     24k→16k          bytes            │
│                interim/final   chunks        PCM S16LE                          │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                              CONTROL SIGNALS                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   MIC_MUTE ◄──── Pipeline Start                                                 │
│   MIC_UNMUTE ◄── TTS Done + Buffer Wait                                         │
│   INTERRUPT ◄─── Interrupt Validator (VAD + Voice + LLM)                        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Components

| Component | File | Model/Library | Chức năng |
|-----------|------|---------------|-----------|
| Audio Preprocessor | `audio_preprocessor.py` | scipy | Lọc nhiễu, chuẩn hóa |
| VAD | `vad_service.py` | Silero VAD | Phát hiện speech start/end |
| Voice Verifier | `voice_verifier.py` | numpy/scipy | Xác nhận tiếng người |
| Streaming ASR | `streaming_asr.py` | Qwen3-ASR-0.6B | Real-time transcription |
| LLM | `llm_service.py` | Groq (Llama-3.3-70B) | Response generation |
| TTS | `tts_service.py` | VieNeu-TTS-v2-Turbo | Speech synthesis |
| Orchestrator | `orchestrator.py` | - | Pipeline coordination |

### 1.3 Cấu hình Audio cơ bản

```python
# config.py
SAMPLE_RATE: int = 16000           # 16kHz - chuẩn cho speech
AUDIO_CHANNELS: int = 1             # Mono
AUDIO_BIT_DEPTH: int = 16           # PCM S16LE (2 bytes/sample)
CHUNK_DURATION_MS: int = 100        # 100ms = 1600 samples/chunk
```

---

## 2. Audio Preprocessing Pipeline

### 2.1 Tầm quan trọng

Audio preprocessing là bước **CRITICAL** quyết định accuracy của toàn bộ pipeline. Mic chất lượng thấp, môi trường ồn, DC offset đều ảnh hưởng nghiêm trọng đến VAD và ASR.

### 2.2 Pipeline xử lý

```
Raw PCM S16LE
      │
      ▼
┌─────────────────┐
│  Decode PCM16   │  int16 → float32 [-1.0, 1.0]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ DC Offset Remove│  audio -= mean(audio)
└────────┬────────┘  → Loại bỏ lệch trung tâm sóng
         │
         ▼
┌─────────────────┐
│ High-Pass Filter│  Butterworth N=5, fc=80Hz
└────────┬────────┘  → Loại rung bàn, gió, hơi thở
         │
         ▼
┌─────────────────┐
│ Low-Pass Filter │  Butterworth N=5, fc=7600Hz
└────────┬────────┘  → Loại nhiễu cao tần, alias
         │
         ▼
┌─────────────────┐
│   Noise Gate    │  if RMS < -40dB → zero out
└────────┬────────┘  → Triệt tiêu nền tĩnh
         │
         ▼
┌─────────────────┐
│ Peak Normalize  │  audio /= max(abs(audio))
└────────┬────────┘  → Chuẩn hóa biên độ
         │
         ▼
   Clean Audio
```

### 2.3 Tham số cấu hình

```python
# config.py
PREPROCESS_HIGH_PASS_HZ: int = 80       # Cắt < 80Hz
PREPROCESS_LOW_PASS_HZ: int = 7600      # Cắt > 7600Hz (< Nyquist 8kHz)
PREPROCESS_NORMALIZE: bool = True       # Bật Peak Normalization
PREPROCESS_NOISE_GATE_DB: float = -40.0 # Ngưỡng gate
```

### 2.4 Implementation chi tiết

```python
# audio_preprocessor.py

class AudioPreprocessor:
    def __init__(self):
        # Pre-compute Butterworth filters (order=5 cho độ dốc sạch)
        self._hp_sos = butter(N=5, Wn=80, btype="highpass", fs=16000, output="sos")
        self._lp_sos = butter(N=5, Wn=7600, btype="lowpass", fs=16000, output="sos")

        # Noise gate threshold: -40dB → linear
        self._noise_gate_linear = 10.0 ** (-40.0 / 20.0)  # ≈ 0.01

    def process_to_float(self, raw: bytes) -> np.ndarray:
        audio = self.decode_pcm16(raw)           # int16 → float32
        audio = self.remove_dc_offset(audio)     # audio - mean
        audio = self.apply_highpass(audio)       # sosfilt HP
        audio = self.apply_lowpass(audio)        # sosfilt LP
        audio = self.apply_noise_gate(audio)     # RMS check
        audio = self.peak_normalize(audio)       # /= max
        return audio

    def apply_noise_gate(self, audio: np.ndarray) -> np.ndarray:
        """Gate đóng nếu RMS < threshold"""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < self._noise_gate_linear:
            return np.zeros_like(audio)
        return audio
```

---

## 3. VAD Service - Silero VADIterator

### 3.1 Tại sao dùng VADIterator?

| Method | Use Case | Latency | Memory |
|--------|----------|---------|--------|
| `get_speech_timestamps()` | Batch processing | High (cần toàn bộ audio) | High |
| **`VADIterator`** | **Real-time streaming** | **Low (32ms/chunk)** | **Low** |

**VADIterator** được thiết kế cho streaming - xử lý từng chunk nhỏ và emit events real-time.

### 3.2 Khởi tạo VADIterator

```python
# vad_service.py

class VADService:
    def __init__(self):
        # Load Silero VAD model
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )

        # Lấy VADIterator class từ utils
        _, _, _, VADIterator_Class, _ = utils

        # Khởi tạo iterator với tham số tối ưu
        self.vad_iterator = VADIterator_Class(
            self.model,
            threshold=0.55,              # Speech probability threshold
            sampling_rate=16000,
            min_silence_duration_ms=500, # 500ms silence → end event
            speech_pad_ms=30,            # Pad 30ms để không mất chữ đầu/cuối
        )

        self.chunk_size = 512  # Optimal: 512 samples = 32ms @ 16kHz
```

### 3.3 Tham số quan trọng

| Parameter | Value | Ý nghĩa |
|-----------|-------|---------|
| `threshold` | **0.55** | Ngưỡng speech probability. Cao hơn = ít false positive, có thể miss speech nhẹ |
| `min_silence_duration_ms` | **500** | Sau 500ms im lặng → emit 'end' event |
| `speech_pad_ms` | **30** | Mở rộng 30ms ở đầu/cuối segment để không mất chữ |
| `chunk_size` | **512** | Silero optimal: 512 samples = 32ms |

### 3.4 Cơ chế xử lý chunk

```python
def process_chunk(self, audio_data: bytes) -> dict:
    """
    Input: 100ms chunk (1600 samples)
    Output: {'event': 'start'|'end'|None, 'is_active': bool}
    """
    # 1. Tiền xử lý
    cleaned = self.preprocessor.process_to_float(audio_data)

    # 2. Chia thành sub-chunks 512 samples (100ms → ~3 sub-chunks)
    event_type = None
    for i in range(0, len(cleaned), 512):
        chunk = cleaned[i:i + 512]
        if len(chunk) < 512:
            chunk = np.pad(chunk, (0, 512 - len(chunk)))

        # 3. Gọi VADIterator
        audio_tensor = torch.from_numpy(chunk)
        speech_dict = self.vad_iterator(audio_tensor, return_seconds=True)

        # 4. Parse events
        if speech_dict:
            if 'start' in speech_dict:
                self.is_speech_active = True
                event_type = 'start'
            elif 'end' in speech_dict:
                self.is_speech_active = False
                event_type = 'end'

    return {
        "event": event_type,
        "is_active": self.is_speech_active,
    }
```

### 3.5 State Machine

```
                    ┌──────────────┐
                    │    IDLE      │
                    │ (is_active   │
         ┌──────────│   = false)   │◄─────────┐
         │          └──────┬───────┘          │
         │                 │                  │
         │    speech_dict  │ 'start'          │ speech_dict
         │    contains     │                  │ contains
         │                 ▼                  │ 'end'
         │          ┌──────────────┐          │
         │          │   SPEAKING   │          │
         └──────────│ (is_active   │──────────┘
      no event      │   = true)    │
                    └──────────────┘
```

---

## 4. Voice Verification Layer

### 4.1 Tại sao cần Voice Verifier?

VAD chỉ phát hiện "có âm thanh giống speech" - không phân biệt được:

- Tiếng người thật vs tiếng TV/radio
- Giọng nói vs tiếng gõ bàn phím
- Speech vs một số loại nhiễu có pattern tương tự

**Voice Verifier** là lớp lọc thứ 2 để confirm đây là tiếng NGƯỜI.

### 4.2 4 đặc trưng phân tích

```
┌─────────────────────────────────────────────────────────────────────┐
│                     VOICE VERIFICATION FEATURES                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────┐ │
│  │   Energy    │   │     ZCR     │   │  Spectral   │   │Spectral │ │
│  │   (RMS dB)  │   │             │   │  Centroid   │   │Flatness │ │
│  ├─────────────┤   ├─────────────┤   ├─────────────┤   ├─────────┤ │
│  │ Weight: 0.3 │   │ Weight: 0.2 │   │ Weight: 0.3 │   │Wght: 0.2│ │
│  │             │   │             │   │             │   │         │ │
│  │ Human: >-35 │   │ Human: <0.15│   │ Human:      │   │Human:   │ │
│  │ Noise: <-35 │   │ Noise: >0.15│   │ 200-4000Hz  │   │< 0.3    │ │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────┘ │
│                                                                      │
│                    Total Score >= 0.5 → IS_HUMAN_VOICE              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.3 Chi tiết từng feature

#### 4.3.1 Energy (RMS in dB)

```python
rms = np.sqrt(np.mean(audio ** 2))
energy_db = 20 * np.log10(max(rms, 1e-10))

# Tiếng người có energy đủ mạnh
if energy_db > -35.0:
    score += 0.3
```

#### 4.3.2 Zero Crossing Rate (ZCR)

```python
signs = np.sign(audio)
zcr = np.mean(np.abs(np.diff(signs)) > 0)

# Nhiễu trắng có ZCR cao, tiếng người ZCR thấp hơn
if zcr < 0.15:
    score += 0.2
```

#### 4.3.3 Spectral Centroid

```python
fft = np.fft.rfft(audio)
magnitude = np.abs(fft)
freqs = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)

spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)

# Tiếng người tập trung 200-4000Hz
if 200 < spectral_centroid < 4000:
    score += 0.3
```

#### 4.3.4 Spectral Flatness

```python
log_mag = np.log(magnitude + 1e-10)
geometric_mean = np.exp(np.mean(log_mag))
arithmetic_mean = np.mean(magnitude)
spectral_flatness = geometric_mean / arithmetic_mean

# Tiếng người có tonal structure (không phẳng như noise)
if spectral_flatness < 0.3:
    score += 0.2
```

### 4.4 Smoothing với History

```python
# Majority voting từ 5 frames gần nhất
self._history.append(is_voice)
if len(self._history) > 5:
    self._history.pop(0)

smoothed_voice = sum(self._history) > len(self._history) / 2
```

### 4.5 Tham số cấu hình

```python
# config.py
VOICE_ENERGY_THRESHOLD_DB: float = -35.0    # Minimum energy
VOICE_ZCR_MAX: float = 0.15                 # Maximum ZCR
VOICE_SPECTRAL_CENTROID_MIN: int = 200      # Hz
VOICE_SPECTRAL_CENTROID_MAX: int = 4000     # Hz
```

---

## 5. Streaming ASR với Interim Results

### 5.1 Concept: Google/Zalo Style Streaming

Thay vì chờ user nói xong mới transcribe, **Streaming ASR** cho phép:

- Hiển thị text **real-time** khi user đang nói
- User thấy feedback ngay → biết hệ thống đang "nghe"
- Text có thể **tự sửa** khi có thêm context

### 5.2 Tham số cấu hình

```python
# orchestrator.py
self.streaming_asr = StreamingASRService(
    asr_service=self.asr,
    on_transcript_update=self._on_transcript_update,
    interim_interval_ms=800,       # Chạy ASR mỗi 800ms
    min_audio_for_interim_ms=600,  # Cần >= 600ms audio mới chạy
)
```

| Parameter | Value | Ý nghĩa |
|-----------|-------|---------|
| `interim_interval_ms` | **800** | Interval giữa các lần chạy interim ASR |
| `min_audio_for_interim_ms` | **600** | Không chạy ASR nếu buffer < 600ms |

### 5.3 State Machine

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STREAMING ASR STATE MACHINE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   VAD 'start'          add_audio()              VAD 'end'           │
│       │                    │                        │                │
│       ▼                    ▼                        ▼                │
│  ┌─────────┐         ┌─────────┐              ┌─────────┐           │
│  │  START  │────────►│ ACTIVE  │─────────────►│  END    │           │
│  │UTTERANCE│         │         │              │UTTERANCE│           │
│  └─────────┘         └────┬────┘              └────┬────┘           │
│       │                   │                        │                 │
│       │                   │ every 800ms            │                 │
│       │                   ▼                        │                 │
│       │            ┌─────────────┐                 │                 │
│       │            │   INTERIM   │                 │                 │
│       │            │    ASR      │                 │                 │
│       │            └──────┬──────┘                 │                 │
│       │                   │                        │                 │
│       │                   ▼                        ▼                 │
│       │          emit INTERIM result       emit FINAL result        │
│       │          (is_final=False)          (is_final=True)          │
│       │                                                              │
│       └──────────── Clear buffer, Start interim_loop ───────────────┘
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.4 Implementation Flow

```python
# streaming_asr.py

class StreamingASRService:
    async def start_utterance(self):
        """Gọi khi VAD trigger 'start'"""
        self._audio_buffer.clear()
        self._is_active = True
        self._last_interim_text = ""
        # Start background interim loop
        self._interim_task = asyncio.create_task(self._interim_loop())

    def add_audio(self, chunk: bytes):
        """Gọi liên tục khi có audio"""
        if self._is_active:
            self._audio_buffer.append(chunk)

    async def _interim_loop(self):
        """Background task chạy ASR định kỳ"""
        while self._is_active:
            await asyncio.sleep(self.interim_interval_ms / 1000)  # 800ms

            # Guard: đủ audio chưa?
            duration_ms = len(self._audio_buffer) * CHUNK_DURATION_MS
            if duration_ms < self.min_audio_for_interim_ms:  # 600ms
                continue

            # Run ASR
            interim_text = await self.asr.transcribe(list(self._audio_buffer))

            # Emit nếu text thay đổi
            if interim_text != self._last_interim_text:
                self._last_interim_text = interim_text
                await self.on_transcript_update(TranscriptUpdate(
                    text=interim_text,
                    is_final=False,
                    confidence=0.8,
                ))

    async def end_utterance(self) -> TranscriptUpdate:
        """Gọi khi VAD trigger 'end'"""
        self._is_active = False
        self._interim_task.cancel()

        # Final ASR
        final_text = await self.asr.transcribe(list(self._audio_buffer))

        return TranscriptUpdate(
            text=final_text,
            is_final=True,
            confidence=1.0,
        )
```

### 5.5 ASR Model: Qwen3-ASR-0.6B

```python
# asr_service.py
from qwen_asr import Qwen3ASRModel

self.model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    dtype=torch.bfloat16,           # Tối ưu VRAM (~1.2GB)
    device_map="cuda:0",
    max_inference_batch_size=8,
    max_new_tokens=256,
)

# Transcribe
results = self.model.transcribe(
    audio=temp_wav_path,
    language="Vietnamese",
)
```

---

## 6. Orchestrator - Pipeline Coordinator

### 6.1 Vai trò

**Orchestrator** là "bộ não" điều phối toàn bộ pipeline:

- Nhận audio chunks từ WebSocket
- Điều phối VAD → Voice Verify → Streaming ASR
- Quản lý state (recording, bot_speaking, mic_muted)
- Xử lý interrupt
- Trigger LLM → TTS pipeline

### 6.2 Thresholds quan trọng

```python
# orchestrator.py

class ConversationOrchestrator:
    def __init__(self, vad, asr, llm, tts):
        # Silence threshold: bao nhiêu chunks im lặng → chốt câu
        self.silence_threshold = max(2, 1200 // 100)  # 12 chunks = 1.2s

        # Interrupt threshold: bao nhiêu chunks speech liên tục → interrupt
        self.interrupt_threshold = max(4, 600 // 100)  # 6 chunks = 600ms

        # Max buffer: giới hạn 15s
        self.max_buffer_chunks = 15000 // 100  # 150 chunks

        # Minimum utterance duration
        self._min_utterance_duration_ms = 500  # 500ms

        # Minimum verified voice chunks
        self._min_verified_chunks = 5  # 5 chunks

        # Post-TTS buffer before unmute
        self._post_tts_buffer_ms = 500  # 500ms
```

### 6.3 handle_audio_in() Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                       handle_audio_in(chunk)                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────┐                                                   │
│   │ Mic Muted?  │───Yes───► RETURN (skip chunk)                     │
│   └──────┬──────┘                                                   │
│          │No                                                         │
│          ▼                                                           │
│   ┌─────────────┐                                                   │
│   │ VAD Process │ → event ('start'/'end'/None), is_active           │
│   └──────┬──────┘                                                   │
│          │                                                           │
│          ├── event='start' ──► Start StreamingASR, is_recording=True │
│          │                                                           │
│          ▼                                                           │
│   ┌─────────────┐                                                   │
│   │Voice Verify │ (if VAD is_speech)                                │
│   └──────┬──────┘                                                   │
│          │                                                           │
│          ▼                                                           │
│   confirmed_speech = is_speech AND is_human_voice                   │
│          │                                                           │
│          ├── Bot Speaking + Confirmed Speech ──► Interrupt Check    │
│          │                                                           │
│          ├── Recording + Not Bot Speaking ──► Buffer Audio          │
│          │                                         │                 │
│          │                                         ▼                 │
│          │                              Add to StreamingASR         │
│          │                                                           │
│          └── event='end' + buffer > 0 ──► Pipeline Guards           │
│                                                │                     │
│                                                ▼                     │
│                                    ┌─────────────────────┐          │
│                                    │ GUARD 1: Pipeline   │          │
│                                    │ already running?    │──Yes──►Skip│
│                                    └──────────┬──────────┘          │
│                                               │No                    │
│                                               ▼                     │
│                                    ┌─────────────────────┐          │
│                                    │ GUARD 2: Duration   │          │
│                                    │ < 500ms?            │──Yes──►Skip│
│                                    └──────────┬──────────┘          │
│                                               │No                    │
│                                               ▼                     │
│                                    ┌─────────────────────┐          │
│                                    │ GUARD 3: Verified   │          │
│                                    │ chunks < 5?         │──Yes──►Skip│
│                                    └──────────┬──────────┘          │
│                                               │No                    │
│                                               ▼                     │
│                                    ┌─────────────────────┐          │
│                                    │ GUARD 4: Transcript │          │
│                                    │ empty/too short?    │──Yes──►Skip│
│                                    └──────────┬──────────┘          │
│                                               │No                    │
│                                               ▼                     │
│                                    RUN PIPELINE (LLM → TTS)         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.4 4 Guard Conditions

```python
# Guard 1: Pipeline đang chạy
if self._is_pipeline_running():
    logger.warning("Skip: Pipeline đang chạy")
    await self.streaming_asr.cancel_utterance()
    return

# Guard 2: Duration quá ngắn
duration_ms = len(self.audio_buffer) * CHUNK_DURATION_MS
if duration_ms < 500:
    logger.warning(f"Skip: Quá ngắn ({duration_ms}ms < 500ms)")
    await self.streaming_asr.cancel_utterance()
    return

# Guard 3: Không đủ verified chunks
if self._verified_speech_count < 5:
    logger.warning(f"Skip: Chỉ có {self._verified_speech_count}/5 verified")
    await self.streaming_asr.cancel_utterance()
    return

# Guard 4: Transcript rỗng
final_result = await self.streaming_asr.end_utterance()
if not final_result or len(final_result.text.strip()) < 2:
    logger.warning("Skip: Empty transcript")
    return
```

---

## 7. Interrupt Detection & Validation

### 7.1 Interrupt Flow

```
User nói chen ────► VAD detect ────► Voice Verify ────► Count chunks
                                                              │
                                                              ▼w
                                                    count >= 6 chunks?
                                                              │
                                              ┌───────────────┴───────────────┐
                                              │No                             │Yes
                                              ▼                               ▼
                                        Continue counting            Quick ASR on buffer
                                                                              │
                                                                              ▼
                                                                    LLM Validate
                                                                              │
                                                    ┌─────────────────────────┴────────────┐
                                                    │ Meaningful                           │ Noise
                                                    ▼                                      ▼
                                            INTERRUPT!                              Reject, reset
                                            - Cancel pipeline
                                            - Clear buffer
                                            - Send INTERRUPT signal
```

### 7.2 Interrupt Validator

```python
# interrupt_validator.py

class InterruptValidator:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = "llama-3.1-8b-instant"  # Fast model

    async def is_meaningful(self, text: str) -> tuple[bool, str]:
        # Quick heuristics TRƯỚC khi gọi LLM

        # 1. Noise patterns → reject
        noise_patterns = ["ừ", "ờ", "à", "uh", "um", "hm", "..."]
        if text.lower() in noise_patterns:
            return False, "Noise pattern"

        # 2. Vietnamese words → accept
        vietnamese_words = ["tôi", "bạn", "gì", "là", "có", "không", ...]
        if len(text) > 10 and any(w in text.lower() for w in vietnamese_words):
            return True, "Contains Vietnamese"

        # 3. LLM validate (timeout 2s)
        response = await self._call_llm(text)
        return response == "YES", "LLM decision"
```

---

## 8. Echo Cancellation & Mic Mute

### 8.1 Vấn đề Echo

Khi bot phát audio TTS qua loa, mic sẽ thu lại → VAD trigger → tạo loop vô hạn.

### 8.2 Giải pháp: Mic Mute với Timing chính xác

```
Pipeline Start ──────► MIC_MUTE
                          │
                          ▼
                    TTS Streaming
                    (tính total duration)
                          │
                          ▼
                    TTS Done
                          │
                          ▼
            Wait (total_duration + 500ms buffer)
                          │
                          ▼
                    MIC_UNMUTE
```

### 8.3 Implementation

```python
# orchestrator.py

async def _run_pipeline_with_text(self, text_query, out_queue):
    # 1. Mute mic ngay
    self._mic_muted = True
    await out_queue.put("MIC_MUTE")

    # 2. LLM → TTS streaming
    total_tts_duration_ms = 0

    async for sentence in self.llm.generate_response_stream(text_query):
        audio = await self.tts.synthesize(sentence)
        if audio:
            # Tính duration: bytes / 2 (16-bit) / sample_rate * 1000
            audio_duration_ms = len(audio) / 2 / 16000 * 1000
            total_tts_duration_ms += audio_duration_ms
            await out_queue.put(audio)

    # 3. Wait cho TTS phát xong
    wait_time = (total_tts_duration_ms / 1000) + (500 / 1000)  # + 500ms buffer
    elapsed = time.perf_counter() - t_first_audio
    remaining = max(0, wait_time - elapsed)
    await asyncio.sleep(remaining)

    # 4. Unmute mic
    self._mic_muted = False
    await out_queue.put("MIC_UNMUTE")
```

### 8.4 Tham số AEC (Echo Cancellation)

```python
# config.py
AEC_ENABLED: bool = True
AEC_REFERENCE_BUFFER_MS: int = 3000         # Buffer 3s TTS audio
AEC_POST_TTS_DELAY_MS: int = 2000           # Delay 2s sau TTS
AEC_ECHO_CORRELATION_THRESHOLD: float = 0.3 # Ngưỡng phát hiện echo
AEC_ECHO_SUPPRESSION_LEVEL: float = 0.05    # Suppression level (gần 0 = gần mute)
```

---

## 9. LLM & TTS Integration

### 9.1 LLM Service (Groq)

```python
# llm_service.py

class LLMService:
    SENTENCE_DELIMITERS = {",", ".", "?", "!", ";", "\n"}

    def __init__(self):
        self.client = AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY,
        )
        self.model = "llama-3.3-70b-versatile"

    async def generate_response_stream(self, prompt: str):
        """Yield từng cụm câu hoàn chỉnh"""
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Bạn là nhân viên tổng đài..."},
                {"role": "user", "content": prompt},
            ],
            stream=True,
            max_tokens=256,
        )

        buffer = ""
        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                buffer += token
                # Yield khi gặp delimiter
                if any(d in token for d in self.SENTENCE_DELIMITERS):
                    yield buffer.strip()
                    buffer = ""
        if buffer.strip():
            yield buffer.strip()
```

### 9.2 TTS Service (VieNeu)

```python
# tts_service.py

class TTSService:
    def __init__(self):
        from vieneu import Vieneu
        self.tts = Vieneu()
        self.output_sample_rate = 24000  # VieNeu output 24kHz
        self.target_sample_rate = 16000  # Pipeline 16kHz

    async def synthesize(self, text: str) -> bytes:
        # Run in thread to not block
        audio = await asyncio.to_thread(self._infer_sync, text)
        return audio

    def _infer_sync(self, text: str) -> bytes:
        # 1. TTS inference
        audio_24k = self.tts.infer(text=text, voice=self.default_voice)

        # 2. Resample 24kHz → 16kHz
        audio_16k = signal.resample(audio_24k, len(audio_24k) * 16000 // 24000)

        # 3. Convert float32 → PCM S16LE
        pcm = (np.clip(audio_16k, -1.0, 1.0) * 32767).astype(np.int16)

        return pcm.tobytes()
```

---

## 10. WebSocket Protocol

### 10.1 Endpoint

```
ws://server:port/agent
```

### 10.2 Client → Server

| Type | Format | Description |
|------|--------|-------------|
| Audio | `bytes` | PCM S16LE, mono, 16kHz, 100ms chunks |

### 10.3 Server → Client

| Type | Format | Description |
|------|--------|-------------|
| Audio | `bytes` | TTS audio PCM S16LE, mono, 16kHz |
| Interrupt | `text: "INTERRUPT"` | Bot bị ngắt lời, clear audio buffer |
| Mic Mute | `text: "MIC_MUTE"` | Mute microphone |
| Mic Unmute | `text: "MIC_UNMUTE"` | Unmute microphone |
| Transcript | `text: JSON` | Real-time transcript update |

### 10.4 Transcript JSON Format

```json
{
  "type": "transcript",
  "text": "xin chào tôi muốn hỏi",
  "is_final": false,
  "confidence": 0.8
}
```

---

## 11. Performance Tuning Guide

### 11.1 Nếu bị MISS SPEECH (không nhận ra tiếng nói)

| Parameter | Action | File |
|-----------|--------|------|
| `VAD_THRESHOLD` | Giảm < 0.55 | config.py |
| `VOICE_ENERGY_THRESHOLD_DB` | Giảm < -35 | config.py |
| `_min_verified_chunks` | Giảm < 5 | orchestrator.py |
| `_min_utterance_duration_ms` | Giảm < 500 | orchestrator.py |

### 11.2 Nếu bị FALSE POSITIVE (noise trigger)

| Parameter | Action | File |
|-----------|--------|------|
| `VAD_THRESHOLD` | Tăng > 0.55 | config.py |
| `VOICE_ENERGY_THRESHOLD_DB` | Tăng > -35 | config.py |
| `_min_verified_chunks` | Tăng > 5 | orchestrator.py |
| `_min_utterance_duration_ms` | Tăng > 500 | orchestrator.py |
| `PREPROCESS_NOISE_GATE_DB` | Tăng > -40 | config.py |

### 11.3 Nếu bị ECHO (bot nghe lại tiếng mình)

| Parameter | Action | File |
|-----------|--------|------|
| `AEC_POST_TTS_DELAY_MS` | Tăng > 2000 | config.py |
| `_post_tts_buffer_ms` | Tăng > 500 | orchestrator.py |
| `AEC_ECHO_SUPPRESSION_LEVEL` | Giảm gần 0 | config.py |

### 11.4 Nếu RESPONSE CHẬM

| Parameter | Action | File |
|-----------|--------|------|
| `interim_interval_ms` | Giảm < 800 | orchestrator.py |
| `min_audio_for_interim_ms` | Giảm < 600 | orchestrator.py |

---

## 12. Troubleshooting

### 12.1 "VAD không trigger start/end"

**Nguyên nhân có thể:**

- Audio quá nhỏ/bị noise gate
- Threshold quá cao

**Debug:**

```python
# Bật debug log
DEBUG_LOG_ALL_CHUNKS: bool = True

# Check preprocessing output
audio = preprocessor.process_to_float(chunk)
print(f"RMS: {np.sqrt(np.mean(audio**2))}")
```

### 12.2 "StreamingASR không emit interim"

**Nguyên nhân có thể:**

- Buffer chưa đủ `min_audio_for_interim_ms`
- `_inference_running` đang True

**Debug:**

```python
print(f"Buffer: {streaming_asr.get_buffer_duration_ms()}ms")
print(f"Is active: {streaming_asr.is_active()}")
```

### 12.3 "Bot tự nghe tiếng mình"

**Nguyên nhân:**

- Mic unmute quá sớm
- AEC không hoạt động

**Fix:**

```python
# Tăng buffer
_post_tts_buffer_ms = 1000  # 1s

# Hoặc tăng AEC delay
AEC_POST_TTS_DELAY_MS = 3000  # 3s
```

### 12.4 "Interrupt không hoạt động"

**Debug:**

```python
# Check interrupt count
print(f"Interrupt count: {self._interrupt_speech_count}")
print(f"Threshold: {self.interrupt_threshold}")

# Check interrupt validator
result, reason = await validator.is_meaningful(text)
print(f"Meaningful: {result}, Reason: {reason}")
```

---

## 📎 Appendix: File Reference

| File | Lines | Key Classes/Functions |
|------|-------|----------------------|
| `config.py` | ~73 | `Settings` |
| `vad_service.py` | ~124 | `VADService.process_chunk()` |
| `voice_verifier.py` | ~137 | `VoiceVerifier.verify()` |
| `streaming_asr.py` | ~242 | `StreamingASRService` |
| `asr_service.py` | ~121 | `ASRService.transcribe()` |
| `orchestrator.py` | ~648 | `ConversationOrchestrator.handle_audio_in()` |
| `llm_service.py` | ~173 | `LLMService.generate_response_stream()` |
| `tts_service.py` | ~124 | `TTSService.synthesize()` |
| `interrupt_validator.py` | ~145 | `InterruptValidator.is_meaningful()` |
| `echo_cancellation.py` | ~425 | `EchoCancellationService`, `ASRTTSCoordinator` |
| `audio_preprocessor.py` | ~143 | `AudioPreprocessor.process_to_float()` |
| `ws_routes.py` | ~77 | `websocket_endpoint()` |

---

> **Note**: Tài liệu này ghi lại version có performance cao nhất. Khi modify code, hãy tham khảo các tham số và logic ở đây để đảm bảo không làm giảm performance.

