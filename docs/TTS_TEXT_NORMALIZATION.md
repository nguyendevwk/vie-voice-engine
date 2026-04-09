# TTS Text Normalization Guide

## Overview

The voice assistant now includes **comprehensive text normalization** optimized for **VieNeu-TTS-v2 Turbo** to:
- ✅ Clean LLM output (remove markdown, code, URLs, emojis, etc.)
- ✅ Ensure TTS-compatible text format
- ✅ Optimize chunk sizes for low latency
- ✅ Handle Vietnamese text properly
- ✅ Expand abbreviations for natural speech

## Architecture

```
LLM Output
    ↓
normalize_for_tts() ─── Clean & normalize
    ↓
split_for_tts() ─────── Split into optimal chunks
    ↓
TTSService.synthesize() ── Generate audio
    ↓
Audio Output
```

## Features

### 1. LLM Output Cleaning

**Removes:**
- ✅ Markdown formatting (`**bold**`, `*italic*`, `# headers`, etc.)
- ✅ Code blocks (```code``` and `inline code`)
- ✅ URLs (https://example.com)
- ✅ Email addresses (user@example.com)
- ✅ Emojis (👋🌍😊)
- ✅ Cyrillic characters (Russian text)
- ✅ Thinking blocks (<think>...</think>)
- ✅ List markers (- item, 1. item)

**Expands:**
- Vietnamese abbreviations (TP → thành phố, HN → Hà Nội, etc.)
- Titles (Dr → Tiến sĩ, Mr → Ông, etc.)
- Numbers and percentages (100% → 100 phần trăm)

### 2. Text Normalization

**Handles:**
- Whitespace normalization (multiple spaces → single space)
- Punctuation cleanup (!!! → !, ,,, → ,)
- Sentence boundary detection
- Minimum length enforcement (≥10 chars)
- Maximum chunk size (≤200 chars for low latency)

### 3. Sentence Splitting

**Optimized for:**
- VieNeu-TTS-v2 Turbo requirements
- Minimum 10 characters per chunk
- Maximum 200 characters per chunk
- Sentence boundary preference
- Merging short sentences
- Splitting long sentences at commas

## Configuration

### Default Configuration (VieNeu-TTS-v2 Turbo)

```python
from voice_assistant.utils.text_utils import VIENEU_V2_TURBO_CONFIG

print(VIENEU_V2_TURBO_CONFIG.min_length)  # 10 chars
print(VIENEU_V2_TURBO_CONFIG.max_length)  # 200 chars
```

### Custom Configuration

```python
from voice_assistant.utils.text_utils import (
    normalize_for_tts,
    split_for_tts,
    TTSNormalizationConfig
)

# Create custom config
custom_config = TTSNormalizationConfig(
    min_length=15,  # Minimum 15 chars
    max_length=150,  # Maximum 150 chars
    remove_emojis=True,
    vietnamese_abbreviations=True,
)

# Normalize text
text = normalize_for_tts("Your text here", config=custom_config)

# Split into chunks
chunks = split_for_tts("Your text here", config=custom_config)
```

## Usage Examples

### Basic Usage

```python
from voice_assistant.utils.text_utils import normalize_for_tts, split_for_tts

# LLM output with markdown and special chars
llm_output = """
## Trả lời

**Đây là câu trả lời:**

- Điểm 1: Thông tin quan trọng
- Điểm 2: Chi tiết bổ sung

Xem thêm: https://example.com
"""

# Normalize
clean_text = normalize_for_tts(llm_output)
print(clean_text)
# Output: "Trả lời Đây là câu trả lời: Điểm 1: Thông tin quan trọng Điểm 2: Chi tiết bổ sung."

# Split into chunks
chunks = split_for_tts(llm_output)
print(chunks)
# Output: ["Trả lời Đây là câu trả lời: Điểm 1: Thông tin quan trọng Điểm 2: Chi tiết bổ sung."]
```

### Vietnamese Abbreviations

```python
text = "TP HCM có trụ sở UBND tại Q.1"
normalized = normalize_for_tts(text)
print(normalized)
# Output: "thành phố Hồ Chí Minh có trụ sở Ủy ban Nhân dân tại Quận 1."
```

### Numbers and Percentages

```python
text = "Tăng 50% lên 1,000,000 VND"
normalized = normalize_for_tts(text)
print(normalized)
# Output: "Tăng 50 phần trăm lên 1000000 đồng."
```

### Emoji and Special Character Removal

```python
text = "Xin chào! 👋 Thời tiết đẹp 🌞"
normalized = normalize_for_tts(text)
print(normalized)
# Output: "Xin chào! Thời tiết đẹp."
```

## Integration with TTS

The normalization is **automatically applied** in the TTS pipeline:

### In TTSService

```python
from voice_assistant.core.tts import TTSService

tts = TTSService()

# Text is automatically normalized before synthesis
audio = await tts.synthesize("**Bold text** with emojis 🎉")
# Internally calls normalize_for_tts() first
```

### In Pipeline

```python
# Pipeline automatically normalizes LLM output
async for sentence in llm.generate_response_stream(...):
    # sentence is normalized before TTS
    await tts.synthesize(sentence)
```

## Performance Impact

### Latency Improvement

| Scenario | Before | After | Improvement |
|----------|--------|-------|-------------|
| Short text (< 10 chars) | TTS error | Skipped | ✅ No errors |
| Markdown text | TTS garble | Clean speech | ✅ Clear audio |
| Long text (> 200 chars) | High latency | Split chunks | ✅ Lower latency |
| Emojis/Special | TTS silence | Removed | ✅ No dead air |

### Chunk Size Distribution

With default config (min=10, max=200):
- **Average chunk size**: ~50-100 characters
- **Optimal for**: VieNeu-TTS-v2 Turbo (< 500ms latency)
- **Chunks per response**: 2-5 (depending on response length)

## Troubleshooting

### TTS Still Failing on Some Text

**Problem**: TTS fails on certain chunks

**Solutions**:
1. Check minimum length (should be ≥10 chars)
2. Ensure text has proper punctuation
3. Remove any remaining special characters

```python
# Debug normalization
text = "Your problematic text here"
normalized = normalize_for_tts(text)
print(f"Original: {text}")
print(f"Normalized: {normalized}")
print(f"Length: {len(normalized)}")
```

### Chunks Too Long or Too Short

**Problem**: Latency too high or TTS failing on short chunks

**Solution**: Adjust chunk size configuration

```python
# For lower latency (smaller chunks)
config = TTSNormalizationConfig(
    min_length=10,
    max_length=100,  # Reduce from 200
)

# For fewer API calls (larger chunks)
config = TTSNormalizationConfig(
    min_length=15,  # Increase from 10
    max_length=300,  # Increase from 200
)
```

### Vietnamese Abbreviations Not Expanding

**Problem**: Abbreviations like "TP" not expanding

**Solution**: Ensure `vietnamese_abbreviations=True` in config

```python
config = TTSNormalizationConfig(
    vietnamese_abbreviations=True,  # Default is True
)
```

## Vietnamese Abbreviation Reference

| Abbreviation | Expansion |
|--------------|-----------|
| TP, Tp, tp | thành phố |
| HCM | Hồ Chí Minh |
| HN | Hà Nội |
| VN | Việt Nam |
| Dr, TS | Tiến sĩ |
| ThS | Thạc sĩ |
| CN | Cử nhân |
| Mr | Ông |
| Mrs | Bà |
| Ms | Cô |
| PGS | Phó Giáo sư |
| GS | Giáo sư |
| TSKS | Tiến sĩ Khoa học |
| UBND | Ủy ban Nhân dân |
| QĐ | Quyết định |
| NĐ | Nghị định |
| TT | Thông tư |
| BCA | Bộ Công an |
| BQP | Bộ Quốc phòng |
| BYT | Bộ Y tế |

## API Reference

### `normalize_for_tts(text, config=None)`

Normalize text for TTS synthesis.

**Parameters:**
- `text` (str): Input text (can contain markdown, emojis, URLs, etc.)
- `config` (TTSNormalizationConfig, optional): Normalization settings

**Returns:**
- `str`: Clean, normalized text ready for TTS

### `split_for_tts(text, config=None)`

Split text into optimal chunks for TTS.

**Parameters:**
- `text` (str): Input text
- `config` (TTSNormalizationConfig, optional): Chunking settings

**Returns:**
- `List[str]`: List of text chunks, each optimized for TTS

### `TTSNormalizationConfig`

Configuration dataclass for text normalization.

**Fields:**
- `min_length`: Minimum chunk length (default: 10)
- `max_length`: Maximum chunk length (default: 200)
- `remove_emojis`: Remove emoji characters (default: True)
- `remove_markdown`: Remove markdown formatting (default: True)
- `remove_urls`: Remove URLs (default: True)
- `remove_code`: Remove code blocks (default: True)
- `remove_cyrillic`: Remove Cyrillic chars (default: True)
- `vietnamese_abbreviations`: Expand VN abbreviations (default: True)

## Resources

- [VieNeu-TTS-v2 Turbo Documentation](VIENEU_TTS_V2_TURBO.md)
- [Voice Assistant README](../README.md)
- [Test File](../tests/test_text_normalization.py)
