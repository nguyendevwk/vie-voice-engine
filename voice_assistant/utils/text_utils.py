"""
Text normalization utilities for TTS and LLM output.

Optimized for VieNeu-TTS-v2 Turbo with:
- Comprehensive LLM output cleaning
- Vietnamese text normalization
- Sentence splitting for optimal TTS latency
- Minimum length enforcement
"""

import re
from typing import List, Optional
from dataclasses import dataclass


THINK_TAG_START = "<think>"
THINK_TAG_END = "</think>"


@dataclass
class TTSNormalizationConfig:
    """Configuration for TTS text normalization."""
    min_length: int = 10  # Minimum chars for TTS
    max_length: int = 200  # Maximum chars per chunk
    prefer_sentence_boundary: bool = True  # Split at sentence boundaries
    ensure_punctuation: bool = True  # Add punctuation if missing
    remove_emojis: bool = True  # Remove emojis
    remove_markdown: bool = True  # Remove markdown
    remove_urls: bool = True  # Remove URLs
    remove_code: bool = True  # Remove code blocks
    remove_cyrillic: bool = True  # Remove Cyrillic chars
    normalize_whitespace: bool = True  # Normalize spaces
    merge_short_sentences: bool = True  # Merge sentences < min_length
    vietnamese_abbreviations: bool = True  # Expand VN abbreviations


# Default config for VieNeu-TTS-v2 Turbo
VIENEU_V2_TURBO_CONFIG = TTSNormalizationConfig(
    min_length=10,  # v2 Turbo needs more context
    max_length=200,  # Optimal for low latency
    prefer_sentence_boundary=True,
    ensure_punctuation=True,
    remove_emojis=True,
    remove_markdown=True,
    remove_urls=True,
    remove_code=True,
    remove_cyrillic=True,
    normalize_whitespace=True,
    merge_short_sentences=True,
    vietnamese_abbreviations=True,
)


def strip_thinking_blocks(text: str) -> str:
    """Remove hidden reasoning blocks and stray think tags from model output."""
    if not text:
        return ""

    # Remove complete reasoning blocks first.
    text = re.sub(r"(?is)<think>.*?</think>", "", text)

    # Remove any leftover standalone think tags.
    text = re.sub(r"(?is)</?think\s*>", "", text)

    return text


def remove_emojis(text: str) -> str:
    """Remove emoji characters that TTS can't pronounce."""
    # Emoji unicode ranges
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)


def remove_markdown(text: str) -> str:
    """Remove markdown formatting."""
    # Code blocks (must be before other patterns)
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]+`', '', text)

    # Headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Bold and italic (order matters!)
    text = re.sub(r'\*\*\*([^*]+)\*\*\*', r'\1', text)  # Bold italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
    text = re.sub(r'___([^_]+)___', r'\1', text)  # Bold italic
    text = re.sub(r'__([^_]+)__', r'\1', text)  # Bold
    text = re.sub(r'_([^_]+)_', r'\1', text)  # Italic

    # Links - keep text, remove URL
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Images - remove completely
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)

    # List markers at start of line
    text = re.sub(r'^\s*[\*\+\-]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Inline bullets at start of text
    text = re.sub(r'^[\*\+\-•]\s*', '', text.strip())

    # Blockquotes
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)

    # Horizontal rules
    text = re.sub(r'^[\*\-_]{3,}\s*$', '', text, flags=re.MULTILINE)

    # Tables
    text = re.sub(r'\|.*\|', '', text)
    text = re.sub(r'^[-]{3,}\s*$', '', text, flags=re.MULTILINE)

    return text


def expand_vietnamese_abbreviations(text: str) -> str:
    """Expand common Vietnamese abbreviations for better TTS."""
    replacements = {
        r'\bTP\b': 'thành phố',
        r'\bTp\b': 'thành phố',
        r'\btp\b': 'thành phố',
        r'\bHCM\b': 'Hồ Chí Minh',
        r'\bHN\b': 'Hà Nội',
        r'\bVN\b': 'Việt Nam',
        r'\bDr\b': 'Tiến sĩ',
        r'\bTS\b': 'Tiến sĩ',
        r'\bThS\b': 'Thạc sĩ',
        r'\bCN\b': 'Cử nhân',
        r'\bMr\b': 'Ông',
        r'\bMrs\b': 'Bà',
        r'\bMs\b': 'Cô',
        r'\bPGS\b': 'Phó Giáo sư',
        r'\bGS\b': 'Giáo sư',
        r'\bTSKS\b': 'Tiến sĩ Khoa học',
        r'\bUBND\b': 'Ủy ban Nhân dân',
        r'\bQĐ\b': 'Quyết định',
        r'\bNĐ\b': 'Nghị định',
        r'\bTT\b': 'Thông tư',
        r'\bBCA\b': 'Bộ Công an',
        r'\bBQP\b': 'Bộ Quốc phòng',
        r'\bBYT\b': 'Bộ Y tế',
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def normalize_numbers(text: str) -> str:
    """Normalize numbers for better TTS reading."""
    # Remove thousand separators
    text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
    
    # Percentages
    text = re.sub(r'(\d+)%', r'\1 phần trăm', text)
    
    # Currency (simple)
    text = re.sub(r'(\d+)\s*USD', r'\1 đô la Mỹ', text)
    text = re.sub(r'(\d+)\s*VND', r'\1 đồng', text)
    
    return text


def normalize_for_tts(
    text: str,
    config: Optional[TTSNormalizationConfig] = None
) -> str:
    """
    Normalize text for TTS synthesis.
    
    Optimized for VieNeu-TTS-v2 Turbo.
    """
    if config is None:
        config = VIENEU_V2_TURBO_CONFIG
    
    if not text or not text.strip():
        return ""

    text = text.strip()

    # Remove thinking blocks
    text = strip_thinking_blocks(text)
    if not text.strip():
        return ""

    # Remove markdown
    if config.remove_markdown:
        text = remove_markdown(text)

    # Remove URLs
    if config.remove_urls:
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)

    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)

    # Remove code blocks (already handled in remove_markdown, but ensure)
    if config.remove_code:
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]+`', '', text)

    # Remove emojis
    if config.remove_emojis:
        text = remove_emojis(text)

    # Remove Cyrillic
    if config.remove_cyrillic:
        text = re.sub(r'[а-яА-ЯёЁ]', '', text)

    # Expand Vietnamese abbreviations
    if config.vietnamese_abbreviations:
        text = expand_vietnamese_abbreviations(text)

    # Normalize numbers
    text = normalize_numbers(text)

    # Normalize whitespace
    if config.normalize_whitespace:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

    # Remove repeated punctuation FIRST
    text = re.sub(r'([.!?])\1+', r'\1', text)  # Multiple ! or ? or .
    text = re.sub(r'[,]{2,}', ',', text)  # Multiple commas
    text = re.sub(r'\s*([.!?])\s*([.!?])*\s*', r'\1', text)  # Mixed punctuation

    # Fix punctuation spacing - remove spaces BEFORE punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # Ensure single space AFTER punctuation if followed by non-space
    text = re.sub(r'([.,!?;:])(?=[^\s])', r'\1 ', text)
    
    # Clean up any trailing space after final punctuation
    text = text.rstrip()
    if text and text[-1] in '.!? ' and (len(text) < 2 or text[-2] not in '.!?'):
        text = text.rstrip()

    # Limit length
    if len(text) > config.max_length * 2:
        text = text[:config.max_length * 2 - 3] + "..."

    # Ensure ending punctuation
    if config.ensure_punctuation and text and text[-1] not in '.!?。':
        text += '.'

    return text


def split_for_tts(
    text: str,
    config: Optional[TTSNormalizationConfig] = None
) -> List[str]:
    """
    Split text into optimal chunks for TTS synthesis.
    
    Optimized for VieNeu-TTS-v2 Turbo with:
    - Minimum length enforcement
    - Sentence boundary preference
    - Merging of short sentences
    """
    if config is None:
        config = VIENEU_V2_TURBO_CONFIG
    
    if not text or not text.strip():
        return []

    # Normalize first
    text = normalize_for_tts(text, config)
    if not text or len(text) < config.min_length:
        return []

    # Vietnamese sentence endings
    sentences = re.split(r'([.!?]+\s*)', text)

    # Combine sentences with their punctuation
    parts = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip()
        punct = sentences[i + 1].strip() if i + 1 < len(sentences) else ''
        
        if sentence:
            combined = (sentence + ' ' + punct).strip()
            parts.append(combined)
    
    # Handle last part
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        last = sentences[-1].strip()
        if config.ensure_punctuation and last and last[-1] not in '.!?':
            last += '.'
        parts.append(last)

    # Merge short sentences
    if config.merge_short_sentences:
        merged = []
        buffer = ""
        
        for part in parts:
            if len(buffer) + len(part) < config.min_length:
                # Buffer is too short, accumulate
                buffer = (buffer + ' ' + part).strip()
            else:
                # Buffer is long enough, emit it
                if buffer:
                    if len(buffer) >= config.min_length:
                        merged.append(buffer)
                    buffer = ""
                
                # Check if current part is long enough
                if len(part) >= config.min_length:
                    merged.append(part)
                else:
                    buffer = part
        
        # Handle remaining buffer
        if buffer:
            if len(buffer) >= config.min_length:
                merged.append(buffer)
            elif merged:
                # Append to last chunk
                merged[-1] = merged[-1] + ' ' + buffer
    
    # Split long chunks at commas
    final_chunks = []
    for chunk in merged:
        if len(chunk) > config.max_length:
            # Split at commas or semicolons
            sub_parts = re.split(r'([,;]\s*)', chunk)
            sub_buffer = ""
            
            for j in range(0, len(sub_parts) - 1, 2):
                sub_part = sub_parts[j].strip()
                sep = sub_parts[j + 1].strip() if j + 1 < len(sub_parts) else ''
                
                if len(sub_buffer) + len(sub_part) <= config.max_length:
                    sub_buffer = (sub_buffer + ' ' + sub_part + ' ' + sep).strip()
                else:
                    if sub_buffer and len(sub_buffer) >= config.min_length:
                        final_chunks.append(sub_buffer)
                    sub_buffer = sub_part + ' ' + sep
            
            if sub_buffer and len(sub_buffer) >= config.min_length:
                final_chunks.append(sub_buffer)
        else:
            if len(chunk) >= config.min_length:
                final_chunks.append(chunk)
    
    return final_chunks


def prepare_for_edge_tts(text: str) -> str:
    """
    Prepare text specifically for edge-tts synthesis.
    Edge-TTS is very picky about text format and fails on:
    - Very short texts (< 10 chars)
    - Texts with only punctuation
    - Texts with special characters
    - Texts without proper sentence endings
    """
    if not text or not text.strip():
        return ""

    text = text.strip()

    # Remove leading bullets/markers
    text = re.sub(r'^[\*\+\-•]\s*', '', text)
    text = re.sub(r'^\d+[\.\)]\s*', '', text)

    # Remove markdown emphasis that might remain
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'_+', '', text)

    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # If text is too short, return empty
    if len(text) < 5:
        return ""

    # Ensure proper ending punctuation
    if text and text[-1] not in '.!?。':
        text += '.'

    return text


def normalize_llm_output(text: str) -> str:
    """
    Normalize LLM output for better readability and TTS.

    Handles:
    - Markdown formatting
    - List formatting
    - Extra whitespace
    - Repeated punctuation
    """
    if not text or not text.strip():
        return ""

    text = strip_thinking_blocks(text)
    if not text.strip():
        return ""

    # Remove markdown
    text = remove_markdown(text)

    # Normalize list items to natural speech
    text = re.sub(r'^\s*[\*\+\-]\s+(.+)', r'\1.', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+(.+)', r'\1.', text, flags=re.MULTILINE)

    # Convert bullet points to comma-separated list
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned_lines.append(line)

    text = ' '.join(cleaned_lines)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Normalize punctuation
    text = re.sub(r'([.!?])\1+', r'\1', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)

    # Add period at end if missing
    text = text.strip()
    if text and text[-1] not in '.!?':
        text += '.'

    return text


def clean_vietnamese_text(text: str) -> str:
    """
    Clean Vietnamese text for better TTS pronunciation.

    Handles:
    - Number normalization
    - Abbreviation expansion
    - Common terms
    """
    if not text:
        return ""

    # Common abbreviations
    replacements = {
        r'\bTP\b': 'thành phố',
        r'\bTp\b': 'thành phố',
        r'\bHCM\b': 'Hồ Chí Minh',
        r'\bHN\b': 'Hà Nội',
        r'\bVN\b': 'Việt Nam',
        r'\bDr\b': 'Tiến sĩ',
        r'\bMr\b': 'Ông',
        r'\bMrs\b': 'Bà',
        r'\bMs\b': 'Cô',
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text
