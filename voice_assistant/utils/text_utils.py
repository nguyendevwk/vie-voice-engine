"""
Text normalization utilities for TTS and LLM output.
Cleans and normalizes Vietnamese text for speech synthesis.
"""

import re
from typing import List


def normalize_for_tts(text: str) -> str:
    """
    Normalize text for TTS synthesis.
    
    Handles:
    - Markdown formatting
    - Special characters
    - Numbers and dates
    - URLs and emails
    - Punctuation
    """
    if not text or not text.strip():
        return ""
    
    # Remove markdown
    text = remove_markdown(text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove code blocks and inline code
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]+`', '', text)
    
    # Remove non-Vietnamese special characters
    text = re.sub(r'[а-яА-ЯёЁ]', '', text)  # Cyrillic
    text = re.sub(r'[^\w\s.,!?;:\-áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđĐÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ\u0300-\u036f]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Normalize punctuation
    text = re.sub(r'([.!?])\1+', r'\1', text)  # Remove repeated punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Remove space before punctuation
    
    # Limit length
    if len(text) > 1000:
        text = text[:997] + "..."
    
    return text


def remove_markdown(text: str) -> str:
    """Remove markdown formatting."""
    # Headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Bold and italic
    text = re.sub(r'\*\*\*([^*]+)\*\*\*', r'\1', text)  # Bold italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
    text = re.sub(r'__([^_]+)__', r'\1', text)  # Bold
    text = re.sub(r'_([^_]+)_', r'\1', text)  # Italic
    
    # Links
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Images
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)
    
    # Lists - remove bullet markers but keep text
    text = re.sub(r'^\s*[\*\+\-]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Also remove inline bullets at start of text
    text = re.sub(r'^[\*\+\-]\s*', '', text.strip())
    
    # Blockquotes
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    
    # Horizontal rules
    text = re.sub(r'^[\*\-_]{3,}\s*$', '', text, flags=re.MULTILINE)
    
    # Tables (simple removal)
    text = re.sub(r'\|.*\|', '', text)
    
    return text


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


def split_into_sentences(text: str, max_length: int = 200) -> List[str]:
    """
    Split text into sentences for streaming TTS.
    
    Args:
        text: Input text
        max_length: Maximum sentence length (chars)
    
    Returns:
        List of sentences
    """
    if not text or not text.strip():
        return []
    
    # Vietnamese sentence endings
    sentences = re.split(r'([.!?]+\s+)', text)
    
    # Combine sentence and its punctuation
    result = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip()
        punct = sentences[i + 1].strip() if i + 1 < len(sentences) else ''
        
        if sentence:
            combined = sentence + punct
            
            # Split long sentences at commas or semicolons
            if len(combined) > max_length:
                parts = re.split(r'([,;]\s+)', combined)
                for j in range(0, len(parts), 2):
                    part = parts[j].strip()
                    sep = parts[j + 1].strip() if j + 1 < len(parts) else ''
                    if part:
                        result.append(part + sep)
            else:
                result.append(combined)
    
    # Handle last part if no punctuation
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        last = sentences[-1].strip()
        if len(last) > max_length:
            parts = re.split(r'([,;]\s+)', last)
            for j in range(0, len(parts), 2):
                part = parts[j].strip()
                sep = parts[j + 1].strip() if j + 1 < len(parts) else ''
                if part:
                    result.append(part + sep)
        else:
            result.append(last)
    
    return [s for s in result if s]


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
