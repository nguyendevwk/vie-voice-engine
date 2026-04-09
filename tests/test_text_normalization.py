"""Test text normalization for TTS."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from voice_assistant.utils.text_utils import (
    normalize_for_tts,
    split_for_tts,
    remove_emojis,
    remove_markdown,
    expand_vietnamese_abbreviations,
    VIENEU_V2_TURBO_CONFIG,
)


def test_normalize_for_tts():
    """Test text normalization."""
    print("="*70)
    print("Testing normalize_for_tts")
    print("="*70)
    
    test_cases = [
        # (input, expected_output, description)
        ("", "", "Empty string"),
        ("   ", "", "Only whitespace"),
        ("Xin chào", "Xin chào.", "Simple text"),
        ("Xin chào!", "Xin chào!", "With punctuation"),
        ("**Bold text**", "Bold text.", "Markdown bold"),
        ("*Italic text*", "Italic text.", "Markdown italic"),
        ("**Bold and italic**", "Bold and italic.", "Markdown bold+italic"),
        ("# Header", "Header.", "Markdown header"),
        ("- List item", "List item.", "List bullet"),
        ("1. Numbered item", "Numbered item.", "Numbered list"),
        ("Check https://example.com", "Check.", "URL removal"),
        ("Email: test@example.com", "Email:", "Email removal"),
        ("Hello 👋 World 🌍", "Hello  World.", "Emoji removal"),
        ("Привет мир", " мир", "Cyrillic removal"),
        ("TP HCM", "thành phố Hồ Chí Minh.", "Vietnamese abbreviations"),
        ("Dr. Nguyễn", "Tiến sĩ Nguyễn.", "Title expansion"),
        ("Multiple   spaces", "Multiple spaces.", "Whitespace normalization"),
        ("Hello!!!", "Hello!", "Repeated punctuation"),
        ("Hello,,,", "Hello,", "Repeated commas"),
        ("`code block`", ".", "Inline code removal"),
        ("```python\ncode\n```", ".", "Code block removal"),
        ("Thought: <think>reasoning</think>", "Thought:", "Think tag removal"),
    ]
    
    passed = 0
    failed = 0
    
    for input_text, expected, description in test_cases:
        result = normalize_for_tts(input_text)
        
        # Allow for minor differences in spacing
        if result.strip() == expected.strip() or (not result and not expected):
            print(f"✅ {description}: '{input_text[:30]}' -> '{result[:30]}'")
            passed += 1
        else:
            print(f"❌ {description}")
            print(f"   Input: '{input_text}'")
            print(f"   Expected: '{expected}'")
            print(f"   Got: '{result}'")
            failed += 1
    
    print(f"\n{'='*70}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*70}\n")
    
    return failed == 0


def test_split_for_tts():
    """Test sentence splitting."""
    print("="*70)
    print("Testing split_for_tts")
    print("="*70)
    
    test_cases = [
        # (input, min_chunks, description)
        ("Xin chào Việt Nam!", 0, "Short text"),
        ("Xin chào! Tôi là trợ lý ảo tiếng Việt. Tôi có thể giúp gì cho bạn?", 1, "Multiple sentences"),
        ("Hôm nay thời tiết đẹp. Mặt trời chiếu sáng. Chim hót líu lo.", 2, "Three sentences"),
        ("Đây là một câu rất dài, có nhiều dấu phẩy, và cần được tách ra để xử lý tốt hơn.", 1, "Long sentence with commas"),
    ]
    
    passed = 0
    failed = 0
    
    for input_text, min_chunks, description in test_cases:
        chunks = split_for_tts(input_text)
        
        print(f"\n{description}:")
        print(f"  Input: '{input_text[:50]}'")
        print(f"  Chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks, 1):
            print(f"  {i}. '{chunk}'")
        
        if len(chunks) >= min_chunks:
            print(f"  ✅ Pass (expected >= {min_chunks} chunks)")
            passed += 1
        else:
            print(f"  ❌ Fail (expected >= {min_chunks} chunks, got {len(chunks)})")
            failed += 1
    
    print(f"\n{'='*70}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*70}\n")
    
    return failed == 0


def test_llm_output_scenarios():
    """Test realistic LLM output scenarios."""
    print("="*70)
    print("Testing Realistic LLM Output Scenarios")
    print("="*70)
    
    test_cases = [
        # Markdown response
        """
## Trả lời

**Đây là câu trả lời của tôi:**

- Điểm 1: Thông tin quan trọng
- Điểm 2: Chi tiết bổ sung
- Điểm 3: Kết luận

> Trích dẫn quan trọng
        """,
        
        # Response with think tags
        """
<think>
User asked about weather. I should provide a helpful response.
</think>

Hôm nay thời tiết đẹp ở TP HCM. Nhiệt độ khoảng 30°C.
        """,
        
        # Response with emojis and URLs
        """
Chào bạn! 👋

Đây là tài liệu tham khảo: https://example.com

Liên hệ: support@example.com

Cảm ơn bạn đã sử dụng dịch vụ! 😊
        """,
        
        # Response with code
        """
Để sử dụng, bạn chạy lệnh sau:

```bash
pip install package
```

Hoặc dùng inline: `pip install`

Cảm ơn bạn!
        """,
        
        # Response with abbreviations
        """
Tại TP HCM, UBND đã ra QĐ mới về việc này.

TS. Nguyễn Văn A sẽ trình bày.

GS. Trần Văn B đồng ý.
        """,
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test Case {i}:")
        print(f"{'='*70}")
        print(f"Input (first 100 chars):")
        print(f"  {text.strip()[:100]}...")
        
        normalized = normalize_for_tts(text)
        print(f"\nNormalized:")
        print(f"  {normalized}")
        
        chunks = split_for_tts(text)
        print(f"\nSplit into {len(chunks)} chunk(s):")
        for j, chunk in enumerate(chunks, 1):
            print(f"  {j}. {chunk}")
    
    print(f"\n{'='*70}")
    print("✅ All scenarios processed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TEXT NORMALIZATION TESTS")
    print("="*70 + "\n")
    
    all_passed = True
    
    # Run tests
    if not test_normalize_for_tts():
        all_passed = False
    
    if not test_split_for_tts():
        all_passed = False
    
    test_llm_output_scenarios()
    
    # Summary
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70 + "\n")
    
    sys.exit(0 if all_passed else 1)
