# LLM Integration Guide

Hướng dẫn tích hợp và mở rộng module LLM cho các task cá nhân.

## 📌 Tổng quan

Module LLM được thiết kế theo kiến trúc mở rộng:

```
┌─────────────────────────────────────────────────────────┐
│                    LLM Service                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │  Providers  │  │  Templates  │  │  Task Handlers  │ │
│  │  - Groq     │  │  - Default  │  │  - Q&A          │ │
│  │  - OpenAI   │  │  - Support  │  │  - Support      │ │
│  │  - Ollama   │  │  - Personal │  │  - Tasks        │ │
│  │  - Custom   │  │  - Custom   │  │  - Custom       │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 🔌 1. Thêm LLM Provider mới

### Sử dụng Provider có sẵn

```python
from voice_assistant.core.llm_providers import (
    GroqProvider,
    OpenAIProvider,
    OllamaProvider
)

# Groq (recommended - fast)
provider = GroqProvider(
    api_key="your_key",
    model="llama-3.3-70b-versatile"
)

# OpenAI
provider = OpenAIProvider(
    api_key="your_key",
    model="gpt-4o-mini"
)

# Ollama (local)
provider = OllamaProvider(
    model="llama3",
    base_url="http://localhost:11434"
)
```

### Tạo Provider tùy chỉnh

```python
from voice_assistant.core.llm_base import BaseLLMProvider, LLMResponse

class MyCustomProvider(BaseLLMProvider):
    """Provider cho API tùy chỉnh."""
    
    def __init__(self, api_key: str, endpoint: str):
        self.api_key = api_key
        self.endpoint = endpoint
    
    async def generate(self, messages, **kwargs) -> LLMResponse:
        # Gọi API của bạn
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.endpoint,
                json={"messages": messages},
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            data = response.json()
        
        return LLMResponse(
            content=data["response"],
            finish_reason="stop"
        )
    
    async def generate_stream(self, messages, **kwargs):
        # Streaming version
        async for chunk in self._call_api_stream(messages):
            yield chunk["text"]

# Đăng ký provider
from voice_assistant.core.llm_base import register_provider
register_provider("my_custom", MyCustomProvider(
    api_key="xxx",
    endpoint="https://my-api.com/generate"
))
```

### Sử dụng OpenAI-Compatible API

Nhiều dịch vụ hỗ trợ OpenAI API format:

```python
from voice_assistant.core.llm_providers import OpenAICompatibleProvider

# Together AI
together = OpenAICompatibleProvider(
    api_key="your_together_key",
    base_url="https://api.together.xyz/v1",
    model="meta-llama/Llama-3-70b-chat-hf"
)

# Anyscale
anyscale = OpenAICompatibleProvider(
    api_key="your_anyscale_key",
    base_url="https://api.endpoints.anyscale.com/v1",
    model="meta-llama/Llama-3-70b-chat-hf"
)

# vLLM local server
vllm = OpenAICompatibleProvider(
    api_key="dummy",  # vLLM không cần key
    base_url="http://localhost:8000/v1",
    model="your-model"
)
```

## 📝 2. Tạo Prompt Template

### Sử dụng Template có sẵn

```python
from voice_assistant.core.llm_base import get_registry

registry = get_registry()

# Vietnamese Assistant (default)
template = registry.get_template("vietnamese_assistant")

# Customer Support
template = registry.get_template("customer_support")

# Personal Assistant
template = registry.get_template("personal_assistant")

# Build messages
messages = template.build_messages(
    query="Tôi cần hỗ trợ",
    company="ABC Corp",
    user_name="Nguyen"
)
```

### Tạo Template tùy chỉnh

```python
from voice_assistant.core.llm_base import BasePromptTemplate, register_template

class EcommerceTemplate(BasePromptTemplate):
    """Template cho trợ lý bán hàng e-commerce."""
    
    name = "ecommerce"
    
    def format_system(self, **kwargs) -> str:
        store_name = kwargs.get("store_name", "cửa hàng")
        products = kwargs.get("products", "sản phẩm")
        
        return f"""Bạn là nhân viên tư vấn bán hàng của {store_name}.
Bạn giúp khách hàng tìm {products} phù hợp.

Nguyên tắc:
- Hỏi nhu cầu khách hàng trước
- Gợi ý sản phẩm phù hợp
- Cung cấp thông tin giá, khuyến mãi
- Hỗ trợ đặt hàng"""
    
    def format_context(self, context: dict) -> str:
        parts = []
        
        if "catalog" in context:
            parts.append(f"Danh mục sản phẩm:\n{context['catalog']}")
        
        if "cart" in context:
            parts.append(f"Giỏ hàng hiện tại:\n{context['cart']}")
        
        if "promotions" in context:
            parts.append(f"Khuyến mãi:\n{context['promotions']}")
        
        return "\n\n".join(parts)

# Đăng ký
register_template(EcommerceTemplate())
```

## 🎯 3. Tạo Task Handler

### Handler đơn giản

```python
from voice_assistant.core.llm_tasks import create_custom_handler
from voice_assistant.core.llm_base import register_handler

# Tạo handler bằng factory function
weather_handler = create_custom_handler(
    name="weather",
    keywords=["thời tiết", "weather", "nhiệt độ", "mưa", "nắng"],
    system_prompt="""Bạn là chuyên gia dự báo thời tiết.
Cung cấp thông tin thời tiết chính xác, ngắn gọn.
Luôn đề cập nhiệt độ và điều kiện thời tiết.""",
    priority=15
)

register_handler(weather_handler)
```

### Handler với Tool Calling

```python
from voice_assistant.core.llm_base import (
    BaseTaskHandler, TaskContext, TaskResult, TaskResponse,
    ToolDefinition
)

class WeatherHandler(BaseTaskHandler):
    """Handler tra cứu thời tiết thực."""
    
    name = "weather"
    description = "Tra cứu thời tiết theo vị trí"
    priority = 20
    
    KEYWORDS = ["thời tiết", "weather", "nhiệt độ", "temperature"]
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def can_handle(self, context: TaskContext) -> float:
        query = context.user_query.lower()
        matches = sum(1 for kw in self.KEYWORDS if kw in query)
        return 0.9 if matches >= 1 else 0.0
    
    async def handle(self, context: TaskContext) -> TaskResponse:
        location = self._extract_location(context.user_query)
        weather = await self._get_weather(location)
        
        return TaskResponse(
            status=TaskResult.SUCCESS,
            content=f"Thời tiết {location}: {weather['temp']}°C, {weather['desc']}",
            data={"weather": weather}
        )
    
    def get_tools(self) -> list:
        return [
            ToolDefinition(
                name="get_weather",
                description="Lấy thông tin thời tiết",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            )
        ]
```

## 🔧 4. Tích hợp vào Pipeline

```python
from voice_assistant.core.llm_extended import ExtendedLLMService

# Tạo service với handlers
llm = ExtendedLLMService()

# Đăng ký handlers tùy chỉnh
llm.register_handler(WeatherHandler(api_key="xxx"))

# Xử lý query
response = await llm.process_query(
    query="Thời tiết Hà Nội?",
    history=[]
)
```

## 📊 5. Best Practices

### Handler Priority

```python
# Cao → Thấp
InterruptHandler:     100  # Luôn check trước
DomainSpecific:       20-30
GeneralHandlers:      10-15
FallbackHandler:      -1
```

### Confidence Scoring

```python
def can_handle(self, context):
    # 0.9-1.0: Perfect match
    # 0.7-0.9: Strong match
    # 0.5-0.7: Possible match
    # <0.5: No match
    pass
```

## 📚 Tài liệu liên quan

- [ARCHITECTURE.md](../ARCHITECTURE.md)
- [API.md](../API.md)

---

*Author: nguyendevwk | License: MIT*
