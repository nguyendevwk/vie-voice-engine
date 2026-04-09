"""
Built-in LLM Providers.

Includes implementations for common LLM APIs:
- Groq (fast inference)
- OpenAI (GPT models)
- Ollama (local models)
- Custom API (any OpenAI-compatible endpoint)

Usage:
    >>> from voice_assistant.core.llm_providers import GroqProvider
    >>> provider = GroqProvider(api_key="...", model="llama-3.3-70b-versatile")
    >>> response = await provider.generate(messages)
"""

import asyncio
from typing import AsyncIterator, List, Dict, Optional, Any

from .llm_base import (
    BaseLLMProvider, LLMResponse, ToolDefinition,
    register_provider
)


class OpenAICompatibleProvider(BaseLLMProvider):
    """
    Provider for OpenAI-compatible APIs.

    Works with OpenAI, Groq, Together AI, Anyscale, vLLM, etc.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_kwargs = kwargs
        self._client = None

    def _ensure_client(self):
        """Lazy initialize client."""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )

    async def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Generate complete response."""
        self._ensure_client()

        response = await self._client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            **self.extra_kwargs
        )

        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            tool_calls=getattr(choice.message, "tool_calls", None),
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            } if response.usage else None,
            raw=response
        )

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream response tokens."""
        self._ensure_client()

        stream = await self._client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            stream=True,
            **self.extra_kwargs
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def supports_tools(self) -> bool:
        return True

    async def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[ToolDefinition],
        **kwargs
    ) -> LLMResponse:
        """Generate with tool calling."""
        self._ensure_client()

        tool_defs = [t.to_dict() for t in tools]

        response = await self._client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            tools=tool_defs,
            tool_choice=kwargs.get("tool_choice", "auto"),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )

        choice = response.choices[0]
        tool_calls = None

        if choice.message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in choice.message.tool_calls
            ]

        return LLMResponse(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            tool_calls=tool_calls,
            raw=response
        )

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.model,
            "provider": "openai_compatible",
            "base_url": self.base_url,
            "supports_tools": True,
            "supports_vision": "vision" in self.model or "4o" in self.model
        }


class GroqProvider(OpenAICompatibleProvider):
    """
    Groq API provider with fast inference.

    Supported models:
    - llama-3.3-70b-versatile (recommended)
    - llama-3.1-8b-instant
    - mixtral-8x7b-32768
    - gemma2-9b-it
    """

    def __init__(
        self,
        api_key: str,
        model: str = "qwen/qwen3-32b",
        **kwargs
    ):
        super().__init__(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            model=model,
            **kwargs
        )

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.model,
            "provider": "groq",
            "supports_tools": True,
            "supports_vision": False,
            "context_length": 131072 if "70b" in self.model else 32768
        }


class OpenAIProvider(OpenAICompatibleProvider):
    """
    OpenAI API provider.

    Supported models:
    - gpt-4o (recommended)
    - gpt-4o-mini (faster)
    - gpt-4-turbo
    - gpt-3.5-turbo
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        **kwargs
    ):
        super().__init__(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            model=model,
            **kwargs
        )

    def supports_vision(self) -> bool:
        return "4o" in self.model or "vision" in self.model

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.model,
            "provider": "openai",
            "supports_tools": True,
            "supports_vision": self.supports_vision(),
            "context_length": 128000 if "4o" in self.model else 16385
        }


class OllamaProvider(BaseLLMProvider):
    """
    Ollama provider for local model inference.

    Install: https://ollama.ai
    Pull models: ollama pull llama3

    Supported models (examples):
    - llama3 (recommended)
    - mistral
    - gemma
    - qwen2
    - phi3
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        **kwargs
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.extra_kwargs = kwargs

    async def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Generate using Ollama API."""
        import httpx

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": kwargs.get("model", self.model),
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", self.temperature),
                        **self.extra_kwargs
                    }
                }
            )
            response.raise_for_status()
            data = response.json()

        return LLMResponse(
            content=data["message"]["content"],
            finish_reason="stop",
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": (
                    data.get("prompt_eval_count", 0) +
                    data.get("eval_count", 0)
                ),
            },
            raw=data
        )

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream response from Ollama."""
        import httpx
        import json

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model": kwargs.get("model", self.model),
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": kwargs.get("temperature", self.temperature),
                        **self.extra_kwargs
                    }
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.model,
            "provider": "ollama",
            "base_url": self.base_url,
            "local": True,
            "supports_tools": False,
            "supports_vision": "llava" in self.model or "vision" in self.model
        }


class CustomAPIProvider(BaseLLMProvider):
    """
    Custom API provider for any endpoint.

    Use this as a base for integrating custom LLM APIs.

    Example:
        >>> provider = CustomAPIProvider(
        ...     endpoint="https://my-api.com/generate",
        ...     headers={"Authorization": "Bearer xxx"},
        ...     request_format=lambda messages: {"prompt": messages[-1]["content"]},
        ...     response_parser=lambda r: r["text"]
        ... )
    """

    def __init__(
        self,
        endpoint: str,
        headers: Dict[str, str] = None,
        request_format: callable = None,
        response_parser: callable = None,
        stream_parser: callable = None,
        **kwargs
    ):
        self.endpoint = endpoint
        self.headers = headers or {}
        self.request_format = request_format or self._default_request
        self.response_parser = response_parser or self._default_response
        self.stream_parser = stream_parser
        self.extra_kwargs = kwargs

    def _default_request(self, messages: List[Dict], **kwargs) -> Dict:
        """Default request format (OpenAI-compatible)."""
        return {
            "messages": messages,
            "stream": kwargs.get("stream", False),
            **self.extra_kwargs,
            **kwargs
        }

    def _default_response(self, data: Dict) -> str:
        """Default response parser."""
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        elif "content" in data:
            return data["content"]
        elif "text" in data:
            return data["text"]
        elif "response" in data:
            return data["response"]
        return str(data)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Generate using custom API."""
        import httpx

        request_data = self.request_format(messages, **kwargs)

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.endpoint,
                json=request_data,
                headers=self.headers
            )
            response.raise_for_status()
            data = response.json()

        content = self.response_parser(data)

        return LLMResponse(
            content=content,
            finish_reason="stop",
            raw=data
        )

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream from custom API."""
        import httpx
        import json

        request_data = self.request_format(messages, stream=True, **kwargs)

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                self.endpoint,
                json=request_data,
                headers=self.headers
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        if line.startswith("data: "):
                            line = line[6:]
                        if line == "[DONE]":
                            break
                        try:
                            data = json.loads(line)
                            if self.stream_parser:
                                text = self.stream_parser(data)
                            else:
                                text = self.response_parser(data)
                            if text:
                                yield text
                        except json.JSONDecodeError:
                            continue

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "custom",
            "provider": "custom_api",
            "endpoint": self.endpoint
        }


# ============================================================================
# Helper functions
# ============================================================================

def create_groq_provider(api_key: str = None, model: str = None) -> GroqProvider:
    """Create Groq provider with defaults from settings."""
    from ..config import settings
    return GroqProvider(
        api_key=api_key or settings.llm.api_key,
        model=model or settings.llm.model
    )


def create_openai_provider(api_key: str = None, model: str = None) -> OpenAIProvider:
    """Create OpenAI provider."""
    import os
    return OpenAIProvider(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        model=model or "gpt-4o-mini"
    )


def create_ollama_provider(model: str = "llama3", base_url: str = None) -> OllamaProvider:
    """Create Ollama provider for local inference."""
    return OllamaProvider(
        model=model,
        base_url=base_url or "http://localhost:11434"
    )


# Auto-register providers if API keys available
def _auto_register_providers():
    """Auto-register providers based on available API keys."""
    from ..config import settings
    import os

    # Groq (primary)
    if settings.llm.api_key:
        register_provider("groq", create_groq_provider())

    # OpenAI (secondary)
    if os.getenv("OPENAI_API_KEY"):
        register_provider("openai", create_openai_provider())

    # Ollama (local fallback)
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if response.status_code == 200:
            register_provider("ollama", create_ollama_provider())
    except Exception:
        pass  # Ollama not available
