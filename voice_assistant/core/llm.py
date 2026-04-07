"""
LLM service with streaming support.
Supports Groq, OpenAI, and compatible APIs.
"""

import asyncio
from typing import AsyncIterator, List, Dict, Optional
from dataclasses import dataclass

from ..config import settings
from ..utils.logging import debug_log, log_llm_token, latency, logger


@dataclass
class Message:
    """Chat message."""
    role: str  # system, user, assistant
    content: str


class LLMService:
    """
    LLM service with streaming token generation.

    Supports:
    - Groq (fast inference)
    - OpenAI
    - Any OpenAI-compatible API
    """

    # Sentence delimiters for chunking TTS
    SENTENCE_DELIMITERS = {",", ".", "?", "!", ";", ":", "\n"}

    def __init__(self, config=None):
        cfg = config or settings.llm
        self.model = cfg.model
        self.max_tokens = cfg.max_tokens
        self.temperature = cfg.temperature
        self.system_prompt = cfg.system_prompt

        self._client = None
        self._config = cfg

    def _ensure_client(self):
        """Lazy initialize API client."""
        if self._client is not None:
            return

        from openai import AsyncOpenAI

        if not self._config.api_key:
            raise ValueError(
                "LLM API key not set. Set GROQ_API_KEY environment variable."
            )

        self._client = AsyncOpenAI(
            api_key=self._config.api_key,
            base_url=self._config.base_url,
        )

        debug_log("LLM client initialized", model=self.model)

    async def generate_response(self, prompt: str, history: List[Message] = None) -> str:
        """Generate complete response (non-streaming)."""
        self._ensure_client()

        messages = self._build_messages(prompt, history)

        with latency.track("llm_generate"):
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

        return response.choices[0].message.content

    async def generate_response_stream(
        self,
        prompt: str,
        history: List[Message] = None,
    ) -> AsyncIterator[str]:
        """
        Stream response as sentence chunks.

        Yields complete sentences/phrases for TTS processing.
        """
        self._ensure_client()

        messages = self._build_messages(prompt, history)

        latency.start("llm_first_token")
        first_token = True

        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
        )

        buffer = ""

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                token = delta.content

                if first_token:
                    latency.end("llm_first_token")
                    first_token = False

                log_llm_token(token)
                buffer += token

                # Yield when we hit a sentence delimiter
                for delim in self.SENTENCE_DELIMITERS:
                    if delim in token:
                        # Find last delimiter position
                        last_pos = max(
                            buffer.rfind(d) for d in self.SENTENCE_DELIMITERS
                            if d in buffer
                        )
                        if last_pos > 0:
                            sentence = buffer[:last_pos + 1].strip()
                            buffer = buffer[last_pos + 1:].strip()
                            if sentence:
                                yield sentence
                        break

        # Yield remaining buffer
        if buffer.strip():
            yield buffer.strip()

        if settings.debug:
            print()  # Newline after token stream

    async def generate_tokens(
        self,
        prompt: str,
        history: List[Message] = None,
    ) -> AsyncIterator[str]:
        """Stream individual tokens (for display)."""
        self._ensure_client()

        messages = self._build_messages(prompt, history)

        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    def _build_messages(
        self,
        prompt: str,
        history: List[Message] = None,
    ) -> List[Dict[str, str]]:
        """Build messages list for API call."""
        messages = [{"role": "system", "content": self.system_prompt}]

        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": prompt})

        return messages


# Lazy singleton
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create LLM service singleton."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
