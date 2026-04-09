"""
LLM service with streaming support.
Supports Groq, OpenAI, and compatible APIs.
"""

import asyncio
from typing import AsyncIterator, List, Dict, Optional
from dataclasses import dataclass

from ..config import settings
from ..utils.logging import debug_log, log_llm_token, latency, logger
from ..utils.text_utils import normalize_llm_output, strip_thinking_blocks
from .llm_extended import ExtendedLLMService


@dataclass
class Message:
    """Chat message."""
    role: str  # system, user, assistant
    content: str


class ExtendedLLMAdapter:
    """Compatibility adapter so pipeline can use ExtendedLLMService transparently."""

    def __init__(self, service: ExtendedLLMService):
        self._service = service
        self.model = settings.llm.model
        self.max_tokens = settings.llm.max_tokens
        self.temperature = settings.llm.temperature

    def _normalize_history(self, history: Optional[List[Message]]) -> List[Dict[str, str]]:
        if not history:
            return []

        normalized: List[Dict[str, str]] = []
        for msg in history:
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content")
            else:
                role = getattr(msg, "role", None)
                content = getattr(msg, "content", None)

            if role and content is not None:
                normalized.append({"role": role, "content": str(content)})

        return normalized

    def _ensure_client(self):
        """Warm up underlying provider client for API server startup."""
        provider = getattr(self._service, "_provider", None)
        if provider and hasattr(provider, "_ensure_client"):
            provider._ensure_client()

    async def generate_response(self, prompt: str, history: List[Message] = None) -> str:
        return await self._service.generate_response(
            prompt,
            history=self._normalize_history(history),
        )

    async def generate_response_stream(
        self,
        prompt: str,
        history: List[Message] = None,
    ) -> AsyncIterator[str]:
        async for sentence in self._service.generate_response_stream(
            prompt,
            history=self._normalize_history(history),
        ):
            yield sentence

    async def generate_tokens(
        self,
        prompt: str,
        history: List[Message] = None,
    ) -> AsyncIterator[str]:
        async for token in self._service.generate_tokens(
            prompt,
            history=self._normalize_history(history),
        ):
            yield token


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

        return normalize_llm_output(strip_thinking_blocks(response.choices[0].message.content or ""))

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
        # Minimum sentence length for TTS (edge-tts fails on short texts)
        MIN_SENTENCE_LENGTH = 5
        is_first_chunk = True

        THINK_START = "<think>"
        THINK_END = "</think>"
        thinking = False
        pending = ""

        def consume_char(char: str) -> str:
            """Filter reasoning tags from the token stream and return visible text."""
            nonlocal thinking, pending

            if thinking:
                pending += char
                if THINK_END.startswith(pending):
                    if pending == THINK_END:
                        thinking = False
                        pending = ""
                    return ""

                pending = pending if THINK_END.startswith(pending) else ""
                return ""

            pending += char

            if THINK_START.startswith(pending):
                if pending == THINK_START:
                    thinking = True
                    pending = ""
                return ""

            if pending and not THINK_START.startswith(pending):
                # Emit the earliest character and keep checking for a tag prefix.
                emitted = pending[0]
                pending = pending[1:]
                return emitted

            return ""

        def flush_pending() -> str:
            """Flush any buffered visible text that is not part of a tag."""
            nonlocal pending
            if thinking or not pending:
                return ""
            if THINK_START.startswith(pending):
                return ""
            emitted = pending
            pending = ""
            return emitted

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                token = delta.content

                if first_token:
                    latency.end("llm_first_token")
                    first_token = False

                log_llm_token(token)

                for char in token:
                    visible_char = consume_char(char)
                    if visible_char:
                        buffer += visible_char

                    # Also flush any buffered visible text that can no longer be a tag.
                    flushed = flush_pending()
                    if flushed:
                        buffer += flushed

                # Yield when we hit a sentence delimiter in visible text AND have enough text.
                while True:
                    last_pos = max(
                        (buffer.rfind(d) for d in self.SENTENCE_DELIMITERS if d in buffer),
                        default=-1,
                    )
                    if last_pos <= 0:
                        break

                    sentence = buffer[:last_pos + 1].strip()
                    remaining = buffer[last_pos + 1:].strip()

                    # Only yield if sentence is long enough for TTS.
                    if len(sentence) >= MIN_SENTENCE_LENGTH:
                        buffer = remaining
                        normalized = normalize_llm_output(strip_thinking_blocks(sentence))
                        if normalized:
                            yield normalized
                        continue

                    # Keep buffering if the visible chunk is too short.
                    break

        # Yield remaining buffer (even if short, it's the last part)
        trailing = flush_pending()
        if trailing:
            buffer += trailing

        if buffer.strip():
            cleaned = normalize_llm_output(strip_thinking_blocks(buffer.strip()))
            if cleaned:
                yield cleaned

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
_llm_service: Optional[object] = None


def get_llm_service() -> object:
    """Get or create LLM service singleton."""
    global _llm_service
    if _llm_service is None:
        if settings.llm.use_extended:
            try:
                extended = ExtendedLLMService(
                    auto_register_handlers=settings.llm.auto_register_task_handlers,
                )
                _llm_service = ExtendedLLMAdapter(extended)
                logger.info("Using Extended LLM service with task routing")
            except Exception as e:
                logger.warning(f"Extended LLM unavailable, fallback to basic service: {e}")
                _llm_service = LLMService()
        else:
            _llm_service = LLMService()
    return _llm_service
