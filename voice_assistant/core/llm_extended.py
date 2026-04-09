"""
Extended LLM Service with Task Routing.

Combines providers, templates, and task handlers for intelligent
query processing with automatic task delegation.

Usage:
    >>> from voice_assistant.core.llm_extended import ExtendedLLMService
    >>> llm = ExtendedLLMService()
    >>> response = await llm.process_query("Thời tiết Hà Nội?")
    >>> print(response.content)
"""

import asyncio
from typing import AsyncIterator, List, Dict, Optional, Any

from .llm_base import (
    Message, TaskContext, TaskResult, TaskResponse, LLMResponse,
    BaseLLMProvider, BasePromptTemplate, BaseTaskHandler,
    get_registry, ToolDefinition
)
from .llm_providers import GroqProvider, create_groq_provider
from .llm_tasks import register_all_handlers

from ..config import settings
from ..utils.logging import logger, debug_log, latency
from ..utils.text_utils import normalize_llm_output, strip_thinking_blocks


class ExtendedLLMService:
    """
    Extended LLM service with intelligent task routing.
    
    Features:
    - Multiple provider support (Groq, OpenAI, Ollama, custom)
    - Task handlers for specialized processing
    - Prompt templates for different use cases
    - Tool/function calling support
    - Streaming responses
    
    Example:
        >>> llm = ExtendedLLMService()
        >>> 
        >>> # Simple query
        >>> response = await llm.process_query("Xin chào")
        >>> print(response.content)
        >>> 
        >>> # With history
        >>> response = await llm.process_query(
        ...     query="Tiếp tục",
        ...     history=[Message("user", "Kể chuyện"), Message("assistant", "...")]
        ... )
        >>> 
        >>> # Streaming
        >>> async for sentence in llm.stream_response("Giải thích AI"):
        ...     print(sentence)
    """
    
    # Sentence delimiters for TTS chunking
    SENTENCE_DELIMITERS = {",", ".", "?", "!", ";", ":", "\n"}
    
    def __init__(
        self,
        provider: BaseLLMProvider = None,
        template: str = "vietnamese_assistant",
        auto_register_handlers: bool = True
    ):
        """
        Initialize extended LLM service.
        
        Args:
            provider: LLM provider (default: Groq)
            template: Prompt template name
            auto_register_handlers: Register built-in handlers
        """
        self._registry = get_registry()
        
        # Set default provider
        if provider:
            self._provider = provider
        else:
            # Try to create Groq provider from settings
            try:
                self._provider = create_groq_provider()
            except Exception as e:
                logger.warning(f"Could not create Groq provider: {e}")
                self._provider = None
        
        # Set template
        self._template = self._registry.get_template(template)
        
        # Auto-register handlers
        if auto_register_handlers:
            register_all_handlers()
        
        # Configuration
        self.max_tokens = settings.llm.max_tokens
        self.temperature = settings.llm.temperature
        self.timeout = settings.pipeline.llm_timeout_s
    
    def set_provider(self, name: str) -> bool:
        """
        Switch to a registered provider.
        
        Args:
            name: Provider name ("groq", "openai", "ollama", etc.)
        
        Returns:
            True if provider found and set
        """
        provider = self._registry.get_provider(name)
        if provider:
            self._provider = provider
            debug_log(f"Switched to provider: {name}")
            return True
        return False
    
    def set_template(self, name: str, **kwargs) -> bool:
        """
        Switch to a registered template.
        
        Args:
            name: Template name
            **kwargs: Template parameters
        
        Returns:
            True if template found and set
        """
        template = self._registry.get_template(name)
        if template:
            self._template = template
            self._template_kwargs = kwargs
            debug_log(f"Switched to template: {name}")
            return True
        return False
    
    def register_provider(self, name: str, provider: BaseLLMProvider):
        """Register a new provider."""
        self._registry.register_provider(name, provider)
    
    def register_handler(self, handler: BaseTaskHandler):
        """Register a new task handler."""
        self._registry.register_handler(handler)
    
    async def process_query(
        self,
        query: str,
        history: List[Message] = None,
        session_data: Dict[str, Any] = None,
        **kwargs
    ) -> TaskResponse:
        """
        Process a user query with intelligent routing.
        
        Args:
            query: User query text
            history: Conversation history
            session_data: Session-specific data
            **kwargs: Additional parameters
        
        Returns:
            TaskResponse with result and content
        """
        if not self._provider:
            return TaskResponse(
                status=TaskResult.FAILED,
                content="LLM provider not configured"
            )
        
        # Create task context
        context = TaskContext(
            user_query=query,
            history=history or [],
            session_data=session_data or {},
            metadata=kwargs
        )
        
        # Find handler
        handler = self._registry.find_handler(context)
        
        if handler:
            debug_log(f"Using handler: {handler.name}")
            
            # Get handler result
            response = await handler.handle(context)
            
            # If handler delegated to LLM, generate response
            if response.status == TaskResult.DELEGATED:
                # Use handler's system prompt if available
                system_prompt = handler.get_system_prompt()
                
                # Generate LLM response
                llm_response = await self._generate_with_template(
                    query=response.content,
                    history=history,
                    system_prompt=system_prompt,
                    **kwargs
                )
                
                return TaskResponse(
                    status=TaskResult.SUCCESS,
                    content=llm_response,
                    data=response.data
                )
            
            return response
        
        # No handler found, use default LLM
        llm_response = await self._generate_with_template(
            query=query,
            history=history,
            **kwargs
        )
        
        return TaskResponse(
            status=TaskResult.SUCCESS,
            content=llm_response
        )
    
    async def generate_response(
        self,
        prompt: str,
        history: List[Message] = None,
        **kwargs
    ) -> str:
        """
        Generate complete response (non-streaming).
        
        Args:
            prompt: User prompt
            history: Conversation history
        
        Returns:
            Complete response text
        """
        response = await self.process_query(prompt, history, **kwargs)
        return response.content
    
    async def generate_response_stream(
        self,
        prompt: str,
        history: List[Message] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream response as sentence chunks for TTS.
        
        Args:
            prompt: User prompt
            history: Conversation history
        
        Yields:
            Complete sentences/phrases
        """
        if not self._provider:
            yield "LLM not configured"
            return

        # Apply task routing for streaming path as well.
        system_prompt = None
        routed_query = prompt
        context = TaskContext(
            user_query=prompt,
            history=history or [],
            session_data=kwargs.get("session_data", {}),
            metadata=kwargs,
        )
        handler = self._registry.find_handler(context)

        if handler:
            debug_log(f"Using handler for stream: {handler.name}")
            response = await handler.handle(context)

            if response.status == TaskResult.SUCCESS:
                normalized = normalize_llm_output(response.content)
                if normalized:
                    yield normalized
                return

            if response.status == TaskResult.DELEGATED:
                routed_query = response.content
                system_prompt = handler.get_system_prompt() or None

        messages = self._build_messages(
            routed_query,
            history,
            system_prompt=system_prompt,
            **kwargs,
        )
        
        latency.start("llm_first_token")
        first_token = True
        buffer = ""

        THINK_START = "<think>"
        THINK_END = "</think>"
        thinking = False
        pending = ""

        def consume_char(char: str) -> str:
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
                emitted = pending[0]
                pending = pending[1:]
                return emitted

            return ""

        def flush_pending() -> str:
            nonlocal pending
            if thinking or not pending:
                return ""
            if THINK_START.startswith(pending):
                return ""
            emitted = pending
            pending = ""
            return emitted
        
        async for token in self._provider.generate_stream(
            messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature)
        ):
            if first_token:
                latency.end("llm_first_token")
                first_token = False
            
            for char in token:
                visible_char = consume_char(char)
                if visible_char:
                    buffer += visible_char

                flushed = flush_pending()
                if flushed:
                    buffer += flushed
            
            # Check for sentence delimiters
            while True:
                last_pos = max(
                    (buffer.rfind(d) for d in self.SENTENCE_DELIMITERS if d in buffer),
                    default=-1,
                )
                if last_pos <= 0:
                    break

                sentence = buffer[:last_pos + 1].strip()
                buffer = buffer[last_pos + 1:].strip()
                if sentence:
                    normalized = normalize_llm_output(strip_thinking_blocks(sentence))
                    if normalized:
                        yield normalized
        
        # Yield remaining buffer
        trailing = flush_pending()
        if trailing:
            buffer += trailing

        if buffer.strip():
            normalized = normalize_llm_output(strip_thinking_blocks(buffer.strip()))
            if normalized:
                yield normalized
    
    async def generate_tokens(
        self,
        prompt: str,
        history: List[Message] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream individual tokens (for display).
        
        Args:
            prompt: User prompt
            history: Conversation history
        
        Yields:
            Individual tokens
        """
        if not self._provider:
            yield "LLM not configured"
            return
        
        messages = self._build_messages(prompt, history, **kwargs)
        
        async for token in self._provider.generate_stream(
            messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature)
        ):
            yield token
    
    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[ToolDefinition],
        history: List[Message] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response with tool calling.
        
        Args:
            prompt: User prompt
            tools: Available tools
            history: Conversation history
        
        Returns:
            LLMResponse with potential tool calls
        """
        if not self._provider:
            return LLMResponse(content="LLM not configured", finish_reason="error")
        
        if not self._provider.supports_tools():
            # Fallback to regular generation
            response = await self._generate_with_template(prompt, history, **kwargs)
            return LLMResponse(content=response, finish_reason="stop")
        
        messages = self._build_messages(prompt, history, **kwargs)
        
        return await self._provider.generate_with_tools(
            messages,
            tools,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature)
        )
    
    async def _generate_with_template(
        self,
        query: str,
        history: List[Message] = None,
        system_prompt: str = None,
        **kwargs
    ) -> str:
        """Generate response using template."""
        messages = self._build_messages(
            query, history,
            system_prompt=system_prompt,
            **kwargs
        )
        
        with latency.track("llm_generate"):
            response = await self._provider.generate(
                messages,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature)
            )
        
        return normalize_llm_output(response.content)
    
    def _build_messages(
        self,
        query: str,
        history: List[Message] = None,
        system_prompt: str = None,
        **kwargs
    ) -> List[Dict[str, str]]:
        """Build messages for API call."""
        # Use custom system prompt or template
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}]
        elif self._template:
            messages = self._template.build_messages(query, history, **kwargs)
            return messages
        else:
            messages = [{"role": "system", "content": settings.llm.system_prompt}]
        
        # Add history
        if history:
            for msg in history:
                if isinstance(msg, Message):
                    messages.append(msg.to_dict())
                else:
                    messages.append(msg)
        
        # Add user query
        messages.append({"role": "user", "content": query})
        
        return messages
    
    def list_providers(self) -> List[str]:
        """List available providers."""
        return self._registry.list_providers()
    
    def list_templates(self) -> List[str]:
        """List available templates."""
        return self._registry.list_templates()
    
    def list_handlers(self) -> List[str]:
        """List registered handlers."""
        return self._registry.list_handlers()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information."""
        if self._provider:
            return self._provider.get_model_info()
        return {"name": "none", "provider": "none"}


# ============================================================================
# Singleton and Factory
# ============================================================================

_extended_llm_service: Optional[ExtendedLLMService] = None


def get_extended_llm_service() -> ExtendedLLMService:
    """Get or create extended LLM service singleton."""
    global _extended_llm_service
    if _extended_llm_service is None:
        _extended_llm_service = ExtendedLLMService()
    return _extended_llm_service


def create_llm_for_task(
    task_type: str,
    provider: str = "groq",
    **kwargs
) -> ExtendedLLMService:
    """
    Factory function to create LLM service for specific tasks.
    
    Args:
        task_type: Task type ("customer_support", "personal", "qa", etc.)
        provider: Provider name
        **kwargs: Template parameters
    
    Returns:
        Configured ExtendedLLMService
    
    Example:
        >>> llm = create_llm_for_task(
        ...     "customer_support",
        ...     company="ABC Corp",
        ...     products="điện thoại và phụ kiện"
        ... )
    """
    llm = ExtendedLLMService()
    
    # Set provider
    if provider:
        llm.set_provider(provider)
    
    # Set template based on task type
    template_map = {
        "customer_support": "customer_support",
        "support": "customer_support",
        "personal": "personal_assistant",
        "assistant": "personal_assistant",
        "qa": "qa",
        "question": "qa",
        "default": "vietnamese_assistant",
    }
    
    template_name = template_map.get(task_type, "vietnamese_assistant")
    llm.set_template(template_name, **kwargs)
    
    return llm
