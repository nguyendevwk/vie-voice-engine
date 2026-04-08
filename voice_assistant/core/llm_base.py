"""
LLM Base Classes and Interfaces.

Provides abstract base classes for implementing custom LLM providers
and task handlers. Designed for easy extension and integration.

Example:
    >>> class MyLLM(BaseLLMProvider):
    ...     async def generate(self, messages):
    ...         return "Response from my LLM"
    
    >>> llm_service = LLMService()
    >>> llm_service.register_provider("my_llm", MyLLM())
"""

import asyncio
from abc import ABC, abstractmethod
from typing import (
    AsyncIterator, List, Dict, Optional, Any, Callable, Awaitable, Union
)
from dataclasses import dataclass, field
from enum import Enum
import json


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Message:
    """Chat message in conversation."""
    role: str  # system, user, assistant, tool
    content: str
    name: Optional[str] = None  # For tool messages
    tool_calls: Optional[List[Dict]] = None  # For assistant tool calls
    tool_call_id: Optional[str] = None  # For tool response
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API-compatible dict."""
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class ToolDefinition:
    """Definition of a callable tool/function."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    handler: Optional[Callable[..., Awaitable[str]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


@dataclass
class LLMResponse:
    """Response from LLM provider."""
    content: str
    finish_reason: str = "stop"  # stop, tool_calls, length, error
    tool_calls: Optional[List[Dict]] = None
    usage: Optional[Dict[str, int]] = None  # tokens used
    raw: Optional[Any] = None  # Raw API response


@dataclass
class TaskContext:
    """Context for task execution."""
    user_query: str
    history: List[Message] = field(default_factory=list)
    session_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskResult(Enum):
    """Result status for task execution."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    NEEDS_INPUT = "needs_input"
    DELEGATED = "delegated"


@dataclass
class TaskResponse:
    """Response from task handler."""
    status: TaskResult
    content: str
    data: Optional[Dict[str, Any]] = None
    next_action: Optional[str] = None
    confidence: float = 1.0


# ============================================================================
# Abstract Base Classes
# ============================================================================

class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Implement this to add support for new LLM backends (local models,
    custom APIs, etc.)
    
    Example:
        >>> class OllamaProvider(BaseLLMProvider):
        ...     def __init__(self, model="llama3"):
        ...         self.model = model
        ...         self.client = ollama.AsyncClient()
        ...     
        ...     async def generate(self, messages, **kwargs):
        ...         response = await self.client.chat(
        ...             model=self.model,
        ...             messages=messages
        ...         )
        ...         return LLMResponse(content=response['message']['content'])
        ...     
        ...     async def generate_stream(self, messages, **kwargs):
        ...         async for chunk in self.client.chat(
        ...             model=self.model,
        ...             messages=messages,
        ...             stream=True
        ...         ):
        ...             yield chunk['message']['content']
    """
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """
        Generate a complete response.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Provider-specific options (temperature, max_tokens, etc.)
        
        Returns:
            LLMResponse with content and metadata
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream response tokens.
        
        Args:
            messages: List of message dicts
            **kwargs: Provider-specific options
        
        Yields:
            Individual tokens or chunks of text
        """
        pass
    
    def supports_tools(self) -> bool:
        """Check if provider supports tool/function calling."""
        return False
    
    def supports_vision(self) -> bool:
        """Check if provider supports vision/images."""
        return False
    
    async def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[ToolDefinition],
        **kwargs
    ) -> LLMResponse:
        """Generate with tool calling support."""
        raise NotImplementedError("Tool calling not supported by this provider")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {"name": "unknown", "context_length": 4096}


class BaseTaskHandler(ABC):
    """
    Abstract base class for task handlers.
    
    Task handlers process specific types of user requests.
    Use this to build custom assistants for specific domains.
    
    Example:
        >>> class WeatherHandler(BaseTaskHandler):
        ...     name = "weather"
        ...     description = "Get weather information"
        ...     
        ...     def can_handle(self, context: TaskContext) -> float:
        ...         keywords = ["thời tiết", "weather", "nhiệt độ", "mưa"]
        ...         query_lower = context.user_query.lower()
        ...         return 0.9 if any(k in query_lower for k in keywords) else 0.0
        ...     
        ...     async def handle(self, context: TaskContext) -> TaskResponse:
        ...         # Call weather API
        ...         weather = await self.get_weather(context.user_query)
        ...         return TaskResponse(
        ...             status=TaskResult.SUCCESS,
        ...             content=f"Thời tiết: {weather}"
        ...         )
    """
    
    name: str = "base"
    description: str = "Base task handler"
    priority: int = 0  # Higher = checked first
    
    @abstractmethod
    def can_handle(self, context: TaskContext) -> float:
        """
        Determine if this handler can process the task.
        
        Args:
            context: Task context with user query and history
        
        Returns:
            Confidence score 0.0 to 1.0 (0 = cannot handle, 1 = perfect match)
        """
        pass
    
    @abstractmethod
    async def handle(self, context: TaskContext) -> TaskResponse:
        """
        Process the task.
        
        Args:
            context: Task context with all relevant information
        
        Returns:
            TaskResponse with result status and content
        """
        pass
    
    def get_system_prompt(self) -> Optional[str]:
        """Get custom system prompt for this handler."""
        return None
    
    def get_tools(self) -> List[ToolDefinition]:
        """Get tools available for this handler."""
        return []


class BasePromptTemplate(ABC):
    """
    Abstract base class for prompt templates.
    
    Use to create reusable prompt structures for different tasks.
    
    Example:
        >>> class CustomerSupportTemplate(BasePromptTemplate):
        ...     def format_system(self, **kwargs) -> str:
        ...         return f'''
        ...         Bạn là nhân viên hỗ trợ khách hàng của {kwargs.get('company', 'công ty')}.
        ...         Hãy trả lời lịch sự và hữu ích.
        ...         '''
        ...     
        ...     def format_user(self, query: str, **kwargs) -> str:
        ...         return f"Khách hàng hỏi: {query}"
    """
    
    name: str = "base"
    
    @abstractmethod
    def format_system(self, **kwargs) -> str:
        """Format system prompt with given parameters."""
        pass
    
    def format_user(self, query: str, **kwargs) -> str:
        """Format user message (default: return as-is)."""
        return query
    
    def format_context(self, context: Dict[str, Any]) -> str:
        """Format additional context to include in prompt."""
        return ""
    
    def build_messages(
        self,
        query: str,
        history: List[Message] = None,
        **kwargs
    ) -> List[Dict[str, str]]:
        """Build complete message list."""
        messages = [{"role": "system", "content": self.format_system(**kwargs)}]
        
        context = kwargs.get("context")
        if context:
            context_str = self.format_context(context)
            if context_str:
                messages[0]["content"] += f"\n\n{context_str}"
        
        if history:
            for msg in history:
                messages.append(msg.to_dict() if isinstance(msg, Message) else msg)
        
        messages.append({"role": "user", "content": self.format_user(query, **kwargs)})
        
        return messages


# ============================================================================
# Built-in Prompt Templates
# ============================================================================

class VietnameseAssistantTemplate(BasePromptTemplate):
    """Default Vietnamese assistant prompt template."""
    
    name = "vietnamese_assistant"
    
    def format_system(self, **kwargs) -> str:
        assistant_name = kwargs.get("assistant_name", "Trợ lý")
        personality = kwargs.get("personality", "thân thiện và hữu ích")
        expertise = kwargs.get("expertise", "")
        
        prompt = f"""Bạn là {assistant_name}, một trợ lý AI {personality}.
Bạn trả lời bằng tiếng Việt, ngắn gọn, rõ ràng.
Giọng văn tự nhiên, phù hợp cho text-to-speech."""
        
        if expertise:
            prompt += f"\nBạn có chuyên môn về: {expertise}."
        
        return prompt
    
    def format_user(self, query: str, **kwargs) -> str:
        return query


class CustomerSupportTemplate(BasePromptTemplate):
    """Customer support prompt template."""
    
    name = "customer_support"
    
    def format_system(self, **kwargs) -> str:
        company = kwargs.get("company", "công ty")
        products = kwargs.get("products", "sản phẩm và dịch vụ")
        policies = kwargs.get("policies", "")
        
        prompt = f"""Bạn là nhân viên hỗ trợ khách hàng của {company}.
Bạn có nhiệm vụ giúp đỡ khách hàng với {products}.

Nguyên tắc:
- Luôn lịch sự, thân thiện
- Trả lời ngắn gọn, dễ hiểu
- Xin lỗi nếu có sự cố
- Hướng dẫn từng bước khi cần
- Không hứa điều không chắc chắn"""
        
        if policies:
            prompt += f"\n\nChính sách: {policies}"
        
        return prompt


class PersonalAssistantTemplate(BasePromptTemplate):
    """Personal assistant prompt template."""
    
    name = "personal_assistant"
    
    def format_system(self, **kwargs) -> str:
        user_name = kwargs.get("user_name", "bạn")
        preferences = kwargs.get("preferences", {})
        
        prompt = f"""Bạn là trợ lý cá nhân của {user_name}.
Bạn giúp quản lý công việc, nhắc nhở và trả lời câu hỏi.

Phong cách:
- Thân thiện như bạn bè
- Ngắn gọn, đi thẳng vào vấn đề
- Chủ động gợi ý khi phù hợp"""
        
        if preferences:
            pref_str = "\n".join(f"- {k}: {v}" for k, v in preferences.items())
            prompt += f"\n\nSở thích của {user_name}:\n{pref_str}"
        
        return prompt
    
    def format_context(self, context: Dict[str, Any]) -> str:
        parts = []
        
        if "tasks" in context:
            tasks_str = "\n".join(f"- {t}" for t in context["tasks"])
            parts.append(f"Công việc cần làm:\n{tasks_str}")
        
        if "calendar" in context:
            parts.append(f"Lịch hôm nay:\n{context['calendar']}")
        
        if "notes" in context:
            parts.append(f"Ghi chú:\n{context['notes']}")
        
        return "\n\n".join(parts)


class QuestionAnswerTemplate(BasePromptTemplate):
    """Q&A knowledge base template."""
    
    name = "qa"
    
    def format_system(self, **kwargs) -> str:
        domain = kwargs.get("domain", "tổng quát")
        knowledge_base = kwargs.get("knowledge_base", "")
        
        prompt = f"""Bạn là chuyên gia trả lời câu hỏi về {domain}.
Trả lời chính xác dựa trên kiến thức có sẵn.
Nếu không biết, hãy nói rõ "Tôi không có thông tin về vấn đề này"."""
        
        if knowledge_base:
            prompt += f"\n\nKiến thức:\n{knowledge_base}"
        
        return prompt


# ============================================================================
# Registry for providers, templates, handlers
# ============================================================================

class ComponentRegistry:
    """Registry for LLM components (providers, templates, handlers)."""
    
    def __init__(self):
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._templates: Dict[str, BasePromptTemplate] = {}
        self._handlers: List[BaseTaskHandler] = []
        
        # Register built-in templates
        self._templates["vietnamese_assistant"] = VietnameseAssistantTemplate()
        self._templates["customer_support"] = CustomerSupportTemplate()
        self._templates["personal_assistant"] = PersonalAssistantTemplate()
        self._templates["qa"] = QuestionAnswerTemplate()
    
    def register_provider(self, name: str, provider: BaseLLMProvider):
        """Register an LLM provider."""
        self._providers[name] = provider
    
    def get_provider(self, name: str) -> Optional[BaseLLMProvider]:
        """Get a registered provider."""
        return self._providers.get(name)
    
    def register_template(self, template: BasePromptTemplate):
        """Register a prompt template."""
        self._templates[template.name] = template
    
    def get_template(self, name: str) -> Optional[BasePromptTemplate]:
        """Get a registered template."""
        return self._templates.get(name)
    
    def register_handler(self, handler: BaseTaskHandler):
        """Register a task handler."""
        self._handlers.append(handler)
        # Sort by priority (highest first)
        self._handlers.sort(key=lambda h: h.priority, reverse=True)
    
    def find_handler(self, context: TaskContext) -> Optional[BaseTaskHandler]:
        """Find best handler for a task context."""
        best_handler = None
        best_confidence = 0.0
        
        for handler in self._handlers:
            confidence = handler.can_handle(context)
            if confidence > best_confidence and confidence > 0.5:  # Threshold
                best_confidence = confidence
                best_handler = handler
        
        return best_handler
    
    def list_providers(self) -> List[str]:
        """List registered provider names."""
        return list(self._providers.keys())
    
    def list_templates(self) -> List[str]:
        """List registered template names."""
        return list(self._templates.keys())
    
    def list_handlers(self) -> List[str]:
        """List registered handler names."""
        return [h.name for h in self._handlers]


# Global registry
_registry = ComponentRegistry()


def get_registry() -> ComponentRegistry:
    """Get the global component registry."""
    return _registry


def register_provider(name: str, provider: BaseLLMProvider):
    """Register a provider in the global registry."""
    _registry.register_provider(name, provider)


def register_template(template: BasePromptTemplate):
    """Register a template in the global registry."""
    _registry.register_template(template)


def register_handler(handler: BaseTaskHandler):
    """Register a handler in the global registry."""
    _registry.register_handler(handler)
