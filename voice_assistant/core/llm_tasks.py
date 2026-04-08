"""
Built-in Task Handlers.

Provides common task handlers for voice assistant use cases.
Use as examples for building custom handlers.

Example:
    >>> from voice_assistant.core.llm_tasks import register_all_handlers
    >>> register_all_handlers()
    >>> 
    >>> # Or register specific handlers
    >>> from voice_assistant.core.llm_tasks import QAHandler
    >>> register_handler(QAHandler())
"""

import re
import json
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime

from .llm_base import (
    BaseTaskHandler, TaskContext, TaskResult, TaskResponse,
    ToolDefinition, register_handler, get_registry
)


class GeneralAssistantHandler(BaseTaskHandler):
    """
    General-purpose assistant handler.
    
    Handles queries that don't match specific handlers.
    Acts as fallback with lowest priority.
    """
    
    name = "general"
    description = "General assistant for open-ended queries"
    priority = -1  # Lowest priority (fallback)
    
    def can_handle(self, context: TaskContext) -> float:
        """Always returns low confidence as fallback."""
        return 0.3  # Low but non-zero
    
    async def handle(self, context: TaskContext) -> TaskResponse:
        """Pass query to LLM without special handling."""
        return TaskResponse(
            status=TaskResult.DELEGATED,
            content=context.user_query,
            data={"handler": "general", "action": "delegate_to_llm"}
        )
    
    def get_system_prompt(self) -> str:
        return """Bạn là trợ lý AI thông minh, thân thiện.
Trả lời câu hỏi ngắn gọn, rõ ràng bằng tiếng Việt.
Phù hợp cho text-to-speech, tránh markdown và ký tự đặc biệt."""


class QAHandler(BaseTaskHandler):
    """
    Question & Answer handler.
    
    Handles factual questions about general knowledge.
    """
    
    name = "qa"
    description = "Question answering for factual queries"
    priority = 10
    
    # Keywords indicating factual questions
    QUESTION_PATTERNS = [
        r"\b(là gì|là ai|ở đâu|khi nào|như thế nào|tại sao|bao nhiêu)\b",
        r"\b(what|who|where|when|how|why)\b",
        r"^\s*(hãy|cho biết|giải thích)\b",
        r"\?\s*$",
    ]
    
    def can_handle(self, context: TaskContext) -> float:
        """Check if query is a factual question."""
        query = context.user_query.lower()
        
        for pattern in self.QUESTION_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return 0.8
        
        # Check for question mark
        if "?" in context.user_query:
            return 0.7
        
        return 0.0
    
    async def handle(self, context: TaskContext) -> TaskResponse:
        """Delegate to LLM with Q&A prompt."""
        return TaskResponse(
            status=TaskResult.DELEGATED,
            content=context.user_query,
            data={"handler": "qa", "template": "qa"}
        )
    
    def get_system_prompt(self) -> str:
        return """Bạn là chuyên gia trả lời câu hỏi.
Trả lời chính xác, súc tích dựa trên kiến thức.
Nếu không chắc chắn, hãy nói rõ.
Tránh đưa ra thông tin sai lệch."""


class CustomerSupportHandler(BaseTaskHandler):
    """
    Customer support handler.
    
    Handles product inquiries, complaints, and support requests.
    """
    
    name = "customer_support"
    description = "Handle customer support queries"
    priority = 20
    
    # Keywords for support queries
    SUPPORT_KEYWORDS = [
        "hỗ trợ", "support", "giúp đỡ",
        "lỗi", "error", "bug", "vấn đề", "problem",
        "khiếu nại", "complain", "phàn nàn",
        "hoàn tiền", "refund", "đổi trả",
        "đơn hàng", "order", "giao hàng", "delivery",
        "tài khoản", "account", "đăng nhập", "login",
    ]
    
    def __init__(self, company_info: Dict[str, Any] = None):
        self.company_info = company_info or {}
    
    def can_handle(self, context: TaskContext) -> float:
        """Check if query is support-related."""
        query = context.user_query.lower()
        
        match_count = sum(1 for kw in self.SUPPORT_KEYWORDS if kw in query)
        
        if match_count >= 2:
            return 0.9
        elif match_count == 1:
            return 0.6
        
        return 0.0
    
    async def handle(self, context: TaskContext) -> TaskResponse:
        """Handle support query."""
        return TaskResponse(
            status=TaskResult.DELEGATED,
            content=context.user_query,
            data={
                "handler": "customer_support",
                "template": "customer_support",
                "company_info": self.company_info
            }
        )
    
    def get_system_prompt(self) -> str:
        company = self.company_info.get("name", "công ty")
        return f"""Bạn là nhân viên hỗ trợ khách hàng của {company}.
Hãy trả lời lịch sự, chuyên nghiệp.
Xin lỗi nếu khách gặp vấn đề.
Hướng dẫn giải quyết từng bước.
Nếu không giải quyết được, hướng dẫn liên hệ hotline."""


class TaskManagerHandler(BaseTaskHandler):
    """
    Task/todo management handler.
    
    Handles creating, listing, and managing tasks.
    """
    
    name = "task_manager"
    description = "Manage tasks and todos"
    priority = 15
    
    TASK_KEYWORDS = [
        "task", "công việc", "việc cần làm", "todo",
        "nhắc nhở", "remind", "reminder",
        "thêm", "add", "tạo", "create",
        "danh sách", "list", "liệt kê",
        "hoàn thành", "done", "xong", "complete",
        "xóa", "delete", "remove",
    ]
    
    def __init__(self):
        self._tasks: List[Dict] = []  # In-memory storage
    
    def can_handle(self, context: TaskContext) -> float:
        """Check if query is task-related."""
        query = context.user_query.lower()
        
        match_count = sum(1 for kw in self.TASK_KEYWORDS if kw in query)
        
        if match_count >= 2:
            return 0.85
        elif match_count == 1:
            return 0.5
        
        return 0.0
    
    async def handle(self, context: TaskContext) -> TaskResponse:
        """Handle task management."""
        query = context.user_query.lower()
        
        # Determine action
        if any(kw in query for kw in ["thêm", "add", "tạo", "create"]):
            return await self._add_task(context)
        elif any(kw in query for kw in ["danh sách", "list", "liệt kê"]):
            return await self._list_tasks(context)
        elif any(kw in query for kw in ["xong", "done", "hoàn thành"]):
            return await self._complete_task(context)
        elif any(kw in query for kw in ["xóa", "delete", "remove"]):
            return await self._delete_task(context)
        
        # Delegate to LLM for interpretation
        return TaskResponse(
            status=TaskResult.DELEGATED,
            content=context.user_query,
            data={"handler": "task_manager", "tasks": self._tasks}
        )
    
    async def _add_task(self, context: TaskContext) -> TaskResponse:
        """Add a new task."""
        # Extract task from query (simple approach)
        task_text = context.user_query
        for prefix in ["thêm task", "thêm công việc", "thêm việc", "add task"]:
            if prefix in task_text.lower():
                task_text = task_text.lower().split(prefix, 1)[1].strip()
                break
        
        task = {
            "id": len(self._tasks) + 1,
            "text": task_text,
            "created_at": datetime.now().isoformat(),
            "completed": False
        }
        self._tasks.append(task)
        
        return TaskResponse(
            status=TaskResult.SUCCESS,
            content=f"Đã thêm công việc: {task_text}",
            data={"task": task}
        )
    
    async def _list_tasks(self, context: TaskContext) -> TaskResponse:
        """List all tasks."""
        if not self._tasks:
            return TaskResponse(
                status=TaskResult.SUCCESS,
                content="Chưa có công việc nào."
            )
        
        pending = [t for t in self._tasks if not t["completed"]]
        completed = [t for t in self._tasks if t["completed"]]
        
        lines = []
        if pending:
            lines.append(f"Có {len(pending)} công việc cần làm:")
            for t in pending:
                lines.append(f"- {t['text']}")
        if completed:
            lines.append(f"Đã hoàn thành {len(completed)} công việc.")
        
        return TaskResponse(
            status=TaskResult.SUCCESS,
            content="\n".join(lines),
            data={"pending": pending, "completed": completed}
        )
    
    async def _complete_task(self, context: TaskContext) -> TaskResponse:
        """Mark task as complete."""
        # Simple: mark last pending task
        pending = [t for t in self._tasks if not t["completed"]]
        if pending:
            pending[-1]["completed"] = True
            return TaskResponse(
                status=TaskResult.SUCCESS,
                content=f"Đã hoàn thành: {pending[-1]['text']}",
                data={"task": pending[-1]}
            )
        return TaskResponse(
            status=TaskResult.FAILED,
            content="Không có công việc nào để hoàn thành."
        )
    
    async def _delete_task(self, context: TaskContext) -> TaskResponse:
        """Delete a task."""
        if self._tasks:
            task = self._tasks.pop()
            return TaskResponse(
                status=TaskResult.SUCCESS,
                content=f"Đã xóa: {task['text']}",
                data={"deleted": task}
            )
        return TaskResponse(
            status=TaskResult.FAILED,
            content="Không có công việc nào để xóa."
        )
    
    def get_tools(self) -> List[ToolDefinition]:
        """Tools for task management."""
        return [
            ToolDefinition(
                name="add_task",
                description="Add a new task to the list",
                parameters={
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "Task description"}
                    },
                    "required": ["task"]
                }
            ),
            ToolDefinition(
                name="list_tasks",
                description="List all tasks",
                parameters={"type": "object", "properties": {}}
            ),
            ToolDefinition(
                name="complete_task",
                description="Mark a task as completed",
                parameters={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "integer", "description": "Task ID"}
                    },
                    "required": ["task_id"]
                }
            ),
        ]


class ConversationHandler(BaseTaskHandler):
    """
    Casual conversation handler.
    
    Handles greetings, small talk, and social interactions.
    """
    
    name = "conversation"
    description = "Handle casual conversation and greetings"
    priority = 5
    
    GREETING_PATTERNS = [
        r"^(xin chào|chào|hello|hi|hey)\b",
        r"^(khỏe không|bạn khỏe|how are you)",
        r"^(tạm biệt|goodbye|bye)\b",
        r"^(cảm ơn|thank|thanks)\b",
    ]
    
    def can_handle(self, context: TaskContext) -> float:
        """Check if query is casual conversation."""
        query = context.user_query.lower().strip()
        
        for pattern in self.GREETING_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return 0.9
        
        # Short queries are often casual
        if len(query.split()) <= 3:
            return 0.4
        
        return 0.0
    
    async def handle(self, context: TaskContext) -> TaskResponse:
        """Handle casual conversation."""
        query = context.user_query.lower().strip()
        
        # Direct responses for common phrases
        if re.search(r"^(xin chào|chào|hello|hi)", query):
            return TaskResponse(
                status=TaskResult.SUCCESS,
                content="Xin chào! Tôi có thể giúp gì cho bạn?",
                confidence=0.95
            )
        
        if re.search(r"(cảm ơn|thank)", query):
            return TaskResponse(
                status=TaskResult.SUCCESS,
                content="Không có gì! Bạn cần gì thêm không?",
                confidence=0.95
            )
        
        if re.search(r"(tạm biệt|bye)", query):
            return TaskResponse(
                status=TaskResult.SUCCESS,
                content="Tạm biệt! Hẹn gặp lại bạn.",
                confidence=0.95
            )
        
        # Delegate to LLM for other casual talk
        return TaskResponse(
            status=TaskResult.DELEGATED,
            content=context.user_query,
            data={"handler": "conversation"}
        )
    
    def get_system_prompt(self) -> str:
        return """Bạn là trợ lý thân thiện.
Trả lời tự nhiên, như đang trò chuyện.
Ngắn gọn, vui vẻ."""


class InterruptHandler(BaseTaskHandler):
    """
    Handler for interrupt commands.
    
    Detects when user wants to stop or interrupt.
    """
    
    name = "interrupt"
    description = "Handle interrupt and stop commands"
    priority = 100  # Highest priority
    
    INTERRUPT_PATTERNS = [
        r"^(dừng|stop|cancel|hủy)\b",
        r"^(im đi|quiet|shut up|im lặng)\b",
        r"^(đủ rồi|enough|ok ok)\b",
        r"^(thôi|never mind|bỏ đi)\b",
    ]
    
    def can_handle(self, context: TaskContext) -> float:
        """Check if this is an interrupt command."""
        query = context.user_query.lower().strip()
        
        for pattern in self.INTERRUPT_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return 1.0  # Highest confidence
        
        return 0.0
    
    async def handle(self, context: TaskContext) -> TaskResponse:
        """Handle interrupt."""
        return TaskResponse(
            status=TaskResult.SUCCESS,
            content="Đã dừng. Bạn cần gì khác không?",
            data={"action": "interrupt"},
            next_action="stop_current"
        )


# ============================================================================
# Registration helpers
# ============================================================================

def register_all_handlers():
    """Register all built-in handlers."""
    handlers = [
        InterruptHandler(),
        CustomerSupportHandler(),
        TaskManagerHandler(),
        QAHandler(),
        ConversationHandler(),
        GeneralAssistantHandler(),  # Fallback
    ]
    
    for handler in handlers:
        register_handler(handler)


def create_custom_handler(
    name: str,
    keywords: List[str],
    system_prompt: str,
    priority: int = 10
) -> BaseTaskHandler:
    """
    Factory function to create a simple custom handler.
    
    Example:
        >>> weather_handler = create_custom_handler(
        ...     name="weather",
        ...     keywords=["thời tiết", "weather", "nhiệt độ"],
        ...     system_prompt="Bạn là chuyên gia dự báo thời tiết...",
        ...     priority=15
        ... )
        >>> register_handler(weather_handler)
    """
    
    class CustomHandler(BaseTaskHandler):
        def can_handle(self, context: TaskContext) -> float:
            query = context.user_query.lower()
            matches = sum(1 for kw in keywords if kw in query)
            if matches >= 2:
                return 0.9
            elif matches == 1:
                return 0.6
            return 0.0
        
        async def handle(self, context: TaskContext) -> TaskResponse:
            return TaskResponse(
                status=TaskResult.DELEGATED,
                content=context.user_query,
                data={"handler": name}
            )
        
        def get_system_prompt(self) -> str:
            return system_prompt
    
    handler = CustomHandler()
    handler.name = name
    handler.description = f"Custom handler for {name}"
    handler.priority = priority
    
    return handler
