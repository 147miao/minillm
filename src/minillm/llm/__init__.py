"""LLM 客户端：真实 OpenAI 兼容网关与 scripted 测试替身。"""

from .client import OpenAICompatClient
from .scripted import ScriptedLLMClient
from .types import (
    LLMClient,
    LLMCompletion,
    LLMDone,
    LLMReasoningDelta,
    LLMResult,
    LLMStreamEvent,
    LLMTextDelta,
    LLMToolCallDelta,
    ToolCall,
    Usage,
    build_request_headers,
)

__all__ = [
    "LLMClient",
    "LLMCompletion",
    "LLMDone",
    "LLMReasoningDelta",
    "LLMResult",
    "LLMStreamEvent",
    "LLMTextDelta",
    "LLMToolCallDelta",
    "OpenAICompatClient",
    "ScriptedLLMClient",
    "ToolCall",
    "Usage",
    "build_request_headers",
]
