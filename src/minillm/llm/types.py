"""LLM 流式事件与客户端协议。

事件序列形如：
    LLMReasoningDelta* -> LLMTextDelta* -> LLMToolCallDelta* -> LLMDone

客户端实现（真实 / scripted）都遵循 LLMClient 协议，上层只依赖这里的类型。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..errors import LLMError

USER_AGENT = "minillm/0.1.0"


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_openai(cls, usage: Any) -> Usage:
        if usage is None:
            return cls()
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
        if not reasoning:
            reasoning = int(getattr(usage, "reasoning_tokens", 0) or 0)
        return cls(
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            reasoning_tokens=reasoning,
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> Usage:
        data = data or {}
        return cls(
            prompt_tokens=int(data.get("prompt_tokens") or 0),
            completion_tokens=int(data.get("completion_tokens") or 0),
            reasoning_tokens=int(data.get("reasoning_tokens") or 0),
            total_tokens=int(data.get("total_tokens") or 0),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
        }


def parse_tool_arguments(raw: str) -> dict[str, Any]:
    """解析模型给出的 JSON 参数；空串视为空对象，非法 JSON 抛 LLMError。"""

    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(f"工具调用参数不是合法 JSON：{raw[:200]}") from exc
    if not isinstance(parsed, dict):
        raise LLMError("工具调用参数必须是 JSON 对象")
    return parsed


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""

    def to_openai(self) -> dict[str, Any]:
        arguments = self.raw_arguments or json.dumps(self.arguments, ensure_ascii=False)
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": arguments},
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "raw_arguments": self.raw_arguments,
        }


@dataclass(slots=True)
class LLMResult:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "reasoning": self.reasoning,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict(),
            "model": self.model,
        }


# plan 6.2 使用 LLMCompletion 命名；二者语义一致，保留别名减少上层分叉
type LLMCompletion = LLMResult


@dataclass(slots=True)
class LLMTextDelta:
    text: str


@dataclass(slots=True)
class LLMReasoningDelta:
    text: str


@dataclass(slots=True)
class LLMToolCallDelta:
    index: int
    id: str | None = None
    name: str | None = None
    args_fragment: str = ""


@dataclass(slots=True)
class LLMDone:
    result: LLMResult


type LLMStreamEvent = LLMTextDelta | LLMReasoningDelta | LLMToolCallDelta | LLMDone


class LLMClient(Protocol):
    """主/小模型统一协议；实现方负责鉴权头与 usage 归一化。"""

    def stream_chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        session_id: str,
        agent_kind: str = "main",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMStreamEvent]: ...

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        session_id: str | None = None,
        agent_kind: str = "small",
    ) -> LLMResult: ...


def build_request_headers(
    base_headers: Mapping[str, str] | None,
    session_id: str | None,
    agent_kind: str,
) -> dict[str, str]:
    """extra headers 作底，每请求覆盖会话/agent 标识与 UA。"""

    headers = {str(key): str(value) for key, value in (base_headers or {}).items()}
    headers["x-opencode-session"] = f"minillm-{session_id}" if session_id else "minillm-anonymous"
    headers["x-minillm-agent-kind"] = agent_kind
    headers["User-Agent"] = USER_AGENT
    return headers
