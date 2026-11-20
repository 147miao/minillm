"""MiniLLM 统一异常体系。

原则：异常只携带可安全落盘/返回给模型的信息，不包含 API Key 等敏感内容。
"""

from __future__ import annotations

from typing import Any


class MiniLLMError(Exception):
    """所有可预期错误的基类。"""

    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail: dict[str, Any] = dict(detail or {})

    def to_dict(self) -> dict[str, Any]:
        return {"message": self.message, **self.detail}


class ConfigError(MiniLLMError):
    """配置缺失或非法（如未设置 LLM_API_KEY）。"""


class LLMError(MiniLLMError):
    """模型网关调用失败、流式解析失败、脚本模式响应耗尽。"""


class ToolError(MiniLLMError):
    """工具执行失败；kind 供模型区分校验错误与运行错误。"""

    def __init__(
        self,
        message: str,
        *,
        kind: str = "tool_error",
        tool: str | None = None,
        errors: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.tool = tool
        self.errors: list[dict[str, Any]] = list(errors or [])

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "tool": self.tool,
            "message": self.message,
            "errors": self.errors,
        }


class BridgeError(MiniLLMError):
    """宿主与 run_code 子进程之间的桥接协议错误。"""


class SessionError(MiniLLMError):
    """会话不存在、事件文件损坏、JSONL 不可解析等持久化错误。"""


class MCPError(MiniLLMError):
    """MCP 服务启动、握手或工具调用失败。"""
