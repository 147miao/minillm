"""API 请求模型与显式 HTTP 错误。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class APIError(Exception):
    """路由层主动抛出的 HTTP 错误，避免被通用 500 处理器吞掉。"""

    def __init__(self, status_code: int, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    workspace: str | None = None


class SendMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = ""
