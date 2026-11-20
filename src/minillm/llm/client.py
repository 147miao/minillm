"""OpenAI 兼容网关客户端（流式 + 非流式）。

- 每次请求携带 x-opencode-session: minillm-<session_id>，Settings 中的 extra
  headers 作为底部配置，被每请求标识覆盖。
- 流式逐片产出 content/reasoning/tool_calls；usage 来自 stream_options.include_usage。
- 异常统一包装为 LLMError，并对消息中的密钥做脱敏。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
from openai import AsyncOpenAI

from ..config import Settings
from ..errors import ConfigError, LLMError, MiniLLMError
from .types import (
    LLMDone,
    LLMReasoningDelta,
    LLMResult,
    LLMStreamEvent,
    LLMTextDelta,
    LLMToolCallDelta,
    ToolCall,
    Usage,
    build_request_headers,
    parse_tool_arguments,
)


def _build_async_http_client(timeout: float) -> httpx.AsyncClient:
    """优先尊重代理环境变量；环境里的 no_proxy 等值被 httpx 判为非法时，
    退化为 trust_env=False，保证本地 OpenAI 兼容网关仍可用。"""

    try:
        return httpx.AsyncClient(timeout=timeout)
    except Exception:  # noqa: BLE001 - 环境代理配置非法时兜底
        return httpx.AsyncClient(timeout=timeout, trust_env=False)


def _delta_reasoning(delta: Any) -> str:
    value = getattr(delta, "reasoning_content", None)
    if value:
        return str(value)
    extra = getattr(delta, "model_extra", None) or {}
    value = extra.get("reasoning_content")
    return str(value) if value else ""


class OpenAICompatClient:
    """真实网关客户端，可用于主模型或小模型。"""

    def __init__(
        self,
        settings: Settings,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        default_model: str | None = None,
    ) -> None:
        self.settings = settings
        self.base_url = base_url or settings.llm_base_url
        secret = api_key if api_key is not None else settings.llm_api_key.get_secret_value()
        if not secret:
            raise ConfigError("未配置 LLM_API_KEY，无法创建 LLM 客户端")
        self._secret = secret
        self.default_model = default_model or settings.llm_model
        self._http_client = _build_async_http_client(settings.llm_timeout)
        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=secret,
            timeout=settings.llm_timeout,
            max_retries=2,
            http_client=self._http_client,
        )

    @classmethod
    def for_small(cls, settings: Settings) -> OpenAICompatClient:
        """小模型客户端；地址/密钥留空时自动回退主模型。"""

        return cls(
            settings,
            base_url=settings.small_base_url,
            api_key=settings.small_api_key.get_secret_value(),
            default_model=settings.small_llm_model,
        )

    # ==== 请求头与错误 ====
    def _headers(self, session_id: str | None, agent_kind: str) -> dict[str, str]:
        return build_request_headers(self.settings.llm_extra_headers, session_id, agent_kind)

    def _safe_error(self, prefix: str, exc: BaseException) -> str:
        message = f"{prefix}: {type(exc).__name__}: {exc}"
        if self._secret:
            message = message.replace(self._secret, "***")
        return message

    # ==== 流式 ====
    async def stream_chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        session_id: str,
        agent_kind: str = "main",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        model_name = model or self.default_model
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_headers": self._headers(session_id, agent_kind),
        }
        if tools:
            kwargs["tools"] = tools
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = self.settings.llm_max_tokens

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        slots: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        usage = Usage()

        try:
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = Usage.from_openai(chunk_usage)
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                if getattr(choice, "finish_reason", None):
                    finish_reason = str(choice.finish_reason)
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                text = getattr(delta, "content", None)
                if text:
                    content_parts.append(str(text))
                    yield LLMTextDelta(str(text))

                reasoning = _delta_reasoning(delta)
                if reasoning:
                    reasoning_parts.append(reasoning)
                    yield LLMReasoningDelta(reasoning)

                for raw_call in getattr(delta, "tool_calls", None) or []:
                    index = int(getattr(raw_call, "index", 0) or 0)
                    slot = slots.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    if getattr(raw_call, "id", None):
                        slot["id"] = str(raw_call.id)
                    function = getattr(raw_call, "function", None)
                    fragment = ""
                    if function is not None:
                        if getattr(function, "name", None):
                            slot["name"] = str(function.name)
                        fragment = str(getattr(function, "arguments", "") or "")
                        slot["arguments"] += fragment
                    yield LLMToolCallDelta(
                        index=index,
                        id=slot["id"] or None,
                        name=slot["name"] or None,
                        args_fragment=fragment,
                    )
        except MiniLLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - 统一包装为 LLMError
            raise LLMError(self._safe_error("LLM 流式请求失败", exc)) from exc

        tool_calls: list[ToolCall] = []
        for index in sorted(slots):
            slot = slots[index]
            if not slot["name"]:
                raise LLMError("LLM 返回了缺少 name 的工具调用分片")
            tool_calls.append(
                ToolCall(
                    id=slot["id"] or f"call_{index}",
                    name=slot["name"],
                    arguments=parse_tool_arguments(slot["arguments"]),
                    raw_arguments=slot["arguments"],
                )
            )

        yield LLMDone(
            LLMResult(
                content="".join(content_parts),
                reasoning="".join(reasoning_parts),
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=usage,
                model=model_name,
            )
        )

    # ==== 非流式（小模型）====
    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        session_id: str | None = None,
        agent_kind: str = "small",
    ) -> LLMResult:
        model_name = model or self.default_model
        try:
            response = await self._client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=False,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers=self._headers(session_id, agent_kind),
            )
        except MiniLLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - 统一包装为 LLMError
            raise LLMError(self._safe_error("LLM 请求失败", exc)) from exc

        choices = getattr(response, "choices", None) or []
        if not choices:
            return LLMResult(
                model=model_name, usage=Usage.from_openai(getattr(response, "usage", None))
            )
        choice = choices[0]
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        reasoning = ""
        if message is not None:
            reasoning = getattr(message, "reasoning_content", None) or ""
            if not reasoning:
                reasoning = (getattr(message, "model_extra", None) or {}).get(
                    "reasoning_content"
                ) or ""
        return LLMResult(
            content=str(content or ""),
            reasoning=str(reasoning or ""),
            finish_reason=getattr(choice, "finish_reason", None),
            usage=Usage.from_openai(getattr(response, "usage", None)),
            model=str(getattr(response, "model", None) or model_name),
        )

    async def close(self) -> None:
        await self._client.close()
