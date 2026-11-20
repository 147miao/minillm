"""Scripted LLM 客户端：离线、可重复的测试替身。

脚本 JSON 结构：
{
  "responses": [
    {"match": {"agent_kind": "main"}, "content": "……", "reasoning": "……",
     "tool_calls": [{"name": "run_code", "arguments": {"code": "..."}}],
     "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}
  ],
  "small_responses": [{"content": "摘要"}]
}

每次调用：
- 按 match（agent_kind/session_id/purpose）取一条尚未使用的响应；
- purpose 取值：stream_chat 用 chat，complete 用 small；
- 把请求上下文追加到 <script>.calls.jsonl；
- 已用下标写入 <script>.state.json，跨进程重启继续消费；
- 脚本耗尽抛 LLMError；complete 只消费 small_responses，缺失即报错。
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from ..config import Settings
from ..errors import ConfigError, LLMError
from ..ids import new_id, utc_now_iso
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

_CHUNK_SIZE = 24


class ScriptedLLMClient:
    """从 JSON 脚本读取响应的 LLMClient 实现。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        path: str | os.PathLike[str] | None = None,
        default_model: str = "scripted",
    ) -> None:
        self.settings = settings
        script_path = path or (settings.minillm_script if settings else None)
        script_path = script_path or os.environ.get("MINILLM_SCRIPT")
        if not script_path:
            raise ConfigError("ScriptedLLMClient 需要脚本路径（MINILLM_SCRIPT 或 path=）")
        self.path = Path(script_path).expanduser()
        if not self.path.exists():
            raise ConfigError(f"脚本不存在：{self.path}")
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"脚本不是合法 JSON：{self.path}") from exc
        if not isinstance(document, dict):
            raise ConfigError("脚本根节点必须是 JSON 对象")
        self.responses = self._as_list(document.get("responses"), "responses")
        self.small_responses = self._as_list(document.get("small_responses"), "small_responses")
        self.default_model = default_model
        self.calls_path = self.path.with_name(self.path.name + ".calls.jsonl")
        self.state_path = self.path.with_name(self.path.name + ".state.json")
        self._lock = threading.RLock()
        self._state = self._load_state()

    @staticmethod
    def _as_list(value: Any, name: str) -> list[dict[str, Any]]:
        if value is None:
            return []
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            raise ConfigError(f"脚本字段 {name} 必须是对象数组")
        return list(value)

    # ==== 计数器状态 ====
    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"used": {"responses": [], "small_responses": []}}
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"used": {"responses": [], "small_responses": []}}
        used = state.get("used") if isinstance(state, dict) else None
        if not isinstance(used, dict):
            used = {}
        return {
            "used": {
                "responses": [int(item) for item in used.get("responses", [])],
                "small_responses": [int(item) for item in used.get("small_responses", [])],
            }
        }

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_name(self.state_path.name + f".{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.state_path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:  # pragma: no cover
                    pass

    def _pick(
        self,
        bucket: str,
        *,
        agent_kind: str,
        session_id: str | None,
        purpose: str,
    ) -> dict[str, Any]:
        responses = self.responses if bucket == "responses" else self.small_responses
        if not responses:
            raise LLMError(f"脚本未提供 {bucket} 响应：{self.path}")
        used = set(self._state["used"].get(bucket, []))
        for index, response in enumerate(responses):
            if index in used:
                continue
            match = response.get("match") or {}
            if not isinstance(match, dict):
                raise ConfigError(f"{bucket}[{index}].match 必须是对象")
            if match.get("agent_kind") and match["agent_kind"] != agent_kind:
                continue
            if match.get("session_id") and match["session_id"] != session_id:
                continue
            if match.get("purpose") and match["purpose"] != purpose:
                continue
            self._state["used"].setdefault(bucket, []).append(index)
            self._state["updated_at"] = utc_now_iso()
            self._save_state()
            return response
        raise LLMError(
            f"脚本响应已耗尽：bucket={bucket} agent_kind={agent_kind} "
            f"session_id={session_id} purpose={purpose} script={self.path}"
        )

    def _record(
        self,
        *,
        agent_kind: str,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        headers: dict[str, str],
        purpose: str,
    ) -> None:
        record = {
            "ts": utc_now_iso(),
            "agent_kind": agent_kind,
            "model": model,
            "messages": messages,
            "tools": tools,
            "headers": headers,
            "purpose": purpose,
        }
        with self._lock:
            self.calls_path.parent.mkdir(parents=True, exist_ok=True)
            with self.calls_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

    def _headers(self, session_id: str | None, agent_kind: str) -> dict[str, str]:
        base = self.settings.llm_extra_headers if self.settings is not None else None
        return build_request_headers(base, session_id, agent_kind)

    @staticmethod
    def _chunks(text: str) -> list[str]:
        if not text:
            return []
        return [text[index : index + _CHUNK_SIZE] for index in range(0, len(text), _CHUNK_SIZE)]

    def _build_tool_calls(self, response: dict[str, Any]) -> list[ToolCall]:
        raw_calls = response.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raise ConfigError("响应的 tool_calls 必须是数组")
        calls: list[ToolCall] = []
        for raw in raw_calls:
            if not isinstance(raw, dict):
                raise ConfigError("tool_calls 的元素必须是对象")
            function = raw.get("function") or {}
            name = raw.get("name") or function.get("name") or ""
            arguments = raw.get("arguments", function.get("arguments", {}))
            raw_arguments = (
                arguments
                if isinstance(arguments, str)
                else json.dumps(arguments, ensure_ascii=False)
            )
            call = ToolCall(
                id=str(raw.get("id") or new_id("call")),
                name=str(name),
                arguments=parse_tool_arguments(raw_arguments),
                raw_arguments=raw_arguments,
            )
            calls.append(call)
        return calls

    # ==== LLMClient 协议 ====
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
        del temperature, max_tokens
        headers = self._headers(session_id, agent_kind)
        model_name = model or self.default_model
        response = self._pick(
            "responses", agent_kind=agent_kind, session_id=session_id, purpose="chat"
        )
        self._record(
            agent_kind=agent_kind,
            model=model_name,
            messages=messages,
            tools=tools,
            headers=headers,
            purpose="chat",
        )
        content = str(response.get("content") or "")
        reasoning = str(response.get("reasoning") or "")
        for chunk in self._chunks(reasoning):
            yield LLMReasoningDelta(chunk)
        for chunk in self._chunks(content):
            yield LLMTextDelta(chunk)
        tool_calls = self._build_tool_calls(response)
        for index, call in enumerate(tool_calls):
            yield LLMToolCallDelta(
                index=index,
                id=call.id,
                name=call.name,
                args_fragment=call.raw_arguments,
            )
        finish_reason = response.get("finish_reason") or ("tool_calls" if tool_calls else "stop")
        yield LLMDone(
            LLMResult(
                content=content,
                reasoning=reasoning,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=Usage.from_dict(response.get("usage")),
                model=str(response.get("model") or model_name),
            )
        )

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
        del temperature, max_tokens
        headers = self._headers(session_id, agent_kind)
        model_name = model or self.default_model
        response = self._pick(
            "small_responses", agent_kind=agent_kind, session_id=session_id, purpose="small"
        )
        self._record(
            agent_kind=agent_kind,
            model=model_name,
            messages=messages,
            tools=[],
            headers=headers,
            purpose="small",
        )
        return LLMResult(
            content=str(response.get("content") or ""),
            reasoning=str(response.get("reasoning") or ""),
            finish_reason=response.get("finish_reason") or "stop",
            usage=Usage.from_dict(response.get("usage")),
            model=str(response.get("model") or model_name),
        )

    async def close(self) -> None:
        return None
