"""从 JSONL 事件重放会话状态，并构造可直接发给 OpenAI 的消息列表。

关键不变量（见 docs/EVENTS.md）：
- assistant 的 tool_calls 必须与紧随其后的 tool 消息成对出现；压缩/裁剪边界
  只允许落在 tool 结果之后，构建消息时会丢弃孤立 tool 与不完整的 tool_calls 组。
- L2 只清除旧 assistant 的 reasoning，并插入一条摘要 system note。
- L3 用摘要 + 最近 kept_message_count 条消息重建视图；原始事件永不删除。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..events import Event, EventType
from .store import SessionStore


@dataclass
class SessionState:
    """重放后的会话视图。"""

    meta: dict[str, Any] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    todos: list[dict[str, Any]] = field(default_factory=list)
    agents: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_skills: list[dict[str, Any]] = field(default_factory=list)
    compactions: list[dict[str, Any]] = field(default_factory=list)
    last_usage: dict[str, Any] = field(default_factory=dict)
    last_seq: int = 0
    pending_runs: dict[str, dict[str, Any]] = field(default_factory=dict)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _public_message(message: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in message.items() if not key.startswith("_")}


def _assistant_message(data: dict[str, Any], seq: int) -> dict[str, Any]:
    tool_calls = data.get("tool_calls") or []
    return {
        "role": "assistant",
        "content": data.get("content") or "",
        "reasoning": data.get("reasoning") or "",
        "tool_calls": list(tool_calls),
        "_seq": seq,
    }


def _run_code_content(data: dict[str, Any]) -> str:
    content = data.get("content")
    if isinstance(content, str) and content:
        return content
    parts: list[str] = []
    stdout = str(data.get("stdout_preview") or "")
    stderr = str(data.get("stderr_preview") or "")
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"[stderr]\n{stderr}")
    if not parts:
        parts.append(f"run_code 结束，exit_code={data.get('exit_code')}")
    return "\n".join(parts)


def _apply_level2(entries: list[dict[str, Any]], data: dict[str, Any]) -> list[dict[str, Any]]:
    upto = _int(data.get("pruned_upto_seq"))
    for message in entries:
        if message.get("role") == "assistant" and _int(message.get("_seq")) <= upto:
            message["reasoning"] = ""
    summary = str(data.get("summary") or "").strip()
    if not summary:
        return entries
    # 只替换旧的 L2 摘要 note；L3 全量摘要必须保留
    entries = [
        message
        for message in entries
        if not (message.get("_compaction_note") and message.get("_compaction_level") == 2)
    ]
    candidates = [
        index
        for index, message in enumerate(entries)
        if _int(message.get("_seq")) > upto and message.get("role") == "user"
    ]
    if not candidates:
        candidates = [
            index
            for index, message in enumerate(entries)
            if _int(message.get("_seq")) > upto and message.get("role") != "tool"
        ]
    if candidates:
        entries.insert(
            candidates[0],
            {
                "role": "system",
                "content": f"[历史上下文摘要] {summary}",
                "_seq": upto,
                "_compaction_note": True,
                "_compaction_level": 2,
            },
        )
    return entries


def _apply_level3(entries: list[dict[str, Any]], data: dict[str, Any]) -> list[dict[str, Any]]:
    kept = max(0, _int(data.get("kept_message_count"), 12))
    tail = entries[-kept:] if kept else []
    summary = str(data.get("summary") or "").strip()
    rebuilt: list[dict[str, Any]] = []
    if summary:
        rebuilt.append(
            {
                "role": "system",
                "content": f"[历史上下文摘要]\n{summary}",
                "_seq": _int(data.get("pruned_upto_seq")),
                "_compaction_note": True,
                "_compaction_level": 3,
            }
        )
    return rebuilt + tail


def reduce_events(meta: dict[str, Any] | None, events: Iterable[Event]) -> SessionState:
    """按顺序重放事件，得到 SessionState。"""

    state = SessionState(meta=dict(meta or {}))
    entries: list[dict[str, Any]] = []

    for event in events:
        state.events.append(event)
        state.last_seq = max(state.last_seq, event.seq)
        data = event.data or {}
        event_type = event.type

        if event_type == EventType.USER_MESSAGE:
            entries.append(
                {"role": "user", "content": data.get("content") or "", "_seq": event.seq}
            )
        elif event_type == EventType.ASSISTANT_MESSAGE:
            entries.append(_assistant_message(data, event.seq))
        elif event_type == EventType.RUN_CODE_STARTED:
            call_id = str(data.get("call_id"))
            state.pending_runs[call_id] = dict(data)
        elif event_type == EventType.RUN_CODE_FINISHED:
            call_id = data.get("call_id")
            entries.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _run_code_content(data),
                    "_seq": event.seq,
                }
            )
            state.pending_runs.pop(str(call_id), None)
        elif event_type == EventType.LLM_USAGE:
            state.last_usage = dict(data)
        elif event_type == EventType.TODO_UPDATED:
            state.todos = list(data.get("items") or [])
        elif event_type == EventType.SKILLS_INJECTED:
            state.active_skills = list(data.get("skills") or [])
        elif event_type == EventType.CONTEXT_COMPACTED:
            state.compactions.append(dict(data))
            level = _int(data.get("level"))
            if level == 2:
                entries = _apply_level2(entries, data)
            elif level == 3:
                entries = _apply_level3(entries, data)
        elif event_type == EventType.SUBAGENT_SPAWNED:
            agent_id = str(data.get("agent_id"))
            record = state.agents.setdefault(agent_id, {"agent_id": agent_id})
            record.update({key: value for key, value in data.items() if key != "agent_id"})
            record.setdefault("status", "running")
        elif event_type == EventType.SUBAGENT_MESSAGE:
            agent_id = str(data.get("agent_id"))
            record = state.agents.setdefault(agent_id, {"agent_id": agent_id})
            record.update({key: value for key, value in data.items() if key != "agent_id"})
        elif event_type == EventType.SUBAGENT_COMPLETED:
            agent_id = str(data.get("agent_id"))
            record = state.agents.setdefault(agent_id, {"agent_id": agent_id})
            record.update({key: value for key, value in data.items() if key != "agent_id"})
            record["status"] = data.get("status") or "completed"

    state.messages = [_public_message(message) for message in entries]
    return state


def _normalize_tool_call(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    call_id = raw.get("id") or ""
    function = raw.get("function")
    if isinstance(function, dict):
        name = function.get("name") or raw.get("name") or ""
        arguments = function.get("arguments", "")
    else:
        name = raw.get("name") or ""
        arguments = raw.get("arguments", "")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _content_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """保证 assistant(tool_calls) 与其 tool 结果相邻成对，丢弃孤立项。"""

    cleaned: list[dict[str, Any]] = []
    index = 0
    total = len(messages)
    while index < total:
        message = messages[index]
        role = message.get("role")
        if role == "tool":
            index += 1
            continue
        if role == "assistant":
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                cleaned.append(message)
                index += 1
                continue
            expected = [call.get("id") for call in tool_calls]
            index += 1
            results: dict[str, dict[str, Any]] = {}
            while index < total and messages[index].get("role") == "tool":
                results[messages[index].get("tool_call_id")] = messages[index]
                index += 1
            if all(call_id and call_id in results for call_id in expected):
                cleaned.append(message)
                cleaned.extend(results[call_id] for call_id in expected)
            continue
        cleaned.append(message)
        index += 1
    return cleaned


def build_messages(
    state: SessionState,
    *,
    include_reasoning: bool = True,
) -> list[dict[str, Any]]:
    """把重放视图转成 OpenAI messages；不生成 system prompt（由 agent 层负责）。"""

    messages: list[dict[str, Any]] = []
    for source in state.messages:
        role = source.get("role")
        if role == "assistant":
            message: dict[str, Any] = {
                "role": "assistant",
                "content": _content_text(source.get("content")),
            }
            tool_calls = [
                normalized
                for normalized in (
                    _normalize_tool_call(call) for call in (source.get("tool_calls") or [])
                )
                if normalized is not None
            ]
            if tool_calls:
                message["tool_calls"] = tool_calls
            reasoning = source.get("reasoning") or source.get("reasoning_content") or ""
            if include_reasoning and reasoning:
                message["reasoning_content"] = reasoning
            messages.append(message)
        elif role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": source.get("tool_call_id"),
                    "content": _content_text(source.get("content")),
                }
            )
        elif role == "system":
            messages.append({"role": "system", "content": _content_text(source.get("content"))})
        elif role == "user":
            messages.append({"role": "user", "content": _content_text(source.get("content"))})
    return _sanitize_messages(messages)


def load_state(session_id: str, *, store: SessionStore | None = None) -> SessionState:
    """读取 meta + 全量事件并重放；store 为空时按当前 Settings 创建。"""

    if store is None:
        from ..config import get_settings
        from .store import SessionStore as _SessionStore

        settings = get_settings()
        store = _SessionStore(settings.data_dir, fsync=settings.wal_fsync)
    return reduce_events(store.meta(session_id), store.load_events(session_id))
