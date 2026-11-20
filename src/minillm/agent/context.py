"""三级上下文压缩（L1 在工具层，L2/L3 在这里）。"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..errors import LLMError
from ..events import EventType
from ..llm.types import LLMClient, LLMResult
from ..persistence.replay import SessionState

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]")
_EMIT = Callable[[str, dict[str, Any]], Awaitable[None]]


@dataclass(slots=True)
class ContextView:
    """一次请求前计算出的消息视图与压缩元数据。"""

    messages: list[dict[str, Any]]
    summary: str = ""
    level: int = 0
    compacted: bool = False
    before_tokens: int = 0
    after_tokens: int = 0
    fallback: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：CJK 2 chars/token，其他 4 chars/token。"""

    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    other = max(0, len(text) - cjk)
    return max(1, math.ceil(cjk / 2 + other / 4))


def estimate_messages_tokens(messages: Iterable[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        total += 4
        content = message.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif content:
            total += estimate_tokens(json.dumps(content, ensure_ascii=False))
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        if isinstance(reasoning, str):
            total += estimate_tokens(reasoning)
        calls = message.get("tool_calls") or []
        for call in calls:
            if isinstance(call, dict):
                function = call.get("function") or {}
                total += estimate_tokens(str(function.get("name") or ""))
                total += estimate_tokens(str(function.get("arguments") or ""))
    return total


def _used_tokens(
    last_usage: dict[str, Any] | None,
    messages: list[dict[str, Any]],
    system_tokens: int,
) -> int:
    prompt = 0
    if isinstance(last_usage, dict):
        prompt = int(last_usage.get("prompt_tokens") or 0)
    estimate = estimate_messages_tokens(messages) + max(0, system_tokens)
    return max(prompt, estimate)


def _safe_cut(messages: list[dict[str, Any]], keep_recent: int) -> int:
    """只在 user 消息前或完整 tool 组之后切，避免拆散 tool_calls 对。"""

    keep = max(1, int(keep_recent))
    user_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "user"
    ]
    if len(user_indices) > keep:
        cut = user_indices[-keep]
        return cut if 0 < cut < len(messages) else 0

    safe: list[int] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant" and not message.get("tool_calls"):
            safe.append(index + 1)
        if role == "tool":
            next_role = messages[index + 1].get("role") if index + 1 < len(messages) else None
            if next_role != "tool":
                safe.append(index + 1)
    boundary = max(0, len(messages) - max(6, keep * 2))
    candidates = sorted({index for index in safe if 0 < index <= boundary})
    return candidates[-1] if candidates else 0


def _event_seqs(state: SessionState, messages: list[dict[str, Any]], cut: int) -> tuple[int, int]:
    """返回 (pruned_upto_seq, cut_user_seq)；只用于事件审计。"""

    user_seqs = [event.seq for event in state.events if event.type == EventType.USER_MESSAGE]
    assistant_seqs = [
        event.seq for event in state.events if event.type == EventType.ASSISTANT_MESSAGE
    ]
    cut_user_seq = 0
    if user_seqs:
        # messages 中 cut 之前的 user 数
        users_before = sum(1 for message in messages[:cut] if message.get("role") == "user")
        if users_before < len(user_seqs):
            cut_user_seq = user_seqs[users_before]
    pruned = 0
    if cut_user_seq:
        prior = [seq for seq in assistant_seqs if seq < cut_user_seq]
        pruned = prior[-1] if prior else 0
    else:
        pruned = assistant_seqs[-1] if assistant_seqs else 0
    return pruned, cut_user_seq


def _render_transcript(messages: list[dict[str, Any]], limit: int = 60_000) -> str:
    lines: list[str] = []
    used = 0
    for message in messages:
        role = str(message.get("role") or "?")
        content = str(message.get("content") or "")
        reasoning = str(message.get("reasoning") or message.get("reasoning_content") or "")
        parts = [f"[{role}] {content}"]
        if reasoning:
            parts.append(f"(推理) {reasoning}")
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict):
                function = call.get("function") or {}
                parts.append(f"(调用) {function.get('name')} {function.get('arguments')}")
        if role == "tool":
            parts.append(f"(tool_call_id={message.get('tool_call_id')})")
        text = "\n".join(parts)
        if used + len(text) > limit:
            text = text[: max(0, limit - used)]
        lines.append(text)
        used += len(text)
        if used >= limit:
            break
    return "\n\n".join(lines)


def _summary_system(summary: str) -> dict[str, Any]:
    return {"role": "system", "content": f"[历史上下文摘要] {summary}"}


def _history_text(messages: list[dict[str, Any]], cut: int) -> str:
    return _render_transcript(messages[:cut])


def _fallback_summary(messages: list[dict[str, Any]], cut: int) -> str:
    files: list[str] = []
    for message in messages[:cut]:
        content = str(message.get("content") or "")
        for match in re.findall(
            r"[A-Za-z0-9_./-]+\.(?:py|ts|tsx|js|json|md|txt|yaml|yml|toml)", content
        ):
            if match not in files:
                files.append(match)
    extra = ("；涉及文件：" + "、".join(files[:8])) if files else ""
    return f"早期 {cut} 条消息已省略（小模型摘要不可用）{extra}。最近步骤保留，继续完成当前任务。"


class ContextManager:
    """按 usage 阈值执行 L2/L3；L1 由工具层完成。"""

    def __init__(self, settings: Settings, small_llm: LLMClient | None = None) -> None:
        self.settings = settings
        self.small_llm = small_llm

    async def prepare(
        self,
        state: SessionState,
        messages: list[dict[str, Any]],
        last_usage: dict[str, Any] | None,
        *,
        session_id: str,
        emit_event: _EMIT | None = None,
        system_tokens: int = 0,
    ) -> ContextView:
        messages = [dict(message) for message in messages]
        budget = max(1, int(self.settings.context_budget))
        used = _used_tokens(last_usage, messages, system_tokens)
        prune_at = budget * float(self.settings.prune_threshold)
        compact_at = budget * float(self.settings.compact_threshold)
        if used < prune_at:
            return ContextView(
                messages=messages,
                before_tokens=used,
                after_tokens=used,
            )

        cut = _safe_cut(messages, self.settings.keep_recent_turns)
        if cut <= 0 or cut >= len(messages):
            return ContextView(
                messages=messages,
                before_tokens=used,
                after_tokens=used,
            )

        pruned_seq, cut_user_seq = _event_seqs(state, messages, cut)
        level = 3 if used >= compact_at else 2
        summary = ""
        fallback: str | None = None
        try:
            summary = await self._summarize(
                level=level,
                messages=messages,
                cut=cut,
                session_id=session_id,
                emit_event=emit_event,
            )
        except Exception as exc:  # noqa: BLE001 - 压缩失败必须退化，不阻塞对话
            logger.warning("上下文压缩失败，使用确定性摘要：%s", exc)
            fallback = "small_model_error"
        if not summary:
            summary = _fallback_summary(messages, cut)
            fallback = fallback or "small_model_error"

        if level == 3:
            new_messages = [_summary_system(summary)] + messages[cut:]
        else:
            old = [self._clear_reasoning(dict(message)) for message in messages[:cut]]
            new_messages = old + [_summary_system(summary)] + messages[cut:]
        kept = len(new_messages) - (1 if level == 3 else len(old))

        after = estimate_messages_tokens(new_messages) + max(0, system_tokens)
        if level == 3 and after >= compact_at:
            new_messages, after = self._hard_truncate(
                new_messages, summary, system_tokens, compact_at
            )
            fallback = "hard_truncate"
            kept = max(0, len(new_messages) - 1)

        if after >= used and fallback is None:
            fallback = "no_token_reduction"

        data = {
            "level": level,
            "before_tokens": used,
            "after_tokens": after,
            "summary": summary,
            "pruned_upto_seq": pruned_seq,
            "kept_message_count": kept,
            "fallback": bool(fallback),
            "fallback_reason": fallback,
            "cut_user_seq": cut_user_seq,
        }
        await self._emit(emit_event, EventType.CONTEXT_COMPACTED, data)
        return ContextView(
            messages=new_messages,
            summary=summary,
            level=level,
            compacted=True,
            before_tokens=used,
            after_tokens=after,
            fallback=fallback,
            meta=data,
        )

    @staticmethod
    def _clear_reasoning(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("role") == "assistant":
            message.pop("reasoning", None)
            message.pop("reasoning_content", None)
        return message

    def _hard_truncate(
        self,
        messages: list[dict[str, Any]],
        summary: str,
        system_tokens: int,
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        keep = max(1, int(self.settings.keep_recent_turns) // 2)
        current = messages
        for _ in range(3):
            cut = _safe_cut(
                current[1:] if current and current[0].get("role") == "system" else current, keep
            )
            if cut <= 0:
                break
            offset = 1 if current and current[0].get("role") == "system" else 0
            new_messages = [_summary_system(summary)] + current[offset + cut :]
            after = estimate_messages_tokens(new_messages) + max(0, system_tokens)
            current = new_messages
            if after < limit:
                return current, after
            keep = max(1, keep - 1)
        after = estimate_messages_tokens(current) + max(0, system_tokens)
        return current, after

    async def _summarize(
        self,
        *,
        level: int,
        messages: list[dict[str, Any]],
        cut: int,
        session_id: str,
        emit_event: _EMIT | None,
    ) -> str:
        if self.small_llm is None:
            raise LLMError("未配置小模型，无法压缩")
        history = _history_text(messages, cut)
        if level == 3:
            instruction = (
                "把下面的早期 Coding Agent 会话压缩成结构化摘要，不超过 500 字："
                "目标、已完成、关键文件/命令、工具结论、未决问题。不要编造。"
            )
        else:
            instruction = (
                "把下面早期 Coding Agent 步骤的思维链压缩成不超过 200 字的推理摘要："
                "保留目标、关键发现、已完成操作、未决问题。不要编造。"
            )
        result: LLMResult = await self.small_llm.complete(
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": history or "（无历史内容）"},
            ],
            temperature=0.0,
            max_tokens=512,
            session_id=session_id,
            agent_kind="small",
        )
        usage = result.usage.to_dict()
        await self._emit(
            emit_event,
            EventType.LLM_USAGE,
            {**usage, "model": result.model, "purpose": "summary"},
        )
        return str(result.content or "").strip()

    @staticmethod
    async def _emit(
        emit_event: _EMIT | None,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        if emit_event is None:
            return
        try:
            await emit_event(event_type, data)
        except Exception:  # noqa: BLE001 - 审计失败不影响压缩
            logger.exception("压缩事件写入失败：%s", event_type)
