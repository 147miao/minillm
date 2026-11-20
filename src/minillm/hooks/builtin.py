"""内置 hook：技能注入与工具审计。"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from ..config import Settings
from ..events import EventType
from ..ids import utc_now_iso
from ..skills.loader import load_skills, render_skill_prompt, select_skills
from .manager import HookContext, HookEvent, HookManager

logger = logging.getLogger("minillm.hooks.builtin")

_SENSITIVE_KEYS = ("key", "token", "secret", "password", "authorization", "cookie")


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(mark in lowered for mark in _SENSITIVE_KEYS):
                redacted[str(key)] = "<redacted>"
            else:
                redacted[str(key)] = _redact(item)
        return redacted
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


def _summarize(value: Any, limit: int = 500) -> str:
    try:
        text = json.dumps(_redact(value), ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001 - 摘要失败不能影响工具
        text = repr(value)
    text = text.replace("\n", " ")
    return text[:limit] + ("..." if len(text) > limit else "")


class SkillsHook:
    """按用户消息关键词选择技能，注入 active_skills / system_parts，并发出事件。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._logger = logging.getLogger("minillm.hooks.skills")

    async def __call__(self, ctx: HookContext, payload: dict[str, Any]) -> dict[str, Any]:
        text = str(payload.get("text") or payload.get("content") or "")
        if not text.strip():
            return payload
        skills = load_skills(self.settings.skills_dir)
        if not skills:
            return payload
        matched = select_skills(text, skills)
        if not matched:
            return payload

        active = payload.setdefault("active_skills", [])
        active_names = set()
        for item in active:
            if isinstance(item, dict):
                active_names.add(str(item.get("name", "")))
            else:
                active_names.add(str(getattr(item, "name", "")))

        new_skills = [skill for skill in matched if skill.name not in active_names]
        if not new_skills:
            return payload

        rendered = render_skill_prompt(new_skills)
        if rendered:
            payload.setdefault("system_parts", []).append(rendered)
        active.extend(skill.to_dict() for skill in new_skills)

        if ctx.emit_event is not None:
            await ctx.emit(
                EventType.SKILLS_INJECTED,
                {"skills": [{"name": skill.name, "path": str(skill.path)} for skill in new_skills]},
            )
        return payload


class AuditHook:
    """把工具调用审计写入 data_dir/audit.jsonl；参数中的敏感字段脱敏。"""

    def __init__(self, settings: Settings, path: Path | None = None) -> None:
        self.settings = settings
        self.path = Path(path) if path is not None else settings.data_dir / "audit.jsonl"
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger("minillm.hooks.audit")
        self._starts: dict[tuple[str, str, str], list[float]] = {}

    @staticmethod
    def _key(ctx: HookContext, payload: dict[str, Any]) -> tuple[str, str, str]:
        return (ctx.session_id, ctx.agent_id or "main", str(payload.get("tool") or ""))

    async def before_tool(self, ctx: HookContext, payload: dict[str, Any]) -> None:
        self._starts.setdefault(self._key(ctx, payload), []).append(time.perf_counter())
        await self._write(
            {
                "ts": utc_now_iso(),
                "session_id": ctx.session_id,
                "agent_id": ctx.agent_id or "main",
                "phase": "before",
                "tool": payload.get("tool"),
                "args": _summarize(payload.get("args")),
            }
        )

    async def after_tool(self, ctx: HookContext, payload: dict[str, Any]) -> None:
        result = payload.get("result")
        ok = getattr(result, "ok", None)
        if ok is None and isinstance(result, dict):
            ok = result.get("ok")
        duration_ms = payload.get("duration_ms")
        if duration_ms is None:
            starts = self._starts.get(self._key(ctx, payload))
            if starts:
                duration_ms = int((time.perf_counter() - starts.pop()) * 1000)
        await self._write(
            {
                "ts": utc_now_iso(),
                "session_id": ctx.session_id,
                "agent_id": ctx.agent_id or "main",
                "phase": "after",
                "tool": payload.get("tool"),
                "ok": ok,
                "duration_ms": duration_ms,
                "result": _summarize(result),
            }
        )

    async def _write(self, record: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False, default=str)
            async with self._lock:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except Exception:  # noqa: BLE001 - 审计失败不影响主流程
            self._logger.exception("写入 audit.jsonl 失败：%s", self.path)


def register_builtin_hooks(
    manager: HookManager,
    settings: Settings,
    *,
    enable_audit: bool = True,
) -> tuple[SkillsHook, AuditHook | None]:
    """注册内置 hook；返回实例便于测试/诊断。"""

    skills_hook = SkillsHook(settings)
    manager.register(HookEvent.SESSION_START, skills_hook)
    manager.register(HookEvent.USER_MESSAGE, skills_hook)

    audit_hook: AuditHook | None = None
    if enable_audit:
        audit_hook = AuditHook(settings)
        manager.register(HookEvent.BEFORE_TOOL, audit_hook.before_tool)
        manager.register(HookEvent.AFTER_TOOL, audit_hook.after_tool)
    return skills_hook, audit_hook
