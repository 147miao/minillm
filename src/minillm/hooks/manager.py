"""Hooks 扩展点。

设计目标：
- 与 agent 层解耦：HookContext 只依赖 Settings 与可选 emit 回调。
- 注册顺序即执行顺序，单个 hook 异常隔离（写入 payload["hook_errors"]）。
- before_tool 支持 deny 与参数修正；after_tool 支持替换 ToolResult。
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings


class HookEvent:
    """Hook 事件名（与 plan.md / docs/EVENTS.md 对齐）。"""

    SESSION_START = "session_start"
    USER_MESSAGE = "user_message"
    BEFORE_LLM = "before_llm"
    AFTER_LLM = "after_llm"
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"
    BEFORE_COMPACT = "before_compact"
    AFTER_COMPACT = "after_compact"
    TURN_END = "turn_end"


ALL_HOOK_EVENTS: frozenset[str] = frozenset(
    value
    for name, value in vars(HookEvent).items()
    if not name.startswith("_") and isinstance(value, str)
)


@dataclass(slots=True)
class HookContext:
    """一次 hook 调用所需的上下文；agent 层负责构造。"""

    session_id: str
    settings: Settings
    state: Any = None
    agent_id: str | None = None
    emit_event: Callable[[str, dict[str, Any]], Awaitable[None] | None] | None = None
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("minillm.hooks"))

    async def emit(self, event_type: str, data: dict[str, Any]) -> None:
        """可选的事件出口；没有配置时静默跳过。"""

        if self.emit_event is None:
            return
        result = self.emit_event(event_type, data)
        if inspect.isawaitable(result):
            await result


@dataclass(slots=True)
class ToolDecision:
    """before_tool 的返回值：deny 非空即拒绝，args 合并进调用参数。"""

    deny: str | None = None
    args: dict[str, Any] | None = None


HookFn = Callable[[HookContext, dict[str, Any]], Awaitable[Any] | Any]


@dataclass(frozen=True, slots=True)
class HookRegistration:
    event: str
    fn: HookFn
    name: str


class HookManager:
    """按事件维护 hook 列表，顺序执行并隔离异常。"""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("minillm.hooks")
        self._hooks: dict[str, list[HookRegistration]] = {}

    def register(self, event: str, fn: HookFn) -> None:
        """注册 hook；非法事件、非 callable、同事件重名都会抛 ValueError/TypeError。"""

        if event not in ALL_HOOK_EVENTS:
            raise ValueError(f"未知 hook 事件：{event}")
        if not callable(fn):
            raise TypeError("hook 必须是 callable")
        module = getattr(fn, "__module__", "") or ""
        qualname = getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn)))
        name = f"{module}.{qualname}" if module else qualname
        hooks = self._hooks.setdefault(event, [])
        if any(reg.name == name for reg in hooks):
            raise ValueError(f"hook 重复注册：{event} -> {name}")
        hooks.append(HookRegistration(event=event, fn=fn, name=name))

    def registered(self, event: str) -> tuple[HookFn, ...]:
        if event not in ALL_HOOK_EVENTS:
            raise ValueError(f"未知 hook 事件：{event}")
        return tuple(reg.fn for reg in self._hooks.get(event, ()))

    def clear(self) -> None:
        self._hooks.clear()

    async def emit(
        self,
        event: str,
        ctx: HookContext,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """顺序执行该事件的所有 hook，返回（可能被修改的）payload。

        before_tool：设置为 payload["denied"] 后立即短路，后续 hook 不再执行。
        after_tool：hook 返回非 None 时替换 payload["result"]。
        其他事件：hook 返回 dict 时合并进 payload。
        """

        if event not in ALL_HOOK_EVENTS:
            raise ValueError(f"未知 hook 事件：{event}")
        data: dict[str, Any] = payload if payload is not None else {}
        for reg in tuple(self._hooks.get(event, ())):
            try:
                result = reg.fn(ctx, data)
                if inspect.isawaitable(result):
                    result = await result
                self._apply(event, data, result)
            except Exception as exc:  # noqa: BLE001 - 单个 hook 不能拖垮主流程
                self._logger.exception("hook 执行失败：%s -> %s", event, reg.name)
                errors = data.setdefault("hook_errors", [])
                errors.append(
                    {
                        "event": event,
                        "hook": reg.name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            if event == HookEvent.BEFORE_TOOL and data.get("denied"):
                break
        return data

    @staticmethod
    def _apply(event: str, data: dict[str, Any], result: Any) -> None:
        if result is None:
            return
        if event == HookEvent.BEFORE_TOOL:
            if isinstance(result, ToolDecision):
                if result.args:
                    data.setdefault("args", {}).update(result.args)
                if result.deny:
                    data["denied"] = result.deny
            elif isinstance(result, dict):
                args = result.get("args")
                if isinstance(args, dict) and args:
                    data.setdefault("args", {}).update(args)
                deny = result.get("deny")
                if deny:
                    data["denied"] = str(deny)
            return
        if event == HookEvent.AFTER_TOOL:
            data["result"] = result
            return
        if isinstance(result, dict):
            data.update(result)
