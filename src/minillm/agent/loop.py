"""ReAct 主循环、Hooks 感知的注册表包装与 run_code 桥接。"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from ..artifacts import ArtifactStore
from ..config import Settings
from ..errors import ConfigError, LLMError, ToolError
from ..events import Event, EventBus, EventType
from ..hooks import HookContext, HookEvent, HookManager
from ..ids import new_id, utc_now_iso
from ..llm.types import (
    LLMClient,
    LLMDone,
    LLMReasoningDelta,
    LLMResult,
    LLMTextDelta,
)
from ..locks import FileLockManager
from ..persistence.replay import SessionState, build_messages
from ..persistence.store import SessionStore
from ..tools.base import ToolContext, ToolResult
from ..tools.registry import ToolRegistry
from .context import ContextManager, ContextView, estimate_tokens
from .prompts import RUN_CODE_TOOL, build_system_prompt

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RunCodeResult:
    """WP3 executor 的返回形状；这里只做结构对齐，不 import runtime。"""

    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    content: str = ""
    artifact: str | None = None
    duration_ms: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    truncated: bool = False
    timed_out: bool = False
    cancelled: bool = False
    error: str | None = None


@runtime_checkable
class RunCodeRunner(Protocol):
    async def execute(self, *, code: str, call_id: str, ctx: ToolContext) -> Any: ...


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class HookAwareRegistry:
    """在 ToolRegistry 外包一层 Hooks；executor 仍按 ToolRegistry 使用。"""

    def __init__(self, registry: ToolRegistry, hooks: HookManager, settings: Settings) -> None:
        self.registry = registry
        self.hooks = hooks
        self.settings = settings
        self.state: SessionState | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.registry, name)

    async def invoke(self, name: str, raw_args: Any, ctx: ToolContext) -> ToolResult:
        spec = self.registry.get(name)
        if spec.kind == "subagent" and not ctx.allow_subagents:
            raise ToolError("子 agent 不能再创建子 agent", kind="denied", tool=name)
        if isinstance(raw_args, BaseModel):
            args = raw_args.model_dump()
        elif isinstance(raw_args, dict):
            args = dict(raw_args)
        elif raw_args is None:
            args = {}
        else:
            raise ToolError("工具参数必须是 JSON 对象", kind="validation", tool=name)

        hook_ctx = HookContext(
            session_id=ctx.session_id,
            settings=ctx.settings,
            state=self.state,
            agent_id=ctx.agent_id,
            emit_event=ctx.emit,
        )
        before = await self.hooks.emit(
            HookEvent.BEFORE_TOOL, hook_ctx, {"tool": name, "args": args}
        )
        if before.get("denied"):
            raise ToolError(str(before["denied"]), kind="denied", tool=name)
        args = before.get("args") if isinstance(before.get("args"), dict) else args

        started = time.perf_counter()
        result = await self.registry.invoke(name, args, ctx)
        duration_ms = int((time.perf_counter() - started) * 1000)
        after = await self.hooks.emit(
            HookEvent.AFTER_TOOL,
            hook_ctx,
            {"tool": name, "args": args, "result": result, "duration_ms": duration_ms},
        )
        final = after.get("result", result)
        if isinstance(final, ToolResult):
            return final
        return ToolResult(ok=True, value=final)


class AgentLoop:
    """一次会话的 ReAct 循环；每个 turn 调用一次 run_turn。"""

    def __init__(
        self,
        settings: Settings,
        store: SessionStore,
        bus: EventBus,
        registry: ToolRegistry,
        hooks: HookManager,
        llm: LLMClient,
        context_manager: ContextManager,
        *,
        session_id: str,
        agent_id: str = "main",
        kind: str = "main",
        allow_subagents: bool = True,
        workspace: Path | None = None,
        locks: FileLockManager | None = None,
        artifacts: ArtifactStore | None = None,
        runner: RunCodeRunner | None = None,
        seed_messages: list[dict[str, Any]] | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.bus = bus
        self.registry = registry
        self.hooks = hooks
        self.llm = llm
        self.context_manager = context_manager
        self.session_id = session_id
        self.agent_id = agent_id
        self.kind = kind
        self.allow_subagents = allow_subagents
        self.workspace = Path(workspace) if workspace else self._workspace_from_meta()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.locks = locks or FileLockManager()
        self.artifacts = artifacts or ArtifactStore(store, session_id, settings)
        self.runner = runner
        self.seed_messages = list(seed_messages or [])
        self._seed_consumed = False
        self.state: SessionState | None = None
        self.hook_registry = HookAwareRegistry(registry, hooks, settings)

    def _workspace_from_meta(self) -> Path:
        meta = self.store.meta(self.session_id)
        raw = meta.get("workspace")
        if raw:
            return Path(str(raw)).expanduser().resolve()
        return (Path(self.settings.workspace_root) / self.session_id).resolve()

    # ==== 事件 ====
    def _record(self, event_type: str, data: dict[str, Any] | None = None) -> Event:
        event = self.store.append(self.session_id, event_type, data or {})
        self.bus.publish(event)
        return event

    async def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        self._record(event_type, data)

    def _publish_delta(self, kind: str, text: str) -> None:
        if not text:
            return
        event_type = "reasoning.delta" if kind == "reasoning" else "assistant.delta"
        self.bus.publish(
            Event(
                seq=0,
                ts=utc_now_iso(),
                session_id=self.session_id,
                type=event_type,
                data={"kind": kind, "text": text},
            )
        )

    def _hook_ctx(self) -> HookContext:
        return HookContext(
            session_id=self.session_id,
            settings=self.settings,
            state=self.state,
            agent_id=self.agent_id,
            emit_event=self._emit,
        )

    # ==== LLM ====
    async def _call_llm(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResult:
        final: LLMResult | None = None
        async for event in self.llm.stream_chat(
            messages=messages,
            tools=tools,
            session_id=self.session_id,
            agent_kind=self.kind,
        ):
            if isinstance(event, LLMTextDelta):
                self._publish_delta("text", event.text)
            elif isinstance(event, LLMReasoningDelta):
                self._publish_delta("reasoning", event.text)
            elif isinstance(event, LLMDone):
                final = event.result
        if final is None:
            raise LLMError("模型流没有返回 LLMDone")
        return final

    # ==== prompt 与压缩 ====
    def _compose_messages(
        self,
        state: SessionState,
        view: ContextView,
        *,
        active_skills: list[dict[str, Any]],
        skill_parts: list[str],
        seed: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str]:
        context_systems = [
            str(message.get("content") or "")
            for message in view.messages
            if message.get("role") == "system"
        ]
        conversation = [
            dict(message) for message in view.messages if message.get("role") != "system"
        ]
        if seed:
            insert_at = len(conversation)
            for index in range(len(conversation) - 1, -1, -1):
                if conversation[index].get("role") == "user":
                    insert_at = index
                    break
            conversation = (
                conversation[:insert_at] + [dict(item) for item in seed] + conversation[insert_at:]
            )
        context_info = {
            "budget": self.settings.context_budget,
            "tokens": view.after_tokens,
            "level": view.level,
        }
        prompt = build_system_prompt(
            self.settings,
            self.hook_registry,
            workspace=self.workspace,
            active_skills=active_skills,
            todos=state.todos,
            is_subagent=self.kind == "subagent",
            context_info=context_info,
            skill_parts=skill_parts,
            context_summary="\n\n".join(part for part in context_systems if part),
        )
        return [{"role": "system", "content": prompt}] + conversation, prompt

    # ==== runner ====
    def _ensure_runner(self) -> RunCodeRunner:
        if self.runner is None:
            try:
                from ..runtime.executor import RunCodeExecutor
            except ImportError as exc:  # pragma: no cover - WP3 未落地时
                raise ConfigError(
                    "run_code executor 尚未就绪（缺少 minillm.runtime.executor）"
                ) from exc
            self.runner = RunCodeExecutor(self.settings, self.hook_registry)
        return self.runner

    def _tool_context(self, call_id: str) -> ToolContext:
        return ToolContext(
            session_id=self.session_id,
            workspace=self.workspace,
            data_dir=Path(self.settings.data_dir),
            settings=self.settings,
            locks=self.locks,
            artifacts=self.artifacts,
            agent_id=self.agent_id,
            kind="subagent" if self.kind == "subagent" else "main",
            allow_subagents=self.allow_subagents,
            emit=self._emit,
            call_id=call_id,
        )

    # ==== 工具结果 ====
    def _result_content(self, result: Any) -> str:
        content = str(_attr(result, "content", "") or "")
        if not content:
            stdout = str(_attr(result, "stdout", "") or "")
            stderr = str(_attr(result, "stderr", "") or "")
            exit_code = int(_attr(result, "exit_code", -1) or 0)
            content = f"[run_code exit={exit_code}]\n{stdout}"
            if stderr:
                content += f"\n[stderr]\n{stderr}"
        limit = max(1, int(self.settings.run_code_output_chars))
        if len(content) > limit:
            clipped, artifact = self.artifacts.cap_text("run_code", content, max_chars=limit)
            if artifact:
                return clipped
            return content[:limit]
        return content

    def _finished_data(self, call_id: str, result: Any, content: str) -> dict[str, Any]:
        return {
            "call_id": call_id,
            "exit_code": int(_attr(result, "exit_code", -1) or -1),
            "stdout_preview": str(_attr(result, "stdout", "") or "")[:2000],
            "stderr_preview": str(_attr(result, "stderr", "") or "")[:2000],
            "content": content,
            "artifact": _attr(result, "artifact", None),
            "duration_ms": int(_attr(result, "duration_ms", 0) or 0),
            "tool_calls": list(_attr(result, "tool_calls", []) or []),
            "truncated": bool(_attr(result, "truncated", False)),
            "timed_out": bool(_attr(result, "timed_out", False)),
            "cancelled": bool(_attr(result, "cancelled", False)),
            "error": _attr(result, "error", None),
            "synthetic": False,
        }

    async def _run_code(self, call: Any, error_budget: int = 3) -> tuple[bool, int]:
        """执行一个 run_code 调用；返回 (是否基础设施错误, 连续错误数)。"""

        call_id = str(getattr(call, "id", "") or new_id("call"))
        arguments = getattr(call, "arguments", None)
        code = ""
        if isinstance(arguments, dict):
            raw = arguments.get("code")
            if isinstance(raw, str):
                code = raw
        if not code.strip():
            self._record(
                EventType.RUN_CODE_FINISHED,
                {
                    "call_id": call_id,
                    "exit_code": -1,
                    "stdout_preview": "",
                    "stderr_preview": "run_code 缺少 code 参数",
                    "content": "[run_code error] 缺少 code 参数，无法执行。",
                    "artifact": None,
                    "duration_ms": 0,
                    "tool_calls": [],
                    "synthetic": False,
                },
            )
            return True, error_budget + 1

        start_seq = self.state.last_seq if self.state else 0
        ctx = self._tool_context(call_id)
        self.hook_registry.state = self.state
        try:
            result = await self._ensure_runner().execute(code=code, call_id=call_id, ctx=ctx)
        except Exception as exc:  # noqa: BLE001 - 执行器故障也要回给模型
            logger.exception("run_code 执行失败")
            events = self.store.load_events(self.session_id, after=start_seq)
            if not any(event.type == EventType.RUN_CODE_STARTED for event in events):
                self._record(EventType.RUN_CODE_STARTED, {"call_id": call_id, "code": code})
            self._record(
                EventType.RUN_CODE_FINISHED,
                {
                    "call_id": call_id,
                    "exit_code": -1,
                    "stdout_preview": "",
                    "stderr_preview": f"{type(exc).__name__}: {exc}",
                    "content": f"[run_code error] {type(exc).__name__}: {exc}",
                    "artifact": None,
                    "duration_ms": 0,
                    "tool_calls": [],
                    "synthetic": False,
                },
            )
            self.state = self.store.load_state(self.session_id)
            return True, error_budget + 1

        content = self._result_content(result)
        events = self.store.load_events(self.session_id, after=start_seq)
        has_started = any(
            event.type == EventType.RUN_CODE_STARTED and str(event.data.get("call_id")) == call_id
            for event in events
        )
        has_finished = any(
            event.type == EventType.RUN_CODE_FINISHED and str(event.data.get("call_id")) == call_id
            for event in events
        )
        if not has_started:
            self._record(EventType.RUN_CODE_STARTED, {"call_id": call_id, "code": code})
        if not has_finished:
            self._record(EventType.RUN_CODE_FINISHED, self._finished_data(call_id, result, content))
        self.state = self.store.load_state(self.session_id)

        infra_error = bool(
            _attr(result, "timed_out", False)
            or _attr(result, "cancelled", False)
            or _attr(result, "error", None)
        )
        return infra_error, (error_budget + 1 if infra_error else 0)

    # ==== 主循环 ====
    async def run_turn(
        self,
        user_text: str,
        *,
        seed_messages: list[dict[str, Any]] | None = None,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        self.state = await self.store.recover(self.session_id)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.hook_registry.state = self.state
        turn_id = turn_id or new_id("turn")
        seed = seed_messages
        if seed is None and not self._seed_consumed and self.seed_messages:
            seed = self.seed_messages
        self._seed_consumed = True

        self._record(EventType.TURN_STARTED, {"turn_id": turn_id})
        self._record(EventType.USER_MESSAGE, {"content": user_text})
        self.state = self.store.load_state(self.session_id)
        try:
            payload = await self.hooks.emit(
                HookEvent.USER_MESSAGE,
                self._hook_ctx(),
                {"text": user_text, "active_skills": [], "system_parts": []},
            )
        except Exception:  # noqa: BLE001 - hook 层已隔离，这里兜底
            logger.exception("user_message hook 失败")
            payload = {"active_skills": [], "system_parts": []}
        self.state = self.store.load_state(self.session_id)
        active_skills = list(payload.get("active_skills") or self.state.active_skills or [])
        skill_parts = [str(part) for part in (payload.get("system_parts") or [])]

        status = "max_steps"
        final = ""
        steps = 0
        consecutive_errors = 0
        cancelled = False
        try:
            async with asyncio.timeout(max(1, int(self.settings.turn_timeout))):
                while steps < max(1, int(self.settings.max_steps)):
                    self.state = self.store.load_state(self.session_id)
                    self.hook_registry.state = self.state
                    conversation = build_messages(
                        self.state, include_reasoning=self.settings.include_reasoning
                    )
                    prelim = build_system_prompt(
                        self.settings,
                        self.hook_registry,
                        workspace=self.workspace,
                        active_skills=active_skills,
                        todos=self.state.todos,
                        is_subagent=self.kind == "subagent",
                        context_info=None,
                        skill_parts=skill_parts,
                    )
                    view = await self.context_manager.prepare(
                        self.state,
                        conversation,
                        self.state.last_usage,
                        session_id=self.session_id,
                        emit_event=self._emit,
                        system_tokens=estimate_tokens(prelim),
                    )
                    self.state = self.store.load_state(self.session_id)
                    outgoing, prompt = self._compose_messages(
                        self.state,
                        view,
                        active_skills=active_skills,
                        skill_parts=skill_parts,
                        seed=seed or [],
                    )
                    seed = []  # seed 只插一次
                    before = await self.hooks.emit(
                        HookEvent.BEFORE_LLM,
                        self._hook_ctx(),
                        {"messages": outgoing, "tools": [RUN_CODE_TOOL]},
                    )
                    messages = (
                        before.get("messages")
                        if isinstance(before.get("messages"), list)
                        else outgoing
                    )
                    tools = (
                        before.get("tools")
                        if isinstance(before.get("tools"), list)
                        else [RUN_CODE_TOOL]
                    )
                    result = await self._call_llm(messages, tools)
                    steps += 1
                    await self.hooks.emit(HookEvent.AFTER_LLM, self._hook_ctx(), {"result": result})
                    self._record(
                        EventType.ASSISTANT_MESSAGE,
                        {
                            "content": result.content,
                            "reasoning": result.reasoning,
                            "tool_calls": [call.to_openai() for call in result.tool_calls],
                            "finish_reason": result.finish_reason,
                        },
                    )
                    usage = result.usage.to_dict()
                    self._record(
                        EventType.LLM_USAGE,
                        {**usage, "model": result.model, "purpose": "chat"},
                    )
                    self.state = self.store.load_state(self.session_id)

                    if not result.tool_calls:
                        final = result.content
                        status = "completed"
                        break

                    for call in result.tool_calls:
                        name = str(getattr(call, "name", "") or "")
                        if name != "run_code":
                            self._record(
                                EventType.RUN_CODE_FINISHED,
                                {
                                    "call_id": str(getattr(call, "id", "") or new_id("call")),
                                    "exit_code": -1,
                                    "stdout_preview": "",
                                    "stderr_preview": f"不支持的 tool：{name}",
                                    "content": f"[run_code error] 只能调用 run_code，收到 {name}",
                                    "artifact": None,
                                    "duration_ms": 0,
                                    "tool_calls": [],
                                    "synthetic": False,
                                },
                            )
                            consecutive_errors += 1
                            if consecutive_errors >= 3:
                                status = "failed"
                                final = "工具连续失败，已停止本轮。"
                                break
                            continue
                        infra_error, consecutive_errors = await self._run_code(
                            call, error_budget=consecutive_errors
                        )
                        del infra_error
                        if consecutive_errors >= 3:
                            status = "failed"
                            final = "工具连续失败，已停止本轮。"
                            break
                    if status == "failed":
                        break
        except TimeoutError:
            status = "timeout"
            final = "本轮执行超时。"
            self._record(
                EventType.AGENT_ERROR,
                {"message": final, "where": "agent_loop", "traceback": None},
            )
        except asyncio.CancelledError:
            cancelled = True
            self._record(EventType.TURN_CANCELLED, {"turn_id": turn_id, "reason": "user_stop"})
            raise
        except Exception as exc:  # noqa: BLE001 - 后台 turn 不能拖垮服务
            status = "failed"
            final = f"本轮失败：{type(exc).__name__}: {exc}"
            logger.exception("agent turn 失败")
            self._record(
                EventType.AGENT_ERROR,
                {"message": str(exc), "where": "agent_loop", "traceback": None},
            )
        finally:
            if not cancelled:
                self._record(EventType.TURN_FINISHED, {"turn_id": turn_id, "status": status})
                try:
                    await self.hooks.emit(
                        HookEvent.TURN_END,
                        self._hook_ctx(),
                        {"turn_id": turn_id, "status": status, "final": final},
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("turn_end hook 失败")

        usage = self.state.last_usage if self.state else {}
        return {
            "session_id": self.session_id,
            "turn_id": turn_id,
            "status": status,
            "ok": status == "completed",
            "final_answer": final,
            "final": final,
            "steps": steps,
            "usage": usage,
        }
