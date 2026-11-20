"""子 agent 管理器：独立会话、上下文继承、并发/等待/停止。

设计要点：
- 子 agent 在独立 Session（独立 events.jsonl）中运行，父会话只记录摘要级事件。
- 子 agent 的 loop 由外部注入（loop_factory），便于离线测试与后续 WP5 的 AgentLoop 集成。
- inherit_context=True 时只把父会话当前压缩视图交给 child loop 的 seed_messages，
  不写入子会话 JSONL；超过预算一半时先小模型摘要，失败退化为安全截断。
- 并发上限由 Semaphore 控制；depth 从父会话 meta 读取，超过 settings.subagent_max_depth 拒绝。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..config import Settings
from ..errors import MiniLLMError
from ..events import Event, EventBus, EventType
from ..ids import new_id, utc_now_iso
from ..llm.types import LLMClient
from ..persistence.replay import SessionState, build_messages
from ..persistence.store import SessionStore

logger = logging.getLogger("minillm.agent.subagents")

ACTIVE_STATUSES = frozenset({"running"})
TERMINAL_STATUSES = frozenset({"completed", "stopped", "failed", "idle"})

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")


class SubagentError(MiniLLMError):
    """子 agent 管理错误；kind 供工具层映射为 ToolError。"""

    def __init__(self, message: str, *, kind: str = "execution") -> None:
        super().__init__(message)
        self.kind = kind


@runtime_checkable
class ChildLoop(Protocol):
    """子 agent 的运行循环；WP5 的 AgentLoop 需要满足该最小契约。"""

    async def run_turn(self, text: str) -> dict[str, Any]: ...


LoopFactory = Callable[[dict[str, Any], str, list[dict[str, Any]]], ChildLoop]


@dataclass
class _Agent:
    """进程内单个子 agent 的运行态。"""

    agent_id: str
    session_id: str
    parent_session_id: str
    task: str
    description: str | None
    inherit_context: bool
    child_loop: ChildLoop
    status: str = "running"
    result_preview: str = ""
    error: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue)
    task_obj: asyncio.Task[Any] | None = None
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
    initial_turn: bool = True
    completion_emitted: bool = False

    def public(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "parent_session_id": self.parent_session_id,
            "task": self.task,
            "description": self.description,
            "status": self.status,
            "result_preview": self.result_preview,
            "error": self.error,
            "inherit_context": self.inherit_context,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class SubagentManager:
    """创建、驱动与回收子 agent。"""

    def __init__(
        self,
        settings: Settings,
        store: SessionStore,
        bus: EventBus | None = None,
        *,
        loop_factory: LoopFactory,
        small_llm: LLMClient | None = None,
        locks: Any = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.bus = bus
        self.loop_factory = loop_factory
        self.small_llm = small_llm
        self.locks = locks
        self._agents: dict[str, _Agent] = {}
        self._semaphore = asyncio.Semaphore(max(1, int(settings.subagent_max_concurrent)))
        self._closed = False

    # ==== 公共 API ====
    async def spawn(
        self,
        parent_session_id: str,
        task: str,
        *,
        description: str | None = None,
        inherit_context: bool = False,
        parent_state: SessionState | None = None,
    ) -> dict[str, Any]:
        """创建子会话并异步启动首轮；立即返回 agent 信息。"""

        task = str(task or "").strip()
        if not task:
            raise SubagentError("task 不能为空", kind="validation")
        if self._closed:
            raise SubagentError("SubagentManager 已关闭", kind="denied")

        parent_meta = self._meta(parent_session_id)
        parent_depth = int(parent_meta.get("depth") or 0)
        if parent_depth >= int(self.settings.subagent_max_depth):
            raise SubagentError(
                f"子 agent 深度超限：parent depth={parent_depth}, "
                f"max={self.settings.subagent_max_depth}",
                kind="denied",
            )

        workspace = str(parent_meta.get("workspace") or "").strip()
        if not workspace:
            ws = self.settings.session_workspace(parent_session_id)
            ws.mkdir(parents=True, exist_ok=True)
            workspace = str(ws)

        agent_id = new_id("agent")
        title = (description or task).strip().replace("\n", " ")[:48]
        child_meta = self.store.create_session(
            {
                "title": title or "子 agent",
                "workspace": workspace,
                "model": self.settings.llm_model,
                "parent_id": parent_session_id,
                "parent_session_id": parent_session_id,
                "agent_id": agent_id,
                "depth": parent_depth + 1,
                "status": "running",
                "task": task,
                "description": description,
                "inherit_context": bool(inherit_context),
            }
        )
        session_id = str(child_meta["session_id"])

        inherited = await self._inherit_messages(
            parent_session_id, parent_state, inherit_context
        )
        try:
            child_loop = self.loop_factory(child_meta, agent_id, inherited)
        except Exception as exc:  # noqa: BLE001 - 工厂失败也要给父会话一个结果
            logger.exception("创建子 agent loop 失败：%s", agent_id)
            self._update_meta(session_id, {"status": "failed"})
            self._append(
                parent_session_id,
                EventType.SUBAGENT_COMPLETED,
                {
                    "agent_id": agent_id,
                    "parent_session_id": parent_session_id,
                    "status": "failed",
                    "result_preview": "",
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                },
            )
            raise SubagentError(f"创建子 agent 失败：{exc}", kind="execution") from exc

        runtime = _Agent(
            agent_id=agent_id,
            session_id=session_id,
            parent_session_id=parent_session_id,
            task=task,
            description=description,
            inherit_context=bool(inherit_context),
            child_loop=child_loop,
        )
        self._agents[agent_id] = runtime
        self._append(
            parent_session_id,
            EventType.SUBAGENT_SPAWNED,
            {
                "agent_id": agent_id,
                "session_id": session_id,
                "parent_session_id": parent_session_id,
                "task": task,
                "description": description,
                "status": "running",
                "inherit_context": bool(inherit_context),
            },
        )
        runtime.done_event.clear()
        runtime.task_obj = asyncio.create_task(
            self._run_agent(runtime, task, initial=True),
            name=f"subagent:{agent_id}",
        )
        return {
            "agent_id": agent_id,
            "session_id": session_id,
            "parent_session_id": parent_session_id,
            "status": "running",
            "task": task,
        }

    async def send_message(self, agent_id: str, message: str) -> dict[str, Any]:
        """空闲时开新轮；运行中排队，当前轮结束后依次处理。"""

        text = str(message or "").strip()
        if not text:
            raise SubagentError("message 不能为空", kind="validation")
        runtime = self._get(agent_id)
        if runtime.status in {"stopped", "failed"}:
            raise SubagentError(
                f"子 agent 状态为 {runtime.status}，无法继续发送", kind="validation"
            )

        self._append(
            runtime.parent_session_id,
            EventType.SUBAGENT_MESSAGE,
            {"agent_id": agent_id, "content": text, "role": "user"},
        )
        if runtime.task_obj is not None and not runtime.task_obj.done():
            runtime.queue.put_nowait(text)
            runtime.updated_at = utc_now_iso()
            return {"agent_id": agent_id, "delivered": True, "status": "running"}

        runtime.status = "running"
        runtime.updated_at = utc_now_iso()
        runtime.completion_emitted = False
        runtime.done_event.clear()
        self._update_meta(runtime.session_id, {"status": "running"})
        runtime.task_obj = asyncio.create_task(
            self._run_agent(runtime, text, initial=False),
            name=f"subagent:{agent_id}",
        )
        return {"agent_id": agent_id, "delivered": True, "status": "running"}

    async def wait(
        self, agent_id: str | None = None, timeout: float = 300.0
    ) -> dict[str, Any]:
        """等待指定/任一子 agent 当前轮结束；超时返回 running。"""

        wait_seconds = max(0.0, float(timeout))
        if agent_id is not None:
            runtime = self._get(agent_id)
            try:
                await asyncio.wait_for(runtime.done_event.wait(), timeout=wait_seconds)
            except TimeoutError:
                pass
            return self._wait_result(runtime)

        if not self._agents:
            raise SubagentError("当前没有子 agent", kind="validation")
        running = [
            runtime
            for runtime in self._agents.values()
            if runtime.task_obj is not None and not runtime.task_obj.done()
        ]
        candidates = running or list(self._agents.values())
        waiters: dict[asyncio.Task[Any], _Agent] = {
            asyncio.create_task(runtime.done_event.wait()): runtime
            for runtime in candidates
        }
        try:
            done, pending = await asyncio.wait(
                waiters.keys(), timeout=wait_seconds, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            for task in waiters:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*waiters, return_exceptions=True)
        if done:
            return self._wait_result(next(iter(waiters[task] for task in done)))
        return self._wait_result(candidates[0])

    def list_agents(self, parent_session_id: str | None = None) -> list[dict[str, Any]]:
        """列出当前进程及磁盘 meta 中的子 agent。"""

        records: dict[str, dict[str, Any]] = {}
        for runtime in self._agents.values():
            if parent_session_id and runtime.parent_session_id != parent_session_id:
                continue
            records[runtime.agent_id] = runtime.public()

        for meta in self.store.list_sessions():
            parent = meta.get("parent_session_id") or meta.get("parent_id")
            if not parent or (parent_session_id and parent != parent_session_id):
                continue
            agent_id = str(meta.get("agent_id") or "")
            if not agent_id or agent_id in records:
                continue
            records[agent_id] = {
                "agent_id": agent_id,
                "session_id": str(meta.get("session_id") or ""),
                "parent_session_id": str(parent),
                "task": str(meta.get("task") or ""),
                "description": meta.get("description"),
                "status": str(meta.get("status") or "idle"),
                "result_preview": str(meta.get("result_preview") or ""),
                "error": meta.get("error"),
                "inherit_context": bool(meta.get("inherit_context")),
                "created_at": str(meta.get("created_at") or ""),
                "updated_at": str(meta.get("updated_at") or ""),
            }
        return sorted(records.values(), key=lambda item: (item["created_at"], item["agent_id"]))

    async def stop(self, agent_id: str) -> dict[str, Any]:
        """取消子 agent 当前任务并落 stopped。"""

        runtime = self._agents.get(agent_id)
        if runtime is None:
            meta = self._find_meta(agent_id)
            if meta is None:
                raise SubagentError(f"未知子 agent：{agent_id}", kind="validation")
            parent = str(meta.get("parent_session_id") or meta.get("parent_id") or "")
            self._update_meta(str(meta["session_id"]), {"status": "stopped"})
            self._append(
                parent,
                EventType.SUBAGENT_COMPLETED,
                {
                    "agent_id": agent_id,
                    "parent_session_id": parent,
                    "status": "stopped",
                    "result_preview": str(meta.get("result_preview") or ""),
                },
            )
            return {"agent_id": agent_id, "status": "stopped"}

        if runtime.task_obj is not None and not runtime.task_obj.done():
            runtime.task_obj.cancel()
            try:
                await runtime.task_obj
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 - stop 不应把异常抛给调用方
                logger.exception("停止子 agent 时任务异常：%s", agent_id)
        runtime.status = "stopped"
        runtime.updated_at = utc_now_iso()
        runtime.done_event.set()
        runtime.task_obj = None
        self._update_meta(runtime.session_id, {"status": "stopped"})
        if not runtime.completion_emitted:
            runtime.completion_emitted = True
            self._append(
                runtime.parent_session_id,
                EventType.SUBAGENT_COMPLETED,
                {
                    "agent_id": agent_id,
                    "parent_session_id": runtime.parent_session_id,
                    "status": "stopped",
                    "result_preview": runtime.result_preview,
                },
            )
        return {"agent_id": agent_id, "status": "stopped"}

    async def close(self) -> None:
        """取消所有子 agent 任务；进程退出前调用。"""

        self._closed = True
        tasks = [
            runtime.task_obj
            for runtime in self._agents.values()
            if runtime.task_obj is not None and not runtime.task_obj.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._agents.clear()

    # ==== 内部：运行 ====
    async def _run_agent(self, runtime: _Agent, first_text: str, *, initial: bool) -> None:
        runtime.status = "running"
        runtime.done_event.clear()
        runtime.updated_at = utc_now_iso()
        runtime.completion_emitted = False
        result: dict[str, Any] = {}
        final_status = "completed" if initial else "idle"
        error: str | None = None
        try:
            pending = [first_text]
            while pending:
                text = pending.pop(0)
                async with self._semaphore:
                    raw = await runtime.child_loop.run_turn(text)
                if isinstance(raw, dict):
                    result = raw
                else:
                    result = {"final": str(raw)}
                self._merge_result(runtime, result)
                while not runtime.queue.empty():
                    pending.append(runtime.queue.get_nowait())
            reported = str(result.get("status") or "")
            if reported in {"failed", "stopped"}:
                final_status = reported
            runtime.status = final_status
        except asyncio.CancelledError:
            runtime.status = "stopped"
            final_status = "stopped"
            error = "cancelled"
        except Exception as exc:  # noqa: BLE001 - 失败要回传给父 agent
            runtime.status = "failed"
            final_status = "failed"
            error = f"{type(exc).__name__}: {exc}"[:500]
            runtime.error = error
            logger.exception("子 agent 执行失败：%s", runtime.agent_id)
        finally:
            runtime.updated_at = utc_now_iso()
            runtime.task_obj = None
            try:
                self._update_meta(
                    runtime.session_id,
                    {
                        "status": runtime.status,
                        "result_preview": runtime.result_preview,
                        "error": runtime.error,
                    },
                )
            except Exception:  # noqa: BLE001 - meta 更新失败不影响事件
                logger.exception("更新子会话 meta 失败：%s", runtime.session_id)
            if not runtime.completion_emitted:
                runtime.completion_emitted = True
                # docs/EVENTS.md 规定 subagent.completed 的 status 只能是
                # completed/stopped/failed；运行态 idle 只用于 list/wait。
                event_status = "completed" if final_status == "idle" else final_status
                data = {
                    "agent_id": runtime.agent_id,
                    "session_id": runtime.session_id,
                    "parent_session_id": runtime.parent_session_id,
                    "status": event_status,
                    "result_preview": runtime.result_preview,
                }
                if error:
                    data["error"] = error
                try:
                    self._append(runtime.parent_session_id, EventType.SUBAGENT_COMPLETED, data)
                except Exception:  # noqa: BLE001 - 父会话可能已删除
                    logger.exception("写 subagent.completed 失败：%s", runtime.agent_id)
            if final_status == "completed" and runtime.result_preview:
                self._maybe_append_child_final(runtime)
            runtime.done_event.set()

    def _merge_result(self, runtime: _Agent, result: dict[str, Any]) -> None:
        preview = self._result_preview(result)
        if preview:
            runtime.result_preview = preview
            runtime.error = None

    @staticmethod
    def _result_preview(result: dict[str, Any]) -> str:
        for key in ("final_answer", "final", "content", "result", "summary", "message"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:500]
        return ""

    def _maybe_append_child_final(self, runtime: _Agent) -> None:
        """子 loop 没有写最终 assistant 消息时补一条；已写则不重复。"""

        try:
            events = self.store.load_events(runtime.session_id)
        except Exception:  # noqa: BLE001
            return
        for event in reversed(events[-20:]):
            if event.type == EventType.ASSISTANT_MESSAGE:
                if str(event.data.get("content") or "").strip():
                    return
        try:
            self._append(
                runtime.session_id,
                EventType.ASSISTANT_MESSAGE,
                {
                    "content": runtime.result_preview,
                    "reasoning": "",
                    "tool_calls": [],
                    "finish_reason": "stop",
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception("写子会话最终消息失败：%s", runtime.session_id)

    def _wait_result(self, runtime: _Agent) -> dict[str, Any]:
        status = runtime.status
        if runtime.task_obj is not None and not runtime.task_obj.done():
            status = "running"
        return {
            "agent_id": runtime.agent_id,
            "session_id": runtime.session_id,
            "status": status,
            "result_preview": runtime.result_preview,
            "error": runtime.error,
        }

    # ==== 内部：上下文继承 ====
    async def _inherit_messages(
        self,
        parent_session_id: str,
        parent_state: SessionState | None,
        inherit_context: bool,
    ) -> list[dict[str, Any]]:
        if not inherit_context:
            return []
        state = parent_state
        if state is None:
            state = self.store.load_state(parent_session_id)
        try:
            messages = build_messages(
                state, include_reasoning=bool(self.settings.include_reasoning)
            )
        except Exception:  # noqa: BLE001 - 继承失败时退化为空上下文
            logger.exception("构建父会话继承消息失败：%s", parent_session_id)
            return []
        if not messages:
            return []
        if self._estimate_tokens(messages) <= int(self.settings.context_budget * 0.5):
            return messages

        summary = await self._summarize_parent(messages)
        if summary:
            return [
                {
                    "role": "system",
                    "content": f"[父会话上下文摘要]\n{summary}",
                }
            ]
        limit = max(2, int(self.settings.keep_recent_turns) * 2)
        return self._safe_tail(messages, limit)

    async def _summarize_parent(self, messages: list[dict[str, Any]]) -> str | None:
        if self.small_llm is None:
            return None
        transcript = json.dumps(messages, ensure_ascii=False)
        prompt = [
            {
                "role": "system",
                "content": (
                    "你是上下文压缩助手。把父会话压缩成可交给子 agent 的摘要，"
                    "保留目标、已完成、关键文件、未决问题与重要工具结论。只输出摘要正文。"
                ),
            },
            {"role": "user", "content": transcript},
        ]
        try:
            result = await self.small_llm.complete(
                messages=prompt,
                temperature=0.0,
                max_tokens=512,
                agent_kind="small",
            )
        except Exception:  # noqa: BLE001 - 摘要失败时调用方退化为截断
            logger.warning("父会话摘要失败，退化为截断", exc_info=True)
            return None
        content = str(getattr(result, "content", "") or "").strip()
        return content or None

    @staticmethod
    def _estimate_tokens(messages: Iterable[dict[str, Any]]) -> int:
        text = json.dumps(list(messages), ensure_ascii=False)
        cjk = len(_CJK_RE.findall(text))
        other = max(0, len(text) - cjk)
        return (cjk + 1) // 2 + (other + 3) // 4

    @staticmethod
    def _safe_tail(messages: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        tail = list(messages[-max(1, limit) :])
        while tail and tail[0].get("role") == "tool":
            tail.pop(0)
        while tail and tail[0].get("role") == "assistant" and tail[0].get("tool_calls"):
            expected = {
                str(call.get("id"))
                for call in tail[0].get("tool_calls") or []
                if isinstance(call, dict)
            }
            present = {
                str(message.get("tool_call_id"))
                for message in tail[1:]
                if message.get("role") == "tool"
            }
            if expected and expected.issubset(present):
                break
            tail.pop(0)
        return tail

    # ==== 内部：持久化 ====
    def _meta(self, session_id: str) -> dict[str, Any]:
        try:
            return self.store.meta(session_id)
        except Exception as exc:  # noqa: BLE001
            raise SubagentError(f"父会话不存在：{session_id}", kind="validation") from exc

    def _find_meta(self, agent_id: str) -> dict[str, Any] | None:
        for meta in self.store.list_sessions():
            if str(meta.get("agent_id") or "") == agent_id:
                return meta
        return None

    def _append(self, session_id: str, event_type: str, data: dict[str, Any]) -> Event:
        event = self.store.append(session_id, event_type, data)
        if self.bus is not None:
            self.bus.publish(event)
        return event

    def _update_meta(self, session_id: str, patch: dict[str, Any]) -> None:
        try:
            meta = self.store.meta(session_id)
        except Exception:  # noqa: BLE001
            return
        meta.update(patch)
        meta["updated_at"] = utc_now_iso()
        path = self.store.meta_path(session_id)
        tmp = path.with_name(f".meta.{os.getpid()}.{new_id('msg')}.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(meta, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                if self.store.fsync:
                    os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:  # pragma: no cover
                    pass

    def _get(self, agent_id: str) -> _Agent:
        runtime = self._agents.get(agent_id)
        if runtime is None:
            raise SubagentError(f"未知子 agent：{agent_id}", kind="validation")
        return runtime
