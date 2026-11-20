"""AgentService：会话、后台 turn、事件流、MCP 与子 agent 的总装。"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from ..errors import ConfigError, SessionError
from ..events import EventBus
from ..ids import new_id
from ..locks import FileLockManager
from ..persistence.store import SessionStore
from ..tools.registry import ToolRegistry
from .context import ContextManager
from .loop import AgentLoop, RunCodeRunner
from .session import create_session as create_session_on_disk
from .subagents import SubagentManager

logger = logging.getLogger(__name__)


class AgentService:
    """单进程内管理所有会话；每个 turn 独立 AgentLoop。"""

    def __init__(
        self,
        settings: Settings,
        *,
        store: SessionStore | None = None,
        bus: EventBus | None = None,
        registry: ToolRegistry | None = None,
        hooks: Any | None = None,
        llm: Any | None = None,
        small_llm: Any | None = None,
        context_manager: ContextManager | None = None,
        runner: RunCodeRunner | None = None,
        subagent_manager: SubagentManager | None = None,
    ) -> None:
        self.settings = settings
        settings.ensure_dirs()
        self.store = store or SessionStore(settings.data_dir, fsync=settings.wal_fsync)
        self.bus = bus or EventBus()
        self.locks = FileLockManager()
        self.llm = llm
        self.small_llm = small_llm
        if self.llm is None or self.small_llm is None:
            from ..bootstrap import build_llm_clients

            main, small = build_llm_clients(settings)
            self.llm = self.llm or main
            self.small_llm = self.small_llm or small
        self.context_manager = context_manager or ContextManager(settings, self.small_llm)
        self.runner = runner
        self.subagents = subagent_manager or SubagentManager(
            settings,
            self.store,
            self.bus,
            loop_factory=self._child_loop_factory,
            small_llm=self.small_llm,
            locks=self.locks,
        )
        if registry is None:
            from ..bootstrap import build_registry

            registry = build_registry(settings, subagent_manager=self.subagents)
        self.registry = registry
        if hooks is None:
            from ..bootstrap import build_hooks

            hooks = build_hooks(settings)
        self.hooks = hooks

        self.mcp_manager: Any = None
        self._started = False
        self._active: dict[str, asyncio.Task[Any]] = {}
        self._turn_ids: dict[str, str] = {}
        self._last_results: dict[str, dict[str, Any]] = {}
        self._closed = False

    # ==== 生命周期 ====
    async def start(self) -> AgentService:
        if self._started:
            return self
        self._started = True
        try:
            from ..mcp import register_mcp_tools

            self.mcp_manager = await register_mcp_tools(self.registry, self.settings)
        except Exception:  # noqa: BLE001 - MCP 失败不阻塞服务
            logger.exception("MCP 初始化失败，跳过外部工具")
        return self

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task in list(self._active.values()):
            task.cancel()
        for task in list(self._active.values()):
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._active.clear()
        try:
            await self.subagents.close()
        except Exception:  # noqa: BLE001
            logger.exception("关闭 subagent manager 失败")
        if self.mcp_manager is not None:
            try:
                await self.mcp_manager.close()
            except Exception:  # noqa: BLE001
                logger.exception("关闭 MCP manager 失败")
        closed: set[int] = set()
        for client in (self.llm, self.small_llm):
            if client is None or id(client) in closed:
                continue
            closed.add(id(client))
            closer = getattr(client, "close", None)
            if callable(closer):
                with contextlib.suppress(Exception):
                    result = closer()
                    if asyncio.iscoroutine(result):
                        await result
        await self.bus.close()

    # ==== 会话 ====
    def create_session(
        self,
        *,
        title: str | None = None,
        workspace: str | Path | None = None,
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        return create_session_on_disk(
            self.store,
            self.settings,
            title=title,
            workspace=workspace,
            parent_id=parent_id,
        )

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions = self.store.list_sessions()
        for item in sessions:
            item["status"] = self._status(str(item.get("session_id") or ""))
        return sessions

    async def get_view(self, session_id: str) -> dict[str, Any]:
        task = self._active.get(session_id)
        if task is not None and not task.done():
            # 正在运行的 turn 不能触发 recover：run_code.started 还没有 finished，
            # 补 synthetic 会把正常执行误判为中断。
            state = self.store.load_state(session_id)
        else:
            state = await self.store.recover(session_id)
        meta = dict(state.meta)
        meta["id"] = str(meta.get("session_id") or session_id)
        meta["status"] = self._status(session_id)
        usage = state.last_usage or {}
        tokens = int(usage.get("prompt_tokens") or 0)
        budget = max(1, int(self.settings.context_budget))
        return {
            "meta": meta,
            "messages": list(state.messages),
            "todos": list(state.todos),
            "agents": self.subagents.list_agents(session_id),
            "context": {
                "tokens": tokens,
                "budget": budget,
                "ratio": (tokens / budget),
                "compactions": list(state.compactions),
            },
            "skills": list(state.active_skills),
            "last_seq": state.last_seq,
            "status": meta["status"],
        }

    async def events(self, session_id: str, after: int = 0) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.store.load_events(session_id, after=after)]

    def subscribe(self, session_id: str) -> Any:
        return self.bus.subscribe(session_id)

    async def submit_message(self, session_id: str, content: str) -> str:
        task = self._active.get(session_id)
        if task is not None and not task.done():
            raise SessionError(
                "会话正在运行，请先 stop 或等待完成",
                detail={"code": "session_busy", "status_code": 409},
            )
        self.store.meta(session_id)
        turn_id = new_id("turn")
        task = asyncio.create_task(self._run_background(session_id, content, turn_id))
        self._active[session_id] = task
        self._turn_ids[session_id] = turn_id
        return turn_id

    async def _run_background(self, session_id: str, content: str, turn_id: str) -> dict[str, Any]:
        try:
            loop = self._make_loop(session_id)
            result = await loop.run_turn(content, turn_id=turn_id)
            self._last_results[session_id] = result
            return result
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 后台任务异常不能崩服务
            logger.exception("后台 turn 失败")
            try:
                self.store.append(
                    session_id,
                    "agent.error",
                    {"message": str(exc), "where": "service", "traceback": None},
                )
                self.store.append(
                    session_id,
                    "turn.finished",
                    {"turn_id": turn_id, "status": "failed"},
                )
            except Exception:  # noqa: BLE001
                pass
            return {
                "session_id": session_id,
                "turn_id": turn_id,
                "status": "failed",
                "ok": False,
                "final_answer": str(exc),
                "final": str(exc),
                "steps": 0,
                "usage": {},
            }
        finally:
            current = self._active.get(session_id)
            if current is asyncio.current_task():
                self._active.pop(session_id, None)

    async def run_turn(self, session_id: str, content: str) -> dict[str, Any]:
        loop = self._make_loop(session_id)
        return await loop.run_turn(content)

    async def stop(self, session_id: str) -> dict[str, Any]:
        task = self._active.get(session_id)
        if task is None or task.done():
            return {"ok": True, "status": "idle"}
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return {"ok": True, "status": "cancelled"}

    # ==== 内部装配 ====
    def _make_loop(
        self,
        session_id: str,
        *,
        kind: str = "main",
        agent_id: str = "main",
        allow_subagents: bool | None = None,
        workspace: str | Path | None = None,
        seed_messages: list[dict[str, Any]] | None = None,
    ) -> AgentLoop:
        meta = self.store.meta(session_id)
        raw = workspace if workspace is not None else meta.get("workspace")
        if raw is None or str(raw).strip() == "":
            target = Path(self.settings.workspace_root) / session_id
        else:
            target = Path(str(raw)).expanduser()
        if allow_subagents is None:
            allow_subagents = kind == "main" and self.settings.subagent_max_depth > 0
        return AgentLoop(
            self.settings,
            self.store,
            self.bus,
            self.registry,
            self.hooks,
            self.llm,
            self.context_manager,
            session_id=session_id,
            agent_id=agent_id,
            kind=kind,
            allow_subagents=allow_subagents,
            workspace=target,
            locks=self.locks,
            runner=self.runner,
            seed_messages=seed_messages,
        )

    def _child_loop_factory(
        self,
        session_meta: dict[str, Any],
        agent_id: str,
        inherited_messages: list[dict[str, Any]],
    ) -> AgentLoop:
        session_id = str(session_meta.get("session_id") or session_meta.get("id") or "")
        if not session_id:
            raise ConfigError("子 agent session_meta 缺少 session_id")
        return self._make_loop(
            session_id,
            kind="subagent",
            agent_id=agent_id,
            allow_subagents=False,
            workspace=session_meta.get("workspace"),
            seed_messages=inherited_messages,
        )

    def _status(self, session_id: str) -> str:
        task = self._active.get(session_id)
        if task is not None and not task.done():
            return "running"
        try:
            meta = self.store.meta(session_id)
        except Exception:  # noqa: BLE001
            return "idle"
        return str(meta.get("status") or "idle")

    def mcp_servers(self) -> list[str]:
        if self.mcp_manager is None:
            return []
        try:
            return list(self.mcp_manager.servers())
        except Exception:  # noqa: BLE001
            return []


def build_service(settings: Settings | None = None) -> AgentService:
    return AgentService(settings or get_settings())


async def run_task(
    task: str,
    *,
    workspace: str | Path | None = None,
    data_dir: str | Path | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """CLI 用：创建临时会话、跑一轮、返回结构化结果。"""

    base = settings or get_settings()
    if data_dir is not None:
        base = base.model_copy(update={"data_dir": Path(data_dir)})
    if workspace is not None and not base.allow_external_workspace:
        base = base.model_copy(update={"allow_external_workspace": True})
    service = AgentService(base)
    try:
        await service.start()
        meta = service.create_session(title=str(task)[:40] or "新会话", workspace=workspace)
        session_id = str(meta["session_id"])
        result = await service.run_turn(session_id, task)
        events = await service.events(session_id, after=0)
        result.update(
            {
                "session_id": session_id,
                "events": [{"seq": item["seq"], "type": item["type"]} for item in events],
            }
        )
        return result
    finally:
        await service.close()


async def chat(
    *,
    settings: Settings | None = None,
    workspace: str | Path | None = None,
) -> int:
    service = AgentService(settings or get_settings())
    await service.start()
    meta = service.create_session(title="终端会话", workspace=workspace)
    session_id = str(meta["session_id"])
    print(f"MiniLLM 会话 {session_id}；输入 /exit 退出。")
    try:
        while True:
            try:
                line = await asyncio.to_thread(input, "你> ")
            except EOFError:
                break
            text = line.strip()
            if not text:
                continue
            if text in {"/exit", "/quit"}:
                break
            result = await service.run_turn(session_id, text)
            print(result.get("final_answer") or result.get("final") or "")
    finally:
        await service.close()
    return 0
