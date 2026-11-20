"""WP6：SubagentManager 与 5 个管理工具的单测（离线、FakeLoop）。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from minillm.agent.subagents import SubagentError, SubagentManager
from minillm.artifacts import ArtifactStore
from minillm.config import Settings
from minillm.errors import ToolError
from minillm.events import EventBus, EventType
from minillm.llm.types import LLMResult
from minillm.locks import FileLockManager
from minillm.persistence.replay import build_messages
from minillm.persistence.store import SessionStore
from minillm.tools.base import ToolContext
from minillm.tools.registry import ToolRegistry
from minillm.tools.subagent_tools import (
    SUBAGENT_TOOL_NAMES,
    build_subagent_specs,
    register_subagent_tools,
)


class FakeSmall:
    """小模型替身：记录调用并返回固定摘要。"""

    def __init__(self, content: str = "SUMMARY") -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any) -> LLMResult:
        self.calls.append(kwargs)
        return LLMResult(content=self.content)


class FakeLoop:
    """记录 seed/calls；支持阻塞、失败、延迟三种行为。"""

    def __init__(
        self, meta: dict[str, Any], agent_id: str, seed: list[dict[str, Any]], behavior: dict
    ) -> None:
        self.meta = meta
        self.agent_id = agent_id
        self.seed = list(seed)
        self.behavior = behavior
        self.calls: list[str] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run_turn(self, text: str) -> dict[str, Any]:
        self.calls.append(text)
        behavior = self.behavior
        behavior["active"] += 1
        behavior["max_active"] = max(behavior["max_active"], behavior["active"])
        self.started.set()
        try:
            if behavior.get("block"):
                await self.release.wait()
            if behavior.get("fail"):
                raise RuntimeError("fake failure")
            if behavior.get("delay"):
                await asyncio.sleep(float(behavior["delay"]))
            return {"final": f"done:{text}", "status": "completed"}
        finally:
            behavior["active"] -= 1


class LoopFactory:
    """把行为配置传给每个 FakeLoop，并保留引用供断言。"""

    def __init__(self, **behavior: Any) -> None:
        self.behavior: dict[str, Any] = {
            "block": False,
            "fail": False,
            "delay": 0.0,
            "active": 0,
            "max_active": 0,
            **behavior,
        }
        self.loops: list[FakeLoop] = []

    def __call__(self, meta: dict, agent_id: str, seed: list[dict]) -> FakeLoop:
        loop = FakeLoop(meta, agent_id, seed, self.behavior)
        self.loops.append(loop)
        return loop


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        llm_api_key="test-key",
        small_llm_api_key="test-small-key",
        data_dir=tmp_path / "data",
        workspace_root=tmp_path / "workspace",
        hooks_dir=tmp_path / "hooks",
        skills_dir=tmp_path / "skills",
        mcp_config=tmp_path / "mcp.json",
        wal_fsync=False,
        context_budget=200000,
        keep_recent_turns=2,
        subagent_max_concurrent=4,
        subagent_max_depth=1,
        llm_model="test-model",
    )


@pytest.fixture
def store(settings: Settings) -> SessionStore:
    return SessionStore(settings.data_dir, fsync=False)


@pytest.fixture
def parent_id(store: SessionStore, tmp_path: Path) -> str:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    meta = store.create_session(
        {"title": "root", "workspace": str(workspace), "model": "test-model", "depth": 0}
    )
    return str(meta["session_id"])


def make_manager(
    settings: Settings,
    store: SessionStore,
    factory: LoopFactory,
    small_llm: Any = None,
    bus: EventBus | None = None,
) -> SubagentManager:
    return SubagentManager(
        settings,
        store,
        bus or EventBus(),
        loop_factory=factory,
        small_llm=small_llm,
    )


def make_ctx(
    settings: Settings, store: SessionStore, parent_id: str, *, allow: bool = True
) -> ToolContext:
    meta = store.meta(parent_id)
    workspace = Path(meta["workspace"])
    return ToolContext(
        session_id=parent_id,
        workspace=workspace,
        data_dir=settings.data_dir,
        settings=settings,
        locks=FileLockManager(),
        artifacts=ArtifactStore(store, parent_id, settings),
        agent_id="main",
        kind="main",
        allow_subagents=allow,
    )


async def _settle() -> None:
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_spawn_creates_isolated_session_and_events(
    settings: Settings, store: SessionStore, parent_id: str
) -> None:
    factory = LoopFactory()
    manager = make_manager(settings, store, factory)
    info = await manager.spawn(parent_id, "inspect files", description="检查")

    assert info["status"] == "running"
    child = store.meta(info["session_id"])
    assert child["parent_id"] == parent_id
    assert child["parent_session_id"] == parent_id
    assert child["agent_id"] == info["agent_id"]
    assert child["depth"] == 1
    assert Path(child["workspace"]) == Path(store.meta(parent_id)["workspace"])

    result = await asyncio.wait_for(manager.wait(info["agent_id"], timeout=2.0), 2.0)
    assert result["status"] == "completed"
    assert result["result_preview"] == "done:inspect files"

    agents = manager.list_agents(parent_id)
    assert len(agents) == 1
    assert agents[0]["agent_id"] == info["agent_id"]
    assert agents[0]["status"] == "completed"
    assert agents[0]["description"] == "检查"

    parent_events = [event.type for event in store.load_events(parent_id)]
    assert EventType.SUBAGENT_SPAWNED in parent_events
    assert EventType.SUBAGENT_COMPLETED in parent_events

    child_events = store.load_events(info["session_id"])
    assert all(event.type != EventType.USER_MESSAGE for event in child_events)
    assert any(event.type == EventType.ASSISTANT_MESSAGE for event in child_events)
    await manager.close()


@pytest.mark.asyncio
async def test_inherit_context_true_and_false(
    settings: Settings, store: SessionStore, parent_id: str
) -> None:
    store.append(parent_id, EventType.USER_MESSAGE, {"content": "父问题"})
    store.append(
        parent_id,
        EventType.ASSISTANT_MESSAGE,
        {
            "content": "父回答",
            "reasoning": "父推理",
            "tool_calls": [],
            "finish_reason": "stop",
        },
    )
    expected = build_messages(store.load_state(parent_id), include_reasoning=True)

    factory = LoopFactory()
    manager = make_manager(settings, store, factory)
    a = await manager.spawn(parent_id, "继承", inherit_context=True)
    b = await manager.spawn(parent_id, "不继承", inherit_context=False)
    await asyncio.wait_for(manager.wait(a["agent_id"], timeout=2.0), 2.0)
    await asyncio.wait_for(manager.wait(b["agent_id"], timeout=2.0), 2.0)

    assert factory.loops[0].seed == expected
    assert factory.loops[0].seed[0]["role"] == "user"
    assert factory.loops[1].seed == []
    await manager.close()


@pytest.mark.asyncio
async def test_inherit_context_over_budget_uses_small_model_summary(
    settings: Settings, store: SessionStore, parent_id: str
) -> None:
    small_settings = settings.model_copy(update={"context_budget": 20, "keep_recent_turns": 1})
    store.append(parent_id, EventType.USER_MESSAGE, {"content": "长问题" * 100})
    store.append(
        parent_id,
        EventType.ASSISTANT_MESSAGE,
        {"content": "长回答" * 100, "reasoning": "", "tool_calls": [], "finish_reason": "stop"},
    )
    small = FakeSmall("SUMMARY")
    factory = LoopFactory()
    manager = make_manager(small_settings, store, factory, small_llm=small)
    info = await manager.spawn(parent_id, "摘要继承", inherit_context=True)
    await asyncio.wait_for(manager.wait(info["agent_id"], timeout=2.0), 2.0)

    seed = factory.loops[0].seed
    assert len(seed) == 1
    assert seed[0]["role"] == "system"
    assert "SUMMARY" in seed[0]["content"]
    assert small.calls and small.calls[0]["agent_kind"] == "small"
    await manager.close()


@pytest.mark.asyncio
async def test_depth_limit_rejects_grandchild(
    settings: Settings, store: SessionStore, parent_id: str
) -> None:
    workspace = store.meta(parent_id)["workspace"]
    child = store.create_session(
        {
            "title": "child",
            "workspace": workspace,
            "model": "test-model",
            "parent_id": parent_id,
            "depth": settings.subagent_max_depth,
        }
    )
    factory = LoopFactory()
    manager = make_manager(settings, store, factory)
    with pytest.raises(SubagentError) as exc:
        await manager.spawn(str(child["session_id"]), "grandchild")
    assert exc.value.kind == "denied"
    assert factory.loops == []
    await manager.close()


@pytest.mark.asyncio
async def test_concurrency_semaphore_limits_turns(
    settings: Settings, store: SessionStore, parent_id: str
) -> None:
    limited = settings.model_copy(update={"subagent_max_concurrent": 1})
    factory = LoopFactory(block=True)
    manager = make_manager(limited, store, factory)
    first = await manager.spawn(parent_id, "first")
    await factory.loops[0].started.wait()
    second = await manager.spawn(parent_id, "second")
    await _settle()
    assert not factory.loops[1].started.is_set()

    factory.loops[0].release.set()
    await factory.loops[1].started.wait()
    factory.loops[1].release.set()
    await asyncio.wait_for(manager.wait(first["agent_id"], timeout=2.0), 2.0)
    await asyncio.wait_for(manager.wait(second["agent_id"], timeout=2.0), 2.0)
    assert factory.behavior["max_active"] == 1
    await manager.close()


@pytest.mark.asyncio
async def test_wait_timeout_returns_running(
    settings: Settings, store: SessionStore, parent_id: str
) -> None:
    factory = LoopFactory(block=True)
    manager = make_manager(settings, store, factory)
    info = await manager.spawn(parent_id, "slow")
    await factory.loops[0].started.wait()

    result = await manager.wait(info["agent_id"], timeout=0.05)
    assert result["status"] == "running"

    factory.loops[0].release.set()
    result = await asyncio.wait_for(manager.wait(info["agent_id"], timeout=2.0), 2.0)
    assert result["status"] == "completed"
    await manager.close()


@pytest.mark.asyncio
async def test_send_message_queues_and_runs_after_current_turn(
    settings: Settings, store: SessionStore, parent_id: str
) -> None:
    factory = LoopFactory(block=True)
    manager = make_manager(settings, store, factory)
    info = await manager.spawn(parent_id, "first")
    await factory.loops[0].started.wait()

    queued = await manager.send_message(info["agent_id"], "second")
    assert queued == {"agent_id": info["agent_id"], "delivered": True, "status": "running"}

    factory.loops[0].release.set()
    result = await asyncio.wait_for(manager.wait(info["agent_id"], timeout=2.0), 2.0)
    assert result["status"] == "completed"
    assert factory.loops[0].calls == ["first", "second"]

    sent = await manager.send_message(info["agent_id"], "third")
    assert sent["status"] == "running"
    result = await asyncio.wait_for(manager.wait(info["agent_id"], timeout=2.0), 2.0)
    assert result["status"] == "idle"
    assert factory.loops[0].calls == ["first", "second", "third"]

    parent_events = [event.type for event in store.load_events(parent_id)]
    assert parent_events.count(EventType.SUBAGENT_MESSAGE) >= 2
    completed_events = [
        event.data
        for event in store.load_events(parent_id)
        if event.type == EventType.SUBAGENT_COMPLETED
    ]
    # 后续的 send_message 轮在运行态是 idle，但事件契约只允许 completed/stopped/failed
    assert completed_events[-1]["status"] == "completed"
    await manager.close()



@pytest.mark.asyncio
async def test_stop_cancels_running_child(
    settings: Settings, store: SessionStore, parent_id: str
) -> None:
    factory = LoopFactory(block=True)
    manager = make_manager(settings, store, factory)
    info = await manager.spawn(parent_id, "long")
    await factory.loops[0].started.wait()

    stopped = await manager.stop(info["agent_id"])
    assert stopped["status"] == "stopped"
    assert store.meta(info["session_id"])["status"] == "stopped"
    completed = [
        event.data
        for event in store.load_events(parent_id)
        if event.type == EventType.SUBAGENT_COMPLETED
    ]
    assert completed and completed[-1]["status"] == "stopped"
    await manager.close()


@pytest.mark.asyncio
async def test_subagent_tools_registration_and_guards(
    settings: Settings, store: SessionStore, parent_id: str
) -> None:
    factory = LoopFactory()
    manager = make_manager(settings, store, factory)
    registry = ToolRegistry()
    register_subagent_tools(registry, manager)
    assert set(SUBAGENT_TOOL_NAMES).issubset(set(registry.names()))
    assert {spec.kind for spec in registry.list_specs("subagent")} == {"subagent"}

    ctx = make_ctx(settings, store, parent_id, allow=True)
    result = await registry.invoke(
        "spawn_subagent", {"task": "tool spawn", "description": "tool"}, ctx
    )
    assert result.ok
    assert result.value["status"] == "running"
    await asyncio.wait_for(manager.wait(result.value["agent_id"], timeout=2.0), 2.0)
    listed = await registry.invoke("list_agents", {}, ctx)
    assert isinstance(listed.value, list) and listed.value

    with pytest.raises(ToolError) as denied:
        await registry.invoke(
            "spawn_subagent",
            {"task": "nope"},
            make_ctx(settings, store, parent_id, allow=False),
        )
    assert denied.value.kind == "denied"

    with pytest.raises(ToolError) as validation:
        await registry.invoke("send_message", {"agent_id": "agent_missing", "message": "x"}, ctx)
    assert validation.value.kind == "validation"

    with pytest.raises(ToolError) as missing:
        await registry.invoke("spawn_subagent", {}, ctx)
    assert missing.value.kind == "validation"

    with pytest.raises(ToolError) as extra:
        await registry.invoke("spawn_subagent", {"task": "x", "extra": 1}, ctx)
    assert extra.value.kind == "validation"
    await manager.close()


@pytest.mark.asyncio
async def test_build_specs_without_manager_lists_five_and_denies(
    settings: Settings, store: SessionStore, parent_id: str
) -> None:
    specs = build_subagent_specs(None)
    assert [spec.name for spec in specs] == list(SUBAGENT_TOOL_NAMES)
    registry = ToolRegistry()
    for spec in specs:
        registry.register(spec)
    assert set(SUBAGENT_TOOL_NAMES).issubset(set(registry.names()))

    ctx = make_ctx(settings, store, parent_id, allow=True)
    with pytest.raises(ToolError) as exc:
        await registry.invoke("list_agents", {}, ctx)
    assert exc.value.kind == "denied"
    assert "未启用" in str(exc.value)


@pytest.mark.asyncio
async def test_close_cancels_all_and_updates_meta(
    settings: Settings, store: SessionStore, parent_id: str
) -> None:
    factory = LoopFactory(block=True)
    manager = make_manager(settings, store, factory)
    info = await manager.spawn(parent_id, "closing")
    await factory.loops[0].started.wait()
    await manager.close()
    assert store.meta(info["session_id"])["status"] == "stopped"
    assert manager.list_agents(parent_id)[0]["status"] == "stopped"


@pytest.mark.asyncio
async def test_all_five_tool_handlers_roundtrip(
    settings: Settings, store: SessionStore, parent_id: str
) -> None:
    factory = LoopFactory()
    manager = make_manager(settings, store, factory)
    registry = ToolRegistry()
    register_subagent_tools(registry, manager)
    ctx = make_ctx(settings, store, parent_id, allow=True)

    spawned = await registry.invoke("spawn_subagent", {"task": "roundtrip"}, ctx)
    agent_id = spawned.value["agent_id"]
    waited = await registry.invoke("wait_agent", {"agent_id": agent_id, "timeout": 2}, ctx)
    assert waited.value["status"] == "completed"

    sent = await registry.invoke("send_message", {"agent_id": agent_id, "message": "again"}, ctx)
    assert sent.value["delivered"] is True
    waited = await registry.invoke("wait_agent", {"agent_id": agent_id, "timeout": 2}, ctx)
    assert waited.value["status"] == "idle"

    listed = await registry.invoke("list_agents", {}, ctx)
    assert listed.value[0]["agent_id"] == agent_id

    stopped = await registry.invoke("stop_agent", {"agent_id": agent_id}, ctx)
    assert stopped.value["status"] == "stopped"
    await manager.close()


@pytest.mark.asyncio
async def test_wait_any_agent_selects_finished_child(
    settings: Settings, store: SessionStore, parent_id: str
) -> None:
    factory = LoopFactory()
    manager = make_manager(settings, store, factory)
    first = await manager.spawn(parent_id, "one")
    second = await manager.spawn(parent_id, "two")
    await _settle()
    result = await asyncio.wait_for(manager.wait(None, timeout=2.0), 2.0)
    assert result["agent_id"] in {first["agent_id"], second["agent_id"]}
    assert result["status"] == "completed"
    await manager.close()


@pytest.mark.asyncio
async def test_child_failure_is_reported_to_parent(
    settings: Settings, store: SessionStore, parent_id: str
) -> None:
    factory = LoopFactory(fail=True)
    manager = make_manager(settings, store, factory)
    info = await manager.spawn(parent_id, "boom")
    result = await asyncio.wait_for(manager.wait(info["agent_id"], timeout=2.0), 2.0)
    assert result["status"] == "failed"
    assert "fake failure" in str(result["error"])
    completed = [
        event.data
        for event in store.load_events(parent_id)
        if event.type == EventType.SUBAGENT_COMPLETED
    ]
    assert completed and completed[-1]["status"] == "failed"
    await manager.close()
