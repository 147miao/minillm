"""AgentLoop / AgentService 端到端（离线）。"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from minillm.agent.context import ContextManager
from minillm.agent.loop import AgentLoop, RunCodeResult
from minillm.agent.service import AgentService
from minillm.config import Settings
from minillm.events import EventBus
from minillm.hooks import HookManager
from minillm.llm.types import (
    LLMDone,
    LLMReasoningDelta,
    LLMResult,
    LLMTextDelta,
    LLMToolCallDelta,
    ToolCall,
    Usage,
)
from minillm.tools import build_core_registry


def _call(code: str = "print(1)", call_id: str = "call_1") -> ToolCall:
    return ToolCall(
        id=call_id,
        name="run_code",
        arguments={"code": code},
        raw_arguments=json.dumps({"code": code}, ensure_ascii=False),
    )


def _result(content: str = "", *, calls: list[ToolCall] | None = None) -> LLMResult:
    return LLMResult(
        content=content,
        tool_calls=list(calls or []),
        finish_reason="tool_calls" if calls else "stop",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="fake",
    )


class FakeLLM:
    def __init__(self, responses: list[LLMResult]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def stream_chat(self, **kwargs: Any):
        self.calls.append(dict(kwargs))
        result = self.responses.pop(0)
        for index in range(0, len(result.content), 4):
            yield LLMTextDelta(result.content[index : index + 4])
        if result.reasoning:
            for index in range(0, len(result.reasoning), 4):
                yield LLMReasoningDelta(result.reasoning[index : index + 4])
        for index, call in enumerate(result.tool_calls):
            yield LLMToolCallDelta(
                index=index,
                id=call.id,
                name=call.name,
                args_fragment=call.raw_arguments,
            )
        yield LLMDone(result)

    async def complete(self, **kwargs: Any) -> LLMResult:
        return LLMResult(content="summary")


class FakeRunner:
    def __init__(self, content: str = "[run_code exit=0]\n1") -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []
        self.started = asyncio.Event()
        self.gate: asyncio.Event | None = None

    async def execute(self, *, code: str, call_id: str, ctx: Any) -> RunCodeResult:
        self.calls.append({"code": code, "call_id": call_id})
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        return RunCodeResult(
            exit_code=0,
            stdout="1",
            content=self.content,
            duration_ms=1,
        )


def _loop(
    settings: Settings,
    store,
    session_id: str,
    workspace,
    llm: FakeLLM,
    runner: FakeRunner,
) -> AgentLoop:
    return AgentLoop(
        settings,
        store,
        EventBus(),
        build_core_registry(),
        HookManager(),
        llm,
        ContextManager(settings, llm),
        session_id=session_id,
        workspace=workspace,
        runner=runner,
    )


async def test_turn_runs_tool_and_final(settings, store, session_id, workspace) -> None:
    llm = FakeLLM([_result(calls=[_call()]), _result("完成")])
    runner = FakeRunner()
    loop = _loop(settings, store, session_id, workspace, llm, runner)
    result = await loop.run_turn("请执行")
    assert result["status"] == "completed"
    assert result["final_answer"] == "完成"
    assert runner.calls == [{"code": "print(1)", "call_id": "call_1"}]
    state = store.load_state(session_id)
    roles = [message["role"] for message in state.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert state.messages[2]["tool_call_id"] == "call_1"
    events = store.load_events(session_id)
    types = [event.type for event in events]
    assert types.count("run_code.started") == 1
    assert types.count("run_code.finished") == 1
    assert types.count("assistant.message") == 2
    assert types.count("turn.finished") == 1
    assert types.count("llm.usage") == 2


async def test_max_steps(settings, store, session_id, workspace) -> None:
    settings = settings.model_copy(update={"max_steps": 2})
    llm = FakeLLM([_result(calls=[_call(call_id=f"c{i}")]) for i in range(3)])
    runner = FakeRunner()
    loop = _loop(settings, store, session_id, workspace, llm, runner)
    result = await loop.run_turn("循环")
    assert result["status"] == "max_steps"
    state = store.load_state(session_id)
    assert sum(1 for message in state.messages if message["role"] == "assistant") == 2
    assert runner.calls and len(runner.calls) == 2


async def test_stop_cancels_and_records(settings, store, session_id, workspace) -> None:
    llm = FakeLLM([_result(calls=[_call()]), _result("不会到达")])
    runner = FakeRunner()
    runner.gate = asyncio.Event()
    loop = _loop(settings, store, session_id, workspace, llm, runner)
    task = asyncio.create_task(loop.run_turn("停"))
    await asyncio.wait_for(runner.started.wait(), timeout=2)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    types = [event.type for event in store.load_events(session_id)]
    assert "turn.cancelled" in types
    assert "turn.finished" not in types


async def test_unsupported_tool_is_rejected(settings, store, session_id, workspace) -> None:
    bad = ToolCall(
        id="call_bad",
        name="read_file",
        arguments={"path": "x"},
        raw_arguments='{"path":"x"}',
    )
    llm = FakeLLM([_result(calls=[bad]), _result("结束")])
    runner = FakeRunner()
    loop = _loop(settings, store, session_id, workspace, llm, runner)
    result = await loop.run_turn("坏工具")
    assert result["status"] == "completed"
    assert runner.calls == []
    finished = [
        event for event in store.load_events(session_id) if event.type == "run_code.finished"
    ]
    assert finished and "只能调用 run_code" in finished[0].data["content"]


async def test_service_scripted_roundtrip(settings, store, session_id, workspace, tmp_path) -> None:
    script = tmp_path / "service.json"
    script.write_text(
        json.dumps(
            {
                "responses": [{"content": "脚本完成"}],
                "small_responses": [{"content": "摘要"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    scripted = settings.model_copy(
        update={
            "minillm_llm": "scripted",
            "minillm_script": str(script),
            "data_dir": tmp_path / "service-data",
            "workspace_root": tmp_path / "service-ws",
        }
    )
    service = AgentService(scripted)
    try:
        await service.start()
        meta = service.create_session(title="测试会话")
        sid = meta["session_id"]
        await service.submit_message(sid, "你好")
        task = service._active[sid]
        await task
        view = await service.get_view(sid)
        assert view["meta"]["session_id"] == sid
        assert view["meta"]["id"] == sid
        assert view["status"] == "idle"
        assert view["messages"][-1]["content"] == "脚本完成"
        events = await service.events(sid)
        assert events and events[0]["type"] == "session.created"
        subscription = service.subscribe(sid)
        assert hasattr(subscription, "__aiter__")
        subscription.close()
        assert await service.stop(sid) == {"ok": True, "status": "idle"}
        assert service.mcp_servers() == []
        assert len(service.registry.names("core")) == 8
        assert len(service.registry.names("subagent")) == 5
    finally:
        await service.close()


async def test_seed_messages_inserted_but_not_persisted(
    settings, store, session_id, workspace
) -> None:
    seed = [
        {"role": "user", "content": "PARENT_CONTEXT"},
        {"role": "assistant", "content": "ACK"},
    ]
    llm = FakeLLM([_result("ok")])
    runner = FakeRunner()
    loop = AgentLoop(
        settings,
        store,
        EventBus(),
        build_core_registry(),
        HookManager(),
        llm,
        ContextManager(settings, llm),
        session_id=session_id,
        workspace=workspace,
        runner=runner,
        seed_messages=seed,
    )
    result = await loop.run_turn("child task")
    assert result["status"] == "completed"
    messages = llm.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    parent_index = next(
        index for index, item in enumerate(messages) if "PARENT_CONTEXT" in str(item.get("content"))
    )
    child_index = next(
        index
        for index, item in enumerate(messages)
        if item.get("role") == "user" and item.get("content") == "child task"
    )
    assert 0 < parent_index < child_index
    state = store.load_state(session_id)
    assert not any("PARENT_CONTEXT" in str(item.get("content")) for item in state.messages)


async def test_get_view_does_not_recover_active_run(settings, tmp_path) -> None:
    script = tmp_path / "active.json"
    script.write_text(
        json.dumps(
            {
                "responses": [
                    {
                        "content": "",
                        "tool_calls": [{"name": "run_code", "arguments": {"code": "print(1)"}}],
                    },
                    {"content": "完成"},
                ],
                "small_responses": [{"content": "摘要"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    scripted = settings.model_copy(
        update={
            "minillm_llm": "scripted",
            "minillm_script": str(script),
            "data_dir": tmp_path / "active-data",
            "workspace_root": tmp_path / "active-ws",
        }
    )
    runner = FakeRunner()
    runner.gate = asyncio.Event()
    service = AgentService(scripted, runner=runner)
    try:
        await service.start()
        meta = service.create_session(title="active")
        sid = meta["session_id"]
        await service.submit_message(sid, "运行")
        await asyncio.wait_for(runner.started.wait(), timeout=2)
        view = await service.get_view(sid)
        assert view["status"] == "running"
        types = [event["type"] for event in await service.events(sid)]
        assert "run_code.finished" not in types
        runner.gate.set()
        await service._active[sid]
        types = [event["type"] for event in await service.events(sid)]
        assert types.count("run_code.finished") == 1
    finally:
        await service.close()
