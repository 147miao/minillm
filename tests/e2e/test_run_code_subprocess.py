"""run_code 子进程执行的端到端测试（真实 worker + 真实工具）。"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import time
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from minillm.runtime.executor import RunCodeExecutor
from minillm.tools import build_core_registry
from minillm.tools.base import ToolSpec


def _ctx(tool_ctx, settings, emitter):
    return dataclasses.replace(tool_ctx, settings=settings, emit=emitter)


def _types(emitter) -> list[str]:
    return [event_type for event_type, _ in emitter.events]


async def test_execute_core_tools_and_events(
    settings, tool_ctx, emitter, registry, workspace: Path
) -> None:
    ctx = _ctx(tool_ctx, settings, emitter)
    code = (
        'print(tools.write_file("a.txt", "hello")["bytes"])\n'
        'print(tools.read_file("a.txt"))'
    )
    result = await RunCodeExecutor(settings, registry).execute(
        code=code, call_id="call_1", ctx=ctx
    )
    assert result.exit_code == 0, result.content
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "hello"
    assert "hello" in result.stdout
    assert result.content.startswith("exit_code: 0")
    assert "write_file" in result.content
    assert "read_file" in result.content
    assert len(result.tool_calls) == 2

    types = _types(emitter)
    assert types[0] == "run_code.started"
    assert types[-1] == "run_code.finished"
    assert types.count("tool.call") == 2, types
    assert types.count("tool.result") == 2, types
    tool_calls = [data for kind, data in emitter.events if kind == "tool.call"]
    tool_results = [data for kind, data in emitter.events if kind == "tool.result"]
    for call, result_event in zip(tool_calls, tool_results, strict=False):
        assert call["call_id"] == result_event["call_id"]
        assert call["run_call_id"] == "call_1"
        assert result_event["ok"] is True

    finished = emitter.events[-1][1]
    assert finished["call_id"] == "call_1"
    assert finished["synthetic"] is False
    assert finished["exit_code"] == 0
    assert "hello" in finished["content"]


async def test_positional_and_keyword_tool_args(
    settings, tool_ctx, emitter, registry, workspace: Path
) -> None:
    (workspace / "a.txt").write_text("line1\nline2\n", encoding="utf-8")
    ctx = _ctx(tool_ctx, settings, emitter)
    code = (
        'print(tools.read_file("a.txt", 1, 1))\n'
        'print(tools.read_file(path="a.txt", limit=1))'
    )
    result = await RunCodeExecutor(settings, registry).execute(
        code=code, call_id="call_args", ctx=ctx
    )
    assert result.exit_code == 0, result.content
    assert "line1" in result.stdout
    assert len(result.tool_calls) == 2


async def test_child_env_strips_secrets(
    monkeypatch, settings, tool_ctx, emitter, registry
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "super-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "super-openai")
    ctx = _ctx(tool_ctx, settings, emitter)
    code = (
        'import os\n'
        'print(os.environ.get("LLM_API_KEY", "MISSING"))\n'
        'print(os.environ.get("OPENAI_API_KEY", "MISSING"))'
    )
    result = await RunCodeExecutor(settings, registry).execute(
        code=code, call_id="call_env", ctx=ctx
    )
    assert result.exit_code == 0, result.content
    assert "MISSING" in result.stdout
    assert "super-secret" not in result.stdout
    assert "super-openai" not in result.stdout
    assert "super-secret" not in result.content


async def test_tool_error_can_be_caught(
    settings, tool_ctx, emitter, registry
) -> None:
    ctx = _ctx(tool_ctx, settings, emitter)
    code = (
        'try:\n'
        '    tools.read_file("missing.txt")\n'
        'except ToolCallError as exc:\n'
        '    print("caught", exc.kind, exc.tool)'
    )
    result = await RunCodeExecutor(settings, registry).execute(
        code=code, call_id="call_caught", ctx=ctx
    )
    assert result.exit_code == 0, result.content
    assert "caught" in result.stdout
    assert "read_file" in result.stdout
    failed = [data for kind, data in emitter.events if kind == "tool.result" and not data["ok"]]
    assert failed and failed[0]["error"]["tool"] == "read_file"


async def test_uncaught_tool_error_and_syntax_error(
    settings, tool_ctx, emitter, registry
) -> None:
    ctx = _ctx(tool_ctx, settings, emitter)
    executor = RunCodeExecutor(settings, registry)

    result = await executor.execute(
        code='tools.read_file("missing.txt")', call_id="call_uncaught", ctx=ctx
    )
    assert result.exit_code == 1
    assert "ToolCallError" in result.stderr or "missing" in result.stderr

    result = await executor.execute(
        code="if True print(1)", call_id="call_syntax", ctx=ctx
    )
    assert result.exit_code == 1
    assert "SyntaxError" in result.stderr


async def test_output_truncation_spills_artifact(
    settings, tool_ctx, emitter, registry
) -> None:
    fast = settings.model_copy(update={"run_code_output_chars": 120})
    ctx = _ctx(tool_ctx, fast, emitter)
    code = 'print("x" * 500)'
    result = await RunCodeExecutor(fast, registry).execute(
        code=code, call_id="call_truncate", ctx=ctx
    )
    assert result.exit_code == 0
    assert result.truncated is True
    assert result.artifact is not None
    assert "[truncated; full output:" in result.content
    full = ctx.artifacts.read_text(result.artifact)
    assert "x" * 500 in full


async def test_idle_timeout_without_tool_calls(
    settings, tool_ctx, emitter, registry
) -> None:
    fast = settings.model_copy(
        update={"run_code_idle_timeout": 1, "run_code_max_timeout": 5}
    )
    ctx = _ctx(tool_ctx, fast, emitter)
    code = 'import time\ntime.sleep(3)\nprint("never")'
    started = time.monotonic()
    result = await RunCodeExecutor(fast, registry).execute(
        code=code, call_id="call_timeout", ctx=ctx
    )
    assert result.timed_out is True
    assert result.exit_code == -1
    assert time.monotonic() - started < 3.0
    finished = emitter.events[-1][1]
    assert finished["timed_out"] is True


class _SleepParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seconds: float = 2.0


async def _sleepy(ctx, params: _SleepParams) -> str:
    await asyncio.sleep(params.seconds)
    return "slept"


def _sleepy_registry():
    registry = build_core_registry()
    registry.register(
        ToolSpec(
            name="sleepy",
            description="sleep then return",
            params=_SleepParams,
            signature="sleepy(seconds: float = 2.0) -> str",
            handler=_sleepy,
        )
    )
    return registry


async def test_tool_inflight_pauses_idle_timeout(
    settings, tool_ctx, emitter
) -> None:
    """工具调用进行中不触发空闲超时（wait_agent 300s 的真实场景）。"""

    fast = settings.model_copy(
        update={"run_code_idle_timeout": 1, "run_code_max_timeout": 10}
    )
    ctx = _ctx(tool_ctx, fast, emitter)
    registry = _sleepy_registry()
    started = time.monotonic()
    result = await RunCodeExecutor(fast, registry).execute(
        code="print(tools.sleepy(2))", call_id="call_inflight", ctx=ctx
    )
    elapsed = time.monotonic() - started
    assert result.timed_out is False, result.content
    assert result.exit_code == 0
    assert "slept" in result.stdout
    assert elapsed >= 2.0
    assert result.tool_calls[0]["name"] == "sleepy"


async def test_cancel_kills_worker(
    settings, tool_ctx, emitter, registry, workspace: Path
) -> None:
    fast = settings.model_copy(
        update={"run_code_idle_timeout": 20, "run_code_max_timeout": 20}
    )
    ctx = _ctx(tool_ctx, fast, emitter)
    executor = RunCodeExecutor(fast, registry)
    code = (
        'import os, time\n'
        'open("pid.txt", "w").write(str(os.getpid()))\n'
        "time.sleep(10)"
    )
    task = asyncio.create_task(
        executor.execute(code=code, call_id="call_cancel", ctx=ctx)
    )
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    pid_text = (workspace / "pid.txt").read_text(encoding="utf-8").strip()
    pid = int(pid_text)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.1)
    else:
        pytest.fail(f"worker {pid} 仍然存活")
    finished = [data for kind, data in emitter.events if kind == "run_code.finished"]
    assert finished and finished[-1]["cancelled"] is True


def test_subagent_tools_filtered_from_worker(settings, tool_ctx, registry) -> None:
    from minillm.tools.base import ToolResult

    class _P(BaseModel):
        model_config = ConfigDict(extra="forbid")

    async def _noop(ctx, params: _P) -> ToolResult:
        return ToolResult(ok=True, value=None)

    registry.register(
        ToolSpec(
            name="spawn_subagent",
            description="fake",
            params=_P,
            signature="spawn_subagent() -> dict",
            handler=_noop,
            kind="subagent",
        )
    )
    executor = RunCodeExecutor(settings, registry)
    assert "spawn_subagent" not in executor._tools_map(tool_ctx)
    allowed = dataclasses.replace(tool_ctx, allow_subagents=True)
    assert "spawn_subagent" in executor._tools_map(allowed)
