"""LLM 契约：只暴露 run_code、scripted 工厂与 bootstrap 计数。"""

from __future__ import annotations

import json
from typing import Any

from minillm.agent.context import ContextManager
from minillm.agent.loop import AgentLoop, RunCodeResult
from minillm.bootstrap import build_llm_clients, build_registry
from minillm.hooks import HookManager
from minillm.llm.types import LLMDone, LLMResult, LLMTextDelta
from minillm.tools import build_core_registry


class FakeLLM:
    def __init__(self, result: LLMResult) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def stream_chat(self, **kwargs: Any):
        self.calls.append(dict(kwargs))
        if self.result.content:
            yield LLMTextDelta(self.result.content)
        yield LLMDone(self.result)

    async def complete(self, **kwargs: Any) -> LLMResult:
        return LLMResult(content="summary")


class FakeRunner:
    async def execute(self, *, code: str, call_id: str, ctx: Any) -> RunCodeResult:
        return RunCodeResult(exit_code=0, stdout=code, content=code)


async def test_llm_only_sees_run_code(settings, store, session_id, workspace) -> None:
    llm = FakeLLM(LLMResult(content="done", finish_reason="stop"))
    loop = AgentLoop(
        settings,
        store,
        __import__("minillm.events", fromlist=["EventBus"]).EventBus(),
        build_core_registry(),
        HookManager(),
        llm,
        ContextManager(settings, llm),
        session_id=session_id,
        workspace=workspace,
        runner=FakeRunner(),
    )
    result = await loop.run_turn("hi")
    assert result["status"] == "completed"
    assert len(llm.calls) == 1
    tools = llm.calls[0]["tools"]
    assert len(tools) == 1
    function = tools[0]["function"]
    assert function["name"] == "run_code"
    assert set(function["parameters"]["properties"]) == {"code"}
    assert function["parameters"]["additionalProperties"] is False
    assert llm.calls[0]["agent_kind"] == "main"


def test_bootstrap_registry_has_core_and_subagent_specs(settings) -> None:
    registry = build_registry(settings)
    assert len(registry.names("core")) == 8
    assert len(registry.names("subagent")) == 5


def test_scripted_factory(settings, tmp_path) -> None:
    script = tmp_path / "script.json"
    script.write_text(
        json.dumps(
            {"responses": [{"content": "x"}], "small_responses": [{"content": "s"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    scripted = settings.model_copy(
        update={"minillm_llm": "scripted", "minillm_script": str(script)}
    )
    main, small = build_llm_clients(scripted)
    assert main is small
    assert type(main).__name__ == "ScriptedLLMClient"
