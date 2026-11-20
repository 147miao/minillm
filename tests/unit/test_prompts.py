"""System prompt 契约测试。"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from minillm.agent.prompts import PROMPT_VERSION, RUN_CODE_TOOL, build_system_prompt
from minillm.tools import CORE_TOOL_NAMES, build_core_registry
from minillm.tools.base import ToolResult, ToolSpec


class _TaskParams(BaseModel):
    task: str


async def _noop(ctx, params):  # noqa: ANN001, ARG001
    return ToolResult(value="ok")


def _registry():
    registry = build_core_registry()
    registry.register(
        ToolSpec(
            name="spawn_subagent",
            description="创建子 agent",
            params=_TaskParams,
            signature="spawn_subagent(task: str) -> dict",
            handler=_noop,
            kind="subagent",
        )
    )
    registry.register(
        ToolSpec(
            name="mcp__demo__echo",
            description="MCP echo",
            params=_TaskParams,
            signature="mcp__demo__echo(task: str) -> str",
            handler=_noop,
            kind="mcp",
        )
    )
    return registry


def test_prompt_version_and_single_tool_contract() -> None:
    assert PROMPT_VERSION == 3
    function = RUN_CODE_TOOL["function"]
    assert function["name"] == "run_code"
    assert set(function["parameters"]["properties"]) == {"code"}
    assert function["parameters"]["additionalProperties"] is False


def test_prompt_contains_core_manual_and_subagent_rules() -> None:
    prompt = build_system_prompt(
        settings=_settings(),
        registry=_registry(),
        workspace=Path("/tmp/ws"),
        active_skills=[{"name": "pytest-debugging"}],
        skill_parts=["SKILL BODY: 先读 traceback"],
        todos=[{"content": "写测试", "status": "in_progress"}],
        context_info={"budget": 1000, "tokens": 120, "level": 0},
    )
    for name in CORE_TOOL_NAMES:
        assert name in prompt
    assert "spawn_subagent" in prompt
    assert "mcp__demo__echo" in prompt
    assert "SKILL BODY: 先读 traceback" in prompt
    assert "写测试" in prompt
    assert "run_code" in prompt
    assert "不能调用其他 function tool" in prompt
    assert "tools.write_file" in prompt
    assert "test-key" not in prompt


def test_subagent_prompt_excludes_subagent_tools() -> None:
    prompt = build_system_prompt(
        settings=_settings(),
        registry=_registry(),
        workspace=Path("/tmp/ws"),
        is_subagent=True,
    )
    assert "spawn_subagent" not in prompt
    assert "不能创建新的子 agent" in prompt
    assert "mcp__demo__echo" in prompt


def test_prompt_summary_and_no_data_dir(tmp_path) -> None:
    settings = _settings().model_copy(update={"data_dir": tmp_path / "secret-data"})
    prompt = build_system_prompt(
        settings=settings,
        registry=_registry(),
        workspace=Path("/tmp/ws"),
        context_info={"budget": 100, "tokens": 90, "level": 3},
        context_summary="历史摘要：已完成 A",
    )
    assert "历史摘要：已完成 A" in prompt
    assert str(settings.data_dir) not in prompt


def _settings():
    from minillm.config import Settings

    return Settings(_env_file=None, llm_api_key="test-key")
