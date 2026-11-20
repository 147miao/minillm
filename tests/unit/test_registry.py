"""ToolRegistry 与 ToolSpec 单元测试。"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, Field

from minillm.errors import ToolError
from minillm.tools import CORE_TOOL_NAMES, build_core_registry
from minillm.tools.base import ToolResult, ToolSpec
from minillm.tools.registry import ToolRegistry


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)


async def _echo(ctx, params: _Params) -> str:
    return params.text


async def _boom(ctx, params: _Params) -> str:
    raise RuntimeError("boom")


async def _tool_error(ctx, params: _Params) -> str:
    raise ToolError("nope", kind="execution")


async def _result(ctx, params: _Params) -> ToolResult:
    return ToolResult(ok=True, value="ok", meta={"x": 1})


async def _long(ctx, params: _Params) -> str:
    return params.text


def _spec(name: str = "echo", **overrides) -> ToolSpec:
    data = {
        "name": name,
        "description": "echo back",
        "params": _Params,
        "signature": f"{name}(text: str) -> str",
        "handler": _echo,
    }
    data.update(overrides)
    return ToolSpec(**data)


def test_register_and_duplicate() -> None:
    registry = ToolRegistry()
    registry.register(_spec())
    assert registry.has("echo")
    assert registry.get("echo").name == "echo"
    with pytest.raises(ValueError, match="重复"):
        registry.register(_spec())
    with pytest.raises(ValueError, match="非法工具名"):
        registry.register(_spec(name="bad name"))
    with pytest.raises(TypeError):
        registry.register("not-a-spec")  # type: ignore[arg-type]


def test_get_unknown_and_validate() -> None:
    registry = ToolRegistry()
    registry.register(_spec())
    with pytest.raises(ToolError) as unknown:
        registry.get("missing")
    assert unknown.value.kind == "validation"

    params = registry.validate("echo", {"text": "hi"})
    assert params.text == "hi"
    with pytest.raises(ToolError) as missing:
        registry.validate("echo", {})
    assert missing.value.kind == "validation"
    assert missing.value.tool == "echo"
    assert any("text" in error["loc"] for error in missing.value.errors)

    with pytest.raises(ToolError) as extra:
        registry.validate("echo", {"text": "hi", "extra": 1})
    assert extra.value.kind == "validation"

    with pytest.raises(ToolError):
        registry.validate("echo", ["not", "dict"])  # type: ignore[arg-type]


def test_manuals_and_filters() -> None:
    registry = ToolRegistry()
    registry.register(_spec())
    registry.register(
        _spec(name="helper", kind="subagent", description="sub helper")
    )
    all_manual = registry.manuals()
    assert "echo(text: str)" in all_manual
    assert "helper(text: str)" in all_manual
    only_sub = registry.manuals(kinds="subagent")
    assert "helper" in only_sub
    assert "echo" not in only_sub
    assert registry.names() == ["echo", "helper"]
    assert registry.names(kinds="core") == ["echo"]
    assert len(registry.list_specs()) == 2
    assert [spec.name for spec in registry.all_specs] == ["echo", "helper"]


async def test_invoke_success(tool_ctx) -> None:
    registry = ToolRegistry()
    registry.register(_spec())
    result = await registry.invoke("echo", {"text": "hello"}, tool_ctx)
    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.value == "hello"


async def test_handler_exceptions_are_wrapped(tool_ctx) -> None:
    registry = ToolRegistry()
    registry.register(_spec(handler=_boom))
    with pytest.raises(ToolError) as wrapped:
        await registry.invoke("echo", {"text": "x"}, tool_ctx)
    assert wrapped.value.kind == "execution"
    assert wrapped.value.tool == "echo"

    registry2 = ToolRegistry()
    registry2.register(_spec(handler=_tool_error))
    with pytest.raises(ToolError) as direct:
        await registry2.invoke("echo", {"text": "x"}, tool_ctx)
    assert direct.value.kind == "execution"
    assert direct.value.tool == "echo"


async def test_tool_result_passthrough(tool_ctx) -> None:
    registry = ToolRegistry()
    registry.register(_spec(handler=_result))
    result = await registry.invoke("echo", {"text": "x"}, tool_ctx)
    assert result.value == "ok"
    assert result.meta == {"x": 1}


async def test_plain_string_is_capped_once(small_ctx) -> None:
    registry = ToolRegistry()
    registry.register(_spec(handler=_long))
    result = await registry.invoke("echo", {"text": "x" * 200}, small_ctx)
    assert result.artifact is not None
    assert result.value.endswith(f"full output: {result.artifact}]")
    assert small_ctx.artifacts.read_text(result.artifact) == "x" * 200


async def test_tool_result_not_double_capped(small_ctx) -> None:
    registry = ToolRegistry()
    registry.register(_spec(handler=_result))
    result = await registry.invoke("echo", {"text": "x"}, small_ctx)
    assert result.value == "ok"
    assert result.artifact is None


async def test_tool_result_capped_without_artifact(small_ctx) -> None:
    async def huge(ctx, params: _Params) -> ToolResult:
        return ToolResult(ok=True, value="y" * 200)

    registry = ToolRegistry()
    registry.register(_spec(handler=huge))
    result = await registry.invoke("echo", {"text": "x"}, small_ctx)
    assert result.artifact is not None
    assert result.value.endswith(f"full output: {result.artifact}]")
    assert small_ctx.artifacts.read_text(result.artifact) == "y" * 200


def test_core_registry_has_eight_tools() -> None:
    registry = build_core_registry()
    assert tuple(registry.names()) == CORE_TOOL_NAMES
    assert len(CORE_TOOL_NAMES) == 8
    assert set(registry.names(kinds="core")) == {
        "read_file",
        "write_file",
        "edit_file",
        "list_files",
        "search_files",
        "run_bash",
        "web_fetch",
        "todo_write",
    }
