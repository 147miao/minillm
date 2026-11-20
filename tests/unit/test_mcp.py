"""MCP stdio 客户端离线测试。"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from minillm.config import Settings
from minillm.errors import MCPError, ToolError
from minillm.mcp import (
    MCPClient,
    MCPServerConfig,
    build_params_model,
    load_mcp_config,
    register_mcp_tools,
    sanitize_tool_name,
)
from minillm.tools.base import ToolSpec

STUB = Path(__file__).resolve().parents[2] / "scripts" / "mcp_stub_server.py"


def _server(
    name: str = "stub",
    *,
    command: str | None = None,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> dict:
    return {
        "name": name,
        "command": command or sys.executable,
        "args": [str(STUB)] if args is None else args,
        "env": env or {},
        "cwd": cwd,
    }


def _write_config(settings: Settings, servers: list[dict]) -> None:
    settings.mcp_config.parent.mkdir(parents=True, exist_ok=True)
    settings.mcp_config.write_text(
        json.dumps({"servers": servers}, ensure_ascii=False), encoding="utf-8"
    )


def _client(timeout: float = 5.0) -> MCPClient:
    return MCPClient(
        MCPServerConfig(name="stub", command=sys.executable, args=[str(STUB)]),
        request_timeout=timeout,
    )


# ==== 配置 ====
def test_load_config_missing(settings: Settings) -> None:
    assert load_mcp_config(settings.mcp_config) == []


def test_load_config_invalid_json(settings: Settings) -> None:
    settings.mcp_config.write_text("{not-json", encoding="utf-8")
    with pytest.raises(MCPError):
        load_mcp_config(settings.mcp_config)


def test_load_config_valid(settings: Settings) -> None:
    _write_config(
        settings,
        [
            {
                "name": "stub",
                "command": "python3",
                "args": ["scripts/mcp_stub_server.py"],
                "env": {"PYTHONUNBUFFERED": "1"},
                "cwd": ".",
            }
        ],
    )
    configs = load_mcp_config(settings.mcp_config)
    assert len(configs) == 1
    assert configs[0].name == "stub"
    assert configs[0].args == ["scripts/mcp_stub_server.py"]
    assert configs[0].env == {"PYTHONUNBUFFERED": "1"}
    assert configs[0].cwd == "."


@pytest.mark.parametrize(
    "servers",
    [
        [{"command": "python3"}],
        [{"name": "x"}],
        [{"name": "x", "command": "python3", "args": "bad"}],
        [{"name": "x", "command": "python3", "env": {"A": 1}}],
        [
            {"name": "x", "command": "python3"},
            {"name": "x", "command": "python3"},
        ],
    ],
)
def test_load_config_validation(settings: Settings, servers: list[dict]) -> None:
    _write_config(settings, servers)
    with pytest.raises(MCPError):
        load_mcp_config(settings.mcp_config)


def test_sanitize_tool_name() -> None:
    assert sanitize_tool_name("my-srv", "bad.tool") == "mcp__my_srv__bad_tool"
    assert sanitize_tool_name("", "") == "mcp__server__tool"


def test_build_params_model_optional_and_extra() -> None:
    model = build_params_model(
        "srv",
        "opt",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "必填"},
                "nums": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["name"],
            "additionalProperties": True,
        },
    )
    instance = model(name="x", extra=1)
    dumped = instance.model_dump(by_alias=True, exclude_unset=True)
    assert dumped == {"name": "x", "extra": 1}
    with pytest.raises(ValidationError):
        model(nums=[1])
    with pytest.raises(ValidationError):
        model(name="x", nums=["bad"])


# ==== 客户端 ====
async def test_client_handshake_list_and_call() -> None:
    client = _client()
    await client.start()
    try:
        assert client.is_alive
        tools = await client.list_tools()
        names = {item.get("name") for item in tools}
        assert {"echo", "add", "slow"} <= names

        echo = await client.call_tool("echo", {"text": "hi"})
        assert echo["isError"] is False
        assert echo["content"][0]["text"] == "hi"

        add = await client.call_tool("add", {"a": 2, "b": 3})
        assert add["content"][0]["text"] == "5"

        unknown = await client.call_tool("nope", {})
        assert unknown["isError"] is True
    finally:
        await client.close()
    assert client.is_alive is False


async def test_client_close_terminates_process() -> None:
    client = _client()
    await client.start()
    assert client.pid is not None
    await client.close()
    assert client.pid is None
    assert client.is_alive is False


async def test_client_request_timeout() -> None:
    client = _client(timeout=5.0)
    await client.start()
    try:
        client.request_timeout = 0.1
        started = asyncio.get_running_loop().time()
        with pytest.raises(MCPError):
            await client.call_tool("slow", {})
        assert asyncio.get_running_loop().time() - started < 2.0
    finally:
        await client.close()


# ==== 管理器与注册 ====
async def test_manager_registers_and_invokes(settings: Settings, registry, tool_ctx) -> None:
    _write_config(settings, [_server()])
    manager = await register_mcp_tools(registry, settings, request_timeout=5.0)
    try:
        assert manager.servers() == ["stub"]
        assert "mcp__stub__echo" in manager.tool_names
        spec = registry.get("mcp__stub__echo")
        assert spec.kind == "mcp"
        assert "mcp__stub__" in spec.signature

        echo = await registry.invoke("mcp__stub__echo", {"text": "hi"}, tool_ctx)
        assert echo.value == "hi"
        add = await registry.invoke("mcp__stub__add", {"a": 1, "b": 2}, tool_ctx)
        assert add.value == "3"

        with pytest.raises(ToolError) as missing:
            await registry.invoke("mcp__stub__echo", {}, tool_ctx)
        assert missing.value.kind == "validation"
        with pytest.raises(ToolError) as extra:
            await registry.invoke(
                "mcp__stub__echo", {"text": "hi", "extra": 1}, tool_ctx
            )
        assert extra.value.kind == "validation"
    finally:
        await manager.close()


async def test_manager_bad_server_is_non_fatal(
    settings: Settings, registry
) -> None:
    _write_config(
        settings,
        [
            _server(name="bad", command="/nonexistent/mcp-server"),
            _server(name="ok"),
        ],
    )
    manager = await register_mcp_tools(registry, settings, request_timeout=5.0)
    try:
        assert "ok" in manager.servers()
        assert "bad" not in manager.servers()
        assert registry.has("mcp__ok__echo")
    finally:
        await manager.close()


async def test_manager_invalid_config_is_non_fatal(settings: Settings, registry) -> None:
    settings.mcp_config.write_text("{bad-json", encoding="utf-8")
    manager = await register_mcp_tools(registry, settings, request_timeout=1.0)
    try:
        assert manager.servers() == []
    finally:
        await manager.close()


async def test_tool_name_collision_is_skipped(settings: Settings, registry) -> None:
    class _Params(BaseModel):
        text: str = ""

    async def _dummy(_ctx, _params):  # noqa: ANN001 - 测试替身
        return "dummy"

    registry.register(
        ToolSpec(
            name="mcp__stub__echo",
            description="dummy",
            params=_Params,
            signature="mcp__stub__echo(text: str) -> str",
            handler=_dummy,
            kind="core",
        )
    )
    _write_config(settings, [_server()])
    manager = await register_mcp_tools(registry, settings, request_timeout=5.0)
    try:
        assert registry.get("mcp__stub__echo").kind == "core"
        assert "mcp__stub__echo" not in manager.tool_names
        assert "mcp__stub__add" in manager.tool_names
    finally:
        await manager.close()
