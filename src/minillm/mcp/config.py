"""MCP server 配置加载。

配置文件格式（mcp.json）：
{
  "servers": [
    {"name": "stub", "command": "python3", "args": ["scripts/mcp_stub_server.py"],
     "env": {"PYTHONUNBUFFERED": "1"}, "cwd": "."}
  ]
}

约定：文件不存在返回空列表；JSON/结构非法抛 MCPError，由 app 启动层决定是否致命。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import MCPError


@dataclass(slots=True)
class MCPServerConfig:
    """一个 MCP server 的启动配置。"""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None


def _as_str_list(value: Any, *, where: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise MCPError(f"{where}.args 必须是字符串数组")
    return list(value)


def _as_env(value: Any, *, where: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise MCPError(f"{where}.env 必须是对象")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise MCPError(f"{where}.env 的 key/value 必须是字符串")
        result[key] = item
    return result


def load_mcp_config(path: str | Path | None) -> list[MCPServerConfig]:
    """读取并校验 MCP 配置；文件不存在返回空列表。"""

    if path is None or str(path).strip() == "":
        return []
    config_path = Path(path).expanduser()
    try:
        if not config_path.exists():
            return []
        if not config_path.is_file():
            raise MCPError(f"MCP 配置不是文件：{config_path}")
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MCPError(f"MCP 配置不是合法 JSON：{config_path}") from exc
    except OSError as exc:
        raise MCPError(f"MCP 配置不可读：{config_path}") from exc

    if not isinstance(raw, dict):
        raise MCPError("MCP 配置根节点必须是对象")
    servers = raw.get("servers")
    if servers is None:
        return []
    if not isinstance(servers, list):
        raise MCPError("MCP 配置 servers 必须是数组")

    result: list[MCPServerConfig] = []
    names: set[str] = set()
    for index, item in enumerate(servers):
        where = f"servers[{index}]"
        if not isinstance(item, dict):
            raise MCPError(f"{where} 必须是对象")
        name = str(item.get("name") or "").strip()
        if not name:
            raise MCPError(f"{where}.name 不能为空")
        if name in names:
            raise MCPError(f"{where}.name 重复：{name}")
        command = str(item.get("command") or "").strip()
        if not command:
            raise MCPError(f"{where}.command 不能为空")
        cwd = item.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise MCPError(f"{where}.cwd 必须是字符串或 null")
        result.append(
            MCPServerConfig(
                name=name,
                command=command,
                args=_as_str_list(item.get("args"), where=where),
                env=_as_env(item.get("env"), where=where),
                cwd=cwd,
            )
        )
        names.add(name)
    return result


__all__ = ["MCPServerConfig", "load_mcp_config"]
