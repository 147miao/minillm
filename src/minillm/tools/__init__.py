"""工具层入口：8 个核心工具与注册表构建。"""

from __future__ import annotations

from . import files, search, shell, todo, web
from .base import (
    AgentKind,
    EmitFn,
    ToolContext,
    ToolKind,
    ToolResult,
    ToolSpec,
    relative_to_workspace,
    resolve_workspace_path,
    workspace_root,
)
from .registry import ToolRegistry, core_kinds

CORE_SPECS: tuple[ToolSpec, ...] = (
    *files.SPECS,
    *search.SPECS,
    *shell.SPECS,
    *web.SPECS,
    *todo.SPECS,
)
CORE_TOOL_NAMES: tuple[str, ...] = tuple(spec.name for spec in CORE_SPECS)


def register_core_tools(registry: ToolRegistry) -> ToolRegistry:
    """把 8 个核心工具注册进给定注册表（重复调用会报错）。"""

    for spec in CORE_SPECS:
        registry.register(spec)
    return registry


def build_core_registry() -> ToolRegistry:
    """构造只含 8 个核心工具的注册表。"""

    return register_core_tools(ToolRegistry())


__all__ = [
    "CORE_SPECS",
    "CORE_TOOL_NAMES",
    "AgentKind",
    "EmitFn",
    "ToolContext",
    "ToolKind",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "build_core_registry",
    "core_kinds",
    "register_core_tools",
    "relative_to_workspace",
    "resolve_workspace_path",
    "workspace_root",
]
