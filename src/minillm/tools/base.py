"""工具层基础类型、路径校验与上下文。"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from ..artifacts import ArtifactStore
from ..config import Settings
from ..errors import ToolError
from ..locks import FileLockManager

logger = logging.getLogger(__name__)

EmitFn = Callable[[str, dict[str, Any]], Awaitable[None]]
ToolKind = Literal["core", "subagent", "mcp"]
AgentKind = Literal["main", "subagent"]


@dataclass(slots=True)
class ToolContext:
    """一次工具调用所需的全部运行期依赖。"""

    session_id: str
    workspace: Path
    data_dir: Path
    settings: Settings
    locks: FileLockManager
    artifacts: ArtifactStore
    agent_id: str = "main"
    kind: AgentKind = "main"
    allow_subagents: bool = False
    emit: EmitFn | None = None
    call_id: str | None = None

    async def emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """向会话事件流发一条审计事件；没有订阅者时静默跳过。"""

        if self.emit is None:
            return
        try:
            await self.emit(event_type, data)
        except Exception:  # pragma: no cover - 审计失败不影响工具结果
            logger.exception("工具事件广播失败: %s", event_type)


@dataclass(slots=True)
class ToolResult:
    """工具执行结果；value 必须是可 JSON 序列化的 Python 值。"""

    ok: bool = True
    value: Any = None
    meta: dict[str, Any] = field(default_factory=dict)
    artifact: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "value": self.value,
            "meta": self.meta,
            "artifact": self.artifact,
        }


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """工具注册表条目。"""

    name: str
    description: str
    params: type[BaseModel]
    signature: str
    handler: Callable[[ToolContext, Any], Awaitable[Any]]
    kind: ToolKind = "core"
    parallel_safe: bool = False


def workspace_root(ctx: ToolContext, *, tool: str) -> Path:
    """返回已解析的 workspace 根目录。"""

    try:
        root = Path(ctx.workspace).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ToolError(
            f"workspace 不存在或不可访问：{ctx.workspace}",
            kind="execution",
            tool=tool,
        ) from exc
    if not root.is_dir():
        raise ToolError(f"workspace 不是目录：{ctx.workspace}", kind="execution", tool=tool)
    return root


def resolve_workspace_path(
    ctx: ToolContext,
    raw: str | None,
    *,
    tool: str,
    must_exist: bool = False,
    expect_dir: bool = False,
) -> Path:
    """把用户/模型给的路径解析为 workspace 内的真实路径。

    任何 ..、绝对路径或符号链接逃逸到 workspace 之外都会被拒绝。
    """

    root = workspace_root(ctx, tool=tool)
    text = "." if raw is None or str(raw).strip() == "" else str(raw).strip()
    try:
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        target = candidate.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ToolError(f"路径非法：{raw!r}", kind="validation", tool=tool) from exc

    if target != root and root not in target.parents:
        raise ToolError(f"路径超出 workspace：{raw}", kind="validation", tool=tool)
    if must_exist and not target.exists():
        raise ToolError(f"路径不存在：{raw}", kind="execution", tool=tool)
    if expect_dir and not target.is_dir():
        raise ToolError(f"路径不是目录：{raw}", kind="execution", tool=tool)
    return target


def relative_to_workspace(ctx: ToolContext, target: Path, *, tool: str) -> str:
    """返回相对 workspace 的 POSIX 路径；根目录返回点。"""

    root = workspace_root(ctx, tool=tool)
    try:
        rel = target.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise ToolError(f"路径超出 workspace：{target}", kind="validation", tool=tool) from exc
    text = rel.as_posix()
    return "." if text in {"", "."} else text
