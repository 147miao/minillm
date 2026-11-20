"""工具注册表：注册、参数校验、统一执行与手册生成。"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ValidationError

from ..errors import ToolError
from .base import ToolContext, ToolKind, ToolResult, ToolSpec

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ToolRegistry:
    """工具注册表。

    注册表只负责校验与执行；Hooks 由 AgentService 在调用前后统一处理，
    避免工具层与扩展点互相耦合。
    """

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    # ==== 注册与查询 ====
    def register(self, spec: ToolSpec) -> None:
        if not isinstance(spec, ToolSpec):
            raise TypeError("register 只接受 ToolSpec")
        if not _NAME_RE.fullmatch(spec.name):
            raise ValueError(f"非法工具名：{spec.name!r}")
        if spec.name in self._specs:
            raise ValueError(f"工具重复注册：{spec.name}")
        if not issubclass(spec.params, BaseModel):
            raise TypeError(f"工具 {spec.name} 的 params 必须是 pydantic 模型")
        if not callable(spec.handler):
            raise TypeError(f"工具 {spec.name} 的 handler 不可调用")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        spec = self._specs.get(name)
        if spec is None:
            raise ToolError(f"未知工具：{name}", kind="validation", tool=name)
        return spec

    def has(self, name: str) -> bool:
        return name in self._specs

    def list_specs(self, kinds: Iterable[str] | str | None = None) -> list[ToolSpec]:
        if kinds is None:
            return list(self._specs.values())
        wanted = {kinds} if isinstance(kinds, str) else set(kinds)
        return [spec for spec in self._specs.values() if spec.kind in wanted]

    @property
    def all_specs(self) -> list[ToolSpec]:
        """全部工具的只读快照（注册顺序）。"""

        return list(self._specs.values())

    def names(self, kinds: Iterable[str] | str | None = None) -> list[str]:
        return [spec.name for spec in self.list_specs(kinds)]

    def manuals(self, kinds: Iterable[str] | str | None = None) -> str:
        """给系统提示用的紧凑工具手册。"""

        return "\n".join(
            f"- {spec.signature}: {spec.description}" for spec in self.list_specs(kinds)
        )

    # ==== 校验 ====
    def validate(self, name: str, raw_args: Any) -> BaseModel:
        spec = self.get(name)
        if raw_args is None:
            raw_args = {}
        if not isinstance(raw_args, dict):
            raise ToolError(
                "工具参数必须是 JSON 对象",
                kind="validation",
                tool=name,
                errors=[{"loc": [], "msg": "必须是对象", "type": "type_error"}],
            )
        try:
            return spec.params.model_validate(raw_args)
        except ValidationError as exc:
            raise ToolError(
                f"工具参数校验失败：{name}",
                kind="validation",
                tool=name,
                errors=_format_errors(exc),
            ) from exc

    # ==== 执行 ====
    async def invoke(self, name: str, raw_args: Any, ctx: ToolContext) -> ToolResult:
        spec = self.get(name)
        params = self.validate(name, raw_args)
        try:
            raw_result = await spec.handler(ctx, params)
        except ToolError as exc:
            if exc.tool is None:
                exc.tool = name
            raise
        except Exception as exc:
            raise ToolError(
                f"工具执行失败：{exc}",
                kind="execution",
                tool=name,
            ) from exc

        if isinstance(raw_result, ToolResult):
            result = raw_result
        else:
            result = ToolResult(ok=True, value=raw_result)
        if (
            result.ok
            and isinstance(result.value, str)
            and result.artifact is None
            and "artifact" not in result.meta
            and len(result.value) > ctx.artifacts.max_chars
        ):
            clipped, artifact = ctx.artifacts.cap_text(spec.name, result.value)
            if artifact is not None:
                result.value = clipped
                result.artifact = artifact
                result.meta.setdefault("artifact", artifact)
                result.meta.setdefault("truncated", True)
        return result


def _format_errors(exc: ValidationError) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for error in exc.errors():
        formatted.append(
            {
                "loc": [str(part) for part in error.get("loc", ())],
                "msg": str(error.get("msg", "")),
                "type": str(error.get("type", "")),
            }
        )
    return formatted


def core_kinds(specs: Iterable[ToolSpec]) -> list[ToolSpec]:
    """便捷函数：筛选 core 工具。"""

    return [spec for spec in specs if spec.kind == "core"]


__all__ = [
    "ToolContext",
    "ToolKind",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "core_kinds",
]
