"""todo_write：会话级待办清单（全量覆盖）。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..events import EventType
from .base import ToolContext, ToolResult, ToolSpec


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TodoItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=2000)
    status: Literal["pending", "in_progress", "completed"] = "pending"


class TodoWriteParams(_Params):
    items: list[TodoItem] = Field(default_factory=list, max_length=100)


async def todo_write(ctx: ToolContext, params: TodoWriteParams) -> ToolResult:
    items = [item.model_dump() for item in params.items]
    await ctx.emit_event(EventType.TODO_UPDATED, {"items": items})
    return ToolResult(ok=True, value=items, meta={"count": len(items)})


SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="todo_write",
        description="全量覆盖会话待办清单（pending/in_progress/completed）",
        params=TodoWriteParams,
        signature="todo_write(items: list[{content: str, status: str}]) -> list[dict]",
        handler=todo_write,
    ),
)
