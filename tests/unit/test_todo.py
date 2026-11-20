"""todo_write 工具测试：状态枚举、覆盖语义、事件。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from minillm.errors import ToolError


async def test_todo_write_valid(registry, tool_ctx) -> None:
    result = await registry.invoke(
        "todo_write",
        {
            "items": [
                {"content": "read code", "status": "completed"},
                {"content": "fix bug", "status": "in_progress"},
                {"content": "run tests", "status": "pending"},
            ]
        },
        tool_ctx,
    )
    assert result.value == [
        {"content": "read code", "status": "completed"},
        {"content": "fix bug", "status": "in_progress"},
        {"content": "run tests", "status": "pending"},
    ]
    assert result.meta == {"count": 3}


async def test_todo_write_replaces(registry, tool_ctx) -> None:
    await registry.invoke(
        "todo_write", {"items": [{"content": "a", "status": "pending"}]}, tool_ctx
    )
    result = await registry.invoke(
        "todo_write", {"items": [{"content": "b", "status": "completed"}]}, tool_ctx
    )
    assert result.value == [{"content": "b", "status": "completed"}]
    cleared = await registry.invoke("todo_write", {"items": []}, tool_ctx)
    assert cleared.value == []


async def test_todo_write_invalid_status(registry, tool_ctx) -> None:
    with pytest.raises(ToolError) as invalid:
        await registry.invoke(
            "todo_write", {"items": [{"content": "x", "status": "doing"}]}, tool_ctx
        )
    assert invalid.value.kind == "validation"


async def test_todo_write_empty_content(registry, tool_ctx) -> None:
    with pytest.raises(ToolError) as invalid:
        await registry.invoke(
            "todo_write", {"items": [{"content": "", "status": "pending"}]}, tool_ctx
        )
    assert invalid.value.kind == "validation"


async def test_todo_write_extra_field(registry, tool_ctx) -> None:
    with pytest.raises(ToolError):
        await registry.invoke(
            "todo_write",
            {"items": [{"content": "x", "status": "pending", "extra": 1}]},
            tool_ctx,
        )


async def test_todo_write_emits_event(registry, tool_ctx, emitter) -> None:
    ctx = replace(tool_ctx, emit=emitter)
    await registry.invoke(
        "todo_write", {"items": [{"content": "x", "status": "in_progress"}]}, ctx
    )
    assert emitter.events == [
        ("todo.updated", {"items": [{"content": "x", "status": "in_progress"}]})
    ]
