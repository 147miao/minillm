"""子 agent 管理工具：spawn / send / wait / list / stop。

这些工具不暴露给 LLM 的 function tools，而是由模型写的 Python 程序通过
§tools.spawn_subagent(...)§ 这类调用使用；ToolContext.allow_subagents 控制可用性。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..agent.subagents import SubagentError, SubagentManager
from ..errors import ToolError
from .base import ToolContext, ToolSpec
from .registry import ToolRegistry


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SpawnSubagentParams(_StrictModel):
    task: str = Field(min_length=1, description="交给子 agent 的任务")
    description: str | None = Field(default=None, description="短标题，便于 UI 展示")
    inherit_context: bool = Field(
        default=False, description="是否把父会话当前上下文交给子 agent"
    )


class SendMessageParams(_StrictModel):
    agent_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class WaitAgentParams(_StrictModel):
    agent_id: str | None = Field(default=None)
    timeout: int = Field(default=300, ge=1, le=600)


class ListAgentsParams(_StrictModel):
    pass


class StopAgentParams(_StrictModel):
    agent_id: str = Field(min_length=1)


SUBAGENT_TOOL_NAMES: tuple[str, ...] = (
    "spawn_subagent",
    "send_message",
    "wait_agent",
    "list_agents",
    "stop_agent",
)


def _manager_for(manager: SubagentManager | None, tool: str) -> SubagentManager:
    if manager is None:
        raise ToolError("子 agent 未启用（当前进程没有 SubagentManager）", kind="denied", tool=tool)
    return manager


def _check_enabled(ctx: ToolContext, manager: SubagentManager | None, tool: str) -> SubagentManager:
    if not ctx.allow_subagents:
        raise ToolError("子 agent 工具在当前上下文不可用", kind="denied", tool=tool)
    return _manager_for(manager, tool)


def _wrap(tool: str, exc: Exception) -> ToolError:
    if isinstance(exc, SubagentError):
        return ToolError(str(exc), kind=exc.kind, tool=tool)
    return ToolError(f"子 agent 操作失败：{exc}", kind="execution", tool=tool)


def _spawn_handler(
    manager: SubagentManager | None,
) -> Callable[[ToolContext, SpawnSubagentParams], Any]:
    async def handler(ctx: ToolContext, params: SpawnSubagentParams) -> dict[str, Any]:
        active = _check_enabled(ctx, manager, "spawn_subagent")
        try:
            return await active.spawn(
                ctx.session_id,
                params.task,
                description=params.description,
                inherit_context=params.inherit_context,
            )
        except SubagentError as exc:
            raise _wrap("spawn_subagent", exc) from exc

    return handler


def _send_handler(
    manager: SubagentManager | None,
) -> Callable[[ToolContext, SendMessageParams], Any]:
    async def handler(ctx: ToolContext, params: SendMessageParams) -> dict[str, Any]:
        active = _check_enabled(ctx, manager, "send_message")
        try:
            return await active.send_message(params.agent_id, params.message)
        except SubagentError as exc:
            raise _wrap("send_message", exc) from exc

    return handler


def _wait_handler(
    manager: SubagentManager | None,
) -> Callable[[ToolContext, WaitAgentParams], Any]:
    async def handler(ctx: ToolContext, params: WaitAgentParams) -> dict[str, Any]:
        active = _check_enabled(ctx, manager, "wait_agent")
        try:
            return await active.wait(params.agent_id, timeout=float(params.timeout))
        except SubagentError as exc:
            raise _wrap("wait_agent", exc) from exc

    return handler


def _list_handler(
    manager: SubagentManager | None,
) -> Callable[[ToolContext, ListAgentsParams], Any]:
    async def handler(ctx: ToolContext, params: ListAgentsParams) -> list[dict[str, Any]]:
        active = _check_enabled(ctx, manager, "list_agents")
        return active.list_agents(ctx.session_id)

    return handler


def _stop_handler(
    manager: SubagentManager | None,
) -> Callable[[ToolContext, StopAgentParams], Any]:
    async def handler(ctx: ToolContext, params: StopAgentParams) -> dict[str, Any]:
        active = _check_enabled(ctx, manager, "stop_agent")
        try:
            return await active.stop(params.agent_id)
        except SubagentError as exc:
            raise _wrap("stop_agent", exc) from exc

    return handler


def build_subagent_specs(manager: SubagentManager | None = None) -> list[ToolSpec]:
    """构造 5 个子 agent 工具；manager=None 时 handler 报“未启用”。"""

    return [
        ToolSpec(
            name="spawn_subagent",
            description="创建一个独立会话的子 agent，异步执行任务；可选是否继承父上下文。",
            params=SpawnSubagentParams,
            signature=(
                "spawn_subagent(task: str, description: str | None = None, "
                "inherit_context: bool = False) -> dict"
            ),
            handler=_spawn_handler(manager),
            kind="subagent",
        ),
        ToolSpec(
            name="send_message",
            description="给子 agent 发消息；空闲时开新轮，运行中排队。",
            params=SendMessageParams,
            signature="send_message(agent_id: str, message: str) -> dict",
            handler=_send_handler(manager),
            kind="subagent",
        ),
        ToolSpec(
            name="wait_agent",
            description="等待指定/任一子 agent 当前轮结束；超时返回 running。",
            params=WaitAgentParams,
            signature=(
                "wait_agent(agent_id: str | None = None, timeout: int = 300) -> dict"
            ),
            handler=_wait_handler(manager),
            kind="subagent",
        ),
        ToolSpec(
            name="list_agents",
            description="列出当前会话的子 agent 及状态、任务与结果摘要。",
            params=ListAgentsParams,
            signature="list_agents() -> list[dict]",
            handler=_list_handler(manager),
            kind="subagent",
        ),
        ToolSpec(
            name="stop_agent",
            description="停止子 agent 当前任务并标记 stopped。",
            params=StopAgentParams,
            signature="stop_agent(agent_id: str) -> dict",
            handler=_stop_handler(manager),
            kind="subagent",
        ),
    ]


def register_subagent_tools(
    registry: ToolRegistry, manager: SubagentManager | None
) -> ToolRegistry:
    """把 5 个子 agent 工具注册进注册表；manager 可为 None 供 CLI 列举。"""

    for spec in build_subagent_specs(manager):
        registry.register(spec)
    return registry


__all__ = [
    "SUBAGENT_TOOL_NAMES",
    "build_subagent_specs",
    "register_subagent_tools",
]
