"""示例 hook：记录工具开始/结束日志。

把本文件复制到 HOOKS_DIR（默认 ./hooks）即可被 load_hook_modules 自动加载。
"""

from __future__ import annotations

import logging
from typing import Any

from minillm.hooks import HookContext, HookEvent

logger = logging.getLogger("minillm.example.audit")


async def before_tool(ctx: HookContext, payload: dict[str, Any]) -> None:
    logger.info(
        "tool start session=%s agent=%s tool=%s args=%s",
        ctx.session_id,
        ctx.agent_id or "main",
        payload.get("tool"),
        payload.get("args"),
    )


async def after_tool(ctx: HookContext, payload: dict[str, Any]) -> None:
    result = payload.get("result")
    logger.info(
        "tool end session=%s agent=%s tool=%s ok=%s",
        ctx.session_id,
        ctx.agent_id or "main",
        payload.get("tool"),
        getattr(result, "ok", None),
    )


def register(manager: Any) -> None:
    """插件入口：注册本示例的两个 hook。"""

    manager.register(HookEvent.BEFORE_TOOL, before_tool)
    manager.register(HookEvent.AFTER_TOOL, after_tool)
