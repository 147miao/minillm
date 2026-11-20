"""运行时装配：工具注册表、Hooks、LLM 客户端与 executor。"""

from __future__ import annotations

import logging
from typing import Any

from .config import Settings
from .errors import ConfigError
from .hooks import HookManager, load_hook_modules, register_builtin_hooks
from .llm.client import OpenAICompatClient
from .llm.scripted import ScriptedLLMClient
from .llm.types import LLMClient
from .tools import build_core_registry
from .tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def build_registry(
    settings: Settings,
    *,
    subagent_manager: Any | None = None,
) -> ToolRegistry:
    """8 个核心工具 + 5 个子 agent 工具。

    - 传 manager：注册可用的 5 个工具；
    - 不传 manager：注册占位 specs，供 CLI tools list / 测试统计（调用会报未启用）。
    MCP 工具由 AgentService 的 async 启动阶段注册。
    """

    registry = build_core_registry()
    try:
        from .tools.subagent_tools import build_subagent_specs, register_subagent_tools
    except ImportError as exc:  # pragma: no cover - WP6 未落地时
        logger.warning("子 agent 工具模块不可用：%s", exc)
        return registry

    if subagent_manager is not None:
        register_subagent_tools(registry, subagent_manager)
    else:
        for spec in build_subagent_specs(None):
            registry.register(spec)
    return registry


def build_hooks(settings: Settings) -> HookManager:
    manager = HookManager()
    register_builtin_hooks(manager, settings)
    try:
        load_hook_modules(manager, settings.hooks_dir)
    except Exception:  # noqa: BLE001 - 插件失败不阻塞启动
        logger.exception("加载用户 Hooks 失败：%s", settings.hooks_dir)
    return manager


def build_llm_clients(settings: Settings) -> tuple[LLMClient, LLMClient]:
    """按 settings.minillm_llm 构造主/小模型；scripted 时二者共用一个实例。"""

    if settings.minillm_llm == "scripted":
        client = ScriptedLLMClient(settings)
        return client, client
    main = OpenAICompatClient(settings)
    small = OpenAICompatClient.for_small(settings)
    return main, small


def build_context_manager(settings: Settings, small_llm: LLMClient) -> Any:
    from .agent.context import ContextManager

    return ContextManager(settings, small_llm)


def build_runner(settings: Settings, registry: ToolRegistry) -> Any:
    try:
        from .runtime.executor import RunCodeExecutor
    except ImportError as exc:  # pragma: no cover - WP3 未落地时
        raise ConfigError("run_code executor 尚未就绪（缺少 minillm.runtime.executor）") from exc
    return RunCodeExecutor(settings, registry)
